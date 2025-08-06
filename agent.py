import asyncio
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
from abc import ABC, abstractmethod
from model import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent


def get_llm():
    return llm


@dataclass
class WorkflowState:
    user_message: str = ""
    messages: List[Any] = field(default_factory=list)
    current_state: str = "code"
    data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 4


class BaseAgent(ABC):
    def __init__(self, name):
        self.name = name
        self.role = "BaseAgent"
        self.llm = get_llm()
    
    @abstractmethod
    async def process(self, state: WorkflowState) -> WorkflowState:
        pass
    
    def add_message(self, state: WorkflowState, msg: str):
        msg = f"{self.name}: {msg}"
        if state.messages is None:
            state.messages = []
        state.messages.append(msg)


class CodingAgent(BaseAgent):
    def __init__(self, name="CodingAgent"):
        super().__init__(name)
        self.role = "CodingAgent"
          
    async def process(self, state: WorkflowState) -> WorkflowState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful coding agent. Your task is to read and understand the user's query, and based on it, generate the necessary code implementation along with relevant test cases.
            
            If previous code and its execution status are provided, use them as a reference to improve or continue the work. If errors occurred, fix them.
            If no prior context is given, generate the code and test cases from scratch according to the user's current request.
            
            Ensure your solution is correct, clean, and follows best practices.
            
            Format your response with clear sections for:
            1. Main Code
            2. Test Cases (if applicable)
            3. Brief explanation
            """),
            ("human", """
            Query: {query}
            Previous Codes: {codes}
            Previous Execution Status: {status}
            Error Message (if any): {error}
            
            Please provide the appropriate code and test cases for the given query.
            """)
        ])
        
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "query": state.user_message,
                "codes": state.data.get('code', 'None'),
                "status": state.data.get('execution_status', 'None'),
                "error": state.data.get('error_message', 'None')
            })
            
            state.data['code'] = response.content
            self.add_message(state, response.content)
            state.current_state = "execute"
        except Exception as e:
            self.add_message(state, f"Code generation failed: {str(e)}")
            state.data['execution_status'] = 'failure'
            state.data['error_message'] = f"Code generation error: {str(e)}"
            state.current_state = "test"
            
        return state


class CodeExecutorAgent(BaseAgent):
    def __init__(self, name="CodeExecutorAgent"):
        super().__init__(name)
        self.role = "ExecutionAgent"
        self._tools_cache = None
    
    async def get_interpreter_tools(self):
        if self._tools_cache is not None:
            return self._tools_cache
            
        try:
            SERVER_URL = "http://127.0.0.1:8000/mcp"
            async with streamablehttp_client(SERVER_URL) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    self._tools_cache = tools
                    return tools
        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            return []
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        try:
            tools = await self.get_interpreter_tools()
            if not tools:
                state.data['execution_status'] = 'failure'
                state.data['error_message'] = 'Could not connect to code execution server'
                self.add_message(state, "Error: Could not connect to code execution server")
                state.current_state = "test"
                return state
            
            agent = create_react_agent(model=self.llm, tools=tools)
            
            try:
                result = await asyncio.wait_for(
                    agent.ainvoke({
                        "messages": [
                            {"role": "system", "content": "Execute the given code and test cases and give status as success or failure. If failure occurs, provide the error message as well."},
                            {"role": "user", "content": f"User query: {state.user_message}\n\nCode to execute: {state.data['code']}"}
                        ]
                    }),
                    timeout=60 
                )
                
                response_content = result["messages"][-1].content
                
                fail_indicators = ['failure', 'failed', 'error', 'exception', 'bug', 'issue', 'traceback']
                if any(indicator.lower() in response_content.lower() for indicator in fail_indicators):
                    state.data['execution_status'] = 'failure'
                    state.data['error_message'] = response_content
                else:
                    state.data['execution_status'] = 'success'
                    state.data['result'] = response_content
                
                self.add_message(state, response_content)
                
            except asyncio.TimeoutError:
                state.data['execution_status'] = 'failure'
                state.data['error_message'] = "Code execution timed out after 60 seconds"
                self.add_message(state, "Execution failed: Timeout after 60 seconds")
                
        except Exception as e:
            state.data['execution_status'] = 'failure'
            state.data['error_message'] = f"Execution error: {str(e)}"
            self.add_message(state, f"Execution failed: {str(e)}")
        
        state.current_state = "test"
        return state


class TestingAgent(BaseAgent):
    def __init__(self, name="TestingAgent"):
        super().__init__(name)
        self.role = "TestingAgent"
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        execution_status = state.data.get("execution_status", "failure")
        
        if execution_status == 'failure':
            state.retry_count += 1
            
            if state.retry_count >= state.max_retries:
                self.add_message(state, f"Maximum retries ({state.max_retries}) reached. Ending workflow.")
                state.current_state = "end"
            else:
                self.add_message(state, f"Execution failed. Retry {state.retry_count}/{state.max_retries}. Going back to code generation.")
                state.current_state = "code"
        else:
            self.add_message(state, "Code executed successfully!")
            state.current_state = "end"
        
        return state


class WorkflowManager:
    def __init__(self):
        self.coding_agent = CodingAgent()
        self.executor_agent = CodeExecutorAgent()
        self.testing_agent = TestingAgent()
    
    def _build_workflow(self):
        workflow = StateGraph(WorkflowState)
        
        workflow.add_node('code', self._code_node)
        workflow.add_node('execute', self._execute_node)
        workflow.add_node('test', self._test_node)
        
        workflow.add_edge(START, 'code')
        workflow.add_edge('code', 'execute')
        workflow.add_edge('execute', 'test')
        
        def decide_next_step(state: WorkflowState) -> str:
            if state.current_state.lower() == "code":
                return "code"
            else:
                return "end"
        
        workflow.add_conditional_edges(
            "test",
            decide_next_step,
            {
                "code": "code",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def _code_node(self, state: WorkflowState) -> WorkflowState:
        return await self.coding_agent.process(state)
    
    async def _execute_node(self, state: WorkflowState) -> WorkflowState:
        return await self.executor_agent.process(state)
    
    async def _test_node(self, state: WorkflowState) -> WorkflowState:
        return await self.testing_agent.process(state)
    
    async def run_workflow(self, user_message: str) -> WorkflowState:
        workflow = self._build_workflow()
        
        initial_state = WorkflowState(
            user_message=user_message,
            messages=[],
            current_state="code",
            data={},
            retry_count=0
        )
        
        try:
            final_state = await workflow.ainvoke(initial_state)
            return final_state
        except Exception as e:
            print(f"Workflow execution error: {e}")
            initial_state.data['error'] = str(e)
            return initial_state


async def main():
    try:
        manager = WorkflowManager()
        
        user_query = "Create a Python function that calculates the factorial of a number and test it with a few examples."
        
        result: WorkflowState = await manager.run_workflow(user_query)
        
        print("=== Workflow Results ===")
        print(f"Final State: {result['current_state']}")
        print(f"Retry Count: {result['retry_count']}")
        print(f"Execution Status: {result['data'].get('execution_status', 'Unknown')}")
        
        print("\n=== Messages ===")
        for msg in result['messages']:
            print(f"{msg}\n")
        
        if result['data'].get('execution_status') == 'success':
            print("=== Final Result ===")
            print(result['data'].get('result', 'No result available'))
            
    except Exception as e:
        print(f"Main execution error: {e}")
        import traceback
        traceback.print_exc()


    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Workflow interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
