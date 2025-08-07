import asyncio
import re
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
    retry_count: int = 1
    max_retries: int = 3


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
        error_context = ""
        if state.retry_count > 1:
            error_msg = state.data.get('error_message', 'Unknown error')
            error_context = f"\n\nPREVIOUS ERROR TO FIX:\n{error_msg}\n\nPlease fix the above error in your code."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a code generator. Generate ONLY executable code.
            STRICT REQUIREMENTS:
            - Generate complete, runnable code
            - Include all imports at the top
            - Use proper syntax
            - Add error handling and test cases
            - Code must be self-contained and executable
            - Do not include explanations outside the code
            OUTPUT ONLY the code, nothing else."""),
            ("human", """{query}{error_context}""")
        ])
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "query": state.user_message,
                "error_context": error_context
            })
            
            content = response.content.strip()
            
            state.data['code'] = content
            self.add_message(state, f"Code generated (attempt {state.retry_count}):\n{state.data['code']}\n\n")
            
            state.current_state = "execute" 
        except Exception as e:
            self.add_message(state, f"Code generation failed: {str(e)}")
            state.data['execution_status'] = 'failure'
            state.data['error_message'] = f"Code generation error: {str(e)}"
            state.current_state = "test"
        
        print(f"\n=== ATTEMPT {state.retry_count} - GENERATED CODE ===")
        print(state.data.get('code', 'No code generated'))
        print("=" * 50)
        return state


class CodeExecutorAgent(BaseAgent):
    def __init__(self, name="CodeExecutorAgent"):
        super().__init__(name)
        self.role = "ExecutionAgent"
        self._tools_cache = None
        self._connection_attempts = 0
        self._max_connection_attempts = 3
    
    def analyze_execution_result(self, response_content: str) -> tuple[bool, str]:
        content = response_content.lower()
        definite_errors = [
            'traceback (most recent call last)',
            'syntaxerror:',
            'nameerror:',
            'indentationerror:',
            'typeerror:',
            'valueerror:',
            'attributeerror:',
            'importerror:',
            'modulenotfounderror:',
            'keyerror:',
            'indexerror:',
            'zerodivisionerror:',
            'execution failed',
            'error occurred',
            'failed to execute',
            'unable to execute',
            'internal error'
            'fail',
            'error',
            'exception'
            
        ]
        
        success_indicators = [
            'success',
            'passed',
            'executed successfully',
            'completed successfully',
            'execution completed',
            'ran successfully',
        ]
        
        if any(error in content for error in definite_errors):
            return False, response_content
            
        if any(success in content for success in success_indicators):
            return True, response_content
            
        if 'sorry' in content or 'unable' in content or 'cannot' in content:
            return False, response_content
            
        return True, response_content
    
    async def get_interpreter_tools(self):
        if self._tools_cache is not None:
            return self._tools_cache
        
        self._connection_attempts += 1
        if self._connection_attempts > self._max_connection_attempts:
            print(f"Max connection attempts ({self._max_connection_attempts}) reached")
            return []
            
        try:
            SERVER_URL = "http://127.0.0.1:8000/mcp"
            print(f"Attempting to connect to MCP server (attempt {self._connection_attempts})...")
            
            async with streamablehttp_client(SERVER_URL) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=10)
                    tools = await asyncio.wait_for(load_mcp_tools(session), timeout=10)
                    self._tools_cache = tools
                    print(f"Successfully connected to MCP server with {len(tools)} tools")
                    return tools
        except asyncio.TimeoutError:
            print(f"Connection timeout on attempt {self._connection_attempts}")
            return []
        except Exception as e:
            print(f"Connection failed on attempt {self._connection_attempts}: {e}")
            return []
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        try:
            tools = await self.get_interpreter_tools()
            if not tools:
                state.data['execution_status'] = 'failure'
                state.data['error_message'] = 'Could not connect to code execution server after multiple attempts'
                self.add_message(state, "Error: Could not connect to MCP server")
                state.current_state = "test"
                return state
            
            agent = create_react_agent(model=self.llm, tools=tools)
            
            execution_message = f"""Execute this code and show me the output:
              {state.data['code']}
              Run the code and tell me:
              1. What output it produces
              2. If there are any errors with details(failure or error or exception)
              3. Whether it executed successfully(success or passed)"""
            
            try:
                print("Executing code...")
                result = await asyncio.wait_for(
                    agent.ainvoke({
                        "messages": [
                            {"role": "system", "content": "Execute the code provided by the user using given tools and report detailed results."},
                            {"role": "user", "content": execution_message}
                        ]
                    }),
                    timeout=45
                )
                
                response_content = result["messages"][-1].content
                print(f"\nExecution response: {response_content[:300]}...")
                
                is_successful, processed_response = self.analyze_execution_result(response_content)
                
                if is_successful:
                    state.data['execution_status'] = 'success'
                    state.data['result'] = processed_response
                    state.data.pop('error_message', None)
                    print("Execution marked as SUCCESS")
                else:
                    state.data['execution_status'] = 'failure'
                    state.data['error_message'] = processed_response
                    print("Execution marked as FAILURE")
                
                self.add_message(state, f"Execution completed: {processed_response[:150]}...")
                
            except asyncio.TimeoutError:
                state.data['execution_status'] = 'failure'
                state.data['error_message'] = "Code execution timed out after 45 seconds"
                self.add_message(state, "Execution failed: Timeout")
                print("Execution TIMEOUT")
                
        except Exception as e:
            state.data['execution_status'] = 'failure'
            state.data['error_message'] = f"Execution system error: {str(e)}"
            self.add_message(state, f"Execution system error: {str(e)}")
            print(f"Execution SYSTEM ERROR: {e}")
        
        state.current_state = "test"
        return state


class TestingAgent(BaseAgent):
    def __init__(self, name="TestingAgent"):
        super().__init__(name)
        self.role = "TestingAgent"
    
    async def process(self, state: WorkflowState) -> WorkflowState:
        execution_status = state.data.get("execution_status", "failure")
        
        print(f"\n=== TEST RESULTS ===")
        print(f"Status: {execution_status}")
        print(f"Attempt: {state.retry_count}")
        
        if execution_status == 'failure':
            state.retry_count += 1
            
            if state.retry_count > state.max_retries:
                print(f"Maximum retries ({state.max_retries}) exceeded")
                self.add_message(state, f"Workflow failed after {state.max_retries} attempts")
                state.current_state = "end"
            else:
                error_msg = state.data.get('error_message', 'Unknown error')
                print(f"Retrying due to: {error_msg[:100]}...")
                self.add_message(state, f"Retry {state.retry_count}: {error_msg[:100]}...")
                state.current_state = "code"
        else:
            print(f"Success on attempt {state.retry_count}")
            self.add_message(state, f"Workflow completed successfully on attempt {state.retry_count}")
            state.current_state = "end"
        
        print("=" * 20)
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
            return "code" if state.current_state.lower() == "code" else "end"
        
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
        
        user_query = "Create a Python program that calculates the factorial of a number and test it with several test cases including edge cases."
        
        print(f"Starting workflow with query: {user_query}")
        result: WorkflowState = await manager.run_workflow(user_query)
        
        print("\n" + "="*60)
        print("FINAL WORKFLOW RESULTS")
        print("="*60)
        print(f"Final State: {result['current_state']}")
        print(f"Total Attempts: {result['retry_count']}")
        print(f"Execution Status: {result['data'].get('execution_status', 'Unknown')}")
        
        if result['data'].get('execution_status') == 'success':
            print("\n WORKFLOW SUCCEEDED")
            print("\n--- EXECUTION RESULT ---")
            print(result['data'].get('result', 'No result available'))
            print("\n--- FINAL CODE ---")
            print(result['data'].get('code', 'No code available'))
        else:
            print("\n WORKFLOW FAILED")
            print("--- ERROR DETAILS ---")
            print(result['data'].get('error_message', 'No error details available'))
            if result['data'].get('code'):
                print("\n--- LAST GENERATED CODE ---")
                print(result['data'].get('code'))
            
    except Exception as e:
        print(f"Main execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")