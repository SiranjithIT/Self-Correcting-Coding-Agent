"""
Code Executor FastMCP server - Multi-language code execution service.
Supports Python, Java, JavaScript (Node.js), and C++.

Run with:
    python code_executor_server.py
"""
import subprocess
import tempfile
import os
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for better schema generation
class CodeSnippet(BaseModel):
    code: str
    language: str
    input: str = ""

mcp = FastMCP("CodeExecutorServer")

class MCPCodeExecutor:
    """
    Multi-language Code Processor and Executor
    Supports Python, Java, JavaScript (Node.js), and C++
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize the code executor
        
        Args:
            timeout (int): Maximum execution time in seconds (default: 10)
        """
        self.timeout = timeout
        self.supported_languages = ['python', 'java', 'javascript', 'cpp', 'c++']
        
        # Language-specific configurations
        self.language_config = {
            'python': {
                'extension': '.py',
                'compile_cmd': None,
                'run_cmd': ['python'],
                'interpreter': True
            },
            'java': {
                'extension': '.java',
                'compile_cmd': ['javac'],
                'run_cmd': ['java'],
                'interpreter': False,
                'class_template': 'public class {class_name} {{\n{code}\n}}'
            },
            'javascript': {
                'extension': '.js',
                'compile_cmd': None,
                'run_cmd': ['node'],
                'interpreter': True
            },
            'cpp': {
                'extension': '.cpp',
                'compile_cmd': ['g++', '-o'],
                'run_cmd': [],
                'interpreter': False
            },
            'c++': {
                'extension': '.cpp',
                'compile_cmd': ['g++', '-o'],
                'run_cmd': [],
                'interpreter': False
            }
        }
    
    def execute_code(self, code: str, language: str, input_data: str = "") -> Dict[str, Any]:
        """
        Execute code in the specified language
        
        Args:
            code (str): The source code to execute
            language (str): Programming language ('python', 'java', 'javascript', 'cpp', 'c++')
            input_data (str): Input data for the program (optional)
            
        Returns:
            Dict containing execution results with keys:
            - success (bool): Whether execution was successful
            - output (str): Program output
            - error (str): Error message if any
            - execution_time (float): Time taken to execute
            - language (str): Language used
        """
        language = language.lower()
        
        if language not in self.supported_languages:
            return {
                'success': False,
                'output': '',
                'error': f'Unsupported language: {language}. Supported: {", ".join(self.supported_languages)}',
                'execution_time': 0.0,
                'language': language
            }
        
        # Normalize c++ to cpp for internal processing
        if language == 'c++':
            language = 'cpp'
        
        start_time = time.time()
        
        try:
            if language == 'python':
                result = self._execute_python(code, input_data)
            elif language == 'java':
                result = self._execute_java(code, input_data)
            elif language == 'javascript':
                result = self._execute_javascript(code, input_data)
            elif language == 'cpp':
                result = self._execute_cpp(code, input_data)
            
            result['execution_time'] = time.time() - start_time
            result['language'] = language
            return result
            
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Execution error: {str(e)}',
                'execution_time': time.time() - start_time,
                'language': language
            }
    
    def _execute_python(self, code: str, input_data: str) -> Dict[str, Any]:
        """Execute Python code"""
        with tempfile.TemporaryDirectory() as temp_dir:
            py_file = os.path.join(temp_dir, 'program.py')
            
            with open(py_file, 'w') as f:
                f.write(code)
            
            logger.info("=== CODE TO EXECUTE ===\n%s\n=======================", code)  
            
            result = subprocess.run(
                ['python', py_file],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
    
    def _execute_java(self, code: str, input_data: str) -> Dict[str, Any]:
        """Execute Java code"""
        # Extract class name or use default
        class_name = self._extract_java_class_name(code) or 'Main'
        
        # If no public class is defined, wrap code in a Main class
        if 'public class' not in code:
            code = f"public class {class_name} {{\n    public static void main(String[] args) {{\n{self._indent_code(code, 8)}\n    }}\n}}"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            java_file = os.path.join(temp_dir, f'{class_name}.java')
            
            with open(java_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ['javac', java_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if compile_result.returncode != 0:
                return {
                    'success': False,
                    'output': '',
                    'error': f'Compilation error: {compile_result.stderr}'
                }
            
            # Run
            run_result = subprocess.run(
                ['java', '-cp', temp_dir, class_name],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                'success': run_result.returncode == 0,
                'output': run_result.stdout,
                'error': run_result.stderr
            }
    
    def _execute_javascript(self, code: str, input_data: str) -> Dict[str, Any]:
        """Execute JavaScript code using Node.js"""
        with tempfile.TemporaryDirectory() as temp_dir:
            js_file = os.path.join(temp_dir, 'program.js')
            
            with open(js_file, 'w') as f:
                f.write(code)
            
            result = subprocess.run(
                ['node', js_file],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
    
    def _execute_cpp(self, code: str, input_data: str) -> Dict[str, Any]:
        """Execute C++ code"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = os.path.join(temp_dir, 'program.cpp')
            exe_file = os.path.join(temp_dir, 'program')
            
            with open(cpp_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ['g++', '-o', exe_file, cpp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if compile_result.returncode != 0:
                return {
                    'success': False,
                    'output': '',
                    'error': f'Compilation error: {compile_result.stderr}'
                }
            
            # Run
            run_result = subprocess.run(
                [exe_file],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                'success': run_result.returncode == 0,
                'output': run_result.stdout,
                'error': run_result.stderr
            }
    
    def _extract_java_class_name(self, code: str) -> Optional[str]:
        """Extract the public class name from Java code"""
        import re
        match = re.search(r'public\s+class\s+(\w+)', code)
        return match.group(1) if match else None
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line for line in code.split('\n'))
    
    def batch_execute(self, code_snippets: list) -> Dict[str, Any]:
        """
        Execute multiple code snippets
        
        Args:
            code_snippets (list): List of dictionaries with 'code', 'language', and optional 'input' keys
            
        Returns:
            Dict with execution results for each snippet
        """
        results = {}
        
        for i, snippet in enumerate(code_snippets):
            if not isinstance(snippet, dict) or 'code' not in snippet or 'language' not in snippet:
                results[f'snippet_{i}'] = {
                    'success': False,
                    'output': '',
                    'error': 'Invalid snippet format. Required keys: code, language',
                    'execution_time': 0.0,
                    'language': 'unknown'
                }
                continue
            
            code = snippet['code']
            language = snippet['language']
            input_data = snippet.get('input', '')
            
            results[f'snippet_{i}'] = self.execute_code(code, language, input_data)
        
        return results
    
    def get_language_info(self) -> Dict[str, Any]:
        """Get information about supported languages"""
        return {
            'supported_languages': self.supported_languages,
            'timeout': self.timeout,
            'language_details': {
                lang: {
                    'extension': config['extension'],
                    'compiled': not config['interpreter'],
                    'interpreter': config['interpreter']
                }
                for lang, config in self.language_config.items()
                if lang != 'c++'  # Exclude duplicate c++
            }
        }

# Create global executor instance
executor = MCPCodeExecutor(timeout=15)

@mcp.tool()
def execute_code(code: str, language: str, input_data: str = "") -> dict:
    """Execute code in the specified programming language (python, java, javascript, cpp, c++)"""
    logger.info(f"Executing {language} code")
    
    try:
        result = executor.execute_code(code, language, input_data)
        
        if result['success']:
            logger.info(f"Code executed successfully in {result['execution_time']:.3f}s")
        else:
            logger.warning(f"Code execution failed: {result['error']}")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error executing code: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'output': '',
            'error': error_msg,
            'execution_time': 0.0,
            'language': language
        }

# Temporarily comment out batch_execute_code if still having issues
# @mcp.tool()
# def batch_execute_code(code_snippets: List[CodeSnippet]) -> dict:
#     """Execute multiple code snippets. Each snippet should be a dict with 'code', 'language', and optional 'input' keys"""
#     logger.info(f"Batch executing {len(code_snippets)} code snippets")
#     
#     try:
#         # Convert Pydantic models to dict format expected by executor
#         snippets_dict = []
#         for snippet in code_snippets:
#             snippets_dict.append({
#                 'code': snippet.code,
#                 'language': snippet.language,
#                 'input': snippet.input
#             })
#         
#         results = executor.batch_execute(snippets_dict)
#         
#         successful = sum(1 for result in results.values() if result['success'])
#         logger.info(f"Batch execution completed: {successful}/{len(results)} successful")
#         
#         return {
#             'success': True,
#             'results': results,
#             'summary': {
#                 'total': len(results),
#                 'successful': successful,
#                 'failed': len(results) - successful
#             }
#         }
#         
#     except Exception as e:
#         error_msg = f"Batch execution error: {str(e)}"
#         logger.error(error_msg)
#         return {
#             'success': False,
#             'error': error_msg,
#             'results': {},
#             'summary': {'total': 0, 'successful': 0, 'failed': 0}
#         }

@mcp.tool()
def get_supported_languages() -> dict:
    """Get information about supported programming languages and configurations"""
    logger.info("Retrieving supported languages information")
    
    try:
        info = executor.get_language_info()
        logger.info(f"Supporting {len(info['supported_languages'])} languages")
        return info
        
    except Exception as e:
        error_msg = f"Error retrieving language info: {str(e)}"
        logger.error(error_msg)
        return {'error': error_msg}

@mcp.tool()
def validate_syntax(code: str, language: str) -> dict:
    """Validate code syntax without executing (for compiled languages like Java and C++)"""
    logger.info(f"Validating {language} syntax")
    
    try:
        language = language.lower()
        if language == 'c++':
            language = 'cpp'
            
        if language not in executor.supported_languages:
            return {
                'valid': False,
                'error': f'Unsupported language: {language}'
            }
        
        # For interpreted languages, we can't easily validate without execution
        if language in ['python', 'javascript']:
            return {
                'valid': True,
                'message': f'Syntax validation not available for {language} (interpreted language)',
                'language': language
            }
        
        # For compiled languages, try compilation only
        if language == 'java':
            result = executor._execute_java(code, "")
            if 'Compilation error' in result.get('error', ''):
                return {
                    'valid': False,
                    'error': result['error'],
                    'language': language
                }
            else:
                return {
                    'valid': True,
                    'message': 'Java code compiled successfully',
                    'language': language
                }
        
        elif language == 'cpp':
            # Try compilation only for C++
            with tempfile.TemporaryDirectory() as temp_dir:
                cpp_file = os.path.join(temp_dir, 'program.cpp')
                exe_file = os.path.join(temp_dir, 'program')
                
                with open(cpp_file, 'w') as f:
                    f.write(code)
                
                compile_result = subprocess.run(
                    ['g++', '-o', exe_file, cpp_file],
                    capture_output=True,
                    text=True,
                    timeout=executor.timeout
                )
                
                if compile_result.returncode != 0:
                    return {
                        'valid': False,
                        'error': f'Compilation error: {compile_result.stderr}',
                        'language': language
                    }
                else:
                    return {
                        'valid': True,
                        'message': 'C++ code compiled successfully',
                        'language': language
                    }
        
    except Exception as e:
        error_msg = f"Syntax validation error: {str(e)}"
        logger.error(error_msg)
        return {
            'valid': False,
            'error': error_msg,
            'language': language
        }

@mcp.resource("code-execution://{language}")
def get_code_execution_resource(language: str) -> str:
    """Get code execution examples and templates for a specific language"""
    language = language.lower()
    
    templates = {
        'python': '''# Python Code Template
print("Hello, World!")

# Input/Output example
name = input("Enter your name: ")
print(f"Hello, {name}!")

# Basic operations
numbers = [1, 2, 3, 4, 5]
squared = [n**2 for n in numbers]
print("Squared numbers:", squared)
''',
        'java': '''// Java Code Template
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        
        // Input/Output example
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter your name: ");
        String name = scanner.nextLine();
        System.out.println("Hello, " + name + "!");
        
        // Basic operations
        int[] numbers = {1, 2, 3, 4, 5};
        System.out.print("Squared numbers: ");
        for (int n : numbers) {
            System.out.print(n * n + " ");
        }
        System.out.println();
        
        scanner.close();
    }
}
''',
        'javascript': '''// JavaScript (Node.js) Code Template
const readline = require('readline');

console.log("Hello, World!");

// Input/Output example
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

rl.question('Enter your name: ', (name) => {
    console.log(`Hello, ${name}!`);
    
    // Basic operations
    const numbers = [1, 2, 3, 4, 5];
    const squared = numbers.map(n => n * n);
    console.log("Squared numbers:", squared);
    
    rl.close();
});
''',
        'cpp': '''// C++ Code Template
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    
    // Input/Output example
    cout << "Enter your name: ";
    string name;
    getline(cin, name);
    cout << "Hello, " << name << "!" << endl;
    
    // Basic operations
    vector<int> numbers = {1, 2, 3, 4, 5};
    cout << "Squared numbers: ";
    for (int n : numbers) {
        cout << n * n << " ";
    }
    cout << endl;
    
    return 0;
}
''',
        'c++': '''// C++ Code Template
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    
    // Input/Output example
    cout << "Enter your name: ";
    string name;
    getline(cin, name);
    cout << "Hello, " << name << "!" << endl;
    
    // Basic operations
    vector<int> numbers = {1, 2, 3, 4, 5};
    cout << "Squared numbers: ";
    for (int n : numbers) {
        cout << n * n << " ";
    }
    cout << endl;
    
    return 0;
}
'''
    }
    
    if language not in templates:
        return f"Error: Unsupported language '{language}'. Supported languages: {', '.join(templates.keys())}"
    
    return f"""Code Execution Template for {language.upper()}

{templates[language]}

Usage Instructions:
1. Use the execute_code tool with your code
2. Provide input data if your program requires user input
3. The timeout is set to {executor.timeout} seconds
4. For compiled languages (Java, C++), compilation errors will be reported
5. For interpreted languages (Python, JavaScript), runtime errors will be shown

Example tool call:
{{
    "code": "<your code here>",
    "language": "{language}",
    "input_data": "optional input data"
}}
"""

@mcp.prompt()
def code_execution_help(language: str = "general", task: str = "basic") -> str:
    """Generate a code execution help prompt for specific language and task"""
    
    task_contexts = {
        "basic": "Write a simple program that demonstrates basic syntax",
        "input": "Create a program that reads user input and processes it",
        "algorithm": "Implement a common algorithm or data structure",
        "file": "Work with file I/O operations",
        "debug": "Debug and fix issues in existing code"
    }
    
    language_tips = {
        "python": "Use clear, readable syntax. Remember Python is indentation-sensitive.",
        "java": "Define a public class with main method. Handle Scanner properly.",
        "javascript": "Use Node.js APIs for I/O. Handle asynchronous operations carefully.",
        "cpp": "Include necessary headers. Use proper memory management.",
        "c++": "Include necessary headers. Use proper memory management."
    }
    
    task_desc = task_contexts.get(task, task_contexts['basic'])
    lang_tip = language_tips.get(language.lower(), "Follow language best practices.")
    
    if language == "general":
        return f"""Code Execution Assistant

Task: {task_desc}

General Guidelines:
- Choose appropriate language for the task
- Handle errors gracefully
- Include comments for clarity
- Test with sample inputs
- Consider edge cases

Supported Languages: Python, Java, JavaScript (Node.js), C++

Use the execute_code tool to run your code with proper error handling and timeout protection.
"""
    else:
        return f"""Code Execution Assistant for {language.upper()}

Task: {task_desc}

Language-specific tips:
{lang_tip}

Use the code execution tools to:
1. validate_syntax() - Check syntax before execution
2. execute_code() - Run your complete program
3. get_supported_languages() - View language configurations

Remember to handle input/output properly and consider the {executor.timeout}s timeout limit.
"""

if __name__ == "__main__":
    logger.info("Starting Code Executor MCP Server...")
    logger.info(f"Supported languages: {', '.join(executor.supported_languages)}")
    logger.info(f"Execution timeout: {executor.timeout} seconds")
    mcp.run(transport="streamable-http")