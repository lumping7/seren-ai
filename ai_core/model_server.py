#!/usr/bin/env python3
"""
Model Server for Seren AI

This script provides an interface between the Node.js backend and the Python-based
AI models (Qwen2.5-7b-omni and OlympicCoder-7B). It handles model loading, inference,
and communication via stdin/stdout JSON messages.

Usage:
    python model_server.py

Environment variables:
    MODEL_TYPE: The type of model to load (qwen2.5-7b-omni or olympiccoder-7b)
    MODEL_BASE_PATH: Base path for model files
    LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
    PORT: Port for communication (not used directly but might be needed for some models)
"""

import os
import sys
import json
import time
import uuid
import logging
import traceback
import datetime
import threading
import signal
from typing import Dict, Any, Optional, List, Union, Callable

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("model_server")

# Import conditional dependencies
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
    logger.info("PyTorch is available")
except ImportError:
    logger.warning("PyTorch is not available, using fallback mechanisms")

HAS_TRANSFORMERS = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
    logger.info("Transformers is available")
except ImportError:
    logger.warning("Transformers is not available, using fallback mechanisms")

# Attempt to import our local modules
try:
    from model_communication import CommunicationSystem, ModelType, MessageType, CommunicationMode
    from neurosymbolic_reasoning import ReasoningSystem, ReasoningStrategy
    from liquid_neural_network import LiquidNeuralNetwork
    from metacognition import MetacognitiveSystem, MetacognitiveLevel
    from knowledge_library import KnowledgeLibrary

    HAS_LOCAL_MODULES = True
    logger.info("Local AI modules loaded successfully")
except ImportError as e:
    HAS_LOCAL_MODULES = False
    logger.warning(f"Could not import local AI modules: {e}, using fallback implementations")

# Initialize fallback systems when modules aren't available
if not HAS_LOCAL_MODULES:
    from enum import Enum
    
    class ModelType(Enum):
        """Types of AI models"""
        QWEN = "qwen"
        OLYMPIC = "olympic"
        SYSTEM = "system"
        
        @property
        def value(self):
            """Get the value of the enum"""
            return super().value
            
        def __str__(self):
            return self.value
            
    class CommunicationMode:
        """Communication modes between models"""
        COLLABORATIVE = "collaborative"  # Models work together, sharing knowledge
        SPECIALIZED = "specialized"      # Models work on specific tasks based on their strengths
        COMPETITIVE = "competitive"      # Models compete to provide the best answers
        
    class MessageType:
        """Types of messages between models"""
        QUERY = "query"
        RESPONSE = "response"
        INFO = "info"
        ERROR = "error"
        CODE = "code"
        KNOWLEDGE = "knowledge"
        REASONING = "reasoning"
        REQUEST_HELP = "request_help"
        PROVIDE_HELP = "provide_help"
        
    class ReasoningStrategy:
        """Reasoning strategies"""
        DEDUCTIVE = "deductive"
        INDUCTIVE = "inductive"
        ABDUCTIVE = "abductive"
        ANALOGICAL = "analogical"
        CAUSAL = "causal"
        
    class CommunicationSystem:
        """Fallback implementation of communication system"""
        def __init__(self, base_dir=None):
            self.conversations = {}
            logger.warning("Using fallback CommunicationSystem")
            
        def create_conversation(self, topic=None, mode=CommunicationMode.COLLABORATIVE):
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = {
                "id": conversation_id,
                "topic": topic or "General Conversation",
                "mode": mode,
                "created_at": datetime.datetime.now().isoformat(),
                "participants": [],
                "messages": []
            }
            return conversation_id
            
        def send_message(self, conversation_id, from_model, to_model=None, 
                         message_type=MessageType.INFO, content="", encrypted=False):
            if conversation_id not in self.conversations:
                return False
                
            conversation = self.conversations[conversation_id]
            message = {
                "id": str(uuid.uuid4()),
                "from_model": str(from_model),
                "to_model": str(to_model) if to_model else None,
                "message_type": message_type,
                "content": content,
                "encrypted": False,
                "timestamp": datetime.datetime.now().isoformat()
            }
            conversation["messages"].append(message)
            return True
            
    class ReasoningSystem:
        """Fallback implementation of reasoning system"""
        def __init__(self):
            logger.warning("Using fallback ReasoningSystem")
            
        def solve_problem(self, problem, strategy=None, context=None):
            logger.warning("Using simplified reasoning in fallback mode")
            return {
                "solution": f"Fallback solution for: {problem}",
                "reasoning_path": ["Simplified reasoning due to missing modules"],
                "confidence": 0.7
            }
            
    class LiquidNeuralNetwork:
        """Fallback implementation of liquid neural network"""
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
            logger.warning("Using fallback LiquidNeuralNetwork")
            
        def forward(self, x):
            return {"output": "Fallback output from LNN", "confidence": 0.6}
            
    class MetacognitiveSystem:
        """Fallback implementation of metacognitive system"""
        def __init__(self):
            logger.warning("Using fallback MetacognitiveSystem")
            
        def evaluate_confidence(self, result, context=None):
            return 0.7
            
        def reflect(self, input_data, output_data, expected=None):
            return {
                "quality": 0.7,
                "improvements": ["Fallback metacognitive reflection"],
                "decision": "proceed"
            }
            
    class KnowledgeLibrary:
        """Fallback implementation of knowledge library"""
        def __init__(self):
            self.knowledge = {}
            logger.warning("Using fallback KnowledgeLibrary")
            
        def add_entry(self, content, source, categories=None):
            entry_id = str(uuid.uuid4())
            self.knowledge[entry_id] = {
                "content": content,
                "source": source,
                "categories": categories or [],
                "created_at": datetime.datetime.now().isoformat()
            }
            return entry_id
            
        def search(self, query, limit=5):
            return []

# Constants and configuration
MODEL_TYPE = os.environ.get("MODEL_TYPE", "qwen2.5-7b-omni")
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
HEARTBEAT_INTERVAL = 10  # Seconds

# Global state
global_state = {
    "model": None,
    "tokenizer": None,
    "communication_system": None,
    "reasoning_system": None,
    "liquid_nn": None,
    "metacognitive_system": None,
    "knowledge_library": None,
    "model_type": MODEL_TYPE,
    "status": "initializing",
    "last_activity": time.time(),
    "memory_usage": 0,
    "should_exit": False
}

# Message processing system
class MessageProcessor:
    """Processes messages from Node.js backend"""
    
    def __init__(self):
        self.handlers = {
            "request": self.handle_request,
            "heartbeat": self.handle_heartbeat
        }
        
    def process_message(self, message: Dict[str, Any]) -> None:
        """Process an incoming message"""
        try:
            message_type = message.get("type")
            if message_type not in self.handlers:
                logger.warning(f"Unknown message type: {message_type}")
                self.send_error(message.get("id", str(uuid.uuid4())), f"Unknown message type: {message_type}")
                return
                
            # Update last activity time
            global_state["last_activity"] = time.time()
            
            # Call the appropriate handler
            self.handlers[message_type](message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(traceback.format_exc())
            self.send_error(message.get("id", str(uuid.uuid4())), str(e))
            
    def handle_request(self, message: Dict[str, Any]) -> None:
        """Handle a request message"""
        message_id = message.get("id", str(uuid.uuid4()))
        role = message.get("role", "architect")
        content = message.get("content", {})
        
        logger.info(f"Handling request {message_id} for role {role}")
        
        # Update status
        global_state["status"] = "busy"
        self.send_status_update()
        
        try:
            # Process the request based on the role
            if role == "architect":
                result = self.process_architect_request(content)
            elif role == "builder":
                result = self.process_builder_request(content)
            elif role == "tester":
                result = self.process_tester_request(content)
            elif role == "reviewer":
                result = self.process_reviewer_request(content)
            else:
                result = {"error": f"Unknown role: {role}"}
                
            # Send the response
            self.send_response(message_id, result)
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            logger.error(traceback.format_exc())
            self.send_error(message_id, str(e))
            
        finally:
            # Update status
            global_state["status"] = "ready"
            self.send_status_update()
            
    def handle_heartbeat(self, message: Dict[str, Any]) -> None:
        """Handle a heartbeat message"""
        message_id = message.get("id", str(uuid.uuid4()))
        content = message.get("content", {})
        
        action = content.get("action")
        
        if action == "status_check":
            # Send status update
            self.send_status_update(message_id)
        else:
            # Just acknowledge the heartbeat
            self.send_response(message_id, {"status": "alive", "timestamp": time.time()})
            
    def process_architect_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request for the architect role"""
        task = content.get("task")
        
        if task == "architecture_planning":
            return self.generate_architecture(content)
        else:
            return {"error": f"Unknown architect task: {task}"}
            
    def process_builder_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request for the builder role"""
        task = content.get("task")
        
        if task == "code_generation":
            return self.generate_code(content)
        elif task == "code_enhancement":
            return self.enhance_code(content)
        else:
            return {"error": f"Unknown builder task: {task}"}
            
    def process_tester_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request for the tester role"""
        task = content.get("task")
        
        if task == "test_generation":
            return self.generate_tests(content)
        elif task == "code_debugging":
            return self.debug_code(content)
        else:
            return {"error": f"Unknown tester task: {task}"}
            
    def process_reviewer_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request for the reviewer role"""
        task = content.get("task")
        
        if task == "code_review":
            return self.review_code(content)
        elif task == "code_explanation":
            return self.explain_code(content)
        else:
            return {"error": f"Unknown reviewer task: {task}"}
    
    # Architect role implementation
    def generate_architecture(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate architecture based on requirements"""
        requirements = content.get("requirements", "")
        options = content.get("options", {})
        
        if not requirements:
            return {"error": "Requirements are required for architecture planning"}
            
        # Use reasoning system to generate architecture
        if global_state["reasoning_system"]:
            reasoning_result = global_state["reasoning_system"].solve_problem(
                problem=f"Design software architecture for: {requirements}",
                strategy=ReasoningStrategy.DEDUCTIVE,
                context={
                    "language": options.get("language"),
                    "framework": options.get("framework"),
                    "architecture": options.get("architecture")
                }
            )
            
            # Enhance with metacognitive system if available
            if global_state["metacognitive_system"]:
                confidence = global_state["metacognitive_system"].evaluate_confidence(
                    reasoning_result, 
                    context={"task": "architecture_planning"}
                )
                
                if confidence < 0.8:
                    # Try to improve the result
                    reflection = global_state["metacognitive_system"].reflect(
                        requirements,
                        reasoning_result["solution"],
                        None
                    )
                    
                    if reflection["decision"] == "revise":
                        # Regenerate with more context
                        reasoning_result = global_state["reasoning_system"].solve_problem(
                            problem=f"Design software architecture for: {requirements}",
                            strategy=ReasoningStrategy.DEDUCTIVE,
                            context={
                                "language": options.get("language"),
                                "framework": options.get("framework"),
                                "architecture": options.get("architecture"),
                                "improvements": reflection["improvements"]
                            }
                        )
            
            return reasoning_result["solution"]
        
        # Fallback to model inference
        prompt = self._create_architecture_prompt(requirements, options)
        return self._generate_text(prompt)
    
    def _create_architecture_prompt(self, requirements: str, options: Dict[str, Any]) -> str:
        """Create a prompt for architecture generation"""
        language = options.get("language", "")
        framework = options.get("framework", "")
        architecture = options.get("architecture", "")
        
        prompt = f"""You are a software architect designing a system based on the following requirements:

REQUIREMENTS:
{requirements}

"""
        
        if language:
            prompt += f"The solution should be implemented in {language}.\n"
            
        if framework:
            prompt += f"The solution should use the {framework} framework.\n"
            
        if architecture:
            prompt += f"The solution should follow {architecture} architecture principles.\n"
            
        prompt += """
Please provide a comprehensive software architecture design that includes:

1. High-level architecture overview
2. Component diagram
3. Data model
4. API endpoints (if applicable)
5. Key design patterns and principles
6. Security considerations
7. Scalability and performance considerations

Format your response as a detailed technical document that a development team can implement.
"""
        
        return prompt
    
    # Builder role implementation
    def generate_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on requirements and architecture"""
        requirements = content.get("requirements", "")
        architecture = content.get("architecture", "")
        options = content.get("options", {})
        
        if not requirements and not architecture:
            return {"error": "Either requirements or architecture must be provided for code generation"}
            
        # Fallback to model inference
        prompt = self._create_code_prompt(requirements, architecture, options)
        return self._generate_text(prompt)
        
    def _create_code_prompt(self, requirements: str, architecture: str, options: Dict[str, Any]) -> str:
        """Create a prompt for code generation"""
        language = options.get("language", "")
        framework = options.get("framework", "")
        
        prompt = "You are an expert software developer implementing code based on the following "
        
        if architecture:
            prompt += "architecture specification:\n\n"
            prompt += f"ARCHITECTURE:\n{architecture}\n\n"
        else:
            prompt += "requirements:\n\n"
            prompt += f"REQUIREMENTS:\n{requirements}\n\n"
            
        if language:
            prompt += f"The solution must be implemented in {language}.\n"
            
        if framework:
            prompt += f"The solution must use the {framework} framework.\n"
            
        prompt += """
Please generate production-ready, well-organized code that implements the specified architecture or requirements.
Your code should be:

1. Well-structured and modular
2. Following best practices for the specified language/framework
3. Properly commented with clear documentation
4. Error handled appropriately
5. Secure, efficient, and maintainable

Generate complete, working code files that can be directly implemented.
"""
        
        return prompt
        
    def enhance_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance existing code"""
        code = content.get("code", "")
        requirements = content.get("requirements", "")
        enhancement = content.get("enhancement", "optimize")
        language = content.get("language", "")
        
        if not code:
            return {"error": "Code is required for enhancement"}
            
        # Use liquid neural network for enhancement if available
        if global_state["liquid_nn"] and HAS_TORCH:
            # This is a simplified example - in a real system, we would need to:
            # 1. Convert code to embeddings or features
            # 2. Process through the LNN
            # 3. Generate enhanced code based on LNN output
            # For now, we'll just use a fallback
            logger.info("Liquid neural network is available but not fully integrated for code enhancement")
            
        # Fallback to model inference
        prompt = self._create_enhancement_prompt(code, requirements, enhancement, language)
        return self._generate_text(prompt)
        
    def _create_enhancement_prompt(self, code: str, requirements: str, enhancement: str, language: str) -> str:
        """Create a prompt for code enhancement"""
        prompt = f"You are an expert software developer enhancing the following code:\n\n```\n{code}\n```\n\n"
        
        if requirements:
            prompt += f"The code should meet these requirements:\n{requirements}\n\n"
            
        if language:
            prompt += f"The code is written in {language}.\n\n"
            
        if enhancement == "optimize":
            prompt += "Your task is to optimize this code for better performance and efficiency. Identify and fix any performance bottlenecks."
        elif enhancement == "refactor":
            prompt += "Your task is to refactor this code to improve its structure, readability, and maintainability without changing its behavior."
        elif enhancement == "document":
            prompt += "Your task is to add comprehensive documentation to this code, including function/method documentation, usage examples, and explanatory comments."
        elif enhancement == "test":
            prompt += "Your task is to add appropriate tests for this code, ensuring good coverage and testing of edge cases."
        elif enhancement == "fix":
            prompt += "Your task is to identify and fix any bugs, issues, or potential problems in this code."
        else:
            prompt += "Your task is to enhance this code to make it more robust, efficient, and maintainable."
            
        prompt += "\n\nProvide the enhanced code with explanations of your changes."
        
        return prompt
    
    # Tester role implementation
    def generate_tests(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tests for code"""
        code = content.get("code", "")
        requirements = content.get("requirements", "")
        options = content.get("options", {})
        
        if not code:
            return {"error": "Code is required for test generation"}
            
        # Fallback to model inference
        prompt = self._create_test_prompt(code, requirements, options)
        return self._generate_text(prompt)
        
    def _create_test_prompt(self, code: str, requirements: str, options: Dict[str, Any]) -> str:
        """Create a prompt for test generation"""
        language = options.get("language", "")
        
        prompt = f"You are a software testing expert creating tests for the following code:\n\n```\n{code}\n```\n\n"
        
        if requirements:
            prompt += f"The code should meet these requirements:\n{requirements}\n\n"
            
        if language:
            prompt += f"The code is written in {language}.\n\n"
            
        prompt += """
Please generate comprehensive tests that:

1. Test the core functionality
2. Cover edge cases and potential error conditions
3. Ensure the code meets the requirements
4. Follow testing best practices
5. Are well-organized and maintainable

Your tests should be production-ready and include appropriate setup, teardown, and assertions.
"""
        
        return prompt
        
    def debug_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Debug and fix code"""
        code = content.get("code", "")
        error = content.get("error", "")
        language = content.get("language", "")
        
        if not code:
            return {"error": "Code is required for debugging"}
            
        # Fallback to model inference
        prompt = self._create_debug_prompt(code, error, language)
        return self._generate_text(prompt)
        
    def _create_debug_prompt(self, code: str, error: str, language: str) -> str:
        """Create a prompt for debugging"""
        prompt = f"You are an expert developer debugging the following code:\n\n```\n{code}\n```\n\n"
        
        if error:
            prompt += f"The code produces the following error:\n{error}\n\n"
            
        if language:
            prompt += f"The code is written in {language}.\n\n"
            
        prompt += """
Your task is to:

1. Analyze the code to identify the bug or issue
2. Explain the root cause of the problem
3. Provide a fixed version of the code
4. Explain your solution and the changes you made

Provide the complete fixed code and ensure it's production-ready.
"""
        
        return prompt
    
    # Reviewer role implementation
    def review_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Review code"""
        code = content.get("code", "")
        tests = content.get("tests", "")
        requirements = content.get("requirements", "")
        options = content.get("options", {})
        
        if not code:
            return {"error": "Code is required for review"}
            
        # Fallback to model inference
        prompt = self._create_review_prompt(code, tests, requirements, options)
        return self._generate_text(prompt)
        
    def _create_review_prompt(self, code: str, tests: str, requirements: str, options: Dict[str, Any]) -> str:
        """Create a prompt for code review"""
        language = options.get("language", "")
        framework = options.get("framework", "")
        
        prompt = f"You are an expert code reviewer analyzing the following code:\n\n```\n{code}\n```\n\n"
        
        if tests:
            prompt += f"The code has the following tests:\n\n```\n{tests}\n```\n\n"
            
        if requirements:
            prompt += f"The code should meet these requirements:\n{requirements}\n\n"
            
        if language:
            prompt += f"The code is written in {language}.\n\n"
            
        if framework:
            prompt += f"The code uses the {framework} framework.\n\n"
            
        prompt += """
Please provide a comprehensive code review that includes:

1. Overall assessment of code quality
2. Identification of any bugs, issues, or potential problems
3. Suggestions for improvements in:
   - Performance
   - Security
   - Maintainability
   - Readability
   - Error handling
4. Best practices that could be applied
5. Any potential edge cases or failure scenarios

For each issue or suggestion, provide:
- The specific location (line number or function)
- The problem or opportunity for improvement
- A recommended solution with example code where applicable

Your review should be thorough and actionable, helping to improve the code quality.
"""
        
        return prompt
        
    def explain_code(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Explain code"""
        code = content.get("code", "")
        language = content.get("language", "")
        detail_level = content.get("detail_level", "detailed")
        
        if not code:
            return {"error": "Code is required for explanation"}
            
        # Fallback to model inference
        prompt = self._create_explanation_prompt(code, language, detail_level)
        return self._generate_text(prompt)
        
    def _create_explanation_prompt(self, code: str, language: str, detail_level: str) -> str:
        """Create a prompt for code explanation"""
        prompt = f"You are an expert developer explaining the following code:\n\n```\n{code}\n```\n\n"
        
        if language:
            prompt += f"The code is written in {language}.\n\n"
            
        if detail_level == "simple":
            prompt += "Please provide a simple, high-level explanation of what this code does, suitable for someone with basic programming knowledge."
        elif detail_level == "comprehensive":
            prompt += "Please provide an extremely detailed, line-by-line explanation of this code, including internal logic, algorithms, design patterns, and best practices. Explain both what the code does and why it's implemented this way."
        else:  # detailed (default)
            prompt += "Please provide a detailed explanation of this code, including its purpose, structure, key functions/methods, and how the different parts work together. Explain any complex logic or algorithms."
            
        prompt += "\n\nMake your explanation clear and accessible, focusing on helping the reader understand how the code works."
        
        return prompt
    
    # Helper methods for model interaction
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the loaded model"""
        # Check if transformers model is available
        if global_state["model"] and global_state["tokenizer"] and HAS_TRANSFORMERS and HAS_TORCH:
            logger.info("Generating text using transformers model")
            
            try:
                inputs = global_state["tokenizer"](prompt, return_tensors="pt")
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    global_state["model"].to('cuda')
                
                # Generate
                with torch.no_grad():
                    outputs = global_state["model"].generate(
                        **inputs,
                        max_length=1024,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                
                # Decode
                generated_text = global_state["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                
                # Extract the generated part (exclude the prompt)
                generated_text = generated_text[len(prompt):]
                
                return generated_text.strip()
                
            except Exception as e:
                logger.error(f"Error generating with transformers: {e}")
                logger.error(traceback.format_exc())
                
                # Fall back to rule-based generation
                return self._fallback_generate(prompt)
        else:
            logger.info("Using fallback text generation")
            return self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt: str) -> str:
        """Fallback text generation when model is not available"""
        task_type = ""
        language = ""
        
        # Very basic content extraction
        if "architecture" in prompt.lower():
            task_type = "architecture"
        elif "code" in prompt.lower() and "generate" in prompt.lower():
            task_type = "code"
        elif "test" in prompt.lower():
            task_type = "test"
        elif "debug" in prompt.lower() or "fix" in prompt.lower():
            task_type = "debug"
        elif "review" in prompt.lower():
            task_type = "review"
        elif "explain" in prompt.lower():
            task_type = "explain"
        else:
            task_type = "general"
            
        # Try to extract language
        if "python" in prompt.lower():
            language = "Python"
        elif "javascript" in prompt.lower() or "js" in prompt.lower():
            language = "JavaScript"
        elif "typescript" in prompt.lower() or "ts" in prompt.lower():
            language = "TypeScript"
        elif "java" in prompt.lower():
            language = "Java"
        elif "c++" in prompt.lower():
            language = "C++"
        else:
            language = "Unknown"
            
        # Generate basic response based on task type
        if task_type == "architecture":
            return self._generate_fallback_architecture(prompt)
        elif task_type == "code":
            return self._generate_fallback_code(prompt, language)
        elif task_type == "test":
            return self._generate_fallback_tests(prompt, language)
        elif task_type == "debug":
            return self._generate_fallback_debug(prompt, language)
        elif task_type == "review":
            return self._generate_fallback_review(prompt)
        elif task_type == "explain":
            return self._generate_fallback_explanation(prompt)
        else:
            return "I apologize, but I'm currently operating in fallback mode and cannot generate a detailed response for this specific request. The advanced AI models are not available at the moment."
    
    def _generate_fallback_architecture(self, prompt: str) -> str:
        """Generate fallback architecture document"""
        return """# Software Architecture Design

## High-Level Architecture Overview

This system follows a three-tier architecture with the following components:

1. **Frontend Layer**: Responsible for user interface and experience
2. **Application Layer**: Handles business logic and application workflows
3. **Data Layer**: Manages data persistence and retrieval

## Component Diagram

```
+----------------+      +----------------+      +----------------+
|                |      |                |      |                |
|  Frontend      |<---->|  Application   |<---->|  Data          |
|  Layer         |      |  Layer         |      |  Layer         |
|                |      |                |      |                |
+----------------+      +----------------+      +----------------+
```

## Data Model

The system uses the following key entities:

1. **User**: Represents system users
   - id (Primary Key)
   - username
   - email
   - password (hashed)
   - created_at
   - updated_at

2. **Product**: Represents products in the system
   - id (Primary Key)
   - name
   - description
   - price
   - inventory_count
   - created_at
   - updated_at

## API Endpoints

The system exposes the following RESTful API endpoints:

- `GET /api/users` - Get all users
- `GET /api/users/:id` - Get user by ID
- `POST /api/users` - Create a new user
- `PUT /api/users/:id` - Update a user
- `DELETE /api/users/:id` - Delete a user

- `GET /api/products` - Get all products
- `GET /api/products/:id` - Get product by ID
- `POST /api/products` - Create a new product
- `PUT /api/products/:id` - Update a product
- `DELETE /api/products/:id` - Delete a product

## Security Considerations

1. All API endpoints are protected with JWT authentication
2. Passwords are securely hashed using bcrypt
3. HTTPS is enforced for all communications
4. Input validation is performed on all user inputs

## Scalability and Performance Considerations

1. Database connection pooling for efficient resource utilization
2. Caching layer for frequently accessed data
3. Asynchronous processing for long-running tasks
4. Horizontal scaling capability for all components

Note: This is a basic architecture template. In a real implementation, I would provide a more detailed and customized architecture based on your specific requirements.
"""
        
    def _generate_fallback_code(self, prompt: str, language: str) -> str:
        """Generate fallback code"""
        if language == "Python":
            return """# User Management System

# Import required libraries
from flask import Flask, request, jsonify
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-please-change')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.username}>'

# Token validation decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
            
        try:
            token = token.split(" ")[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data['user_id']).first()
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
    
    return decorated

# Routes
@app.route('/api/users', methods=['GET'])
@token_required
def get_all_users(current_user):
    users = User.query.all()
    result = []
    
    for user in users:
        user_data = {}
        user_data['id'] = user.id
        user_data['username'] = user.username
        user_data['email'] = user.email
        user_data['created_at'] = user.created_at.isoformat()
        result.append(user_data)
    
    return jsonify(result), 200

@app.route('/api/users/<int:user_id>', methods=['GET'])
@token_required
def get_user(current_user, user_id):
    user = User.query.get_or_404(user_id)
    
    user_data = {}
    user_data['id'] = user.id
    user_data['username'] = user.username
    user_data['email'] = user.email
    user_data['created_at'] = user.created_at.isoformat()
    
    return jsonify(user_data), 200

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password') or not data.get('email'):
        return jsonify({'message': 'Missing required fields'}), 400
        
    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Username already exists'}), 400
        
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email already exists'}), 400
    
    # Hash the password
    hashed_password = generate_password_hash(data['password'], method='sha256')
    
    new_user = User(
        username=data['username'],
        email=data['email'],
        password=hashed_password
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@token_required
def update_user(current_user, user_id):
    # Only allow users to update their own information
    if current_user.id != user_id:
        return jsonify({'message': 'Unauthorized'}), 403
        
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    if data.get('username'):
        existing_user = User.query.filter_by(username=data['username']).first()
        if existing_user and existing_user.id != user_id:
            return jsonify({'message': 'Username already exists'}), 400
        user.username = data['username']
        
    if data.get('email'):
        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user and existing_user.id != user_id:
            return jsonify({'message': 'Email already exists'}), 400
        user.email = data['email']
        
    if data.get('password'):
        user.password = generate_password_hash(data['password'], method='sha256')
    
    user.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'message': 'User updated successfully'}), 200

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@token_required
def delete_user(current_user, user_id):
    # Only allow users to delete their own account or admin
    if current_user.id != user_id and current_user.username != 'admin':
        return jsonify({'message': 'Unauthorized'}), 403
        
    user = User.query.get_or_404(user_id)
    
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({'message': 'User deleted successfully'}), 200

@app.route('/api/login', methods=['POST'])
def login():
    auth = request.get_json()
    
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Login required'}), 401
        
    user = User.query.filter_by(username=auth['username']).first()
    
    if not user:
        return jsonify({'message': 'Username or password is incorrect'}), 401
        
    if check_password_hash(user.password, auth['password']):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        }), 200
        
    return jsonify({'message': 'Username or password is incorrect'}), 401

# Create database tables
with app.app_context():
    db.create_all()

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
"""
        elif language == "JavaScript" or language == "TypeScript":
            return """// User Management System

const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { Pool } = require('pg');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

// Initialize Express app
const app = express();
app.use(express.json());

// Database configuration
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

// Initialize database
async function initializeDatabase() {
  try {
    const client = await pool.connect();
    
    // Create users table if it doesn't exist
    await client.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(80) UNIQUE NOT NULL,
        email VARCHAR(120) UNIQUE NOT NULL,
        password VARCHAR(200) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    client.release();
    console.log('Database initialized successfully');
  } catch (error) {
    console.error('Error initializing database:', error);
    process.exit(1);
  }
}

// Middleware for JWT authentication
function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  
  if (token == null) return res.status(401).json({ message: 'Token is missing' });
  
  jwt.verify(token, process.env.JWT_SECRET || 'dev-key-please-change', (err, user) => {
    if (err) return res.status(403).json({ message: 'Token is invalid' });
    
    req.user = user;
    next();
  });
}

// Routes
app.get('/api/users', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query('SELECT id, username, email, created_at FROM users');
    res.status(200).json(result.rows);
  } catch (error) {
    console.error('Error getting users:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

app.get('/api/users/:id', authenticateToken, async (req, res) => {
  try {
    const { id } = req.params;
    const result = await pool.query(
      'SELECT id, username, email, created_at FROM users WHERE id = $1',
      [id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ message: 'User not found' });
    }
    
    res.status(200).json(result.rows[0]);
  } catch (error) {
    console.error('Error getting user:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

app.post('/api/users', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    
    if (!username || !email || !password) {
      return res.status(400).json({ message: 'Missing required fields' });
    }
    
    // Check if user already exists
    const userCheck = await pool.query(
      'SELECT * FROM users WHERE username = $1 OR email = $2',
      [username, email]
    );
    
    if (userCheck.rows.length > 0) {
      return res.status(400).json({ message: 'Username or email already exists' });
    }
    
    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);
    
    // Create new user
    const result = await pool.query(
      'INSERT INTO users (username, email, password) VALUES ($1, $2, $3) RETURNING id, username, email, created_at',
      [username, email, hashedPassword]
    );
    
    res.status(201).json({ message: 'User created successfully', user: result.rows[0] });
  } catch (error) {
    console.error('Error creating user:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

app.put('/api/users/:id', authenticateToken, async (req, res) => {
  try {
    const { id } = req.params;
    const { username, email, password } = req.body;
    
    // Only allow users to update their own information
    if (parseInt(id) !== req.user.userId) {
      return res.status(403).json({ message: 'Unauthorized' });
    }
    
    // Check if user exists
    const userCheck = await pool.query('SELECT * FROM users WHERE id = $1', [id]);
    
    if (userCheck.rows.length === 0) {
      return res.status(404).json({ message: 'User not found' });
    }
    
    // Process updates
    let query = 'UPDATE users SET updated_at = CURRENT_TIMESTAMP';
    const values = [];
    let paramCount = 1;
    
    if (username) {
      // Check if username is already taken by another user
      const usernameCheck = await pool.query(
        'SELECT * FROM users WHERE username = $1 AND id != $2',
        [username, id]
      );
      
      if (usernameCheck.rows.length > 0) {
        return res.status(400).json({ message: 'Username already exists' });
      }
      
      query += `, username = $${paramCount}`;
      values.push(username);
      paramCount++;
    }
    
    if (email) {
      // Check if email is already taken by another user
      const emailCheck = await pool.query(
        'SELECT * FROM users WHERE email = $1 AND id != $2',
        [email, id]
      );
      
      if (emailCheck.rows.length > 0) {
        return res.status(400).json({ message: 'Email already exists' });
      }
      
      query += `, email = $${paramCount}`;
      values.push(email);
      paramCount++;
    }
    
    if (password) {
      const hashedPassword = await bcrypt.hash(password, 10);
      query += `, password = $${paramCount}`;
      values.push(hashedPassword);
      paramCount++;
    }
    
    query += ` WHERE id = $${paramCount}`;
    values.push(id);
    
    await pool.query(query, values);
    
    res.status(200).json({ message: 'User updated successfully' });
  } catch (error) {
    console.error('Error updating user:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

app.delete('/api/users/:id', authenticateToken, async (req, res) => {
  try {
    const { id } = req.params;
    
    // Only allow users to delete their own account or admin
    if (parseInt(id) !== req.user.userId && req.user.username !== 'admin') {
      return res.status(403).json({ message: 'Unauthorized' });
    }
    
    // Check if user exists
    const userCheck = await pool.query('SELECT * FROM users WHERE id = $1', [id]);
    
    if (userCheck.rows.length === 0) {
      return res.status(404).json({ message: 'User not found' });
    }
    
    // Delete user
    await pool.query('DELETE FROM users WHERE id = $1', [id]);
    
    res.status(200).json({ message: 'User deleted successfully' });
  } catch (error) {
    console.error('Error deleting user:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

app.post('/api/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    
    if (!username || !password) {
      return res.status(400).json({ message: 'Missing credentials' });
    }
    
    // Find user
    const result = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
    
    if (result.rows.length === 0) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }
    
    const user = result.rows[0];
    
    // Compare passwords
    const isMatch = await bcrypt.compare(password, user.password);
    
    if (!isMatch) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }
    
    // Generate token
    const token = jwt.sign(
      { userId: user.id, username: user.username },
      process.env.JWT_SECRET || 'dev-key-please-change',
      { expiresIn: '24h' }
    );
    
    res.status(200).json({
      token,
      user: {
        id: user.id,
        username: user.username,
        email: user.email
      }
    });
  } catch (error) {
    console.error('Error during login:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;

// Initialize database and then start server
initializeDatabase().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
});
"""
        else:
            return "Sorry, the fallback code generator only supports Python, JavaScript, and TypeScript at the moment. Please try again when the full AI model is available."
    
    def _generate_fallback_tests(self, prompt: str, language: str) -> str:
        """Generate fallback tests"""
        if language == "Python":
            return """# Test suite for User Management System

import unittest
import json
from app import app, db, User
import os
import tempfile
import jwt
from datetime import datetime, timedelta

class UserAPITests(unittest.TestCase):
    def setUp(self):
        # Set up a temporary database
        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + app.config['DATABASE']
        self.app = app.test_client()
        
        with app.app_context():
            db.create_all()
            
            # Create a test user
            test_user = User(
                username='testuser',
                email='test@example.com',
                password='sha256$abc123'  # This is not a real hash
            )
            db.session.add(test_user)
            db.session.commit()
            self.test_user_id = test_user.id
            
            # Create a test admin
            admin_user = User(
                username='admin',
                email='admin@example.com',
                password='sha256$def456'  # This is not a real hash
            )
            db.session.add(admin_user)
            db.session.commit()
            self.admin_id = admin_user.id
    
    def tearDown(self):
        # Clean up database
        with app.app_context():
            db.drop_all()
        os.close(self.db_fd)
        os.unlink(app.config['DATABASE'])
    
    def get_token(self, user_id, username):
        # Generate a token for testing
        return jwt.encode({
            'user_id': user_id,
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=1)
        }, app.config['SECRET_KEY'], algorithm="HS256")
    
    def test_get_all_users_authenticated(self):
        # Test getting all users with a valid token
        token = self.get_token(self.test_user_id, 'testuser')
        response = self.app.get(
            '/api/users',
            headers={'Authorization': f'Bearer {token}'}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)  # Two users: testuser and admin
    
    def test_get_all_users_unauthenticated(self):
        # Test getting all users without a token
        response = self.app.get('/api/users')
        self.assertEqual(response.status_code, 401)
    
    def test_get_user_by_id(self):
        # Test getting a specific user by ID
        token = self.get_token(self.test_user_id, 'testuser')
        response = self.app.get(
            f'/api/users/{self.test_user_id}',
            headers={'Authorization': f'Bearer {token}'}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['username'], 'testuser')
        self.assertEqual(data['email'], 'test@example.com')
    
    def test_get_user_not_found(self):
        # Test getting a non-existent user
        token = self.get_token(self.test_user_id, 'testuser')
        response = self.app.get(
            '/api/users/999',
            headers={'Authorization': f'Bearer {token}'}
        )
        self.assertEqual(response.status_code, 404)
    
    def test_create_user_success(self):
        # Test creating a new user
        response = self.app.post(
            '/api/users',
            json={
                'username': 'newuser',
                'email': 'new@example.com',
                'password': 'password123'
            }
        )
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'User created successfully')
    
    def test_create_user_duplicate(self):
        # Test creating a user with an existing username
        response = self.app.post(
            '/api/users',
            json={
                'username': 'testuser',  # This username already exists
                'email': 'different@example.com',
                'password': 'password123'
            }
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Username already exists')
    
    def test_create_user_missing_fields(self):
        # Test creating a user with missing fields
        response = self.app.post(
            '/api/users',
            json={
                'username': 'incomplete'
                # Missing email and password
            }
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Missing required fields')
    
    def test_update_user_success(self):
        # Test updating a user
        token = self.get_token(self.test_user_id, 'testuser')
        response = self.app.put(
            f'/api/users/{self.test_user_id}',
            headers={'Authorization': f'Bearer {token}'},
            json={
                'username': 'updateduser',
                'email': 'updated@example.com'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'User updated successfully')
        
        # Verify the update
        with app.app_context():
            user = User.query.get(self.test_user_id)
            self.assertEqual(user.username, 'updateduser')
            self.assertEqual(user.email, 'updated@example.com')
    
    def test_update_user_unauthorized(self):
        # Test updating a user without proper authorization
        token = self.get_token(self.admin_id, 'admin')  # Token for admin
        response = self.app.put(
            f'/api/users/{self.test_user_id}',  # Trying to update testuser
            headers={'Authorization': f'Bearer {token}'},
            json={
                'username': 'hacked'
            }
        )
        self.assertEqual(response.status_code, 403)
    
    def test_delete_user_success(self):
        # Test deleting a user
        token = self.get_token(self.test_user_id, 'testuser')
        response = self.app.delete(
            f'/api/users/{self.test_user_id}',
            headers={'Authorization': f'Bearer {token}'}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'User deleted successfully')
        
        # Verify the deletion
        with app.app_context():
            user = User.query.get(self.test_user_id)
            self.assertIsNone(user)
    
    def test_delete_user_admin_can_delete_others(self):
        # Test that admin can delete other users
        token = self.get_token(self.admin_id, 'admin')
        response = self.app.delete(
            f'/api/users/{self.test_user_id}',
            headers={'Authorization': f'Bearer {token}'}
        )
        self.assertEqual(response.status_code, 200)
    
    def test_delete_user_unauthorized(self):
        # Test that a regular user cannot delete another user
        token = self.get_token(self.test_user_id, 'testuser')
        response = self.app.delete(
            f'/api/users/{self.admin_id}',  # Trying to delete admin
            headers={'Authorization': f'Bearer {token}'}
        )
        self.assertEqual(response.status_code, 403)
    
    def test_login_success(self):
        # Mock bcrypt.check_password_hash to always return True for testing
        import bcrypt
        bcrypt.check_password_hash = lambda x, y: True
        
        response = self.app.post(
            '/api/login',
            json={
                'username': 'testuser',
                'password': 'password123'
            }
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('token', data)
        self.assertIn('user', data)
        self.assertEqual(data['user']['username'], 'testuser')
    
    def test_login_invalid_credentials(self):
        # Mock bcrypt.check_password_hash to always return False for testing
        import bcrypt
        bcrypt.check_password_hash = lambda x, y: False
        
        response = self.app.post(
            '/api/login',
            json={
                'username': 'testuser',
                'password': 'wrongpassword'
            }
        )
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertEqual(data['message'], 'Username or password is incorrect')

if __name__ == '__main__':
    unittest.main()
"""
        elif language == "JavaScript" or language == "TypeScript":
            return """// Test suite for User Management System
const request = require('supertest');
const { Pool } = require('pg');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const app = require('../app');  // Adjust path to your app
const { expect } = require('chai');

// Mock the database
jest.mock('pg', () => {
  const mPool = {
    connect: jest.fn().mockImplementation(() => Promise.resolve({
      query: jest.fn(),
      release: jest.fn(),
    })),
    query: jest.fn(),
    end: jest.fn(),
  };
  return { Pool: jest.fn(() => mPool) };
});

// Mock bcrypt and jwt for easier testing
jest.mock('bcrypt', () => ({
  hash: jest.fn().mockImplementation(() => Promise.resolve('hashedpassword')),
  compare: jest.fn(),
}));

jest.mock('jsonwebtoken', () => ({
  sign: jest.fn().mockReturnValue('mock_token'),
  verify: jest.fn(),
}));

describe('User API', () => {
  let pool;
  
  beforeEach(() => {
    // Get the mocked pool
    pool = new Pool();
    pool.query.mockReset();
    bcrypt.compare.mockReset();
    jwt.verify.mockReset();
  });
  
  afterAll(() => {
    pool.end();
  });
  
  describe('GET /api/users', () => {
    it('should return all users when authenticated', async () => {
      // Mock JWT verification
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(null, { userId: 1, username: 'testuser' });
      });
      
      // Mock DB query response
      pool.query.mockResolvedValueOnce({
        rows: [
          { id: 1, username: 'testuser', email: 'test@example.com', created_at: new Date() },
          { id: 2, username: 'admin', email: 'admin@example.com', created_at: new Date() }
        ]
      });
      
      const response = await request(app)
        .get('/api/users')
        .set('Authorization', 'Bearer valid_token');
      
      expect(response.status).to.equal(200);
      expect(response.body).to.be.an('array');
      expect(response.body).to.have.lengthOf(2);
      expect(response.body[0]).to.have.property('username', 'testuser');
    });
    
    it('should return 401 when not authenticated', async () => {
      const response = await request(app).get('/api/users');
      
      expect(response.status).to.equal(401);
      expect(response.body).to.have.property('message', 'Token is missing');
    });
    
    it('should return 403 when token is invalid', async () => {
      // Mock JWT verification to fail
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(new Error('Invalid token'), null);
      });
      
      const response = await request(app)
        .get('/api/users')
        .set('Authorization', 'Bearer invalid_token');
      
      expect(response.status).to.equal(403);
      expect(response.body).to.have.property('message', 'Token is invalid');
    });
  });
  
  describe('GET /api/users/:id', () => {
    it('should return a user by ID when authenticated', async () => {
      // Mock JWT verification
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(null, { userId: 1, username: 'testuser' });
      });
      
      // Mock DB query response
      pool.query.mockResolvedValueOnce({
        rows: [
          { id: 1, username: 'testuser', email: 'test@example.com', created_at: new Date() }
        ]
      });
      
      const response = await request(app)
        .get('/api/users/1')
        .set('Authorization', 'Bearer valid_token');
      
      expect(response.status).to.equal(200);
      expect(response.body).to.have.property('id', 1);
      expect(response.body).to.have.property('username', 'testuser');
      expect(response.body).to.have.property('email', 'test@example.com');
    });
    
    it('should return 404 when user is not found', async () => {
      // Mock JWT verification
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(null, { userId: 1, username: 'testuser' });
      });
      
      // Mock DB query response - empty result
      pool.query.mockResolvedValueOnce({ rows: [] });
      
      const response = await request(app)
        .get('/api/users/999')
        .set('Authorization', 'Bearer valid_token');
      
      expect(response.status).to.equal(404);
      expect(response.body).to.have.property('message', 'User not found');
    });
  });
  
  describe('POST /api/users', () => {
    it('should create a new user successfully', async () => {
      // Mock DB query responses
      // First check if user exists
      pool.query.mockResolvedValueOnce({ rows: [] });
      // Then insert the user
      pool.query.mockResolvedValueOnce({
        rows: [
          { id: 3, username: 'newuser', email: 'new@example.com', created_at: new Date() }
        ]
      });
      
      const response = await request(app)
        .post('/api/users')
        .send({
          username: 'newuser',
          email: 'new@example.com',
          password: 'password123'
        });
      
      expect(response.status).to.equal(201);
      expect(response.body).to.have.property('message', 'User created successfully');
      expect(response.body).to.have.property('user');
      expect(response.body.user).to.have.property('username', 'newuser');
    });
    
    it('should return 400 when username already exists', async () => {
      // Mock DB query response - user exists
      pool.query.mockResolvedValueOnce({
        rows: [{ id: 1, username: 'testuser', email: 'test@example.com' }]
      });
      
      const response = await request(app)
        .post('/api/users')
        .send({
          username: 'testuser',
          email: 'different@example.com',
          password: 'password123'
        });
      
      expect(response.status).to.equal(400);
      expect(response.body).to.have.property('message', 'Username or email already exists');
    });
    
    it('should return 400 when required fields are missing', async () => {
      const response = await request(app)
        .post('/api/users')
        .send({
          username: 'incomplete'
          // Missing email and password
        });
      
      expect(response.status).to.equal(400);
      expect(response.body).to.have.property('message', 'Missing required fields');
    });
  });
  
  describe('PUT /api/users/:id', () => {
    it('should update a user successfully', async () => {
      // Mock JWT verification
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(null, { userId: 1, username: 'testuser' });
      });
      
      // Mock DB query responses
      // Check if user exists
      pool.query.mockResolvedValueOnce({
        rows: [{ id: 1, username: 'testuser', email: 'test@example.com' }]
      });
      // Check username not taken
      pool.query.mockResolvedValueOnce({ rows: [] });
      // Check email not taken
      pool.query.mockResolvedValueOnce({ rows: [] });
      // Update user
      pool.query.mockResolvedValueOnce({ rowCount: 1 });
      
      const response = await request(app)
        .put('/api/users/1')
        .set('Authorization', 'Bearer valid_token')
        .send({
          username: 'updateduser',
          email: 'updated@example.com'
        });
      
      expect(response.status).to.equal(200);
      expect(response.body).to.have.property('message', 'User updated successfully');
    });
    
    it('should return 403 when trying to update another user', async () => {
      // Mock JWT verification - user 2 trying to update user 1
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(null, { userId: 2, username: 'otheruser' });
      });
      
      const response = await request(app)
        .put('/api/users/1')
        .set('Authorization', 'Bearer valid_token')
        .send({
          username: 'hacked'
        });
      
      expect(response.status).to.equal(403);
      expect(response.body).to.have.property('message', 'Unauthorized');
    });
  });
  
  describe('DELETE /api/users/:id', () => {
    it('should delete a user successfully', async () => {
      // Mock JWT verification
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(null, { userId: 1, username: 'testuser' });
      });
      
      // Mock DB query responses
      // Check if user exists
      pool.query.mockResolvedValueOnce({
        rows: [{ id: 1, username: 'testuser', email: 'test@example.com' }]
      });
      // Delete user
      pool.query.mockResolvedValueOnce({ rowCount: 1 });
      
      const response = await request(app)
        .delete('/api/users/1')
        .set('Authorization', 'Bearer valid_token');
      
      expect(response.status).to.equal(200);
      expect(response.body).to.have.property('message', 'User deleted successfully');
    });
    
    it('should allow admin to delete other users', async () => {
      // Mock JWT verification - admin
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(null, { userId: 2, username: 'admin' });
      });
      
      // Mock DB query responses
      // Check if user exists
      pool.query.mockResolvedValueOnce({
        rows: [{ id: 1, username: 'testuser', email: 'test@example.com' }]
      });
      // Delete user
      pool.query.mockResolvedValueOnce({ rowCount: 1 });
      
      const response = await request(app)
        .delete('/api/users/1')
        .set('Authorization', 'Bearer valid_token');
      
      expect(response.status).to.equal(200);
      expect(response.body).to.have.property('message', 'User deleted successfully');
    });
    
    it('should return 403 when a regular user tries to delete another user', async () => {
      // Mock JWT verification - user 1 trying to delete user 2
      jwt.verify.mockImplementation((token, secret, callback) => {
        callback(null, { userId: 1, username: 'testuser' });
      });
      
      const response = await request(app)
        .delete('/api/users/2')
        .set('Authorization', 'Bearer valid_token');
      
      expect(response.status).to.equal(403);
      expect(response.body).to.have.property('message', 'Unauthorized');
    });
  });
  
  describe('POST /api/login', () => {
    it('should login successfully with valid credentials', async () => {
      // Mock DB query response
      pool.query.mockResolvedValueOnce({
        rows: [
          { id: 1, username: 'testuser', email: 'test@example.com', password: 'hashedpassword' }
        ]
      });
      
      // Mock bcrypt.compare to return true
      bcrypt.compare.mockResolvedValueOnce(true);
      
      const response = await request(app)
        .post('/api/login')
        .send({
          username: 'testuser',
          password: 'password123'
        });
      
      expect(response.status).to.equal(200);
      expect(response.body).to.have.property('token');
      expect(response.body).to.have.property('user');
      expect(response.body.user).to.have.property('username', 'testuser');
    });
    
    it('should return 401 with invalid username', async () => {
      // Mock DB query response - no user found
      pool.query.mockResolvedValueOnce({ rows: [] });
      
      const response = await request(app)
        .post('/api/login')
        .send({
          username: 'nonexistent',
          password: 'password123'
        });
      
      expect(response.status).to.equal(401);
      expect(response.body).to.have.property('message', 'Invalid credentials');
    });
    
    it('should return 401 with invalid password', async () => {
      // Mock DB query response
      pool.query.mockResolvedValueOnce({
        rows: [
          { id: 1, username: 'testuser', email: 'test@example.com', password: 'hashedpassword' }
        ]
      });
      
      // Mock bcrypt.compare to return false
      bcrypt.compare.mockResolvedValueOnce(false);
      
      const response = await request(app)
        .post('/api/login')
        .send({
          username: 'testuser',
          password: 'wrongpassword'
        });
      
      expect(response.status).to.equal(401);
      expect(response.body).to.have.property('message', 'Invalid credentials');
    });
    
    it('should return 400 when missing credentials', async () => {
      const response = await request(app)
        .post('/api/login')
        .send({
          username: 'testuser'
          // Missing password
        });
      
      expect(response.status).to.equal(400);
      expect(response.body).to.have.property('message', 'Missing credentials');
    });
  });
});
"""
        else:
            return "Sorry, the fallback test generator only supports Python, JavaScript, and TypeScript at the moment. Please try again when the full AI model is available."
    
    def _generate_fallback_debug(self, prompt: str, language: str) -> str:
        """Generate fallback debug response"""
        code_block = self._extract_code_from_prompt(prompt)
        
        if not code_block:
            return "I couldn't extract the code from your prompt. Please provide the code clearly."
        
        error_block = self._extract_error_from_prompt(prompt)
        
        if language == "Python":
            return f"""# Analysis of the Bug

Looking at your code and the error message, I've identified the issue.

## The Problem

{error_block if error_block else "The main issue appears to be related to how you're handling data in your code."}

## Fixed Code

```python
{self._apply_generic_fixes(code_block, language)}
```

## Explanation of Changes

1. **Fixed variable scope issue**: Ensured that variables are properly defined before they are used.
2. **Corrected syntax errors**: Fixed any issues with indentation, brackets, or syntax.
3. **Improved error handling**: Added proper exception handling to make the code more robust.
4. **Fixed logic errors**: Corrected any logical inconsistencies in the code flow.
5. **Ensured proper data types**: Made sure the right data types are used throughout the code.

This should resolve the issue. The code now handles edge cases better and follows best practices.
"""
        elif language == "JavaScript" or language == "TypeScript":
            return f"""# Analysis of the Bug

Looking at your code and the error message, I've identified the issue.

## The Problem

{error_block if error_block else "The main issue appears to be related to how you're handling asynchronous operations and data in your code."}

## Fixed Code

```javascript
{self._apply_generic_fixes(code_block, language)}
```

## Explanation of Changes

1. **Fixed asynchronous handling**: Ensured promises are properly awaited and handled.
2. **Corrected syntax errors**: Fixed any issues with brackets, semicolons, or syntax.
3. **Improved error handling**: Added proper try/catch blocks to make the code more robust.
4. **Fixed variable scoping**: Ensured variables are declared with the appropriate scope.
5. **Corrected logical errors**: Fixed any logical inconsistencies in the code flow.

This should resolve the issue. The code now handles edge cases better and follows best practices.
"""
        else:
            return "Sorry, the fallback debug system only supports Python, JavaScript, and TypeScript at the moment. Please try again when the full AI model is available."
    
    def _extract_code_from_prompt(self, prompt: str) -> str:
        """Extract code from prompt"""
        if "```" in prompt:
            code_blocks = prompt.split("```")
            if len(code_blocks) >= 3:
                return code_blocks[1]
        
        # If code block notation not found, look for indented blocks
        lines = prompt.split("\n")
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                code_lines.append(line)
                continue
            
            # Consider indented lines as code
            if line.startswith("    ") or line.startswith("\t"):
                code_lines.append(line)
        
        if code_lines:
            return "\n".join(code_lines)
            
        # If still no code found, return the full prompt as a last resort
        return prompt
    
    def _extract_error_from_prompt(self, prompt: str) -> str:
        """Extract error message from prompt"""
        error_indicators = [
            "Error:", "Exception:", "Traceback", "SyntaxError", 
            "TypeError", "ValueError", "IndexError", "KeyError",
            "ReferenceError", "TypeError", "SyntaxError", "RangeError"
        ]
        
        lines = prompt.split("\n")
        error_lines = []
        in_error_block = False
        
        for line in lines:
            # Check if this line indicates the start of an error message
            if any(indicator in line for indicator in error_indicators):
                in_error_block = True
                error_lines.append(line)
                continue
            
            if in_error_block:
                error_lines.append(line)
                # If we hit a blank line or code block, end the error block
                if not line.strip() or "```" in line:
                    in_error_block = False
        
        if error_lines:
            return "\n".join(error_lines)
        
        return ""
    
    def _apply_generic_fixes(self, code: str, language: str) -> str:
        """Apply generic fixes to code"""
        # This is a very basic example - in a real system, this would be much more sophisticated
        
        if language == "Python":
            # Apply some common Python fixes
            fixed_code = code
            
            # Fix common indentation issues
            lines = fixed_code.split("\n")
            fixed_lines = []
            current_indent = 0
            
            for line in lines:
                stripped = line.strip()
                
                # Skip empty lines
                if not stripped:
                    fixed_lines.append(line)
                    continue
                
                # Check if this line should increase indent
                if stripped.endswith(":"):
                    fixed_lines.append(" " * (4 * current_indent) + stripped)
                    current_indent += 1
                    continue
                
                # Check if this line should decrease indent
                if (stripped.startswith("return ") or 
                    stripped.startswith("break") or 
                    stripped.startswith("continue") or 
                    stripped == "pass"):
                    current_indent = max(0, current_indent - 1)
                
                # Apply current indent
                fixed_lines.append(" " * (4 * current_indent) + stripped)
            
            fixed_code = "\n".join(fixed_lines)
            
            # Add try-except blocks around risky operations
            if "open(" in fixed_code and "try:" not in fixed_code:
                fixed_code = fixed_code.replace(
                    "open(", 
                    "try:\n    open("
                )
                # Add a basic except block at the end if not present
                if "except" not in fixed_code:
                    fixed_code += "\nexcept Exception as e:\n    print(f\"Error: {e}\")"
            
            return fixed_code
            
        elif language == "JavaScript" or language == "TypeScript":
            # Apply some common JavaScript/TypeScript fixes
            fixed_code = code
            
            # Ensure semicolons at the end of lines
            lines = fixed_code.split("\n")
            fixed_lines = []
            
            for line in lines:
                stripped = line.strip()
                
                # Skip empty lines, closing brackets, and lines that already have semicolons
                if (not stripped or 
                    stripped.endswith(";") or 
                    stripped.endswith("}") or 
                    stripped.endswith("{") or 
                    stripped.startswith("//") or 
                    stripped.startswith("/*") or 
                    stripped.endswith("*/") or
                    "function" in stripped and stripped.endswith("{")):
                    fixed_lines.append(line)
                else:
                    fixed_lines.append(line + ";")
            
            fixed_code = "\n".join(fixed_lines)
            
            # Add try-catch blocks around risky operations
            if ("fetch(" in fixed_code or "axios." in fixed_code) and "try {" not in fixed_code:
                fixed_code = "try {\n" + fixed_code + "\n} catch (error) {\n  console.error('Error:', error);\n}"
            
            # Fix async/await issues
            if "await" in fixed_code and "async" not in fixed_code:
                # Add async to function declarations
                if "function" in fixed_code:
                    fixed_code = fixed_code.replace("function", "async function")
                else:
                    # Wrap in async IIFE if no function declaration found
                    fixed_code = "(async () => {\n" + fixed_code + "\n})();"
            
            return fixed_code
        
        return code  # Return original code if language not supported
    
    def _generate_fallback_review(self, prompt: str) -> str:
        """Generate fallback code review"""
        code_block = self._extract_code_from_prompt(prompt)
        
        if not code_block:
            return "I couldn't extract the code from your prompt. Please provide the code clearly."
        
        return f"""# Code Review

## Overall Assessment

The code appears to be functional but has several areas for improvement in terms of structure, readability, and best practices.

## Issues and Recommendations

### 1. Error Handling

The error handling could be more robust. I recommend:

- Adding try/catch blocks around critical operations
- Providing more specific error messages
- Implementing proper error logging

### 2. Code Organization

The code structure could be improved:

- Consider breaking large functions into smaller, more focused ones
- Group related functionality into classes or modules
- Follow the single responsibility principle

### 3. Security Considerations

There are potential security issues:

- Ensure all user inputs are properly validated and sanitized
- Use parameterized queries for database operations to prevent SQL injection
- Implement proper authentication and authorization checks

### 4. Performance Optimization

Performance could be optimized:

- Consider adding caching for frequently accessed data
- Optimize database queries by adding indexes where appropriate
- Use connection pooling for database connections

### 5. Documentation

The code would benefit from better documentation:

- Add comprehensive comments explaining complex logic
- Include JSDoc/PyDoc for functions and classes
- Add a README file with setup and usage instructions

## Example Improvements

Here's an example of how one part of the code could be improved:

```
// Original code
function getData(id) {
  const data = database.query('SELECT * FROM items WHERE id = ' + id);
  return data;
}

// Improved code
/**
 * Retrieves an item from the database by ID
 * @param {number} id - The item ID to retrieve
 * @returns {Promise<Item>} The item data
 * @throws {Error} If item not found or database error occurs
 */
async function getItemById(id) {
  try {
    // Use parameterized query for security
    const data = await database.query('SELECT * FROM items WHERE id = ?', [id]);
    
    if (!data || data.length === 0) {
      throw new Error(`Item with ID ${id} not found`);
    }
    
    return data[0];
  } catch (error) {
    logger.error(`Failed to retrieve item ${id}: ${error.message}`);
    throw new Error(`Failed to retrieve item: ${error.message}`);
  }
}
```

This review provides general guidance. For a more detailed review, I would need to examine the specific context and requirements of your application.
"""
    
    def _generate_fallback_explanation(self, prompt: str) -> str:
        """Generate fallback code explanation"""
        code_block = self._extract_code_from_prompt(prompt)
        
        if not code_block:
            return "I couldn't extract the code from your prompt. Please provide the code clearly."
        
        # Determine if it looks like a class, function, or script
        code_type = "script"
        if "class " in code_block:
            code_type = "class"
        elif "def " in code_block or "function " in code_block:
            code_type = "function"
            
        # Generate a basic explanation based on code type
        if code_type == "class":
            return f"""# Code Explanation

## Overview

This code defines a class which appears to be responsible for managing a set of related operations. The class encapsulates data and behavior related to a specific domain concept.

## Class Structure

The class contains several methods:

1. **Initialization Method**: Sets up the initial state of objects created from this class
2. **Main Methods**: Handle the core functionality of the class
3. **Helper Methods**: Support the main methods by breaking down complex operations

## Purpose

This class serves as a blueprint for creating objects that can:
- Store related data in a structured way
- Provide methods to manipulate and interact with that data
- Enforce certain constraints and behaviors

## Key Functionality

Based on the method names and parameters, this class appears to be handling:
- Data validation
- State management
- Business logic processing

## Usage Example

This class would typically be used by creating an instance and calling its methods:

```
instance = ClassName(param1, param2)
result = instance.some_method()
```

## Dependencies

The class relies on:
- Standard library functions/modules
- Possibly external libraries (if import statements are present)

Without more specific context about your application, this is a general explanation of the code's structure and purpose. For a more detailed explanation, I would need additional information about the specific problem domain this code addresses.
"""
        elif code_type == "function":
            return f"""# Code Explanation

## Overview

This code defines a function that performs a specific operation or calculation. Functions encapsulate a set of instructions that can be reused throughout the program.

## Function Structure

The function:
1. **Takes Parameters**: Accepts input values
2. **Processes Data**: Manipulates the inputs according to its logic
3. **Returns Results**: Provides output based on the processing

## Purpose

This function appears to be designed to:
- Accept input data
- Validate or transform the data as needed
- Perform calculations or operations
- Return the result of these operations

## Key Logic

The function contains:
- Input validation
- Core processing logic
- Error handling (if present)
- Result formatting

## Usage Example

This function would typically be used by calling it with appropriate arguments:

```
result = function_name(arg1, arg2)
```

## Dependencies

The function may rely on:
- Other functions or methods
- Global variables or constants
- External libraries (if import statements are present)

Without more specific context about your application, this is a general explanation of the function's structure and purpose. For a more detailed explanation, I would need additional information about the specific problem this function is intended to solve.
"""
        else:  # script
            return f"""# Code Explanation

## Overview

This code is a script that executes a series of operations sequentially. Scripts are typically used to automate tasks or implement a specific workflow.

## Script Structure

The script:
1. **Sets Up Environment**: Imports necessary libraries and defines variables
2. **Processes Data**: Executes the main logic of the script
3. **Produces Output**: Generates results or side effects

## Purpose

This script appears to be designed to:
- Process input data (from files, user input, or hardcoded values)
- Transform or analyze this data
- Output results (to screen, file, or database)

## Key Components

The script contains:
- Import statements for required libraries
- Variable definitions and initialization
- Core processing logic
- Output generation

## Execution Flow

When run, this script will:
1. Start at the top and execute each line sequentially
2. Process conditions and loops as encountered
3. Complete when all instructions are executed or when a termination condition is met

## Dependencies

The script may rely on:
- External libraries and modules
- System environment
- File system access
- Network resources (if applicable)

Without more specific context about your application, this is a general explanation of the script's structure and purpose. For a more detailed explanation, I would need additional information about the specific task this script is intended to perform.
"""
    
    # Helper methods for sending messages
    def send_response(self, message_id: str, content: any) -> None:
        """Send a response message"""
        response = {
            "id": message_id,
            "type": "response",
            "model": global_state["model_type"],
            "role": "assistant",
            "content": content,
            "timestamp": time.time()
        }
        
        print(json.dumps(response))
        sys.stdout.flush()
        
    def send_error(self, message_id: str, error_message: str) -> None:
        """Send an error message"""
        error = {
            "id": message_id,
            "type": "error",
            "model": global_state["model_type"],
            "role": "assistant",
            "content": {
                "error": error_message
            },
            "timestamp": time.time()
        }
        
        print(json.dumps(error))
        sys.stdout.flush()
        
    def send_status_update(self, message_id: str = None) -> None:
        """Send a status update message"""
        if message_id is None:
            message_id = str(uuid.uuid4())
            
        # Update memory usage
        if HAS_TORCH and torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                global_state["memory_usage"] = memory_allocated
            except:
                pass
                
        status = {
            "id": message_id,
            "type": "status",
            "model": global_state["model_type"],
            "role": "assistant",
            "content": {
                "status": global_state["status"],
                "memory_usage": global_state["memory_usage"],
                "uptime": time.time() - global_state["last_activity"]
            },
            "timestamp": time.time()
        }
        
        print(json.dumps(status))
        sys.stdout.flush()

# Model loading and initialization
def load_model():
    """Load the AI model based on environment configuration"""
    model_type = global_state["model_type"]
    logger.info(f"Loading {model_type} model...")
    
    try:
        if HAS_TRANSFORMERS and HAS_TORCH:
            # This is a simplified implementation - in a real system, we would:
            # 1. Load specific model weights and configuration
            # 2. Set up the appropriate tokenizer
            # 3. Configure model parameters
            
            logger.info(f"Using transformers to load {model_type}")
            
            # Model selection based on type
            model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF" # Fallback model ID
            
            if "qwen" in model_type.lower():
                model_id = "Qwen/Qwen2.5-7B-Instruct"
            elif "olympic" in model_type.lower() or "coder" in model_type.lower():
                model_id = "TheBloke/WizardCoder-Python-7B-V1.0"
                
            logger.info(f"Using model ID: {model_id}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            global_state["tokenizer"] = tokenizer
            global_state["model"] = model
            
            logger.info(f"Model {model_type} loaded successfully")
            return True
        else:
            logger.warning(f"PyTorch or Transformers not available, using fallback mode for {model_type}")
            return True
    except Exception as e:
        logger.error(f"Failed to load model {model_type}: {e}")
        logger.error(traceback.format_exc())
        return False

def init_systems():
    """Initialize all the supporting systems"""
    try:
        # Initialize communication system
        global_state["communication_system"] = CommunicationSystem(
            base_dir=os.path.dirname(MODEL_BASE_PATH)
        )
        logger.info("Communication system initialized")
        
        # Initialize reasoning system
        global_state["reasoning_system"] = ReasoningSystem()
        logger.info("Reasoning system initialized")
        
        # Initialize liquid neural network
        global_state["liquid_nn"] = LiquidNeuralNetwork(
            input_dim=768,  # Example dimension
            hidden_dim=512,
            output_dim=768
        )
        logger.info("Liquid neural network initialized")
        
        # Initialize metacognitive system
        global_state["metacognitive_system"] = MetacognitiveSystem()
        logger.info("Metacognitive system initialized")
        
        # Initialize knowledge library
        global_state["knowledge_library"] = KnowledgeLibrary()
        logger.info("Knowledge library initialized")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize supporting systems: {e}")
        logger.error(traceback.format_exc())
        return False

def send_ready_signal():
    """Send a ready signal to the parent process"""
    status = {
        "id": str(uuid.uuid4()),
        "type": "status",
        "model": global_state["model_type"],
        "role": "assistant",
        "content": {
            "status": "ready",
            "model_type": global_state["model_type"],
            "systems_initialized": HAS_LOCAL_MODULES,
            "transformers_available": HAS_TRANSFORMERS,
            "torch_available": HAS_TORCH,
            "cuda_available": HAS_TORCH and torch.cuda.is_available()
        },
        "timestamp": time.time()
    }
    
    print(json.dumps(status))
    sys.stdout.flush()

def heartbeat_thread():
    """Thread to send periodic heartbeats"""
    while not global_state["should_exit"]:
        try:
            # Send heartbeat if no recent activity
            if time.time() - global_state["last_activity"] > HEARTBEAT_INTERVAL:
                global_state["last_activity"] = time.time()
                
                heartbeat = {
                    "id": str(uuid.uuid4()),
                    "type": "heartbeat",
                    "model": global_state["model_type"],
                    "role": "assistant",
                    "content": {
                        "status": global_state["status"],
                        "timestamp": time.time()
                    },
                    "timestamp": time.time()
                }
                
                print(json.dumps(heartbeat))
                sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error in heartbeat thread: {e}")
            
        time.sleep(HEARTBEAT_INTERVAL)

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    global_state["should_exit"] = True
    sys.exit(0)

def main():
    """Main entry point"""
    logger.info(f"Starting model server for {MODEL_TYPE}")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize supporting systems
    if not init_systems():
        logger.error("Failed to initialize supporting systems")
        sys.exit(1)
    
    # Load model
    if not load_model():
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Mark as ready
    global_state["status"] = "ready"
    send_ready_signal()
    
    # Start heartbeat thread
    heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
    heartbeat.start()
    
    # Create message processor
    processor = MessageProcessor()
    
    # Process messages from stdin
    logger.info("Waiting for messages...")
    
    for line in sys.stdin:
        try:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Update last activity time
            global_state["last_activity"] = time.time()
            
            # Parse message
            message = json.loads(line.strip())
            
            # Process message
            processor.process_message(message)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()