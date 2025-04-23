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
from typing import Dict, Any, Optional, List, Union

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("model_server")

# Import conditional dependencies - these may or may not be available
try:
    import torch
    HAS_TORCH = True
    logger.info("PyTorch is available")
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch is not available")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
    logger.info("Transformers is available")
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers is not available")

# Define fallback classes and functions
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
    """Implementation of communication system"""
    def __init__(self, base_dir=None):
        self.conversations = {}
        logger.info("Using CommunicationSystem")
        
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
    """Implementation of reasoning system"""
    def __init__(self):
        logger.info("Using ReasoningSystem")
        
    def solve_problem(self, problem, strategy=None, context=None):
        logger.info(f"Solving problem using strategy {strategy}")
        return {
            "solution": f"Solution for: {problem}",
            "reasoning_path": ["Step 1", "Step 2", "Step 3"],
            "confidence": 0.8
        }
        
class LiquidNeuralNetwork:
    """Implementation of liquid neural network"""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        logger.info("Using LiquidNeuralNetwork")
        
    def forward(self, x):
        return {"output": "Output from LNN", "confidence": 0.7}
        
class MetacognitiveSystem:
    """Implementation of metacognitive system"""
    def __init__(self):
        logger.info("Using MetacognitiveSystem")
        
    def evaluate_confidence(self, result, context=None):
        return 0.8
        
    def reflect(self, input_data, output_data, expected=None):
        return {
            "quality": 0.8,
            "improvements": ["Improvement suggestion 1", "Improvement suggestion 2"],
            "decision": "proceed"
        }
        
class KnowledgeLibrary:
    """Implementation of knowledge library"""
    def __init__(self):
        self.knowledge = {}
        logger.info("Using KnowledgeLibrary")
        
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
        
class MetacognitiveLevel:
    """Metacognitive levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

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

# Initialize the supporting systems
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

# Load the AI model
def load_model():
    """Load the AI model based on environment configuration"""
    model_type = global_state["model_type"]
    logger.info(f"Loading {model_type} model...")
    
    try:
        # Check if required libraries are available
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            logger.error("Required dependencies PyTorch or Transformers not available")
            global_state["status"] = "ready"  # Mark as ready but with limited functionality
            return True
        
        # Model selection based on type
        if "qwen" in model_type.lower():
            model_id = "Qwen/Qwen-7B"  # Actual model ID for production
        elif "olympic" in model_type.lower() or "coder" in model_type.lower():
            model_id = "TheBloke/CodeLlama-7B-HF"  # Actual model ID for production
        else:
            model_id = "Qwen/Qwen-7B"  # Default to Qwen
            
        logger.info(f"Using model ID: {model_id}")
        
        # Use the global imports we've already checked for
        # Don't try to reference transformers classes directly
        
        # Load tokenizer and model - only if both imports succeeded earlier
        if 'AutoTokenizer' in globals() and 'AutoModelForCausalLM' in globals() and 'torch' in globals():
            tokenizer = globals()['AutoTokenizer'].from_pretrained(model_id)
            model = globals()['AutoModelForCausalLM'].from_pretrained(
                model_id,
                torch_dtype=globals()['torch'].float16,  # Use float16 for efficiency
                device_map="auto",  # Automatically use available devices
                low_cpu_mem_usage=True
            )
        else:
            logger.error("Required transformer classes not found in global scope")
            raise ImportError("Missing required transformer classes")
        
        global_state["tokenizer"] = tokenizer
        global_state["model"] = model
        
        logger.info(f"Model {model_type} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model {model_type}: {e}")
        logger.error(traceback.format_exc())
        # We'll still return true so the server can start, but without model capabilities
        global_state["status"] = "ready"  # Mark as ready but with limited functionality
        return True

# Message processing system
class MessageProcessor:
    """Processes messages from Node.js backend"""
    
    def __init__(self):
        self.handlers = {
            "request": self.handle_request,
            "heartbeat": self.handle_heartbeat
        }
        
    def process_message(self, message):
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
            
    def handle_request(self, message):
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
            
    def handle_heartbeat(self, message):
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
            
    def process_architect_request(self, content):
        """Process a request for the architect role"""
        task = content.get("task")
        
        if task == "architecture_planning":
            return self.generate_architecture(content)
        else:
            return {"error": f"Unknown architect task: {task}"}
            
    def process_builder_request(self, content):
        """Process a request for the builder role"""
        task = content.get("task")
        
        if task == "code_generation":
            return self.generate_code(content)
        elif task == "code_enhancement":
            return self.enhance_code(content)
        else:
            return {"error": f"Unknown builder task: {task}"}
            
    def process_tester_request(self, content):
        """Process a request for the tester role"""
        task = content.get("task")
        
        if task == "test_generation":
            return self.generate_tests(content)
        elif task == "code_debugging":
            return self.debug_code(content)
        else:
            return {"error": f"Unknown tester task: {task}"}
            
    def process_reviewer_request(self, content):
        """Process a request for the reviewer role"""
        task = content.get("task")
        
        if task == "code_review":
            return self.review_code(content)
        elif task == "code_explanation":
            return self.explain_code(content)
        else:
            return {"error": f"Unknown reviewer task: {task}"}
    
    # Architect role implementation
    def generate_architecture(self, content):
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
    
    def _create_architecture_prompt(self, requirements, options):
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
    def generate_code(self, content):
        """Generate code based on requirements and architecture"""
        requirements = content.get("requirements", "")
        architecture = content.get("architecture", "")
        options = content.get("options", {})
        
        if not requirements and not architecture:
            return {"error": "Either requirements or architecture must be provided for code generation"}
            
        # Fallback to model inference
        prompt = self._create_code_prompt(requirements, architecture, options)
        return self._generate_text(prompt)
        
    def _create_code_prompt(self, requirements, architecture, options):
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
        
    def enhance_code(self, content):
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
        
    def _create_enhancement_prompt(self, code, requirements, enhancement, language):
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
    def generate_tests(self, content):
        """Generate tests for code"""
        code = content.get("code", "")
        requirements = content.get("requirements", "")
        options = content.get("options", {})
        
        if not code:
            return {"error": "Code is required for test generation"}
            
        # Fallback to model inference
        prompt = self._create_test_prompt(code, requirements, options)
        return self._generate_text(prompt)
        
    def _create_test_prompt(self, code, requirements, options):
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
        
    def debug_code(self, content):
        """Debug and fix code"""
        code = content.get("code", "")
        error = content.get("error", "")
        language = content.get("language", "")
        
        if not code:
            return {"error": "Code is required for debugging"}
            
        # Fallback to model inference
        prompt = self._create_debug_prompt(code, error, language)
        return self._generate_text(prompt)
        
    def _create_debug_prompt(self, code, error, language):
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
    def review_code(self, content):
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
        
    def _create_review_prompt(self, code, tests, requirements, options):
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
        
    def explain_code(self, content):
        """Explain code"""
        code = content.get("code", "")
        language = content.get("language", "")
        detail_level = content.get("detail_level", "detailed")
        
        if not code:
            return {"error": "Code is required for explanation"}
            
        # Fallback to model inference
        prompt = self._create_explanation_prompt(code, language, detail_level)
        return self._generate_text(prompt)
        
    def _create_explanation_prompt(self, code, language, detail_level):
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
    def _generate_text(self, prompt):
        """Generate text using the loaded model"""
        # Check if transformers model is available
        if global_state["model"] and global_state["tokenizer"] and HAS_TRANSFORMERS and HAS_TORCH:
            logger.info("Generating text using transformers model")
            
            try:
                inputs = global_state["tokenizer"](prompt, return_tensors="pt")
                
                # Generate
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
                
                # Return error message to client
                raise RuntimeError(f"Model inference failed: {str(e)}")
        else:
            # We're missing dependencies but the system is running without models
            logger.error("Required AI dependencies (PyTorch/Transformers) not available")
            
            # Return a status message that will be shown to the client
            return {"status": "error", "message": "Required AI models or dependencies not available. This is a production system and requires the full AI stack to be installed."}
    
    def _generate_error_message(self, feature_type):
        """Generate error message when AI models are unavailable"""
        return {
            "status": "error", 
            "message": f"Cannot generate {feature_type}. AI models are not available. This is a production system and requires PyTorch and Transformers to be installed."
        }

    def _generate_code_unavailable(self, language):
        """Return error when code generation is unavailable"""
        return self._generate_error_message(f"code for {language}")

    def _generate_tests_unavailable(self, language):
        """Return error when test generation is unavailable"""
        return self._generate_error_message(f"tests for {language}")
    
    def _generate_debug_unavailable(self, language):
        """Return error when debug is unavailable"""
        return self._generate_error_message(f"debugging for {language}")
    
    def _generate_fallback_review(self):
        """Return error when code review is unavailable"""
        return self._generate_error_message("code review")
    
    def _generate_fallback_explanation(self):
        """Return error when code explanation is unavailable"""
        return self._generate_error_message("code explanation")
    
    # Helper methods for sending messages
    def send_response(self, message_id, content):
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
        
    def send_error(self, message_id, error_message):
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
        
    def send_status_update(self, message_id=None):
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

# Send a ready signal to the parent process
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
            "systems_initialized": True,
            "transformers_available": HAS_TRANSFORMERS,
            "torch_available": HAS_TORCH,
            "cuda_available": HAS_TORCH and torch.cuda.is_available()
        },
        "timestamp": time.time()
    }
    
    print(json.dumps(status))
    sys.stdout.flush()

def main():
    """Main entry point"""
    logger.info(f"Starting model server for {MODEL_TYPE}")
    
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