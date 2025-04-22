"""
Model-to-Model Communication System

Enables AI models to dynamically communicate with each other when stuck or requiring assistance.
This is a critical component for enabling a truly intelligent dev team where models can collaborate
on complex problems in real-time.
"""

import os
import sys
import json
import logging
import time
import enum
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class MessageType(str, enum.Enum):
    """Types of communication messages between models"""
    QUESTION = "question"
    ANSWER = "answer"
    SUGGESTION = "suggestion"
    CLARIFICATION = "clarification"
    CODE_REVIEW = "code_review"
    ERROR_HELP = "error_help"
    PLANNING = "planning"
    OPTIMIZATION = "optimization"
    VERIFICATION = "verification"
    DEBUGGING = "debugging"

class ModelType(str, enum.Enum):
    """Types of models that can communicate"""
    QWEN25_OMNI = "qwen25_omni"
    OLYMPIC_CODER = "olympic_coder"
    LLAMA3 = "llama3"
    GEMMA3 = "gemma3"
    HYBRID = "hybrid"

class CommunicationSystem:
    """
    Model-to-Model Communication System
    
    Enables models to communicate with each other to solve problems collaboratively.
    
    Key capabilities:
    1. Allow models to ask questions to each other
    2. Enable code review and optimization suggestions
    3. Support error debugging assistance
    4. Track and analyze communication patterns
    5. Balance model strengths and specializations
    """
    
    def __init__(self):
        """Initialize the communication system"""
        # Store all messages
        self.messages = []
        
        # Track active exchanges (conversations)
        self.exchanges = {}
        
        # Track pending questions and answers
        self.pending_questions = {}
        
        # Configure question thresholds - when to ask for help
        self.question_thresholds = {
            ModelType.QWEN25_OMNI: {
                "confidence_threshold": 0.7,  # Below this, ask for help
                "complexity_threshold": 0.8,  # Above this, consider asking for help
                "code_size_threshold": 200,  # Lines of code above which to ask for review
                "error_unknown_threshold": 0.6  # Ask for help if error understanding is below this
            },
            ModelType.OLYMPIC_CODER: {
                "confidence_threshold": 0.6,
                "complexity_threshold": 0.75,
                "code_size_threshold": 300,
                "error_unknown_threshold": 0.5
            },
            ModelType.LLAMA3: {
                "confidence_threshold": 0.65,
                "complexity_threshold": 0.8,
                "code_size_threshold": 250,
                "error_unknown_threshold": 0.55
            },
            ModelType.GEMMA3: {
                "confidence_threshold": 0.7,
                "complexity_threshold": 0.85,
                "code_size_threshold": 200,
                "error_unknown_threshold": 0.6
            }
        }
        
        # Configure model specializations
        self.model_specializations = {
            ModelType.QWEN25_OMNI: {
                "strengths": ["omnidirectional reasoning", "cross-domain synthesis", "complex problem decomposition"],
                "weaknesses": ["specific code implementations", "framework-specific details"]
            },
            ModelType.OLYMPIC_CODER: {
                "strengths": ["code generation", "debugging", "optimization", "framework knowledge"],
                "weaknesses": ["high-level design", "user experience", "cross-domain synthesis"]
            },
            ModelType.LLAMA3: {
                "strengths": ["logical reasoning", "technical analysis", "code generation"],
                "weaknesses": ["creative solutions", "human-centered design"]
            },
            ModelType.GEMMA3: {
                "strengths": ["creative ideation", "human-centered design", "ethical reasoning"],
                "weaknesses": ["complex code generation", "technical problem solving"]
            }
        }
        
        logger.info("Model Communication System initialized")
    
    def ask_question(
        self,
        from_model: ModelType,
        to_model: ModelType,
        content: str,
        message_type: MessageType,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        One model asks a question to another model
        
        Args:
            from_model: Model asking the question
            to_model: Model being asked
            content: Question text
            message_type: Type of question/message
            context: Additional context
            metadata: Additional metadata
            
        Returns:
            Message object with unique ID
        """
        # Create a unique ID for this message
        message_id = str(uuid.uuid4())
        exchange_id = str(uuid.uuid4())
        
        # Create the message
        message = {
            "id": message_id,
            "exchange_id": exchange_id,
            "from_model": from_model,
            "to_model": to_model,
            "content": content,
            "type": message_type,
            "context": context or {},
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Add to messages list
        self.messages.append(message)
        
        # Create a new exchange
        self.exchanges[exchange_id] = {
            "id": exchange_id,
            "started_at": datetime.now().isoformat(),
            "messages": [message_id],
            "status": "active",
            "from_model": from_model,
            "to_model": to_model,
            "type": message_type
        }
        
        # Add to pending questions
        self.pending_questions[message_id] = message
        
        logger.info(f"Model {from_model} asked {to_model} a {message_type} question: {content[:50]}...")
        
        return message
    
    def answer_question(
        self,
        question_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a pending question
        
        Args:
            question_id: ID of the question to answer
            content: Answer text
            metadata: Additional metadata
            
        Returns:
            Answer message object
        """
        # Find the question
        question = self.pending_questions.get(question_id)
        
        if not question:
            logger.warning(f"Question {question_id} not found in pending questions")
            return {
                "error": "Question not found",
                "id": str(uuid.uuid4()),
                "status": "error"
            }
        
        # Create a unique ID for the answer
        message_id = str(uuid.uuid4())
        
        # Create the answer message
        answer = {
            "id": message_id,
            "exchange_id": question["exchange_id"],
            "from_model": question["to_model"],
            "to_model": question["from_model"],
            "content": content,
            "type": MessageType.ANSWER,
            "reply_to": question_id,
            "context": question["context"],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Add to messages list
        self.messages.append(answer)
        
        # Update exchange
        exchange = self.exchanges.get(question["exchange_id"])
        if exchange:
            exchange["messages"].append(message_id)
            exchange["last_activity"] = datetime.now().isoformat()
            
            # If it's been resolved, mark the exchange as completed
            exchange["status"] = "completed"
        
        # Update question status
        question["status"] = "answered"
        
        # Remove from pending questions
        del self.pending_questions[question_id]
        
        logger.info(f"Model {answer['from_model']} answered question from {answer['to_model']}: {content[:50]}...")
        
        return answer
    
    def should_ask_for_help(
        self,
        model: ModelType,
        confidence: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determine if a model should ask for help
        
        Args:
            model: The model considering asking for help
            confidence: Model's confidence in its current solution (0-1)
            context: Additional context about the task
            code: Code being generated or reviewed
            error: Error message if applicable
            
        Returns:
            Decision with reasoning
        """
        # Get thresholds for this model
        thresholds = self.question_thresholds.get(model, self.question_thresholds[ModelType.QWEN25_OMNI])
        
        # Initialize decision
        decision = {
            "should_ask": False,
            "reasons": [],
            "recommended_model": None,
            "recommendation_reasons": []
        }
        
        # Check confidence threshold
        if confidence < thresholds["confidence_threshold"]:
            decision["should_ask"] = True
            decision["reasons"].append(f"Low confidence: {confidence:.2f} < {thresholds['confidence_threshold']:.2f}")
        
        # Check code complexity if code is provided
        if code:
            code_lines = len(code.split("\n"))
            if code_lines > thresholds["code_size_threshold"]:
                decision["should_ask"] = True
                decision["reasons"].append(f"Complex code: {code_lines} lines > {thresholds['code_size_threshold']} threshold")
        
        # Check error comprehension if error is provided
        if error and context and "error_understanding" in context:
            error_understanding = context["error_understanding"]
            if error_understanding < thresholds["error_unknown_threshold"]:
                decision["should_ask"] = True
                decision["reasons"].append(f"Error comprehension: {error_understanding:.2f} < {thresholds['error_unknown_threshold']:.2f}")
        
        # If should ask, determine which model to ask
        if decision["should_ask"]:
            recommended_model = self._determine_best_helper(model, context, code, error)
            decision["recommended_model"] = recommended_model
            
            # Add reasons for recommendation
            if recommended_model:
                specializations = self.model_specializations.get(recommended_model, {})
                strengths = specializations.get("strengths", [])
                
                if strengths:
                    decision["recommendation_reasons"].append(f"Model {recommended_model} has strengths in: {', '.join(strengths[:3])}")
        
        return decision
    
    def _determine_best_helper(
        self,
        asking_model: ModelType,
        context: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
        error: Optional[str] = None
    ) -> ModelType:
        """Determine the best model to help with a given problem"""
        # In a real implementation, this would be more sophisticated
        
        # Default to complementary model pairs
        if asking_model == ModelType.QWEN25_OMNI:
            return ModelType.OLYMPIC_CODER
        elif asking_model == ModelType.OLYMPIC_CODER:
            return ModelType.QWEN25_OMNI
        elif asking_model == ModelType.LLAMA3:
            return ModelType.GEMMA3
        elif asking_model == ModelType.GEMMA3:
            return ModelType.LLAMA3
        else:
            # Default to Qwen for comprehensive understanding
            return ModelType.QWEN25_OMNI
    
    def get_exchange_messages(self, exchange_id: str) -> List[Dict[str, Any]]:
        """Get all messages in an exchange"""
        exchange = self.exchanges.get(exchange_id)
        
        if not exchange:
            logger.warning(f"Exchange {exchange_id} not found")
            return []
        
        # Collect all messages in this exchange
        exchange_messages = []
        for message_id in exchange["messages"]:
            for message in self.messages:
                if message["id"] == message_id:
                    exchange_messages.append(message)
                    break
        
        # Sort by timestamp
        exchange_messages.sort(key=lambda m: m["timestamp"])
        
        return exchange_messages
    
    def get_active_exchanges(self) -> List[Dict[str, Any]]:
        """Get all active exchanges"""
        active_exchanges = []
        
        for exchange_id, exchange in self.exchanges.items():
            if exchange["status"] == "active":
                # Add exchange with message count
                exchange_copy = dict(exchange)
                exchange_copy["message_count"] = len(exchange["messages"])
                active_exchanges.append(exchange_copy)
        
        return active_exchanges
    
    def get_pending_questions(self) -> List[Dict[str, Any]]:
        """Get all pending questions"""
        return list(self.pending_questions.values())
    
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific message by ID"""
        for message in self.messages:
            if message["id"] == message_id:
                return message
        
        return None
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about model communication"""
        # Count messages by type
        message_types = {}
        for message in self.messages:
            message_type = message["type"]
            if message_type not in message_types:
                message_types[message_type] = 0
            message_types[message_type] += 1
        
        # Count exchanges by model pair
        model_pairs = {}
        for exchange in self.exchanges.values():
            pair = f"{exchange['from_model']}->{exchange['to_model']}"
            if pair not in model_pairs:
                model_pairs[pair] = 0
            model_pairs[pair] += 1
        
        # Calculate average response time
        response_times = []
        for exchange_id, exchange in self.exchanges.items():
            if exchange["status"] == "completed":
                messages = self.get_exchange_messages(exchange_id)
                if len(messages) >= 2:
                    question_time = datetime.fromisoformat(messages[0]["timestamp"])
                    answer_time = datetime.fromisoformat(messages[1]["timestamp"])
                    response_time = (answer_time - question_time).total_seconds()
                    response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_messages": len(self.messages),
            "total_exchanges": len(self.exchanges),
            "active_exchanges": len([e for e in self.exchanges.values() if e["status"] == "active"]),
            "pending_questions": len(self.pending_questions),
            "message_types": message_types,
            "model_pairs": model_pairs,
            "avg_response_time": avg_response_time
        }
    
    def create_optimization_suggestion(
        self,
        from_model: ModelType,
        to_model: ModelType,
        code: str,
        suggestions: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a suggestion for code optimization
        
        Args:
            from_model: Model providing the suggestion
            to_model: Model receiving the suggestion
            code: Code being optimized
            suggestions: List of specific optimization suggestions
            context: Additional context
            
        Returns:
            Suggestion message object
        """
        # Create a formatted suggestion message
        content = f"I noticed some optimization opportunities in your code:\n\n"
        for i, suggestion in enumerate(suggestions):
            content += f"{i+1}. {suggestion}\n"
        
        # Create the suggestion message
        return self.ask_question(
            from_model=from_model,
            to_model=to_model,
            content=content,
            message_type=MessageType.OPTIMIZATION,
            context={
                "code": code,
                **(context or {})
            },
            metadata={
                "suggestion_count": len(suggestions)
            }
        )
    
    def create_error_help_request(
        self,
        from_model: ModelType,
        to_model: ModelType,
        code: str,
        error: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a request for help with an error
        
        Args:
            from_model: Model asking for help
            to_model: Model being asked
            code: Code with the error
            error: Error message
            context: Additional context
            
        Returns:
            Error help request message object
        """
        # Create a formatted error help request
        content = f"I'm encountering the following error and need help understanding/fixing it:\n\n"
        content += f"Error: {error}\n\n"
        content += f"Here's the code that's causing the issue:\n\n```\n{code}\n```\n\n"
        content += "Can you help me understand what's going wrong and how to fix it?"
        
        # Create the error help request message
        return self.ask_question(
            from_model=from_model,
            to_model=to_model,
            content=content,
            message_type=MessageType.ERROR_HELP,
            context={
                "code": code,
                "error": error,
                **(context or {})
            },
            metadata={
                "error_length": len(error),
                "code_length": len(code)
            }
        )
    
    def create_code_review_request(
        self,
        from_model: ModelType,
        to_model: ModelType,
        code: str,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a request for code review
        
        Args:
            from_model: Model asking for review
            to_model: Model being asked
            code: Code to review
            goal: Goal of the code
            context: Additional context
            
        Returns:
            Code review request message object
        """
        # Create a formatted code review request
        content = f"I've written code for the following goal: {goal}\n\n"
        content += f"Can you review this code and suggest improvements?\n\n```\n{code}\n```\n\n"
        content += "I'm particularly interested in: correctness, efficiency, readability, and best practices."
        
        # Create the code review request message
        return self.ask_question(
            from_model=from_model,
            to_model=to_model,
            content=content,
            message_type=MessageType.CODE_REVIEW,
            context={
                "code": code,
                "goal": goal,
                **(context or {})
            },
            metadata={
                "code_length": len(code),
                "language": self._detect_language(code)
            }
        )
    
    def create_clarification_request(
        self,
        from_model: ModelType,
        to_model: ModelType,
        topic: str,
        specific_question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a request for clarification on a topic
        
        Args:
            from_model: Model asking for clarification
            to_model: Model being asked
            topic: General topic
            specific_question: Specific question about the topic
            context: Additional context
            
        Returns:
            Clarification request message object
        """
        # Create a formatted clarification request
        content = f"I need clarification about: {topic}\n\n"
        content += f"Specifically: {specific_question}\n\n"
        content += "Can you help me understand this better?"
        
        # Create the clarification request message
        return self.ask_question(
            from_model=from_model,
            to_model=to_model,
            content=content,
            message_type=MessageType.CLARIFICATION,
            context={
                "topic": topic,
                **(context or {})
            },
            metadata={
                "topic": topic
            }
        )
    
    def _detect_language(self, code: str) -> str:
        """Detect the programming language of code"""
        # Simple detection based on keywords and patterns
        code = code.strip().lower()
        
        # Check for Python indicators
        if ("def " in code or "import " in code or "class " in code or
                code.startswith("#!/usr/bin/env python") or ".py" in code):
            return "python"
        
        # Check for JavaScript indicators
        if ("function " in code or "const " in code or "let " in code or "var " in code or
                "() =>" in code or code.endswith(".js") or "export " in code):
            return "javascript"
        
        # Check for TypeScript indicators
        if ("interface " in code or ": string" in code or ": number" in code or 
                code.endsWith(".ts") or "as any" in code):
            return "typescript"
        
        # Check for Shell indicators
        if (code.startswith("#!/bin/bash") or code.startswith("#!/bin/sh") or
                "if [ " in code or "while [ " in code or "for i in " in code):
            return "shell"
        
        # Check for HTML indicators
        if ("<html" in code or "<!doctype html" in code or "</div>" in code or
                code.endswith(".html") or "<body" in code):
            return "html"
        
        # Check for SQL indicators
        if ("select " in code or "from " in code or "where " in code or
                "join " in code or "group by " in code or "order by " in code):
            return "sql"
        
        # Default to "unknown"
        return "unknown"
    
    def simulate_model_response(
        self,
        question: Dict[str, Any]
    ) -> str:
        """
        Simulate a model's response to a question (for development/testing)
        
        Args:
            question: Question message object
            
        Returns:
            Simulated response text
        """
        # This is for demonstration/testing only
        # In a real implementation, this would call the actual model API
        
        question_type = question["type"]
        context = question.get("context", {})
        
        if question_type == MessageType.ERROR_HELP:
            return self.simulate_error_help_response(question)
        
        elif question_type == MessageType.CODE_REVIEW:
            return self.simulate_code_review_response(question)
        
        elif question_type == MessageType.OPTIMIZATION:
            return self.simulate_optimization_response(question)
        
        elif question_type == MessageType.CLARIFICATION:
            return self.simulate_clarification_response(question)
        
        else:
            return self.simulate_general_response(question)
    
    def simulate_error_help_response(self, question: Dict[str, Any]) -> str:
        """Simulate an error help response"""
        context = question.get("context", {})
        error = context.get("error", "Unknown error")
        
        response = f"I've analyzed the error you're encountering:\n\n"
        response += f"```\n{error}\n```\n\n"
        
        # Generate a plausible solution based on common error patterns
        if "undefined" in error.lower():
            response += "This appears to be a reference error. The variable or function you're trying to use hasn't been defined or is out of scope.\n\n"
            response += "Potential solutions:\n"
            response += "1. Check the variable name for typos\n"
            response += "2. Make sure the variable is defined before it's used\n"
            response += "3. Verify that imports/dependencies are correctly set up\n"
        
        elif "syntax" in error.lower():
            response += "This is a syntax error. There's something malformed in your code structure.\n\n"
            response += "Potential solutions:\n"
            response += "1. Check for missing parentheses, brackets, or braces\n"
            response += "2. Verify that all strings are properly closed\n"
            response += "3. Look for missing commas or semicolons\n"
        
        elif "type" in error.lower():
            response += "This is a type error. You're trying to perform an operation on a value of the wrong type.\n\n"
            response += "Potential solutions:\n"
            response += "1. Check the types of the variables involved\n"
            response += "2. Use appropriate type conversion functions\n"
            response += "3. Add error handling for unexpected types\n"
        
        else:
            response += "After analyzing the code and error, I believe the issue might be:\n\n"
            response += "1. Incorrect function usage or parameter order\n"
            response += "2. Missing dependency or import\n"
            response += "3. Logic error in the algorithm\n\n"
            response += "I'd suggest adding some debug output to trace the execution flow and variable values around where the error occurs."
        
        return response
    
    def simulate_code_review_response(self, question: Dict[str, Any]) -> str:
        """Simulate a code review response"""
        context = question.get("context", {})
        code = context.get("code", "")
        goal = context.get("goal", "Unknown goal")
        
        response = f"I've reviewed your code for the goal: {goal}\n\n"
        response += "Here's my feedback:\n\n"
        
        # Generate plausible review based on code size and language
        code_length = len(code.split("\n"))
        language = question.get("metadata", {}).get("language", "unknown")
        
        if code_length < 10:
            response += "Your code is concise, which is good. Some observations:\n\n"
        else:
            response += "Your code is fairly complex. Here are my observations:\n\n"
        
        # Add language-specific suggestions
        if language == "python":
            response += "Strengths:\n"
            response += "1. Your solution addresses the core functionality\n"
            
            response += "\nImprovement opportunities:\n"
            response += "1. Consider adding docstrings to functions\n"
            response += "2. Use list comprehensions for more concise code where appropriate\n"
            response += "3. Add type hints to improve code readability and catch potential errors\n"
            
            if "try" not in code:
                response += "4. Add error handling with try/except blocks for robust operation\n"
            
            if code_length > 20:
                response += "5. Consider breaking down larger functions into smaller, more focused ones\n"
        
        elif language in ["javascript", "typescript"]:
            response += "Strengths:\n"
            response += "1. The solution approaches the problem effectively\n"
            
            response += "\nImprovement opportunities:\n"
            response += "1. Use const for variables that don't get reassigned\n"
            response += "2. Consider using array methods like map/filter/reduce for cleaner code\n"
            response += "3. Add JSDoc comments for better documentation\n"
            
            if "try" not in code:
                response += "4. Add error handling with try/catch blocks\n"
            
            if language == "typescript" and ": any" in code:
                response += "5. Avoid using 'any' type in TypeScript; specify more precise types\n"
        
        else:
            response += "Strengths:\n"
            response += "1. The code is structured logically\n"
            response += "2. The solution addresses the stated goal\n"
            
            response += "\nImprovement opportunities:\n"
            response += "1. Add more comments to explain complex logic\n"
            response += "2. Consider edge cases and error handling\n"
            response += "3. Review variable naming for clarity\n"
        
        response += "\nOverall, the code is on the right track. With the suggested improvements, it will be more robust and maintainable."
        
        return response
    
    def simulate_optimization_response(self, question: Dict[str, Any]) -> str:
        """Simulate an optimization suggestion response"""
        context = question.get("context", {})
        suggestions = question.get("content", "").split("\n")[1:]  # Skip the first line
        
        response = "Thanks for the optimization suggestions. Here's my feedback on them:\n\n"
        
        for i, suggestion in enumerate(suggestions):
            if not suggestion.strip():
                continue
                
            # Strip the numbering
            if '. ' in suggestion:
                suggestion = suggestion.split('. ', 1)[1]
            
            response += f"Suggestion {i+1}: {suggestion}\n"
            response += "I agree with this suggestion. I'll implement this change in the next iteration.\n\n"
        
        response += "I appreciate your insights - these optimizations will definitely improve the code quality and performance."
        
        return response
    
    def simulate_clarification_response(self, question: Dict[str, Any]) -> str:
        """Simulate a clarification response"""
        context = question.get("context", {})
        topic = context.get("topic", "Unknown topic")
        
        response = f"I'm happy to clarify about {topic}.\n\n"
        
        # Generate a plausible explanation based on common topics
        if "api" in topic.lower():
            response += "APIs (Application Programming Interfaces) allow different software systems to communicate with each other. They define methods and data formats that applications can use to request and exchange information.\n\n"
            response += "Key concepts in API design:\n"
            response += "1. **Endpoints**: URLs that represent different resources or actions\n"
            response += "2. **HTTP Methods**: GET, POST, PUT, DELETE for different operations\n"
            response += "3. **Request/Response formats**: Typically JSON or XML\n"
            response += "4. **Authentication**: Methods to verify the identity of clients\n"
            response += "5. **Rate limiting**: Controlling how many requests can be made\n\n"
            response += "Does this help with your specific question, or would you like me to elaborate on a particular aspect?"
        
        elif "algorithm" in topic.lower():
            response += "Algorithms are step-by-step procedures for solving problems or accomplishing tasks. They're fundamental to computer science and programming.\n\n"
            response += "Important characteristics of algorithms:\n"
            response += "1. **Correctness**: Produces the right output for all valid inputs\n"
            response += "2. **Efficiency**: Uses resources (time, memory) optimally\n"
            response += "3. **Clarity**: Can be understood and implemented correctly\n"
            response += "4. **Finiteness**: Terminates after a finite number of steps\n\n"
            response += "Common algorithm categories include sorting, searching, graph algorithms, dynamic programming, and divide-and-conquer approaches.\n\n"
            response += "Is there a specific type of algorithm you're interested in understanding better?"
        
        else:
            response += f"{topic} is an important concept in software development.\n\n"
            response += "Let me break it down into key components:\n"
            response += "1. **Core principles**: What problems it solves and why it exists\n"
            response += "2. **Common implementations**: How it's typically used in practice\n"
            response += "3. **Best practices**: How to use it effectively\n"
            response += "4. **Common pitfalls**: What to avoid when working with it\n\n"
            response += "Would you like me to elaborate on any of these aspects specifically?"
        
        return response
    
    def simulate_general_response(self, question: Dict[str, Any]) -> str:
        """Simulate a general response"""
        content = question.get("content", "")
        
        response = "I've considered your question, and here's my perspective:\n\n"
        
        if "?" in content:
            # Extract the question
            question_parts = content.split("?")
            actual_question = question_parts[0] + "?"
            
            response += f"Regarding '{actual_question}'\n\n"
            response += "This is a great question. Based on my understanding:\n\n"
            response += "1. This question touches on an important concept in development\n"
            response += "2. There are multiple approaches we could consider\n"
            response += "3. The best solution depends on specific project requirements\n\n"
            response += "Would you like me to elaborate on any particular aspect of this topic?"
        else:
            response += "I've analyzed your statement and have some thoughts:\n\n"
            response += "1. Your perspective makes sense in many contexts\n"
            response += "2. There are some additional factors worth considering\n"
            response += "3. We could explore some alternative approaches as well\n\n"
            response += "Let me know if you'd like to dive deeper into any of these points."
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get basic status information"""
        return {
            "total_messages": len(self.messages),
            "active_exchanges": len([e for e in self.exchanges.values() if e["status"] == "active"]),
            "pending_questions": len(self.pending_questions)
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            **self.get_communication_stats(),
            "model_thresholds": {str(model): thresholds for model, thresholds in self.question_thresholds.items()},
            "model_specializations": {str(model): spec for model, spec in self.model_specializations.items()},
            "recent_messages": self.messages[-10:] if self.messages else []
        }