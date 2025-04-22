"""
AI Engine for Seren

Provides the core AI capabilities by orchestrating locally hosted models,
including Qwen2.5-omni-7b and OlympicCoder-7B for offline operation.
"""

import os
import sys
import json
import logging
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from datetime import datetime
import threading
import queue

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import model manager for handling local models
from ai_core.model_manager import model_manager, ModelType, ModelMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class CollaborationMode(Enum):
    """Modes of AI collaboration"""
    COLLABORATIVE = "collaborative"  # Models work together on the same task
    SPECIALIZED = "specialized"      # Models work on different aspects of a task
    COMPETITIVE = "competitive"      # Models compete to produce the best solution

class AIEngine:
    """
    AI Engine for Seren
    
    Provides the core AI capabilities by orchestrating locally hosted models:
    - Query processing and response generation
    - Code generation and analysis
    - Multi-model collaboration
    - Content creation and summarization
    - Knowledge access and reasoning
    
    Leverages locally hosted models:
    - Qwen2.5-omni-7b: Advanced multi-purpose language model
    - OlympicCoder-7B: Specialized code generation model
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the AI engine"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Internal state
        self.initialized = False
        self.primary_model = ModelType.QWEN
        self.secondary_model = ModelType.OLYMPIC
        self.collaboration_mode = CollaborationMode.COLLABORATIVE
        
        # Model preferences by task
        self.task_preferences = {
            "code_generation": ModelType.OLYMPIC,
            "code_explanation": ModelType.OLYMPIC,
            "general_query": ModelType.QWEN,
            "creative_writing": ModelType.QWEN,
            "reasoning": ModelType.QWEN,
            "planning": ModelType.QWEN
        }
        
        # Conversation contexts
        self.conversations = {}
        
        # Stats
        self.stats = {
            "total_queries": 0,
            "total_code_generations": 0,
            "model_usage": {model_type.value: 0 for model_type in ModelType},
            "average_response_time": 0,
            "total_response_time": 0
        }
        
        logger.info("AI Engine initialized")
    
    def initialize_models(
        self,
        primary_model: str = "qwen",
        secondary_model: str = "olympic",
        collaboration_mode: str = "collaborative"
    ) -> bool:
        """
        Initialize AI models
        
        Args:
            primary_model: Primary model type
            secondary_model: Secondary model type
            collaboration_mode: Mode of collaboration
            
        Returns:
            Success status
        """
        try:
            # Convert parameters to enums
            self.primary_model = ModelType(primary_model)
            self.secondary_model = ModelType(secondary_model)
            self.collaboration_mode = CollaborationMode(collaboration_mode)
            
            logger.info(f"Initializing models: {primary_model} (primary), {secondary_model} (secondary), "
                       f"mode: {collaboration_mode}")
            
            # Initialize model manager (this doesn't load models yet, just prepares for loading)
            # Models will be loaded on-demand
            
            # Mark as initialized
            self.initialized = True
            
            # Log system information
            self._log_system_info()
            
            return True
        
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            return False
    
    def _log_system_info(self) -> None:
        """Log system information"""
        import platform
        import psutil
        
        # System info
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        
        # CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        logger.info(f"CPU: {cpu_count} physical cores, {cpu_logical} logical cores")
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        logger.info(f"Memory: {memory_gb:.2f} GB total")
        
        # GPU info (if available)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"GPU {i} Memory: {memory:.2f} GB")
            else:
                logger.info("No GPU detected")
        except ImportError:
            logger.info("PyTorch not available, GPU info not collected")
    
    def process_query(
        self,
        query: str,
        mode: str = "default",
        context: Dict[str, Any] = None,
        settings: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a general query
        
        Args:
            query: User query
            mode: Processing mode
            context: Additional context
            settings: Processing settings
            
        Returns:
            Response data
        """
        if not self.initialized:
            logger.warning("AI Engine not initialized, initializing with defaults")
            self.initialize_models()
        
        start_time = time.time()
        
        # Update stats
        self.stats["total_queries"] += 1
        
        # Process context
        context = context or {}
        settings = settings or {}
        
        # Get conversation ID or create a new one
        conversation_id = context.get("conversation_id")
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat()
            }
        elif conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat()
            }
        
        # Add user message to conversation
        self.conversations[conversation_id]["messages"].append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Determine appropriate model based on query and mode
        model_type = self._select_model_for_query(query, mode)
        logger.info(f"Selected {model_type.value} model for query")
        
        # Update stats
        self.stats["model_usage"][model_type.value] += 1
        
        # Prepare messages
        messages = []
        
        # Add system message
        system_message = settings.get("system_message", "You are Seren, a powerful AI assistant. Be concise and helpful.")
        messages.append({"role": "system", "content": system_message})
        
        # Try to get relevant knowledge from the knowledge library
        try:
            from ai_core.knowledge.library import knowledge_library
            # Get relevant knowledge for the query
            knowledge_context = knowledge_library.extract_context_for_query(query)
            if knowledge_context:
                # Add knowledge to system message to provide context
                knowledge_prompt = f"\nThe following information from my knowledge library may be helpful:\n{knowledge_context}\n\nPlease use this information in your response when relevant."
                system_message += knowledge_prompt
                # Update the system message in the messages list
                messages[0]["content"] = system_message
        except ImportError:
            # Knowledge library not available, continue without it
            pass
        
        # Add conversation history
        conversation_messages = self.conversations[conversation_id]["messages"]
        max_history = settings.get("max_history", 10)
        
        # Take only the most recent messages up to max_history
        recent_messages = conversation_messages[-max_history*2:] if len(conversation_messages) > max_history*2 else conversation_messages
        
        for msg in recent_messages:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Generate response
        try:
            temperature = settings.get("temperature", 0.7)
            max_length = settings.get("max_tokens", 1024)
            
            response_text = model_manager.chat(
                model_type=model_type,
                messages=messages,
                max_length=max_length,
                temperature=temperature
            )
            
            # Add assistant message to conversation
            self.conversations[conversation_id]["messages"].append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Try to learn from this interaction
            try:
                from ai_core.knowledge.self_learning import self_learning_system, LearningPriority
                # Check if this is a particularly informative exchange that we should learn from
                if len(query) > 100 and len(response_text) > 200:
                    # Add the conversation to the learning queue with low priority
                    self_learning_system.add_learning_task(
                        content=f"User query: {query}\n\nAI response: {response_text}",
                        source=f"conversation:{conversation_id}",
                        priority=LearningPriority.LOW,
                        metadata={"interaction_type": "query_response"}
                    )
            except ImportError:
                # Self-learning system not available, continue without it
                pass
            
            # Calculate and update response time
            response_time = time.time() - start_time
            self.stats["total_response_time"] += response_time
            total_queries = self.stats["total_queries"]
            self.stats["average_response_time"] = self.stats["total_response_time"] / total_queries
            
            # Prepare response
            response = {
                "text": response_text,
                "conversation_id": conversation_id,
                "model": model_type.value,
                "response_time": response_time
            }
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "error": str(e),
                "conversation_id": conversation_id
            }
    
    def _select_model_for_query(self, query: str, mode: str) -> ModelType:
        """
        Select the appropriate model for a query
        
        Args:
            query: User query
            mode: Processing mode
            
        Returns:
            Selected model type
        """
        # If mode specifies a particular model, use it
        if mode == "code":
            return ModelType.OLYMPIC
        elif mode == "general":
            return ModelType.QWEN
        
        # Check for code-related queries
        code_indicators = [
            "code", "function", "script", "program", "algorithm",
            "implement", "coding", "compile", "runtime", "syntax",
            "javascript", "python", "java", "c++", "typescript",
            "html", "css", "sql", "php", "rust", "golang"
        ]
        
        # Estimate if this is a code query
        lower_query = query.lower()
        code_score = sum(1 for indicator in code_indicators if indicator in lower_query)
        
        # If strong code indicators, use Olympic
        if code_score >= 2 or "write a" in lower_query and any(indicator in lower_query for indicator in code_indicators):
            return ModelType.OLYMPIC
        
        # Default to primary model (Qwen)
        return self.primary_model
    
    def generate_code(
        self,
        specification: str,
        language: str = "python",
        generate_tests: bool = False,
        mode: str = "standard"
    ) -> Dict[str, Any]:
        """
        Generate code based on a specification
        
        Args:
            specification: Code specification
            language: Programming language
            generate_tests: Whether to generate tests
            mode: Generation mode
            
        Returns:
            Generated code
        """
        if not self.initialized:
            logger.warning("AI Engine not initialized, initializing with defaults")
            self.initialize_models()
        
        start_time = time.time()
        
        # Update stats
        self.stats["total_code_generations"] += 1
        self.stats["model_usage"][ModelType.OLYMPIC.value] += 1
        
        # Create prompt for generation
        if generate_tests:
            prompt = f"""Generate {language} code based on this specification: 
{specification}

Please include:
1. The main implementation
2. Tests to verify the implementation
3. Brief inline comments

Use best practices for {language}.
"""
        else:
            prompt = f"""Generate {language} code based on this specification:
{specification}

Please include:
1. The main implementation
2. Brief inline comments

Use best practices for {language}.
"""
        
        try:
            # Generate code using OlympicCoder model
            generated_code = model_manager.generate_code(
                prompt=prompt,
                language=language,
                max_length=2048,
                temperature=0.3  # Lower temperature for code
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Prepare response
            response = {
                "code": generated_code,
                "language": language,
                "model": ModelType.OLYMPIC.value,
                "response_time": response_time
            }
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "error": str(e)
            }
    
    def analyze_code(
        self,
        code: str,
        language: str = "python",
        analysis_type: str = "review"
    ) -> Dict[str, Any]:
        """
        Analyze code
        
        Args:
            code: Code to analyze
            language: Programming language
            analysis_type: Type of analysis (review, bugs, optimize)
            
        Returns:
            Analysis results
        """
        if not self.initialized:
            logger.warning("AI Engine not initialized, initializing with defaults")
            self.initialize_models()
        
        start_time = time.time()
        
        # Determine prompt based on analysis type
        if analysis_type == "review":
            prompt = f"""Review this {language} code and provide feedback:
```{language}
{code}
```

Please analyze:
1. Code quality and style
2. Potential bugs or issues
3. Architecture and design
4. Performance considerations

Provide specific suggestions for improvement.
"""
        elif analysis_type == "bugs":
            prompt = f"""Find potential bugs and issues in this {language} code:
```{language}
{code}
```

List all potential bugs, edge cases not handled, and logic errors.
For each issue, explain why it's a problem and how to fix it.
"""
        elif analysis_type == "optimize":
            prompt = f"""Optimize this {language} code:
```{language}
{code}
```

Suggest optimizations for:
1. Performance improvements
2. Memory efficiency
3. Readability and maintainability
4. Algorithmic improvements

For each suggestion, explain the benefit and provide example code.
"""
        else:
            prompt = f"""Analyze this {language} code:
```{language}
{code}
```

Provide a comprehensive analysis including structure, functionality,
potential issues, and suggestions for improvement.
"""
        
        # Update stats
        self.stats["model_usage"][ModelType.OLYMPIC.value] += 1
        
        try:
            # Generate analysis using OlympicCoder model
            analysis_text = model_manager.generate_text(
                model_type=ModelType.OLYMPIC,
                prompt=prompt,
                max_length=2048,
                temperature=0.5
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Prepare response
            response = {
                "analysis": analysis_text,
                "language": language,
                "analysis_type": analysis_type,
                "model": ModelType.OLYMPIC.value,
                "response_time": response_time
            }
            
            return response
        
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {
                "error": str(e)
            }
    
    def explain_code(
        self,
        code: str,
        language: str = "python",
        detail_level: str = "medium"
    ) -> Dict[str, Any]:
        """
        Explain code
        
        Args:
            code: Code to explain
            language: Programming language
            detail_level: Level of detail (high, medium, low)
            
        Returns:
            Explanation
        """
        if not self.initialized:
            logger.warning("AI Engine not initialized, initializing with defaults")
            self.initialize_models()
        
        start_time = time.time()
        
        # Determine level of detail
        if detail_level == "high":
            detail_instruction = "Provide a highly detailed line-by-line explanation."
        elif detail_level == "low":
            detail_instruction = "Provide a brief high-level overview."
        else:  # medium
            detail_instruction = "Explain the main components and flow of the code."
        
        prompt = f"""Explain this {language} code:
```{language}
{code}
```

{detail_instruction}
Focus on helping someone understand how the code works and its purpose.
"""
        
        # Update stats
        self.stats["model_usage"][ModelType.OLYMPIC.value] += 1
        
        try:
            # Generate explanation using OlympicCoder model
            explanation_text = model_manager.generate_text(
                model_type=ModelType.OLYMPIC,
                prompt=prompt,
                max_length=2048,
                temperature=0.5
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Prepare response
            response = {
                "explanation": explanation_text,
                "language": language,
                "detail_level": detail_level,
                "model": ModelType.OLYMPIC.value,
                "response_time": response_time
            }
            
            return response
        
        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            return {
                "error": str(e)
            }
    
    def collaborative_response(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a collaborative response using both models
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Collaborative response
        """
        if not self.initialized:
            logger.warning("AI Engine not initialized, initializing with defaults")
            self.initialize_models()
        
        # Import communication system
        from ai_core.model_communication import communication_system, ModelType as CommModelType
        
        start_time = time.time()
        context = context or {}
        
        # Create or continue conversation
        conversation_id = context.get("conversation_id")
        if not conversation_id:
            topic = f"Collaborative response to: {query[:50]}..."
            
            # Convert our model types to communication system model types
            model1 = CommModelType(self.primary_model.value)
            model2 = CommModelType(self.secondary_model.value)
            
            conversation_id = communication_system.create_conversation(
                topic=topic,
                participants=[model1, model2],
                context=context
            )
        
        # Update stats for both models
        self.stats["model_usage"][self.primary_model.value] += 1
        self.stats["model_usage"][self.secondary_model.value] += 1
        
        # Step 1: Primary model generates initial response
        primary_prompt = f"""Generate a response to the following query:
Query: {query}

Provide a comprehensive, accurate, and helpful response.
"""
        primary_response = model_manager.generate_text(
            model_type=self.primary_model,
            prompt=primary_prompt,
            max_length=1024,
            temperature=0.7
        )
        
        # Record in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType(self.primary_model.value),
            message_type="response",
            content=primary_response
        )
        
        # Step 2: Secondary model reviews and improves
        secondary_prompt = f"""Review and improve the following response to the query:

Query: {query}

Initial response:
{primary_response}

Provide any corrections, additional information, or improvements to make the response better.
Focus on adding technical accuracy, practical examples, or filling knowledge gaps.
"""
        
        secondary_response = model_manager.generate_text(
            model_type=self.secondary_model,
            prompt=secondary_prompt,
            max_length=1024,
            temperature=0.7
        )
        
        # Record in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType(self.secondary_model.value),
            message_type="improvement",
            content=secondary_response
        )
        
        # Step 3: Primary model integrates improvements
        final_prompt = f"""Integrate these suggested improvements into the original response:

Original query: {query}

Initial response:
{primary_response}

Suggested improvements:
{secondary_response}

Provide a final, integrated response that incorporates the valuable additions while maintaining coherence and flow.
"""
        
        final_response = model_manager.generate_text(
            model_type=self.primary_model,
            prompt=final_prompt,
            max_length=1536,
            temperature=0.7
        )
        
        # Record in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType(self.primary_model.value),
            message_type="final_response",
            content=final_response
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Prepare response
        response = {
            "text": final_response,
            "conversation_id": conversation_id,
            "draft": primary_response,
            "revision": secondary_response,
            "collaborative_mode": self.collaboration_mode.value,
            "response_time": response_time
        }
        
        return response
    
    def specialized_response(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a specialized response with models handling different aspects
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Specialized response
        """
        if not self.initialized:
            logger.warning("AI Engine not initialized, initializing with defaults")
            self.initialize_models()
        
        # Import communication system
        from ai_core.model_communication import communication_system, ModelType as CommModelType
        
        start_time = time.time()
        context = context or {}
        
        # Create or continue conversation
        conversation_id = context.get("conversation_id")
        if not conversation_id:
            topic = f"Specialized response to: {query[:50]}..."
            
            # Convert our model types to communication system model types
            model1 = CommModelType(self.primary_model.value)
            model2 = CommModelType(self.secondary_model.value)
            
            conversation_id = communication_system.create_conversation(
                topic=topic,
                participants=[model1, model2],
                context=context
            )
        
        # Update stats for both models
        self.stats["model_usage"][self.primary_model.value] += 1
        self.stats["model_usage"][self.secondary_model.value] += 1
        
        # Step 1: Qwen (Primary) creates high-level plan and explanation
        planning_prompt = f"""Break down the following query into key components that need to be addressed:
Query: {query}

1. Identify the main topic and sub-topics
2. Outline a structured approach to answering this query
3. Identify any code or technical components that need to be included
4. Create a high-level explanation of concepts

Focus only on planning and conceptual explanations, not implementation details.
"""
        
        planning_response = model_manager.generate_text(
            model_type=ModelType.QWEN,
            prompt=planning_prompt,
            max_length=1024,
            temperature=0.7
        )
        
        # Record in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType.QWEN,
            message_type="planning",
            content=planning_response
        )
        
        # Step 2: OlympicCoder provides technical implementation
        implementation_prompt = f"""Based on this planning outline, provide the technical implementation details:

Query: {query}

Planning outline:
{planning_response}

Focus on:
1. Any code examples or implementations needed
2. Technical specifics and best practices
3. Implementation tips and troubleshooting advice

Provide practical, working examples where relevant.
"""
        
        implementation_response = model_manager.generate_text(
            model_type=ModelType.OLYMPIC,
            prompt=implementation_prompt,
            max_length=1024,
            temperature=0.5
        )
        
        # Record in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType.OLYMPIC,
            message_type="implementation",
            content=implementation_response
        )
        
        # Step 3: Qwen integrates everything into a coherent response
        integration_prompt = f"""Create a complete, integrated response to the original query by combining the conceptual explanation with the technical implementation:

Original query: {query}

Conceptual explanation and plan:
{planning_response}

Technical implementation:
{implementation_response}

Create a well-structured, comprehensive response that flows naturally between conceptual understanding and practical implementation.
"""
        
        final_response = model_manager.generate_text(
            model_type=ModelType.QWEN,
            prompt=integration_prompt,
            max_length=1536,
            temperature=0.7
        )
        
        # Record in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType.QWEN,
            message_type="final_response",
            content=final_response
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Prepare response
        response = {
            "text": final_response,
            "conversation_id": conversation_id,
            "planning": planning_response,
            "implementation": implementation_response,
            "specialized_mode": "planning_implementation",
            "response_time": response_time
        }
        
        return response
    
    def competitive_response(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate competitive responses from both models
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Best response and alternatives
        """
        if not self.initialized:
            logger.warning("AI Engine not initialized, initializing with defaults")
            self.initialize_models()
        
        # Import communication system
        from ai_core.model_communication import communication_system, ModelType as CommModelType
        
        start_time = time.time()
        context = context or {}
        
        # Create or continue conversation
        conversation_id = context.get("conversation_id")
        if not conversation_id:
            topic = f"Competitive response to: {query[:50]}..."
            
            # Convert our model types to communication system model types
            model1 = CommModelType(self.primary_model.value)
            model2 = CommModelType(self.secondary_model.value)
            system = CommModelType.SYSTEM
            
            conversation_id = communication_system.create_conversation(
                topic=topic,
                participants=[model1, model2, system],
                context=context
            )
        
        # Update stats for both models
        self.stats["model_usage"][self.primary_model.value] += 1
        self.stats["model_usage"][self.secondary_model.value] += 1
        
        # Both models generate responses independently
        prompt = f"""Generate the best possible response to this query:
Query: {query}

Provide a comprehensive, accurate, and helpful response.
Focus on clarity, accuracy, and actionable information.
"""
        
        # Get response from primary model (Qwen)
        qwen_response = model_manager.generate_text(
            model_type=ModelType.QWEN,
            prompt=prompt,
            max_length=1024,
            temperature=0.7
        )
        
        # Record in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType.QWEN,
            message_type="response",
            content=qwen_response
        )
        
        # Get response from secondary model (Olympic)
        olympic_response = model_manager.generate_text(
            model_type=ModelType.OLYMPIC,
            prompt=prompt,
            max_length=1024,
            temperature=0.7
        )
        
        # Record in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType.OLYMPIC,
            message_type="response",
            content=olympic_response
        )
        
        # System evaluates both responses
        evaluation_prompt = f"""Compare these two responses to the query and determine which is better:

Original query: {query}

Response A (Qwen):
{qwen_response}

Response B (Olympic):
{olympic_response}

Evaluate based on:
1. Accuracy and correctness
2. Comprehensiveness
3. Clarity and structure
4. Usefulness and practicality

First, analyze each response separately.
Then, compare them directly and declare a winner with clear reasoning.
"""
        
        # Use Qwen for evaluation
        evaluation = model_manager.generate_text(
            model_type=ModelType.QWEN,
            prompt=evaluation_prompt,
            max_length=1024,
            temperature=0.5
        )
        
        # Record evaluation in conversation
        communication_system.add_message(
            conversation_id=conversation_id,
            from_model=CommModelType.SYSTEM,
            message_type="evaluation",
            content=evaluation
        )
        
        # Determine winner (simplistic approach)
        winner = None
        if "Response A" in evaluation and "winner" in evaluation.lower():
            if "Response A" in evaluation.lower().split("winner")[1]:
                winner = "qwen"
            elif "Response B" in evaluation.lower().split("winner")[1]:
                winner = "olympic"
        
        # If can't determine winner, use primary model
        if not winner:
            winner = self.primary_model.value
        
        # Get winning response
        winning_response = qwen_response if winner == "qwen" else olympic_response
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Prepare response
        response = {
            "text": winning_response,
            "conversation_id": conversation_id,
            "qwen_response": qwen_response,
            "olympic_response": olympic_response,
            "evaluation": evaluation,
            "winner": winner,
            "competitive_mode": True,
            "response_time": response_time
        }
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the AI engine"""
        # Get model manager status
        model_status = model_manager.get_status()
        
        # Compile engine status
        status = {
            "operational": self.initialized,
            "primary_model": self.primary_model.value if self.initialized else None,
            "secondary_model": self.secondary_model.value if self.initialized else None,
            "collaboration_mode": self.collaboration_mode.value if self.initialized else None,
            "conversations": len(self.conversations),
            "stats": {
                "total_queries": self.stats["total_queries"],
                "total_code_generations": self.stats["total_code_generations"],
                "model_usage": self.stats["model_usage"],
                "average_response_time": self.stats["average_response_time"]
            },
            "models": model_status
        }
        
        return status

# Initialize AI engine - the actual model initialization happens on-demand
ai_engine = AIEngine()