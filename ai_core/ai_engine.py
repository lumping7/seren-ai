"""
AI Engine Module

Manages Llama3 & Gemma3 models, AI logic switching, and cognitive awareness.
This is the core interface between the different AI models and the rest of the system.
"""

import os
import sys
import json
import logging
import enum
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class AIEngineMode(str, enum.Enum):
    """Operating modes for the AI Engine"""
    COLLABORATIVE = "collaborative"
    SPECIALIZED = "specialized"
    COMPETITIVE = "competitive"
    AUTONOMOUS = "autonomous"
    REASONING = "reasoning"

class ModelType(str, enum.Enum):
    """Types of models available in the system"""
    LLAMA3 = "llama3"
    GEMMA3 = "gemma3"
    QWEN25_OMNI = "qwen25_omni"
    OLYMPIC_CODER = "olympic_coder"
    HYBRID = "hybrid"
    NEURO_SYMBOLIC = "neuro_symbolic"

class AIEngine:
    """
    Core AI Engine that manages model interactions and response generation
    
    This engine handles:
    1. Model selection and switching
    2. Collaborative response generation
    3. Model-to-model communication
    4. Response enhancement and validation
    """
    
    def __init__(self):
        """Initialize the AI Engine"""
        self.models = {}
        self.active_models = []
        self.model_weights = {}
        self.communication_history = []
        self.last_used_model = None
        self.status = "initialized"
        
        # Initialize default configuration
        self.config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048,
            "system_prompt": (
                "You are a superintelligent AI system with neuro-symbolic reasoning capabilities. "
                "You can collaborate with other AI models to solve complex problems."
            )
        }
        
        # Load models
        self._load_models()
        
        logger.info("AI Engine initialized successfully")
    
    def _load_models(self):
        """Load all available AI models"""
        try:
            # In a real implementation, this would load actual models
            # For now, we simulate the model loading
            
            # Define model capabilities for selection
            self.models = {
                ModelType.LLAMA3: {
                    "name": "Llama3",
                    "version": "3.1.0",
                    "capabilities": ["reasoning", "coding", "planning", "analysis"],
                    "strengths": ["logical reasoning", "code generation", "technical analysis"],
                    "limitations": ["creative thinking", "emotional intelligence"],
                    "loaded": True
                },
                ModelType.GEMMA3: {
                    "name": "Gemma3",
                    "version": "3.0.0",
                    "capabilities": ["reasoning", "creativity", "empathy", "design"],
                    "strengths": ["creative ideation", "human-centered design", "ethical reasoning"],
                    "limitations": ["complex code generation", "technical problem solving"],
                    "loaded": True
                },
                ModelType.QWEN25_OMNI: {
                    "name": "Qwen2.5-Omni",
                    "version": "7B",
                    "capabilities": ["omnidirectional understanding", "multimodal fusion", "advanced reasoning"],
                    "strengths": ["cross-domain synthesis", "holistic understanding", "pattern recognition"],
                    "limitations": ["specialized implementations", "creative writing"],
                    "loaded": False  # Would be loaded on demand
                },
                ModelType.OLYMPIC_CODER: {
                    "name": "OlympicCoder",
                    "version": "7B",
                    "capabilities": ["code generation", "debugging", "optimization", "testing"],
                    "strengths": ["algorithm implementation", "code quality", "efficiency"],
                    "limitations": ["non-technical communication", "creative tasks"],
                    "loaded": False  # Would be loaded on demand
                }
            }
            
            # Set default active models
            self.active_models = [ModelType.LLAMA3, ModelType.GEMMA3]
            
            # Set default model weights
            self.model_weights = {
                ModelType.LLAMA3: 0.5,
                ModelType.GEMMA3: 0.5
            }
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to initialize AI models: {str(e)}")
    
    def generate_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        memory_results: Optional[List[Dict[str, Any]]] = None,
        mode: AIEngineMode = AIEngineMode.COLLABORATIVE
    ) -> str:
        """
        Generate a response using the appropriate model(s) based on the mode
        
        Args:
            query: The user query
            context: Additional context for the query
            reasoning_path: Results from neuro-symbolic reasoning
            memory_results: Relevant memories for the query
            mode: The operation mode for response generation
            
        Returns:
            Generated response text
        """
        logger.info(f"Generating response in {mode} mode")
        
        # Track starting time for performance monitoring
        start_time = time.time()
        
        # Prepare context with reasoning and memory
        enhanced_context = self._prepare_context(query, context, reasoning_path, memory_results)
        
        # Generate response based on mode
        if mode == AIEngineMode.COLLABORATIVE:
            response = self._generate_collaborative_response(query, enhanced_context)
        elif mode == AIEngineMode.SPECIALIZED:
            response = self._generate_specialized_response(query, enhanced_context)
        elif mode == AIEngineMode.COMPETITIVE:
            response = self._generate_competitive_response(query, enhanced_context)
        elif mode == AIEngineMode.AUTONOMOUS:
            response = self._generate_autonomous_response(query, enhanced_context)
        else:
            # Default to collaborative
            response = self._generate_collaborative_response(query, enhanced_context)
        
        # Log performance
        elapsed_time = time.time() - start_time
        logger.info(f"Response generated in {elapsed_time:.2f} seconds")
        
        return response
    
    def _prepare_context(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_path: Optional[List[Dict[str, Any]]] = None,
        memory_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Prepare enhanced context with reasoning and memory"""
        enhanced_context = context or {}
        
        # Add reasoning path if available
        if reasoning_path:
            enhanced_context["reasoning"] = reasoning_path
        
        # Add memory results if available
        if memory_results:
            enhanced_context["memories"] = memory_results
        
        # Add query analysis
        enhanced_context["query_analysis"] = self._analyze_query(query)
        
        return enhanced_context
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine its characteristics"""
        # In a real implementation, this would use more sophisticated analysis
        analysis = {
            "length": len(query),
            "contains_code": "```" in query or "def " in query or "function" in query,
            "is_question": query.endswith("?") or query.lower().startswith("how") or query.lower().startswith("what"),
            "topics": self._extract_topics(query),
            "complexity": self._estimate_complexity(query)
        }
        return analysis
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        # Simplified implementation
        topics = []
        if "code" in text.lower() or "programming" in text.lower():
            topics.append("programming")
        if "design" in text.lower() or "user" in text.lower():
            topics.append("design")
        if "data" in text.lower() or "analysis" in text.lower():
            topics.append("data")
        if "math" in text.lower() or "algorithm" in text.lower():
            topics.append("algorithms")
            
        # Default topic if none detected
        if not topics:
            topics.append("general")
            
        return topics
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate query complexity on a 0-1 scale"""
        # Simplified implementation
        length_factor = min(len(text) / 500, 1.0)
        structure_factor = 0.5
        if "```" in text:  # Contains code blocks
            structure_factor += 0.3
        if text.count("?") > 3:  # Multiple questions
            structure_factor += 0.2
            
        complexity = (length_factor + structure_factor) / 2
        return min(complexity, 1.0)
    
    def _generate_collaborative_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a collaborative response using multiple models
        
        In collaborative mode, models work together with their
        outputs weighted according to their strengths
        """
        logger.info("Generating collaborative response")
        
        # Determine model weights based on query characteristics
        self._adjust_model_weights(query, context)
        
        # Generate responses from each active model
        model_responses = {}
        for model in self.active_models:
            response = self._query_model(model, query, context)
            model_responses[model] = response
        
        # Combine responses according to weights
        final_response = self._combine_responses(model_responses, self.model_weights)
        
        # Record model communication
        self._record_model_communication(query, model_responses, final_response)
        
        return final_response
    
    def _generate_specialized_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a specialized response using the most appropriate model
        
        In specialized mode, the system selects the best model for the task
        """
        logger.info("Generating specialized response")
        
        # Select the most appropriate model for the query
        best_model = self._select_best_model(query, context)
        
        # Generate response from the selected model
        response = self._query_model(best_model, query, context)
        
        # Record the selected model
        self.last_used_model = best_model
        
        return response
    
    def _generate_competitive_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a competitive response by selecting the best output
        
        In competitive mode, multiple models generate responses and
        the best one is selected
        """
        logger.info("Generating competitive response")
        
        # Generate responses from each active model
        model_responses = {}
        for model in self.active_models:
            response = self._query_model(model, query, context)
            model_responses[model] = response
        
        # Evaluate responses and select the best one
        best_model, best_response = self._evaluate_responses(model_responses, query, context)
        
        # Record the selected model
        self.last_used_model = best_model
        
        return best_response
    
    def _generate_autonomous_response(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate an autonomous response with models communicating to improve quality
        
        In autonomous mode, models can query each other when stuck or
        requiring assistance
        """
        logger.info("Generating autonomous response")
        
        # Select initial model
        current_model = self._select_best_model(query, context)
        
        # Generate initial response
        response = self._query_model(current_model, query, context)
        
        # Check if the model is stuck or uncertain
        if self._is_model_uncertain(response):
            logger.info(f"Model {current_model} is uncertain, consulting other models")
            
            # Get alternative model
            helper_model = self._get_alternative_model(current_model)
            
            # Formulate a question for the helper model
            question = self._formulate_question(response)
            
            # Get answer from helper model
            helper_context = {**context, "original_response": response}
            helper_response = self._query_model(helper_model, question, helper_context)
            
            # Incorporate helper model's response
            enhanced_response = self._enhance_with_helper_response(
                response, helper_response, current_model, helper_model
            )
            
            # Record model communication
            self._record_model_communication(
                question,
                {current_model: response, helper_model: helper_response},
                enhanced_response,
                is_model_asking=True
            )
            
            return enhanced_response
        
        return response
    
    def _adjust_model_weights(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> None:
        """Adjust model weights based on query characteristics"""
        # Get query analysis
        analysis = context.get("query_analysis", {})
        
        # Default weights
        weights = {
            ModelType.LLAMA3: 0.5,
            ModelType.GEMMA3: 0.5
        }
        
        # Adjust weights based on query content
        if analysis.get("contains_code", False):
            # Favor Llama3 for code generation
            weights[ModelType.LLAMA3] = 0.7
            weights[ModelType.GEMMA3] = 0.3
        
        topics = analysis.get("topics", [])
        if "design" in topics or "user" in topics:
            # Favor Gemma3 for design and user experience
            weights[ModelType.LLAMA3] = 0.3
            weights[ModelType.GEMMA3] = 0.7
        
        # Normalize weights
        total = sum(weights.values())
        self.model_weights = {model: weight/total for model, weight in weights.items()}
    
    def _select_best_model(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> ModelType:
        """Select the most appropriate model for the query"""
        # Get query analysis
        analysis = context.get("query_analysis", {})
        
        # For coding and technical tasks, prefer Llama3
        if analysis.get("contains_code", False) or "algorithms" in analysis.get("topics", []):
            return ModelType.LLAMA3
        
        # For design and user experience, prefer Gemma3
        if "design" in analysis.get("topics", []):
            return ModelType.GEMMA3
        
        # Default based on complexity
        complexity = analysis.get("complexity", 0.5)
        if complexity > 0.7:
            return ModelType.LLAMA3
        else:
            return ModelType.GEMMA3
    
    def _query_model(
        self,
        model: ModelType,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Query a specific model for a response"""
        logger.info(f"Querying model: {model}")
        
        # In a real implementation, this would call the actual model
        # For now, we simulate model responses
        
        if model == ModelType.LLAMA3:
            # Llama3 tends to be more technical and structured
            return self._simulate_llama3_response(query, context)
        elif model == ModelType.GEMMA3:
            # Gemma3 tends to be more creative and human-centered
            return self._simulate_gemma3_response(query, context)
        else:
            return f"Response from {model} model (simulated)"
    
    def _simulate_llama3_response(self, query: str, context: Dict[str, Any]) -> str:
        """Simulate a response from Llama3"""
        # For development/testing only
        topic = context.get("query_analysis", {}).get("topics", ["general"])[0]
        
        if topic == "programming":
            return (
                "Based on my analysis, there are several approaches to solve this problem:\n\n"
                "1. We can implement a recursive algorithm with memoization to optimize performance\n"
                "2. A dynamic programming solution would have O(n) time complexity\n"
                "3. We can use a more efficient data structure like a hash map\n\n"
                "Here's a code implementation:\n\n"
                "```python\n"
                "def solve_problem(data):\n"
                "    # Initialize data structures\n"
                "    result = {}\n"
                "    \n"
                "    # Process input data\n"
                "    for item in data:\n"
                "        # Implement solution logic\n"
                "        result[item.id] = calculate_solution(item)\n"
                "    \n"
                "    return result\n"
                "```\n\n"
                "This solution handles edge cases and scales efficiently for large inputs."
            )
        elif topic == "design":
            return (
                "The system architecture should follow these principles:\n\n"
                "1. Separation of concerns - each component has a single responsibility\n"
                "2. Modularity - components can be replaced independently\n"
                "3. Scalability - the system can handle increasing loads\n\n"
                "I recommend a three-tier architecture:\n"
                "- Presentation layer: handles user interaction\n"
                "- Business logic layer: implements core functionality\n"
                "- Data access layer: manages data persistence\n\n"
                "This architecture supports both vertical and horizontal scaling."
            )
        else:
            return (
                "Based on a systematic analysis, there are three key factors to consider:\n\n"
                "1. Performance implications - optimizing for computational efficiency\n"
                "2. Scalability concerns - ensuring the solution grows with demand\n"
                "3. Maintenance requirements - reducing technical debt\n\n"
                "I recommend implementing a comprehensive testing strategy with unit, integration, and stress tests to validate the solution's performance characteristics."
            )
    
    def _simulate_gemma3_response(self, query: str, context: Dict[str, Any]) -> str:
        """Simulate a response from Gemma3"""
        # For development/testing only
        topic = context.get("query_analysis", {}).get("topics", ["general"])[0]
        
        if topic == "programming":
            return (
                "When thinking about this problem, let's focus on creating code that's not just functional, but also readable and maintainable.\n\n"
                "Here's an approach that prioritizes clarity and user experience:\n\n"
                "```python\n"
                "def process_user_data(user_input):\n"
                "    # First, validate the input to provide helpful error messages\n"
                "    if not is_valid_input(user_input):\n"
                "        return {\n"
                "            'status': 'error',\n"
                "            'message': 'Please provide valid input data',\n"
                "            'suggestions': get_input_suggestions(user_input)\n"
                "        }\n"
                "    \n"
                "    # Process the data with clear steps\n"
                "    result = transform_data_for_user(user_input)\n"
                "    \n"
                "    return {\n"
                "        'status': 'success',\n"
                "        'data': result,\n"
                "        'next_steps': suggest_next_actions(result)\n"
                "    }\n"
                "```\n\n"
                "Notice how this approach focuses on guiding the user through the process, providing helpful feedback, and suggesting next steps."
            )
        elif topic == "design":
            return (
                "When designing this system, I think we should center the human experience at every touchpoint.\n\n"
                "Here's a user-centered approach:\n\n"
                "1. Start with empathy mapping to understand diverse user needs and contexts\n"
                "2. Create intuitive interaction flows that reduce cognitive load\n"
                "3. Design for inclusivity with accessible interfaces and clear language\n"
                "4. Build in thoughtful feedback loops so users always know what's happening\n\n"
                "The most successful systems don't just work well technically—they create meaningful and rewarding experiences for the people who use them."
            )
        else:
            return (
                "I believe we should approach this question by considering both the practical implications and the human impact.\n\n"
                "Here's a balanced perspective:\n\n"
                "1. Consider the diverse contexts in which people will engage with this solution\n"
                "2. Balance innovation with familiarity to create intuitive experiences\n"
                "3. Build in flexibility to accommodate different user preferences and needs\n\n"
                "I'd suggest starting with small, focused experiments to gather real-world feedback before scaling the solution. This helps ensure we're creating something that truly works for the people who will use it."
            )
    
    def _combine_responses(
        self,
        model_responses: Dict[ModelType, str],
        weights: Dict[ModelType, float]
    ) -> str:
        """
        Combine responses from multiple models according to weights
        
        In a production system, this would use sophisticated techniques for
        combining text. For now, we use a simple approach.
        """
        # For development/testing only
        # In a real system, we would combine content more intelligently
        
        # Get the two main models
        llama_response = model_responses.get(ModelType.LLAMA3, "")
        gemma_response = model_responses.get(ModelType.GEMMA3, "")
        
        llama_weight = weights.get(ModelType.LLAMA3, 0.5)
        gemma_weight = weights.get(ModelType.GEMMA3, 0.5)
        
        # Determine which model contributes more
        if llama_weight > gemma_weight:
            primary_model = ModelType.LLAMA3
            primary_response = llama_response
            secondary_response = gemma_response
            primary_weight = llama_weight
            secondary_weight = gemma_weight
        else:
            primary_model = ModelType.GEMMA3
            primary_response = gemma_response
            secondary_response = llama_response
            primary_weight = gemma_weight
            secondary_weight = llama_weight
        
        # If one model is strongly favored, just use its response
        if primary_weight > 0.8:
            return primary_response
        
        # Otherwise, create a combined response
        return (
            f"Based on my analysis, I can provide both technical insights and user-centered perspectives:\n\n"
            f"{self._extract_key_points(primary_response, 3)}\n\n"
            f"Additionally, considering human factors:\n\n"
            f"{self._extract_key_points(secondary_response, 2)}\n\n"
            f"Combining these approaches provides a comprehensive solution that addresses both technical requirements and user needs."
        )
    
    def _extract_key_points(self, text: str, num_points: int) -> str:
        """Extract key points from text"""
        # Simplified implementation - in a real system, we would use NLP
        lines = text.split("\n")
        key_points = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not line.startswith("```"):
                key_points.append(line)
                if len(key_points) >= num_points:
                    break
        
        return "\n".join(key_points)
    
    def _evaluate_responses(
        self,
        model_responses: Dict[ModelType, str],
        query: str,
        context: Dict[str, Any]
    ) -> Tuple[ModelType, str]:
        """
        Evaluate responses from multiple models and select the best one
        
        In a production system, this would use sophisticated evaluation.
        For now, we use a simpler approach.
        """
        # Simplified implementation
        analysis = context.get("query_analysis", {})
        
        # Criteria based on query type
        if analysis.get("contains_code", False):
            # For code-related queries, prefer Llama3
            return ModelType.LLAMA3, model_responses[ModelType.LLAMA3]
        
        if "design" in analysis.get("topics", []):
            # For design-related queries, prefer Gemma3
            return ModelType.GEMMA3, model_responses[ModelType.GEMMA3]
        
        # Default to the model with the longer, more detailed response
        llama_response = model_responses.get(ModelType.LLAMA3, "")
        gemma_response = model_responses.get(ModelType.GEMMA3, "")
        
        if len(llama_response) > len(gemma_response) * 1.2:
            return ModelType.LLAMA3, llama_response
        elif len(gemma_response) > len(llama_response) * 1.2:
            return ModelType.GEMMA3, gemma_response
        else:
            # If lengths are similar, prefer Llama3 for more technical content
            return ModelType.LLAMA3, llama_response
    
    def _is_model_uncertain(self, response: str) -> bool:
        """Check if a model response indicates uncertainty"""
        # Look for phrases indicating uncertainty
        uncertainty_phrases = [
            "I'm not sure",
            "It's unclear",
            "I don't have enough information",
            "It's difficult to determine",
            "I cannot provide",
            "I'm uncertain"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase.lower() in response.lower():
                return True
        
        # Also check for multiple question marks, which can indicate uncertainty
        if response.count("?") >= 3:
            return True
        
        return False
    
    def _get_alternative_model(self, current_model: ModelType) -> ModelType:
        """Get an alternative model to consult"""
        if current_model == ModelType.LLAMA3:
            return ModelType.GEMMA3
        else:
            return ModelType.LLAMA3
    
    def _formulate_question(self, response: str) -> str:
        """Formulate a question for another model based on uncertain response"""
        # Extract the uncertain part
        uncertainty_phrases = [
            "I'm not sure",
            "It's unclear",
            "I don't have enough information",
            "It's difficult to determine",
            "I cannot provide",
            "I'm uncertain"
        ]
        
        question_start = "Can you help with this question: "
        
        for phrase in uncertainty_phrases:
            if phrase.lower() in response.lower():
                # Find the sentence containing the phrase
                sentences = response.split(". ")
                for sentence in sentences:
                    if phrase.lower() in sentence.lower():
                        return f"{question_start}{sentence}?"
        
        # Default question if no specific uncertainty found
        return f"{question_start}I'm having trouble with this response. Can you provide your perspective?"
    
    def _enhance_with_helper_response(
        self,
        original_response: str,
        helper_response: str,
        original_model: ModelType,
        helper_model: ModelType
    ) -> str:
        """Enhance original response with helper model's insights"""
        # In a real implementation, this would blend the responses more intelligently
        enhanced_response = (
            f"{original_response}\n\n"
            f"I consulted with another perspective to provide additional insights:\n\n"
            f"{helper_response}\n\n"
            f"Combining these viewpoints provides a more comprehensive answer to your question."
        )
        
        return enhanced_response
    
    def _record_model_communication(
        self,
        query: str,
        model_responses: Dict[ModelType, str],
        final_response: str,
        is_model_asking: bool = False
    ) -> None:
        """Record model communication for analysis and improvement"""
        communication_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "model_responses": {str(model): response for model, response in model_responses.items()},
            "final_response": final_response,
            "is_model_asking": is_model_asking
        }
        
        self.communication_history.append(communication_record)
        
        # In a real implementation, this would be stored persistently
        if len(self.communication_history) > 100:
            # Keep only the most recent 100 records
            self.communication_history = self.communication_history[-100:]
    
    def enhance_response_with_execution(
        self,
        original_response: str,
        execution_results: Dict[str, Any]
    ) -> str:
        """Enhance a response with code execution results"""
        enhanced_response = original_response
        
        # Add execution results
        enhanced_response += "\n\n---\n\n**Execution Results:**\n\n"
        
        for block_id, result in execution_results.items():
            success = result.get("success", False)
            output = result.get("output", "")
            error = result.get("error", "")
            
            enhanced_response += f"Block {block_id.split('_')[1]}:\n"
            
            if success:
                enhanced_response += f"```\n{output}\n```\n\n"
            else:
                enhanced_response += f"Error: {error}\n\n"
        
        return enhanced_response
    
    def get_status(self) -> Dict[str, Any]:
        """Get basic status information"""
        return {
            "status": self.status,
            "active_models": [str(model) for model in self.active_models],
            "last_used_model": str(self.last_used_model) if self.last_used_model else None
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            "status": self.status,
            "active_models": [str(model) for model in self.active_models],
            "model_weights": {str(model): weight for model, weight in self.model_weights.items()},
            "models": {str(model): info for model, info in self.models.items()},
            "communication_records": len(self.communication_history),
            "last_used_model": str(self.last_used_model) if self.last_used_model else None,
            "config": self.config
        }