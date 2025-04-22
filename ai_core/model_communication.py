"""
Model Communication System for Seren

Provides mechanisms for advanced inter-model communication, collaboration,
and information exchange within the OpenManus architecture.
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

# Import security components
from security.quantum_encryption import quantum_encryption, SecurityLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of models in the system"""
    QWEN = "qwen"           # Qwen model (previously Llama)
    OLYMPIC = "olympic"     # Olympic Coder model (previously Gemma)
    HYBRID = "hybrid"       # Hybrid model (combined capabilities)
    SYSTEM = "system"       # System-generated messages

class MessageType(Enum):
    """Types of inter-model messages"""
    QUESTION = "question"      # Question from one model to another
    ANSWER = "answer"          # Answer to a question
    SUGGESTION = "suggestion"  # Suggestion for improvement
    CRITIQUE = "critique"      # Critique of a solution
    REFINEMENT = "refinement"  # Refinement of a previous solution
    STATUS = "status"          # Status update
    ERROR = "error"            # Error message
    SYSTEM = "system"          # System message

class CollaborationMode(Enum):
    """Modes of model collaboration"""
    COLLABORATIVE = "collaborative"  # Models work together on the same task
    SPECIALIZED = "specialized"      # Models work on different aspects of a task
    COMPETITIVE = "competitive"      # Models compete to produce the best solution

class CommunicationSystem:
    """
    Model Communication System for Seren
    
    Provides mechanisms for advanced inter-model communication, collaboration,
    and information exchange:
    - Model-to-model messaging
    - Multi-agent conversations
    - Structured knowledge exchange
    - Collaborative problem-solving
    - Query and response patterns
    
    Bleeding-edge capabilities:
    1. Context-aware communication
    2. Cross-model knowledge synthesis
    3. Parallel collaborative reasoning
    4. Self-reflection and model criticism
    5. Structured output negotiation
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the communication system"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Conversations registry
        self.conversations = {}
        
        # Active participants
        self.participants = {
            ModelType.QWEN.value: True,
            ModelType.OLYMPIC.value: True,
            ModelType.HYBRID.value: False,
            ModelType.SYSTEM.value: True
        }
        
        # Message queues (model-specific)
        self.message_queues = {
            ModelType.QWEN.value: queue.Queue(),
            ModelType.OLYMPIC.value: queue.Queue(),
            ModelType.HYBRID.value: queue.Queue(),
            ModelType.SYSTEM.value: queue.Queue()
        }
        
        # Message history
        self.message_history = []
        
        # Communication stats
        self.stats = {
            "messages_sent": 0,
            "messages_by_type": {message_type.value: 0 for message_type in MessageType},
            "messages_by_model": {model_type.value: 0 for model_type in ModelType},
            "conversations_created": 0,
            "average_response_time": 0
        }
        
        # Start message processors
        self._start_message_processors()
        
        # Current collaboration mode
        self.collaboration_mode = CollaborationMode.COLLABORATIVE
        
        logger.info("Communication System initialized")
    
    def _start_message_processors(self) -> None:
        """Start the message processing threads"""
        # In a real implementation, this would start actual threads
        # For simulation, we'll just set up placeholders
        self.message_processors = {}
        
        # This would be threaded in a real implementation
        # for model_type in ModelType:
        #     processor = threading.Thread(
        #         target=self._process_messages,
        #         args=(model_type.value,),
        #         daemon=True
        #     )
        #     processor.start()
        #     self.message_processors[model_type.value] = processor
    
    def _process_messages(self, model_type: str) -> None:
        """
        Process messages for a specific model
        
        Args:
            model_type: Type of model to process messages for
        """
        # In a real implementation, this would be a thread function
        message_queue = self.message_queues.get(model_type)
        
        if not message_queue:
            logger.error(f"No message queue for model type: {model_type}")
            return
        
        while True:
            try:
                # Get next message from queue
                message = message_queue.get(timeout=1)
                
                # Process message (in a real implementation, this would route to the model)
                logger.info(f"Processing message for {model_type}: {message['id']}")
                
                # Mark as done
                message_queue.task_done()
            
            except queue.Empty:
                # No messages, continue
                continue
            
            except Exception as e:
                logger.error(f"Error processing message for {model_type}: {str(e)}")
    
    def create_conversation(
        self,
        topic: str,
        participants: List[Union[ModelType, str]],
        context: Dict[str, Any] = None
    ) -> str:
        """
        Create a new conversation
        
        Args:
            topic: Conversation topic
            participants: List of participants
            context: Additional context
            
        Returns:
            Conversation ID
        """
        # Generate conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Convert participants to string if needed
        participant_values = []
        for participant in participants:
            if isinstance(participant, ModelType):
                participant_values.append(participant.value)
            else:
                # Validate participant
                try:
                    ModelType(participant)
                    participant_values.append(participant)
                except ValueError:
                    logger.warning(f"Invalid participant type: {participant}")
        
        # Create conversation
        conversation = {
            "id": conversation_id,
            "topic": topic,
            "participants": participant_values,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "context": context or {},
            "status": "active"
        }
        
        # Store conversation
        self.conversations[conversation_id] = conversation
        
        # Update stats
        self.stats["conversations_created"] += 1
        
        # Add system message
        self.add_message(
            conversation_id=conversation_id,
            from_model=ModelType.SYSTEM.value,
            message_type=MessageType.SYSTEM.value,
            content=f"Conversation started: {topic}",
            metadata={
                "participants": participant_values
            }
        )
        
        logger.info(f"Conversation created: {conversation_id} - {topic}")
        
        return conversation_id
    
    def add_message(
        self,
        conversation_id: str,
        from_model: Union[ModelType, str],
        message_type: Union[MessageType, str],
        content: str,
        to_model: Union[ModelType, str, None] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Add a message to a conversation
        
        Args:
            conversation_id: Conversation ID
            from_model: Source model
            message_type: Type of message
            content: Message content
            to_model: Target model (if applicable)
            metadata: Additional metadata
            
        Returns:
            Added message
        """
        # Check if conversation exists
        if conversation_id not in self.conversations:
            logger.error(f"Conversation not found: {conversation_id}")
            return {"error": f"Conversation not found: {conversation_id}"}
        
        # Convert model types to string if needed
        from_model_value = from_model.value if isinstance(from_model, ModelType) else from_model
        to_model_value = to_model.value if isinstance(to_model, ModelType) else to_model
        
        # Convert message type to string if needed
        message_type_value = message_type.value if isinstance(message_type, MessageType) else message_type
        
        # Generate message ID
        message_id = str(uuid.uuid4())
        
        # Create message
        message = {
            "id": message_id,
            "conversation_id": conversation_id,
            "from": from_model_value,
            "to": to_model_value,
            "type": message_type_value,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to conversation
        conversation = self.conversations[conversation_id]
        conversation["messages"].append(message)
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Add to message history
        self.message_history.append(message)
        
        # Update stats
        self.stats["messages_sent"] += 1
        self.stats["messages_by_type"][message_type_value] = self.stats["messages_by_type"].get(message_type_value, 0) + 1
        self.stats["messages_by_model"][from_model_value] = self.stats["messages_by_model"].get(from_model_value, 0) + 1
        
        # If directed message, add to recipient's queue
        if to_model_value and to_model_value in self.message_queues:
            self.message_queues[to_model_value].put(message)
        
        logger.info(f"Message added to conversation {conversation_id}: {message_id}")
        
        return message
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        conversation = self.conversations.get(conversation_id)
        
        if conversation:
            # Make a copy with messages sorted by timestamp
            result = conversation.copy()
            result["messages"] = sorted(result["messages"], key=lambda m: m["timestamp"])
            return result
        
        return None
    
    def get_messages(
        self,
        conversation_id: str,
        from_model: Union[ModelType, str, None] = None,
        to_model: Union[ModelType, str, None] = None,
        message_type: Union[MessageType, str, None] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation
        
        Args:
            conversation_id: Conversation ID
            from_model: Filter by source model
            to_model: Filter by target model
            message_type: Filter by message type
            limit: Maximum number of messages to return
            
        Returns:
            Filtered messages
        """
        # Check if conversation exists
        if conversation_id not in self.conversations:
            logger.error(f"Conversation not found: {conversation_id}")
            return []
        
        # Get conversation
        conversation = self.conversations[conversation_id]
        
        # Convert model types to string if needed
        from_model_value = from_model.value if isinstance(from_model, ModelType) else from_model
        to_model_value = to_model.value if isinstance(to_model, ModelType) else to_model
        
        # Convert message type to string if needed
        message_type_value = message_type.value if isinstance(message_type, MessageType) else message_type
        
        # Filter messages
        messages = conversation["messages"]
        
        if from_model_value:
            messages = [m for m in messages if m["from"] == from_model_value]
        
        if to_model_value:
            messages = [m for m in messages if m["to"] == to_model_value]
        
        if message_type_value:
            messages = [m for m in messages if m["type"] == message_type_value]
        
        # Sort by timestamp
        messages = sorted(messages, key=lambda m: m["timestamp"])
        
        # Apply limit
        if limit and limit > 0:
            messages = messages[-limit:]
        
        return messages
    
    def ask_question(
        self,
        from_model: Union[ModelType, str],
        to_model: Union[ModelType, str],
        content: str,
        context: Dict[str, Any] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question from one model to another
        
        Args:
            from_model: Source model
            to_model: Target model
            content: Question content
            context: Additional context
            conversation_id: Existing conversation ID
            
        Returns:
            Question message
        """
        # Convert model types to string if needed
        from_model_value = from_model.value if isinstance(from_model, ModelType) else from_model
        to_model_value = to_model.value if isinstance(to_model, ModelType) else to_model
        
        # Validate models
        if from_model_value not in self.participants or not self.participants[from_model_value]:
            logger.error(f"Source model not available: {from_model_value}")
            return {"error": f"Source model not available: {from_model_value}"}
        
        if to_model_value not in self.participants or not self.participants[to_model_value]:
            logger.error(f"Target model not available: {to_model_value}")
            return {"error": f"Target model not available: {to_model_value}"}
        
        # Create or get conversation
        if conversation_id and conversation_id in self.conversations:
            # Check if models are participants
            conversation = self.conversations[conversation_id]
            if from_model_value not in conversation["participants"]:
                conversation["participants"].append(from_model_value)
            if to_model_value not in conversation["participants"]:
                conversation["participants"].append(to_model_value)
        else:
            # Create new conversation
            conversation_id = self.create_conversation(
                topic=f"Question from {from_model_value} to {to_model_value}",
                participants=[from_model_value, to_model_value],
                context=context
            )
        
        # Add question message
        message = self.add_message(
            conversation_id=conversation_id,
            from_model=from_model_value,
            to_model=to_model_value,
            message_type=MessageType.QUESTION.value,
            content=content,
            metadata={
                "context": context or {}
            }
        )
        
        # In a real implementation, this would trigger the model to generate a response
        # For simulation, we'll add a system placeholder message
        answer = self.add_message(
            conversation_id=conversation_id,
            from_model=to_model_value,
            to_model=from_model_value,
            message_type=MessageType.ANSWER.value,
            content=f"This is a simulated response from {to_model_value} to the question: {content[:50]}...",
            metadata={
                "simulated": True
            }
        )
        
        return message
    
    def send_suggestion(
        self,
        from_model: Union[ModelType, str],
        to_model: Union[ModelType, str],
        content: str,
        context: Dict[str, Any] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a suggestion from one model to another
        
        Args:
            from_model: Source model
            to_model: Target model
            content: Suggestion content
            context: Additional context
            conversation_id: Existing conversation ID
            
        Returns:
            Suggestion message
        """
        # Convert model types to string if needed
        from_model_value = from_model.value if isinstance(from_model, ModelType) else from_model
        to_model_value = to_model.value if isinstance(to_model, ModelType) else to_model
        
        # Validate models
        if from_model_value not in self.participants or not self.participants[from_model_value]:
            logger.error(f"Source model not available: {from_model_value}")
            return {"error": f"Source model not available: {from_model_value}"}
        
        if to_model_value not in self.participants or not self.participants[to_model_value]:
            logger.error(f"Target model not available: {to_model_value}")
            return {"error": f"Target model not available: {to_model_value}"}
        
        # Create or get conversation
        if conversation_id and conversation_id in self.conversations:
            # Check if models are participants
            conversation = self.conversations[conversation_id]
            if from_model_value not in conversation["participants"]:
                conversation["participants"].append(from_model_value)
            if to_model_value not in conversation["participants"]:
                conversation["participants"].append(to_model_value)
        else:
            # Create new conversation
            conversation_id = self.create_conversation(
                topic=f"Suggestion from {from_model_value} to {to_model_value}",
                participants=[from_model_value, to_model_value],
                context=context
            )
        
        # Add suggestion message
        message = self.add_message(
            conversation_id=conversation_id,
            from_model=from_model_value,
            to_model=to_model_value,
            message_type=MessageType.SUGGESTION.value,
            content=content,
            metadata={
                "context": context or {}
            }
        )
        
        return message
    
    def send_critique(
        self,
        from_model: Union[ModelType, str],
        to_model: Union[ModelType, str],
        content: str,
        target_message_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a critique from one model to another
        
        Args:
            from_model: Source model
            to_model: Target model
            content: Critique content
            target_message_id: ID of the message being critiqued
            context: Additional context
            conversation_id: Existing conversation ID
            
        Returns:
            Critique message
        """
        # Convert model types to string if needed
        from_model_value = from_model.value if isinstance(from_model, ModelType) else from_model
        to_model_value = to_model.value if isinstance(to_model, ModelType) else to_model
        
        # Validate models
        if from_model_value not in self.participants or not self.participants[from_model_value]:
            logger.error(f"Source model not available: {from_model_value}")
            return {"error": f"Source model not available: {from_model_value}"}
        
        if to_model_value not in self.participants or not self.participants[to_model_value]:
            logger.error(f"Target model not available: {to_model_value}")
            return {"error": f"Target model not available: {to_model_value}"}
        
        # Create or get conversation
        if conversation_id and conversation_id in self.conversations:
            # Check if models are participants
            conversation = self.conversations[conversation_id]
            if from_model_value not in conversation["participants"]:
                conversation["participants"].append(from_model_value)
            if to_model_value not in conversation["participants"]:
                conversation["participants"].append(to_model_value)
        else:
            # Create new conversation
            conversation_id = self.create_conversation(
                topic=f"Critique from {from_model_value} to {to_model_value}",
                participants=[from_model_value, to_model_value],
                context=context
            )
        
        # Add critique message
        message = self.add_message(
            conversation_id=conversation_id,
            from_model=from_model_value,
            to_model=to_model_value,
            message_type=MessageType.CRITIQUE.value,
            content=content,
            metadata={
                "target_message_id": target_message_id,
                "context": context or {}
            }
        )
        
        return message
    
    def collaborate(
        self,
        task: str,
        models: List[Union[ModelType, str]],
        mode: Union[CollaborationMode, str] = CollaborationMode.COLLABORATIVE,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Initiate collaborative work between models
        
        Args:
            task: Task description
            models: Models to involve in collaboration
            mode: Collaboration mode
            context: Additional context
            
        Returns:
            Collaboration setup information
        """
        # Convert model types to string if needed
        model_values = []
        for model in models:
            if isinstance(model, ModelType):
                model_values.append(model.value)
            else:
                # Validate model
                try:
                    ModelType(model)
                    model_values.append(model)
                except ValueError:
                    logger.warning(f"Invalid model type: {model}")
        
        # Convert mode to string if needed
        mode_value = mode.value if isinstance(mode, CollaborationMode) else mode
        
        # Validate mode
        try:
            collaboration_mode = CollaborationMode(mode_value)
        except ValueError:
            logger.error(f"Invalid collaboration mode: {mode_value}")
            return {"error": f"Invalid collaboration mode: {mode_value}"}
        
        # Set current collaboration mode
        self.collaboration_mode = collaboration_mode
        
        # Create a conversation for the collaboration
        conversation_id = self.create_conversation(
            topic=f"Collaboration: {task}",
            participants=model_values + [ModelType.SYSTEM.value],
            context=context
        )
        
        # Add system message about the collaboration
        self.add_message(
            conversation_id=conversation_id,
            from_model=ModelType.SYSTEM.value,
            message_type=MessageType.SYSTEM.value,
            content=f"Collaboration started in {mode_value} mode: {task}",
            metadata={
                "task": task,
                "mode": mode_value,
                "participants": model_values
            }
        )
        
        # Add task assignment message
        if mode_value == CollaborationMode.COLLABORATIVE.value:
            # All models work on the same task
            message = f"All models should collaborate on the task: {task}"
        
        elif mode_value == CollaborationMode.SPECIALIZED.value:
            # Split the task based on specialization
            message = f"Task will be divided based on specialization: {task}"
            
            # Example specialization
            for model in model_values:
                if model == ModelType.QWEN.value:
                    self.add_message(
                        conversation_id=conversation_id,
                        from_model=ModelType.SYSTEM.value,
                        to_model=model,
                        message_type=MessageType.SYSTEM.value,
                        content=f"Your specialization: Planning and architecture for {task}"
                    )
                elif model == ModelType.OLYMPIC.value:
                    self.add_message(
                        conversation_id=conversation_id,
                        from_model=ModelType.SYSTEM.value,
                        to_model=model,
                        message_type=MessageType.SYSTEM.value,
                        content=f"Your specialization: Implementation and testing for {task}"
                    )
        
        elif mode_value == CollaborationMode.COMPETITIVE.value:
            # Models compete
            message = f"All models should compete to provide the best solution for: {task}"
        
        # Add the mode-specific system message
        self.add_message(
            conversation_id=conversation_id,
            from_model=ModelType.SYSTEM.value,
            message_type=MessageType.SYSTEM.value,
            content=message
        )
        
        # Return collaboration information
        return {
            "conversation_id": conversation_id,
            "mode": mode_value,
            "participants": model_values,
            "task": task
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the communication system"""
        total_messages = self.stats["messages_sent"]
        
        # Calculate average response time
        avg_response_time = 0
        if "ANSWER" in self.stats["messages_by_type"] and self.stats["messages_by_type"]["ANSWER"] > 0:
            # In a real implementation, this would be calculated from actual response times
            # Here we just use a placeholder
            avg_response_time = 1.5  # seconds
        
        return {
            "operational": True,
            "models": {
                model: {"active": active}
                for model, active in self.participants.items()
            },
            "collaboration_mode": self.collaboration_mode.value,
            "conversations": len(self.conversations),
            "stats": {
                "total_messages": total_messages,
                "messages_by_type": self.stats["messages_by_type"],
                "messages_by_model": self.stats["messages_by_model"],
                "average_response_time": avg_response_time
            }
        }

# Initialize communication system
communication_system = CommunicationSystem()