"""
Model Communication System

Enables seamless communication between AI models, allowing them to collaborate,
ask questions of each other, and synthesize combined insights.
"""

import os
import sys
import json
import logging
import time
import uuid
import threading
import queue
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from enum import Enum
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Define model types
class ModelType(Enum):
    """Types of AI models in the system"""
    QWEN = "qwen"              # Qwen2.5-omni-7b
    OLYMPIC = "olympic"        # OlympicCoder-7B
    HYBRID = "hybrid"          # Combined model
    SPECIALIZED = "specialized"  # Task-specific model
    SYSTEM = "system"          # System-generated messages

# Define message types
class MessageType(Enum):
    """Types of messages exchanged between models"""
    QUESTION = "question"      # Question to another model
    ANSWER = "answer"          # Answer to a question
    SUGGESTION = "suggestion"  # Suggestion for improvement
    CRITIQUE = "critique"      # Critique of a solution
    CLARIFICATION = "clarification"  # Request for clarification
    EXPLANATION = "explanation"  # Explanation of approach
    CODE = "code"              # Code snippet
    ERROR = "error"            # Error report
    SOLUTION = "solution"      # Solution proposal
    PLANNING = "planning"      # Planning discussion
    SYSTEM = "system"          # System message

# Define message priority
class MessagePriority(Enum):
    """Priority levels for messages"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

# Define message states
class MessageState(Enum):
    """States a message can be in"""
    PENDING = "pending"        # Waiting to be processed
    DELIVERED = "delivered"    # Delivered to recipient
    READ = "read"              # Read by recipient
    ANSWERED = "answered"      # Answered by recipient
    EXPIRED = "expired"        # No longer relevant
    FAILED = "failed"          # Failed to process

class CommunicationSystem:
    """
    Model Communication System
    
    Enables seamless, structured communication between different AI models
    in the system, allowing for collaborative problem-solving and knowledge sharing.
    
    Bleeding-edge capabilities:
    1. Multi-model dialogue with contextual awareness
    2. Structured knowledge exchange with semantic routing
    3. Interruption and priority handling for critical insights
    4. Backpropagation of insights across conversation history
    5. Parallel reasoning paths with synthesis
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the communication system"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Conversations and messages storage
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, Dict[str, Any]] = {}
        
        # Message queue and processing thread
        self.message_queue = queue.PriorityQueue()
        self.message_thread = threading.Thread(target=self._message_processor, daemon=True)
        self.message_thread.start()
        
        # Message handlers for different model types
        self.message_handlers: Dict[ModelType, Callable] = {}
        
        # Configure default timeouts
        self.default_answer_timeout = 60  # seconds
        
        # Message templates for different message types
        self.message_templates = self._load_message_templates()
        
        # Metrics tracking
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "questions_asked": 0,
            "questions_answered": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "model_interactions": {
                ModelType.QWEN.value: 0,
                ModelType.OLYMPIC.value: 0,
                ModelType.HYBRID.value: 0,
                ModelType.SPECIALIZED.value: 0,
                ModelType.SYSTEM.value: 0
            }
        }
        
        logger.info("Model Communication System initialized")
    
    def _load_message_templates(self) -> Dict[MessageType, str]:
        """Load message templates for different message types"""
        return {
            MessageType.QUESTION: "Question from {from_model}: {content}",
            MessageType.ANSWER: "Answer from {from_model}: {content}",
            MessageType.SUGGESTION: "Suggestion from {from_model}: {content}",
            MessageType.CRITIQUE: "Critique from {from_model}: {content}",
            MessageType.CLARIFICATION: "Clarification request from {from_model}: {content}",
            MessageType.EXPLANATION: "Explanation from {from_model}: {content}",
            MessageType.CODE: "Code from {from_model}: {content}",
            MessageType.ERROR: "Error report from {from_model}: {content}",
            MessageType.SOLUTION: "Solution from {from_model}: {content}",
            MessageType.PLANNING: "Planning from {from_model}: {content}",
            MessageType.SYSTEM: "System message: {content}"
        }
    
    def register_message_handler(self, model_type: ModelType, handler: Callable) -> None:
        """Register a message handler for a specific model type"""
        self.message_handlers[model_type] = handler
        logger.info(f"Registered message handler for {model_type.value}")
    
    def create_conversation(
        self,
        topic: str,
        context: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create a new conversation
        
        Args:
            topic: The conversation topic
            context: Contextual information for the conversation
            metadata: Additional metadata
            
        Returns:
            Conversation ID
        """
        # Generate conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Create conversation object
        conversation = {
            "id": conversation_id,
            "topic": topic,
            "context": context or {},
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_ids": [],
            "participants": set(),
            "status": "active"
        }
        
        # Store conversation
        self.conversations[conversation_id] = conversation
        
        logger.info(f"Created conversation {conversation_id}: {topic}")
        
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation details"""
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return None
        
        conversation = self.conversations[conversation_id].copy()
        
        # Convert participant set to list for serialization
        conversation["participants"] = list(conversation["participants"])
        
        return conversation
    
    def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get messages in a conversation"""
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return []
        
        conversation = self.conversations[conversation_id]
        message_ids = conversation["message_ids"]
        
        # Apply pagination
        paginated_ids = message_ids[offset:offset + limit]
        
        # Get messages
        messages = []
        for message_id in paginated_ids:
            if message_id in self.messages:
                messages.append(self.messages[message_id])
        
        return messages
    
    def ask_question(
        self,
        from_model: Union[ModelType, str],
        to_model: Union[ModelType, str],
        content: str,
        message_type: Union[MessageType, str] = MessageType.QUESTION,
        priority: Union[MessagePriority, str] = MessagePriority.NORMAL,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        answer_timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Ask a question to another model
        
        Args:
            from_model: The asking model
            to_model: The model being asked
            content: Question content
            message_type: Type of message
            priority: Message priority
            conversation_id: Optional conversation ID
            context: Contextual information
            metadata: Additional metadata
            answer_timeout: Timeout for answer in seconds
            
        Returns:
            Message object
        """
        # Convert string types to enum values if needed
        if isinstance(from_model, str):
            from_model = ModelType(from_model)
        
        if isinstance(to_model, str):
            to_model = ModelType(to_model)
        
        if isinstance(message_type, str):
            message_type = MessageType(message_type)
        
        if isinstance(priority, str):
            priority = MessagePriority(priority)
        
        # Create new conversation if needed
        topic = f"Question from {from_model.value} to {to_model.value}"
        if not conversation_id:
            conversation_id = self.create_conversation(
                topic=topic,
                context=context,
                metadata=metadata
            )
        elif conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found, creating new one")
            conversation_id = self.create_conversation(
                topic=topic,
                context=context,
                metadata=metadata
            )
        
        # Update conversation participants
        conversation = self.conversations[conversation_id]
        conversation["participants"].add(from_model.value)
        conversation["participants"].add(to_model.value)
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Generate message ID
        message_id = str(uuid.uuid4())
        
        # Create message object
        message = {
            "id": message_id,
            "conversation_id": conversation_id,
            "from_model": from_model.value,
            "to_model": to_model.value,
            "content": content,
            "message_type": message_type.value,
            "priority": priority.value,
            "sent_at": datetime.now().isoformat(),
            "delivered_at": None,
            "read_at": None,
            "answered_at": None,
            "state": MessageState.PENDING.value,
            "answer_id": None,
            "context": context or {},
            "metadata": metadata or {}
        }
        
        # Store message
        self.messages[message_id] = message
        
        # Add to conversation
        conversation["message_ids"].append(message_id)
        
        # Add to message queue for processing
        # Priority queue items are tuples of (priority_value, timestamp, message_id)
        priority_value = {
            MessagePriority.LOW.value: 3,
            MessagePriority.NORMAL.value: 2,
            MessagePriority.HIGH.value: 1,
            MessagePriority.CRITICAL.value: 0
        }.get(priority.value, 2)
        
        self.message_queue.put((
            priority_value,
            time.time(),
            message_id
        ))
        
        # Update metrics
        self.metrics["messages_sent"] += 1
        self.metrics["questions_asked"] += 1
        self.metrics["model_interactions"][from_model.value] += 1
        
        logger.info(f"Question asked: {message_id} from {from_model.value} to {to_model.value}")
        
        return message
    
    def answer_question(
        self,
        question_id: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Answer a question
        
        Args:
            question_id: ID of the question to answer
            content: Answer content
            metadata: Additional metadata
            
        Returns:
            Answer message object
        """
        # Check if question exists
        if question_id not in self.messages:
            logger.warning(f"Question {question_id} not found")
            return {
                "error": "Question not found",
                "question_id": question_id
            }
        
        # Get question
        question = self.messages[question_id]
        
        # Check if already answered
        if question["state"] == MessageState.ANSWERED.value:
            logger.warning(f"Question {question_id} already answered")
            return {
                "error": "Question already answered",
                "question_id": question_id,
                "answer_id": question["answer_id"]
            }
        
        # Get conversation
        conversation_id = question["conversation_id"]
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return {
                "error": "Conversation not found",
                "conversation_id": conversation_id
            }
        
        conversation = self.conversations[conversation_id]
        
        # Generate answer ID
        answer_id = str(uuid.uuid4())
        
        # Create answer message
        answer = {
            "id": answer_id,
            "conversation_id": conversation_id,
            "from_model": question["to_model"],
            "to_model": question["from_model"],
            "content": content,
            "message_type": MessageType.ANSWER.value,
            "priority": question["priority"],
            "sent_at": datetime.now().isoformat(),
            "delivered_at": datetime.now().isoformat(),
            "read_at": None,
            "state": MessageState.DELIVERED.value,
            "question_id": question_id,
            "context": question["context"],
            "metadata": metadata or {}
        }
        
        # Store answer
        self.messages[answer_id] = answer
        
        # Update question
        question["state"] = MessageState.ANSWERED.value
        question["answer_id"] = answer_id
        question["answered_at"] = datetime.now().isoformat()
        
        # Add to conversation
        conversation["message_ids"].append(answer_id)
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Calculate response time
        sent_time = datetime.fromisoformat(question["sent_at"])
        answered_time = datetime.fromisoformat(question["answered_at"])
        response_time = (answered_time - sent_time).total_seconds()
        
        # Update metrics
        self.metrics["messages_sent"] += 1
        self.metrics["questions_answered"] += 1
        self.metrics["model_interactions"][question["to_model"]] += 1
        self.metrics["total_response_time"] += response_time
        if self.metrics["questions_answered"] > 0:
            self.metrics["average_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["questions_answered"]
            )
        
        logger.info(f"Question {question_id} answered with {answer_id}")
        
        return answer
    
    def send_message(
        self,
        from_model: Union[ModelType, str],
        to_model: Union[ModelType, str],
        content: str,
        message_type: Union[MessageType, str],
        priority: Union[MessagePriority, str] = MessagePriority.NORMAL,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Send a general message to another model
        
        Args:
            from_model: The sending model
            to_model: The receiving model
            content: Message content
            message_type: Type of message
            priority: Message priority
            conversation_id: Optional conversation ID
            context: Contextual information
            metadata: Additional metadata
            
        Returns:
            Message object
        """
        # Convert string types to enum values if needed
        if isinstance(from_model, str):
            from_model = ModelType(from_model)
        
        if isinstance(to_model, str):
            to_model = ModelType(to_model)
        
        if isinstance(message_type, str):
            message_type = MessageType(message_type)
        
        if isinstance(priority, str):
            priority = MessagePriority(priority)
        
        # Create new conversation if needed
        topic = f"Communication from {from_model.value} to {to_model.value}"
        if not conversation_id:
            conversation_id = self.create_conversation(
                topic=topic,
                context=context,
                metadata=metadata
            )
        elif conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found, creating new one")
            conversation_id = self.create_conversation(
                topic=topic,
                context=context,
                metadata=metadata
            )
        
        # Update conversation participants
        conversation = self.conversations[conversation_id]
        conversation["participants"].add(from_model.value)
        conversation["participants"].add(to_model.value)
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Generate message ID
        message_id = str(uuid.uuid4())
        
        # Create message object
        message = {
            "id": message_id,
            "conversation_id": conversation_id,
            "from_model": from_model.value,
            "to_model": to_model.value,
            "content": content,
            "message_type": message_type.value,
            "priority": priority.value,
            "sent_at": datetime.now().isoformat(),
            "delivered_at": None,
            "read_at": None,
            "state": MessageState.PENDING.value,
            "context": context or {},
            "metadata": metadata or {}
        }
        
        # Store message
        self.messages[message_id] = message
        
        # Add to conversation
        conversation["message_ids"].append(message_id)
        
        # Add to message queue for processing
        # Priority queue items are tuples of (priority_value, timestamp, message_id)
        priority_value = {
            MessagePriority.LOW.value: 3,
            MessagePriority.NORMAL.value: 2,
            MessagePriority.HIGH.value: 1,
            MessagePriority.CRITICAL.value: 0
        }.get(priority.value, 2)
        
        self.message_queue.put((
            priority_value,
            time.time(),
            message_id
        ))
        
        # Update metrics
        self.metrics["messages_sent"] += 1
        self.metrics["model_interactions"][from_model.value] += 1
        
        logger.info(f"Message sent: {message_id} from {from_model.value} to {to_model.value}")
        
        return message
    
    def _message_processor(self) -> None:
        """Worker thread to process messages in the queue"""
        while True:
            try:
                # Get next message from the queue
                _, _, message_id = self.message_queue.get()
                
                # Check if message exists
                if message_id not in self.messages:
                    logger.warning(f"Message {message_id} not found in queue")
                    self.message_queue.task_done()
                    continue
                
                # Get message
                message = self.messages[message_id]
                
                # Check message state
                if message["state"] != MessageState.PENDING.value:
                    logger.warning(f"Message {message_id} not in PENDING state")
                    self.message_queue.task_done()
                    continue
                
                # Get target model
                to_model = message["to_model"]
                
                # Deliver message
                self._deliver_message(message)
                
                # Mark as delivered
                message["delivered_at"] = datetime.now().isoformat()
                message["state"] = MessageState.DELIVERED.value
                
                # Update metrics
                self.metrics["messages_received"] += 1
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                time.sleep(1)  # Avoid busy loop in case of repeated errors
    
    def _deliver_message(self, message: Dict[str, Any]) -> None:
        """Deliver a message to its target model"""
        to_model_str = message["to_model"]
        
        try:
            # Convert to ModelType enum if possible
            to_model = ModelType(to_model_str)
            
            # Check if we have a registered handler for this model
            if to_model in self.message_handlers:
                # Call the handler
                self.message_handlers[to_model](message)
                logger.info(f"Message {message['id']} delivered to handler for {to_model_str}")
            else:
                # No handler, just log the message
                logger.info(f"Message {message['id']} delivered to {to_model_str} (no handler)")
                
                # Format using template if available
                message_type = message["message_type"]
                if message_type in self.message_templates:
                    template = self.message_templates[message_type]
                    formatted = template.format(
                        from_model=message["from_model"],
                        to_model=message["to_model"],
                        content=message["content"]
                    )
                    logger.info(f"Message content: {formatted}")
                else:
                    logger.info(f"Message content: {message['content']}")
        
        except ValueError:
            # Not a valid ModelType
            logger.warning(f"Unknown model type: {to_model_str}")
            logger.info(f"Message {message['id']} content: {message['content']}")
    
    def mark_message_read(self, message_id: str) -> bool:
        """Mark a message as read"""
        if message_id not in self.messages:
            logger.warning(f"Message {message_id} not found")
            return False
        
        message = self.messages[message_id]
        
        if message["state"] not in [MessageState.DELIVERED.value, MessageState.READ.value]:
            logger.warning(f"Message {message_id} not in DELIVERED or READ state")
            return False
        
        message["read_at"] = datetime.now().isoformat()
        message["state"] = MessageState.READ.value
        
        logger.info(f"Message {message_id} marked as read")
        
        return True
    
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a message by ID"""
        if message_id not in self.messages:
            logger.warning(f"Message {message_id} not found")
            return None
        
        return self.messages[message_id]
    
    def get_unread_messages(
        self,
        model: Union[ModelType, str],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get unread messages for a model
        
        Args:
            model: The model to get messages for
            limit: Maximum number of messages to return
            
        Returns:
            List of unread messages
        """
        # Convert string to enum if needed
        if isinstance(model, str):
            model = ModelType(model)
        
        model_value = model.value
        
        # Find unread messages
        unread = []
        for message_id, message in self.messages.items():
            if (
                message["to_model"] == model_value and
                message["state"] == MessageState.DELIVERED.value and
                not message["read_at"]
            ):
                unread.append(message)
        
        # Sort by priority and time
        unread.sort(
            key=lambda m: (
                {
                    MessagePriority.LOW.value: 3,
                    MessagePriority.NORMAL.value: 2,
                    MessagePriority.HIGH.value: 1,
                    MessagePriority.CRITICAL.value: 0
                }.get(m["priority"], 2),
                m["sent_at"]
            )
        )
        
        # Apply limit
        return unread[:limit]
    
    def get_unanswered_questions(
        self,
        model: Union[ModelType, str],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get unanswered questions for a model
        
        Args:
            model: The model to get questions for
            limit: Maximum number of questions to return
            
        Returns:
            List of unanswered questions
        """
        # Convert string to enum if needed
        if isinstance(model, str):
            model = ModelType(model)
        
        model_value = model.value
        
        # Find unanswered questions
        unanswered = []
        for message_id, message in self.messages.items():
            if (
                message["to_model"] == model_value and
                message["message_type"] == MessageType.QUESTION.value and
                message["state"] not in [MessageState.ANSWERED.value, MessageState.EXPIRED.value, MessageState.FAILED.value]
            ):
                unanswered.append(message)
        
        # Sort by priority and time
        unanswered.sort(
            key=lambda m: (
                {
                    MessagePriority.LOW.value: 3,
                    MessagePriority.NORMAL.value: 2,
                    MessagePriority.HIGH.value: 1,
                    MessagePriority.CRITICAL.value: 0
                }.get(m["priority"], 2),
                m["sent_at"]
            )
        )
        
        # Apply limit
        return unanswered[:limit]
    
    def close_conversation(self, conversation_id: str) -> bool:
        """Close a conversation"""
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return False
        
        conversation = self.conversations[conversation_id]
        conversation["status"] = "closed"
        conversation["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Conversation {conversation_id} closed")
        
        return True
    
    def archive_conversation(self, conversation_id: str) -> bool:
        """Archive a conversation"""
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return False
        
        conversation = self.conversations[conversation_id]
        conversation["status"] = "archived"
        conversation["updated_at"] = datetime.now().isoformat()
        
        logger.info(f"Conversation {conversation_id} archived")
        
        return True
    
    def search_conversations(
        self,
        query: str,
        models: List[Union[ModelType, str]] = None,
        message_types: List[Union[MessageType, str]] = None,
        status: str = "active",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversations
        
        Args:
            query: Search query
            models: Filter by models involved
            message_types: Filter by message types
            status: Filter by conversation status
            limit: Maximum number of conversations to return
            
        Returns:
            List of matching conversations
        """
        # Convert models to string values if needed
        model_values = None
        if models:
            model_values = []
            for model in models:
                if isinstance(model, ModelType):
                    model_values.append(model.value)
                else:
                    model_values.append(model)
        
        # Convert message types to string values if needed
        message_type_values = None
        if message_types:
            message_type_values = []
            for message_type in message_types:
                if isinstance(message_type, MessageType):
                    message_type_values.append(message_type.value)
                else:
                    message_type_values.append(message_type)
        
        # Find matching conversations
        results = []
        
        for conversation_id, conversation in self.conversations.items():
            # Filter by status
            if status and conversation["status"] != status:
                continue
            
            # Filter by models
            if model_values:
                participants = conversation["participants"]
                if not any(model in participants for model in model_values):
                    continue
            
            # Check if query matches conversation topic
            topic_match = query.lower() in conversation["topic"].lower()
            
            # Check if query matches any message content
            content_match = False
            message_type_match = True
            
            for message_id in conversation["message_ids"]:
                if message_id in self.messages:
                    message = self.messages[message_id]
                    
                    # Check content match
                    if query.lower() in message["content"].lower():
                        content_match = True
                    
                    # Check message type match if needed
                    if message_type_values and message["message_type"] not in message_type_values:
                        message_type_match = False
            
            # Include if any match criteria
            if (topic_match or content_match) and message_type_match:
                # Create a copy with participants as list
                conv_copy = conversation.copy()
                conv_copy["participants"] = list(conv_copy["participants"])
                results.append(conv_copy)
        
        # Sort by last updated time (newest first)
        results.sort(key=lambda c: c["updated_at"], reverse=True)
        
        # Apply limit
        return results[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset metrics counters"""
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "questions_asked": 0,
            "questions_answered": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "model_interactions": {
                ModelType.QWEN.value: 0,
                ModelType.OLYMPIC.value: 0,
                ModelType.HYBRID.value: 0,
                ModelType.SPECIALIZED.value: 0,
                ModelType.SYSTEM.value: 0
            }
        }
        
        logger.info("Communication metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get communication system status"""
        return {
            "active_conversations": len([c for c in self.conversations.values() if c["status"] == "active"]),
            "total_conversations": len(self.conversations),
            "total_messages": len(self.messages),
            "pending_messages": self.message_queue.qsize(),
            "metrics": {
                "messages_sent": self.metrics["messages_sent"],
                "questions_asked": self.metrics["questions_asked"],
                "questions_answered": self.metrics["questions_answered"],
                "average_response_time": self.metrics["average_response_time"]
            }
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed communication system status"""
        # Count message states
        message_states = {}
        for state in MessageState:
            message_states[state.value] = 0
        
        for message in self.messages.values():
            state = message["state"]
            message_states[state] = message_states.get(state, 0) + 1
        
        # Count message types
        message_types = {}
        for message_type in MessageType:
            message_types[message_type.value] = 0
        
        for message in self.messages.values():
            message_type = message["message_type"]
            message_types[message_type] = message_types.get(message_type, 0) + 1
        
        return {
            "active_conversations": len([c for c in self.conversations.values() if c["status"] == "active"]),
            "closed_conversations": len([c for c in self.conversations.values() if c["status"] == "closed"]),
            "archived_conversations": len([c for c in self.conversations.values() if c["status"] == "archived"]),
            "total_conversations": len(self.conversations),
            "total_messages": len(self.messages),
            "pending_messages": self.message_queue.qsize(),
            "message_states": message_states,
            "message_types": message_types,
            "metrics": self.metrics
        }

# Initialize communication system
communication_system = CommunicationSystem()