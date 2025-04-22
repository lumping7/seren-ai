"""
Model Communication System for Seren

Enables secure communication and knowledge sharing between different AI models.
"""

import os
import sys
import json
import logging
import time
import uuid
import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# For security
try:
    from security.quantum_encryption import encrypt_message, decrypt_message
    has_quantum_security = True
    logger.info("Quantum encryption module loaded successfully")
except ImportError:
    logger.warning("Quantum encryption module not available. Using fallback encryption.")
    has_quantum_security = False
    
    # Fallback if security module not available
    def encrypt_message(message, recipient=None):
        logger.debug(f"Using fallback encryption for message to {recipient}")
        return message

    def decrypt_message(encrypted_message, recipient=None):
        logger.debug(f"Using fallback decryption for message to {recipient}")
        return encrypted_message

# Local imports
try:
    from ai_core.model_manager import ModelType
    has_model_manager = True
except ImportError:
    has_model_manager = False
    # Create a proper Enum class for ModelType if model_manager not available
    from enum import Enum, auto
    
    class ModelType(Enum):
        """Types of AI models"""
        QWEN = "qwen"
        OLYMPIC = "olympic"
        SYSTEM = "system"
        
        @property
        def value(self):
            """Get the value of the enum"""
            return self._value_
            
        def __str__(self):
            return self.value

# Import knowledge library with fallback
try:
    from ai_core.knowledge.library import knowledge_library, KnowledgeSource
    has_knowledge_lib = True
except ImportError:
    has_knowledge_lib = False
    logger.warning("Knowledge library not available. Knowledge sharing capabilities will be limited.")
    
    # Create minimal fallback implementations if needed
    class KnowledgeSource:
        USER = "user"
        SYSTEM = "system"
        MODEL = "model"
        EXTERNAL = "external"
        CONVERSATION = "conversation"  # Adding this for use in share_knowledge
    
    # Create a more robust knowledge library fallback
    class KnowledgeLibraryFallback:
        """Fallback implementation of knowledge library with required methods"""
        
        def __init__(self):
            self.next_id = 1
            self.entries = {}
            logger.warning("Using fallback knowledge library implementation")
            
        def add_knowledge_entry(self, content, source_type, source_reference="", categories=None, created_by="system", metadata=None):
            """Add a knowledge entry to the library"""
            entry_id = f"entry_{self.next_id}"
            self.next_id += 1
            self.entries[entry_id] = {
                "id": entry_id,
                "content": content,
                "source_type": source_type,
                "source_reference": source_reference,
                "categories": categories or [],
                "created_by": created_by,
                "created_at": datetime.datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            logger.info(f"Knowledge entry {entry_id} added to fallback library")
            return entry_id
            
        def retrieve_knowledge(self, query=None, categories=None, limit=5, source_type=None, created_by=None):
            """Retrieve knowledge entries based on criteria"""
            logger.warning("Using simplified knowledge retrieval in fallback mode")
            return []
            
        def extract_knowledge(self, text, source_type=None, source_reference=None, categories=None):
            """Extract knowledge from text"""
            logger.warning("Knowledge extraction not available in fallback mode")
            return []
            
    # Initialize fallback knowledge library
    knowledge_library = KnowledgeLibraryFallback()

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

class CommunicationMode:
    """Communication modes between models"""
    COLLABORATIVE = "collaborative"  # Models work together, sharing knowledge
    SPECIALIZED = "specialized"      # Models work on specific tasks based on their strengths
    COMPETITIVE = "competitive"      # Models compete to provide the best answers

class CommunicationSystem:
    """
    Communication System for Seren
    
    Enables secure communication and knowledge sharing between AI models:
    - Message passing between models
    - Collaborative problem solving
    - Knowledge sharing and integration
    - Secure communication with encryption
    - Conversation history tracking
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the communication system
        
        Args:
            base_dir: Base directory for storing conversation data
        """
        # Set base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create directories
        self.comm_dir = os.path.join(self.base_dir, "data", "communications")
        os.makedirs(self.comm_dir, exist_ok=True)
        
        # Conversations storage
        self.conversations = {}  # id -> conversation data
        
        # Load existing conversations
        self._load_conversations()
        
        logger.info("Communication System initialized")
    
    def _load_conversations(self) -> None:
        """Load existing conversations"""
        conv_file = os.path.join(self.comm_dir, "conversations.json")
        if os.path.exists(conv_file):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    self.conversations = json.load(f)
                logger.info(f"Loaded {len(self.conversations)} conversations")
            except Exception as e:
                logger.error(f"Error loading conversations: {str(e)}")
    
    def _save_conversations(self) -> None:
        """Save conversations"""
        try:
            conv_file = os.path.join(self.comm_dir, "conversations.json")
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving conversations: {str(e)}")
    
    def create_conversation(self, topic: str = None, mode: str = CommunicationMode.COLLABORATIVE) -> str:
        """
        Create a new conversation
        
        Args:
            topic: Topic of the conversation
            mode: Communication mode
            
        Returns:
            Conversation ID
        """
        # Generate ID
        conversation_id = str(uuid.uuid4())
        
        # Initialize conversation
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "topic": topic or "General Conversation",
            "mode": mode,
            "created_at": datetime.datetime.now().isoformat(),
            "participants": [],
            "messages": []
        }
        
        # Save conversations
        self._save_conversations()
        
        logger.info(f"Created conversation {conversation_id}")
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Conversation data or None if not found
        """
        return self.conversations.get(conversation_id)
    
    def add_participant(self, conversation_id: str, model_type: ModelType) -> bool:
        """
        Add a participant to a conversation
        
        Args:
            conversation_id: ID of the conversation
            model_type: Type of the participating model
            
        Returns:
            Success status
        """
        if conversation_id not in self.conversations:
            logger.error(f"Conversation not found: {conversation_id}")
            return False
        
        conversation = self.conversations[conversation_id]
        model_id = model_type.value
        
        if model_id not in conversation["participants"]:
            conversation["participants"].append(model_id)
            self._save_conversations()
            logger.info(f"Added {model_id} to conversation {conversation_id}")
        
        return True
    
    def send_message(
        self,
        conversation_id: str,
        from_model: ModelType,
        to_model: Optional[ModelType] = None,
        message_type: str = MessageType.INFO,
        content: str = "",
        encrypted: bool = True
    ) -> bool:
        """
        Send a message in a conversation
        
        Args:
            conversation_id: ID of the conversation
            from_model: Sender model
            to_model: Recipient model (None for broadcast)
            message_type: Type of message
            content: Message content
            encrypted: Whether to encrypt the message
            
        Returns:
            Success status
        """
        if conversation_id not in self.conversations:
            logger.error(f"Conversation not found: {conversation_id}")
            return False
        
        conversation = self.conversations[conversation_id]
        
        # Add sender to participants if not already
        from_id = from_model.value
        if from_id not in conversation["participants"]:
            conversation["participants"].append(from_id)
        
        # Add recipient to participants if specified and not already
        to_id = to_model.value if to_model else None
        if to_id and to_id not in conversation["participants"]:
            conversation["participants"].append(to_id)
        
        # Encrypt content if needed
        if encrypted:
            encrypted_content = encrypt_message(content, to_id)
        else:
            encrypted_content = content
        
        # Create message
        message = {
            "id": str(uuid.uuid4()),
            "from_model": from_id,
            "to_model": to_id,  # None for broadcast
            "message_type": message_type,
            "content": encrypted_content,
            "encrypted": encrypted,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to conversation
        conversation["messages"].append(message)
        
        # Save conversations
        self._save_conversations()
        
        logger.info(f"Sent message from {from_id} to {to_id or 'all'} in conversation {conversation_id}")
        return True
    
    def get_messages(
        self,
        conversation_id: str,
        for_model: ModelType,
        decrypt: bool = True,
        message_types: List[str] = None,
        from_model: ModelType = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation
        
        Args:
            conversation_id: ID of the conversation
            for_model: Model retrieving the messages
            decrypt: Whether to decrypt encrypted messages
            message_types: Filter by message types
            from_model: Filter by sender
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        if conversation_id not in self.conversations:
            logger.error(f"Conversation not found: {conversation_id}")
            return []
        
        conversation = self.conversations[conversation_id]
        messages = conversation["messages"]
        
        # Filter messages
        filtered_messages = []
        for message in messages:
            # Filter by recipient
            to_model = message.get("to_model")
            if to_model and to_model != for_model.value:
                continue
            
            # Filter by message type
            if message_types and message.get("message_type") not in message_types:
                continue
            
            # Filter by sender
            if from_model and message.get("from_model") != from_model.value:
                continue
            
            # Clone the message to avoid modifying the original
            filtered_message = message.copy()
            
            # Decrypt content if needed
            if decrypt and message.get("encrypted", False):
                content = message.get("content", "")
                filtered_message["content"] = decrypt_message(content, for_model.value)
            
            filtered_messages.append(filtered_message)
        
        # Apply limit if specified
        if limit is not None:
            filtered_messages = filtered_messages[-limit:]
        
        return filtered_messages
    
    def share_knowledge(
        self,
        conversation_id: str,
        from_model: ModelType,
        to_model: Optional[ModelType] = None,
        content: str = "",
        source_reference: str = "",
        categories: List[str] = None,
        add_to_library: bool = True
    ) -> bool:
        """
        Share knowledge between models
        
        Args:
            conversation_id: ID of the conversation
            from_model: Model sharing the knowledge
            to_model: Model to share with (None for all)
            content: Knowledge content
            source_reference: Reference to the source
            categories: Knowledge categories
            add_to_library: Whether to add to the knowledge library
            
        Returns:
            Success status
        """
        # Verify knowledge library is correctly initialized
        if not hasattr(knowledge_library, "add_knowledge_entry"):
            logger.error("Knowledge library is not properly initialized.")
            return False
        
        # Add to knowledge library if requested
        entry_id = None
        if add_to_library:
            entry_id = knowledge_library.add_knowledge_entry(
                content=content,
                source_type=KnowledgeSource.CONVERSATION,
                source_reference=source_reference or f"conversation:{conversation_id}",
                categories=categories or ["shared_knowledge"],
                created_by=from_model.value,
                metadata={
                    "conversation_id": conversation_id,
                    "shared_at": datetime.datetime.now().isoformat(),
                    "shared_with": to_model.value if to_model else "all"
                }
            )
        
        # Prepare knowledge message
        message_content = {
            "knowledge_content": content,
            "source_reference": source_reference,
            "categories": categories or ["shared_knowledge"],
            "entry_id": entry_id,
            "shared_at": datetime.datetime.now().isoformat()
        }
        
        # Send as knowledge message
        return self.send_message(
            conversation_id=conversation_id,
            from_model=from_model,
            to_model=to_model,
            message_type=MessageType.KNOWLEDGE,
            content=json.dumps(message_content),
            encrypted=True
        )
    
    def request_help(
        self,
        conversation_id: str,
        from_model: ModelType,
        to_model: ModelType,
        query: str,
        context: Dict[str, Any] = None
    ) -> bool:
        """
        Request help from another model
        
        Args:
            conversation_id: ID of the conversation
            from_model: Model requesting help
            to_model: Model to request help from
            query: Help query
            context: Additional context
            
        Returns:
            Success status
        """
        # Prepare help request
        request_content = {
            "query": query,
            "context": context or {},
            "requested_at": datetime.datetime.now().isoformat()
        }
        
        # Send as help request
        return self.send_message(
            conversation_id=conversation_id,
            from_model=from_model,
            to_model=to_model,
            message_type=MessageType.REQUEST_HELP,
            content=json.dumps(request_content),
            encrypted=True
        )
    
    def provide_help(
        self,
        conversation_id: str,
        from_model: ModelType,
        to_model: ModelType,
        request_id: str,
        response: str,
        knowledge_refs: List[str] = None
    ) -> bool:
        """
        Provide help to another model
        
        Args:
            conversation_id: ID of the conversation
            from_model: Model providing help
            to_model: Model to help
            request_id: ID of the help request message
            response: Help response
            knowledge_refs: References to knowledge entries
            
        Returns:
            Success status
        """
        # Prepare help response
        response_content = {
            "request_id": request_id,
            "response": response,
            "knowledge_refs": knowledge_refs or [],
            "provided_at": datetime.datetime.now().isoformat()
        }
        
        # Send as help response
        return self.send_message(
            conversation_id=conversation_id,
            from_model=from_model,
            to_model=to_model,
            message_type=MessageType.PROVIDE_HELP,
            content=json.dumps(response_content),
            encrypted=True
        )
    
    def collaborative_response(
        self,
        query: str,
        models: List[ModelType],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a collaborative response from multiple models
        
        Args:
            query: User query
            models: List of models to collaborate
            context: Additional context
            
        Returns:
            Collaborative response
        """
        if len(models) < 2:
            logger.warning("Collaborative response requires at least 2 models")
            return {"error": "Not enough models for collaboration"}
        
        # Create conversation
        conversation_id = self.create_conversation(
            topic=f"Collaborative: {query[:50]}...",
            mode=CommunicationMode.COLLABORATIVE
        )
        
        # Add participants
        for model in models:
            self.add_participant(conversation_id, model)
        
        # Prepare response
        response = {
            "conversation_id": conversation_id,
            "query": query,
            "mode": CommunicationMode.COLLABORATIVE,
            "models": [model.value for model in models],
            "responses": {},
            "combined_response": "",
            "knowledge_used": []
        }
        
        # Process query with all models and enable collaboration
        primary_model = models[0]
        secondary_models = models[1:]
        
        # 1. Get initial responses from each model
        logger.info(f"Getting initial responses from {len(models)} models")
        
        for model in models:
            # Send the query message to each model
            self.send_message(
                conversation_id=conversation_id,
                from_model=ModelType.SYSTEM,
                to_model=model,
                message_type=MessageType.QUERY,
                content=query,
                encrypted=True
            )
            
            # In a real system, this would wait for the model to process and respond
            # Here we'll directly fetch model outputs from the model relay system
            try:
                from ai_core.model_relay import model_relay
                model_response = model_relay.get_model_response(model.value, query, context)
                
                # Store the response
                response["responses"][model.value] = {
                    "content": model_response,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting response from model {model.value}: {str(e)}")
                response["responses"][model.value] = {
                    "content": f"Error: {str(e)}",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": True
                }
        
        # 2. Share knowledge between models
        logger.info("Sharing knowledge between models")
        shared_knowledge = []
        
        if hasattr(knowledge_library, "extract_knowledge"):
            # For each model response, extract knowledge and share with other models
            for model in models:
                model_value = model.value
                if model_value in response["responses"] and not response["responses"][model_value].get("error", False):
                    # Extract knowledge from response
                    knowledge_entries = knowledge_library.extract_knowledge(
                        content=response["responses"][model_value]["content"],
                        source_reference=f"model_response:{model_value}",
                        categories=["collaborative_response"]
                    )
                    
                    # Share each knowledge entry with other models
                    for entry in knowledge_entries:
                        shared_knowledge.append(entry)
                        # Share with all other models
                        for other_model in [m for m in models if m != model]:
                            self.share_knowledge(
                                conversation_id=conversation_id,
                                from_model=model,
                                to_model=other_model,
                                content=entry["content"],
                                source_reference=entry["source_reference"],
                                categories=entry["categories"]
                            )
        
        # Record the shared knowledge
        response["knowledge_used"] = shared_knowledge
        
        # 3. Have models collaborate to refine the response
        logger.info("Collaborating to refine response")
        refinement_prompt = f"""
        Query: {query}
        
        Here are the initial responses from all models:
        {json.dumps(response["responses"], indent=2)}
        
        Please provide a refined response that incorporates the best elements
        of all initial responses and the shared knowledge.
        """
        
        # Ask primary model to create a refined response
        try:
            from ai_core.model_relay import model_relay
            refined_response = model_relay.get_model_response(
                primary_model.value, 
                refinement_prompt,
                {
                    "conversation_id": conversation_id,
                    "knowledge": shared_knowledge,
                    "collaboration_type": "refinement"
                }
            )
            
            # Store the refined response
            response["combined_response"] = refined_response
        except Exception as e:
            logger.error(f"Error getting refined response: {str(e)}")
            # Fall back to concatenating responses if refinement fails
            response["combined_response"] = "\n\n".join([
                f"Response from {model_id}:\n{data['content']}"
                for model_id, data in response["responses"].items()
                if not data.get("error", False)
            ])
        
        # Mark processing as complete
        response["completed_at"] = datetime.datetime.now().isoformat()
        response["status"] = "success"
        
        return response
    
    def export_shared_knowledge(
        self,
        model_id: str,
        output_file: str,
        format: str = "plain_text"
    ) -> bool:
        """
        Export knowledge shared by a specific model
        
        Args:
            model_id: ID of the model
            output_file: Path to output file
            format: Output format ("plain_text" or "json")
            
        Returns:
            Success status
        """
        # Verify knowledge library is correctly initialized
        if not hasattr(knowledge_library, "get_entries_by_model"):
            logger.error("Knowledge library is not properly initialized for export.")
            return False
        
        # Get entries created by the model
        entries = knowledge_library.get_entries_by_model(model_id)
        
        if not entries:
            logger.warning(f"No knowledge entries found for model {model_id}")
            return False
        
        # Export in requested format
        if format.lower() == "json":
            return knowledge_library.export_entries_to_json(entries, output_file)
        else:
            return knowledge_library.export_entries_to_plain_text(entries, output_file)
    
    def search_shared_knowledge(
        self,
        query: str,
        model_id: str = None,
        categories: List[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge shared by models
        
        Args:
            query: Search query
            model_id: Filter by model that created the knowledge
            categories: Filter by categories
            limit: Maximum number of results
            
        Returns:
            List of matching knowledge entries
        """
        # Verify knowledge library is correctly initialized
        if not hasattr(knowledge_library, "search_knowledge"):
            logger.error("Knowledge library is not properly initialized for search.")
            return []
        
        # Search knowledge
        entries = knowledge_library.search_knowledge(query, limit=limit, categories=categories)
        
        # Filter by model if specified
        if model_id:
            entries = [entry for entry in entries if entry.created_by == model_id]
        
        # Convert to dicts for return
        results = []
        for entry in entries:
            results.append({
                "id": entry.id,
                "content": entry.content,
                "source_type": entry.source_type,
                "source_reference": entry.source_reference,
                "categories": entry.categories,
                "created_by": entry.created_by,
                "metadata": entry.metadata
            })
        
        return results

# Initialize communication system
communication_system = CommunicationSystem()