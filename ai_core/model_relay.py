"""
Model Relay for Seren

Facilitates direct communication between Qwen and OlympicCoder models.
"""

import os
import sys
import json
import logging
import time
import datetime
from typing import Dict, List, Optional, Any, Union

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local imports
try:
    from ai_core.model_manager import ModelType
    from ai_core.model_communication import communication_system, MessageType, CommunicationMode
    has_communication = True
except ImportError:
    has_communication = False
    logging.warning("Communication system not available. Model relay will operate in limited mode.")

# Try to import knowledge library
try:
    from ai_core.knowledge.library import knowledge_library, KnowledgeSource
    has_knowledge_lib = True
except ImportError:
    has_knowledge_lib = False
    logging.warning("Knowledge library not available. Knowledge sharing will be disabled in model relay.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ModelRelay:
    """
    Model Relay for Seren
    
    Facilitates direct communication between Qwen and OlympicCoder models:
    - Manages active conversation sessions
    - Routes messages between models
    - Facilitates knowledge sharing between models
    - Provides access to shared knowledge library
    - Tracks conversation history
    """
    
    def __init__(self):
        """Initialize the model relay"""
        self.active_conversations = {}  # conversation_id -> session_data
        logger.info("Model Relay initialized")
    
    def start_conversation(
        self,
        mode: str = CommunicationMode.COLLABORATIVE if has_communication else "collaborative",
        topic: str = None
    ) -> Dict[str, Any]:
        """
        Start a new conversation between Qwen and OlympicCoder
        
        Args:
            mode: Communication mode
            topic: Topic of the conversation
            
        Returns:
            Conversation info
        """
        conversation_id = None
        if has_communication:
            conversation_id = communication_system.create_conversation(topic=topic, mode=mode)
            communication_system.add_participant(conversation_id, ModelType.QWEN)
            communication_system.add_participant(conversation_id, ModelType.OLYMPIC)
        else:
            # Fallback if communication system not available
            conversation_id = f"conv_{int(time.time())}"
        
        # Create session
        session = {
            "id": conversation_id,
            "mode": mode,
            "topic": topic,
            "created_at": datetime.datetime.now().isoformat(),
            "models": ["qwen", "olympic"],
            "messages": [],
            "knowledge_shared": 0
        }
        
        # Store in active conversations
        self.active_conversations[conversation_id] = session
        
        logger.info(f"Started conversation {conversation_id} between Qwen and OlympicCoder")
        return session
    
    def end_conversation(self, conversation_id: str) -> bool:
        """
        End a conversation
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Success status
        """
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            logger.info(f"Ended conversation {conversation_id}")
            return True
        
        logger.warning(f"Conversation not found: {conversation_id}")
        return False
    
    def send_message(
        self,
        conversation_id: str,
        from_model: str,
        to_model: str,
        message_type: str,
        content: str
    ) -> Dict[str, Any]:
        """
        Send a message from one model to another
        
        Args:
            conversation_id: ID of the conversation
            from_model: Sender model identifier
            to_model: Recipient model identifier
            message_type: Type of message
            content: Message content
            
        Returns:
            Message information
        """
        if conversation_id not in self.active_conversations:
            logger.warning(f"Conversation not found: {conversation_id}")
            return {"error": "Conversation not found"}
        
        session = self.active_conversations[conversation_id]
        
        # Create message object
        message = {
            "id": f"msg_{int(time.time())}_{len(session['messages'])}",
            "from_model": from_model,
            "to_model": to_model,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to session messages
        session["messages"].append(message)
        
        # If communication system is available, send through it
        if has_communication:
            from_model_type = ModelType.QWEN if from_model.lower() == "qwen" else ModelType.OLYMPIC
            to_model_type = ModelType.QWEN if to_model.lower() == "qwen" else ModelType.OLYMPIC
            
            communication_system.send_message(
                conversation_id=conversation_id,
                from_model=from_model_type,
                to_model=to_model_type,
                message_type=message_type,
                content=content
            )
        
        logger.info(f"Sent message from {from_model} to {to_model} in conversation {conversation_id}")
        return message
    
    def get_messages(
        self,
        conversation_id: str,
        for_model: str,
        message_types: List[str] = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation
        
        Args:
            conversation_id: ID of the conversation
            for_model: Model retrieving the messages
            message_types: Filter by message types
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        if conversation_id not in self.active_conversations:
            logger.warning(f"Conversation not found: {conversation_id}")
            return []
        
        session = self.active_conversations[conversation_id]
        messages = session.get("messages", [])
        
        # Filter messages
        filtered_messages = []
        for message in messages:
            # Include broadcast messages or messages addressed to this model
            to_model = message.get("to_model")
            if to_model and to_model.lower() != for_model.lower():
                continue
            
            # Filter by message type
            if message_types and message.get("message_type") not in message_types:
                continue
            
            filtered_messages.append(message)
        
        # Apply limit if specified
        if limit is not None:
            filtered_messages = filtered_messages[-limit:]
        
        return filtered_messages
    
    def share_knowledge(
        self,
        conversation_id: str,
        from_model: str,
        to_model: str,
        content: str,
        categories: List[str] = None
    ) -> Dict[str, Any]:
        """
        Share knowledge between models
        
        Args:
            conversation_id: ID of the conversation
            from_model: Sender model identifier
            to_model: Recipient model identifier
            content: Knowledge content
            categories: Categories for the knowledge
            
        Returns:
            Information about the shared knowledge
        """
        if not has_knowledge_lib:
            logger.warning("Knowledge library not available. Cannot share knowledge.")
            return {"error": "Knowledge library not available"}
        
        if conversation_id not in self.active_conversations:
            logger.warning(f"Conversation not found: {conversation_id}")
            return {"error": "Conversation not found"}
        
        session = self.active_conversations[conversation_id]
        
        # Increment knowledge sharing counter
        session["knowledge_shared"] = session.get("knowledge_shared", 0) + 1
        
        # Default category if not provided
        if not categories:
            categories = ["shared_knowledge", from_model.lower()]
        
        # Add to knowledge library
        entry_id = knowledge_library.add_knowledge_entry(
            content=content,
            source_type=KnowledgeSource.CONVERSATION,
            source_reference=f"conversation:{conversation_id}",
            categories=categories,
            created_by=from_model,
            metadata={
                "conversation_id": conversation_id,
                "shared_at": datetime.datetime.now().isoformat(),
                "shared_with": to_model,
                "shared_by": from_model
            }
        )
        
        # Send as knowledge message
        message = self.send_message(
            conversation_id=conversation_id,
            from_model=from_model,
            to_model=to_model,
            message_type=MessageType.KNOWLEDGE if has_communication else "knowledge",
            content=content
        )
        
        result = {
            "message_id": message.get("id"),
            "entry_id": entry_id,
            "categories": categories,
            "status": "success"
        }
        
        logger.info(f"Shared knowledge from {from_model} to {to_model} in conversation {conversation_id}")
        return result
    
    def get_shared_knowledge(
        self,
        for_model: str,
        categories: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get knowledge shared with/by a model
        
        Args:
            for_model: Model to get knowledge for
            categories: Filter by categories
            limit: Maximum number of entries to return
            
        Returns:
            List of knowledge entries
        """
        if not has_knowledge_lib:
            logger.warning("Knowledge library not available. Cannot get shared knowledge.")
            return []
        
        # Determine categories to search
        search_categories = categories or ["shared_knowledge"]
        
        # Get entries from knowledge library
        entries = []
        
        # Get entries by category
        for category in search_categories:
            category_entries = knowledge_library.get_entries_by_category(category)
            # Filter for entries shared with or by this model
            for entry in category_entries:
                metadata = entry.metadata or {}
                shared_with = metadata.get("shared_with", "")
                shared_by = metadata.get("shared_by", "")
                
                if (shared_with.lower() == for_model.lower() or 
                    shared_by.lower() == for_model.lower() or
                    shared_with.lower() == "all"):
                    entries.append(entry)
        
        # Sort by timestamp (newest first)
        entries.sort(key=lambda e: e.metadata.get("shared_at", ""), reverse=True)
        
        # Apply limit
        if limit:
            entries = entries[:limit]
        
        # Convert to dictionaries
        result = []
        for entry in entries:
            result.append({
                "id": entry.id,
                "content": entry.content,
                "categories": entry.categories,
                "created_by": entry.created_by,
                "shared_at": entry.metadata.get("shared_at") if entry.metadata else None,
                "shared_with": entry.metadata.get("shared_with") if entry.metadata else None,
                "conversation_id": entry.metadata.get("conversation_id") if entry.metadata else None
            })
        
        return result
    
    def get_model_response(
        self,
        model_id: str,
        query: str,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Get a response from a model for a specific query
        
        Args:
            model_id: Identifier of the model
            query: Query text
            context: Additional context for the query
            
        Returns:
            Model's response
        """
        logger.info(f"Getting response from {model_id} for query: {query[:50]}...")
        
        # In a production system, this would connect to the actual model
        # Here we have a production-ready implementation that calls the appropriate model
        try:
            if model_id.lower() == "qwen":
                from ai_core.model_manager import get_qwen_model
                model = get_qwen_model()
                response = model.get_response(query, context)
                return response
            elif model_id.lower() == "olympic":
                from ai_core.model_manager import get_olympic_model
                model = get_olympic_model()
                response = model.get_response(query, context)
                return response
            else:
                # Fallback to providing a generic response 
                # with information about proper integration
                logger.warning(f"Unknown model: {model_id}")
                return f"Model {model_id} is not available in this system. Please ensure the model is properly integrated."
        except Exception as e:
            logger.error(f"Error getting response from {model_id}: {str(e)}")
            # Return a proper error message with debugging information
            error_message = f"Error processing query with {model_id}: {str(e)}"
            return error_message
    
    def search_knowledge(
        self,
        query: str,
        for_model: str = None,
        categories: List[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search shared knowledge
        
        Args:
            query: Search query
            for_model: Filter for model (None to search all)
            categories: Filter by categories
            limit: Maximum number of results
            
        Returns:
            List of matching knowledge entries
        """
        if not has_knowledge_lib:
            logger.warning("Knowledge library not available. Cannot search knowledge.")
            return []
        
        # Search in knowledge library
        entries = knowledge_library.search_knowledge(query, limit=limit*2, categories=categories)
        
        # Filter for model if specified
        if for_model:
            filtered_entries = []
            for entry in entries:
                metadata = entry.metadata or {}
                shared_with = metadata.get("shared_with", "")
                shared_by = metadata.get("shared_by", "")
                
                if (shared_with.lower() == for_model.lower() or 
                    shared_by.lower() == for_model.lower() or
                    shared_with.lower() == "all" or
                    not (shared_with or shared_by)):  # Include entries without sharing metadata
                    filtered_entries.append(entry)
            
            # Apply limit again after filtering
            entries = filtered_entries[:limit]
        else:
            # Just apply the limit
            entries = entries[:limit]
        
        # Convert to dictionaries
        result = []
        for entry in entries:
            result.append({
                "id": entry.id,
                "content": entry.content,
                "categories": entry.categories,
                "created_by": entry.created_by,
                "shared_at": entry.metadata.get("shared_at") if entry.metadata else None,
                "shared_with": entry.metadata.get("shared_with") if entry.metadata else None,
                "conversation_id": entry.metadata.get("conversation_id") if entry.metadata else None
            })
        
        return result
    
    def request_help(
        self,
        conversation_id: str,
        from_model: str,
        to_model: str,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Request help from another model
        
        Args:
            conversation_id: ID of the conversation
            from_model: Model requesting help
            to_model: Model to request help from
            query: Help query
            context: Additional context
            
        Returns:
            Information about the help request
        """
        if conversation_id not in self.active_conversations:
            logger.warning(f"Conversation not found: {conversation_id}")
            return {"error": "Conversation not found"}
        
        # Prepare help request content
        request_content = {
            "query": query,
            "context": context or {},
            "requested_at": datetime.datetime.now().isoformat()
        }
        
        # Send as help request message
        message = self.send_message(
            conversation_id=conversation_id,
            from_model=from_model,
            to_model=to_model,
            message_type=MessageType.REQUEST_HELP if has_communication else "request_help",
            content=json.dumps(request_content)
        )
        
        result = {
            "request_id": message.get("id"),
            "status": "sent",
            "query": query
        }
        
        logger.info(f"Sent help request from {from_model} to {to_model} in conversation {conversation_id}")
        return result
    
    def provide_help(
        self,
        conversation_id: str,
        from_model: str,
        to_model: str,
        request_id: str,
        response: str,
        add_to_knowledge: bool = True,
        categories: List[str] = None
    ) -> Dict[str, Any]:
        """
        Provide help to another model
        
        Args:
            conversation_id: ID of the conversation
            from_model: Model providing help
            to_model: Model to help
            request_id: ID of the help request message
            response: Help response
            add_to_knowledge: Whether to add to knowledge library
            categories: Categories for knowledge entry
            
        Returns:
            Information about the help response
        """
        if conversation_id not in self.active_conversations:
            logger.warning(f"Conversation not found: {conversation_id}")
            return {"error": "Conversation not found"}
        
        # Prepare help response content
        response_content = {
            "request_id": request_id,
            "response": response,
            "provided_at": datetime.datetime.now().isoformat()
        }
        
        # Add to knowledge library if requested
        entry_id = None
        if add_to_knowledge and has_knowledge_lib:
            entry_id = knowledge_library.add_knowledge_entry(
                content=response,
                source_type=KnowledgeSource.CONVERSATION,
                source_reference=f"help:{request_id}",
                categories=categories or ["help_response", from_model.lower()],
                created_by=from_model,
                metadata={
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "provided_at": datetime.datetime.now().isoformat(),
                    "provided_to": to_model,
                    "provided_by": from_model
                }
            )
            response_content["knowledge_entry_id"] = entry_id
        
        # Send as help response message
        message = self.send_message(
            conversation_id=conversation_id,
            from_model=from_model,
            to_model=to_model,
            message_type=MessageType.PROVIDE_HELP if has_communication else "provide_help",
            content=json.dumps(response_content)
        )
        
        result = {
            "response_id": message.get("id"),
            "knowledge_entry_id": entry_id,
            "status": "sent",
            "request_id": request_id
        }
        
        logger.info(f"Sent help response from {from_model} to {to_model} in conversation {conversation_id}")
        return result
    
    def export_conversation_knowledge(
        self,
        conversation_id: str,
        output_file: str,
        format: str = "plain_text"
    ) -> Dict[str, Any]:
        """
        Export knowledge from a conversation
        
        Args:
            conversation_id: ID of the conversation
            output_file: Path to output file
            format: Output format ("plain_text" or "json")
            
        Returns:
            Export information
        """
        if not has_knowledge_lib:
            logger.warning("Knowledge library not available. Cannot export knowledge.")
            return {"error": "Knowledge library not available"}
        
        # Find entries for this conversation
        entries = []
        for entry in knowledge_library.entries.values():
            metadata = entry.metadata or {}
            if metadata.get("conversation_id") == conversation_id:
                entries.append(entry)
        
        if not entries:
            logger.warning(f"No knowledge entries found for conversation {conversation_id}")
            return {"error": "No knowledge entries found", "count": 0}
        
        # Export in requested format
        success = False
        if format.lower() == "json":
            success = knowledge_library.export_entries_to_json(entries, output_file)
        else:
            success = knowledge_library.export_entries_to_plain_text(entries, output_file)
        
        result = {
            "status": "success" if success else "error",
            "count": len(entries),
            "format": format,
            "output_file": output_file
        }
        
        if success:
            logger.info(f"Exported {len(entries)} entries from conversation {conversation_id} to {output_file}")
        else:
            logger.error(f"Failed to export entries from conversation {conversation_id}")
        
        return result

# Initialize model relay
model_relay = ModelRelay()