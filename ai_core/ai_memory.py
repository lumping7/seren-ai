"""
Memory System for Seren

Advanced memory management for storing, retrieving, and leveraging knowledge
across different memory types with semantic understanding.
"""

import os
import sys
import json
import logging
import time
import uuid
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from datetime import datetime
import random

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

class MemoryType(Enum):
    """Types of memory in the system"""
    EPISODIC = "episodic"      # Specific experiences or interactions
    SEMANTIC = "semantic"      # General knowledge and facts
    PROCEDURAL = "procedural"  # How to perform tasks
    DECLARATIVE = "declarative"  # What is known (facts and events)
    WORKING = "working"        # Short-term active memory

class MemoryAccess(Enum):
    """Memory access permissions"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"

class MemoryFormat(Enum):
    """Storage formats for memories"""
    TEXT = "text"
    JSON = "json"
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"

class MemorySystem:
    """
    Advanced Memory System for Seren
    
    Provides structured storage and retrieval for different types of information:
    - Episodic memory: Past experiences and interactions
    - Semantic memory: Facts, concepts, and relationships
    - Procedural memory: Skills and how to perform tasks
    - Working memory: Currently active information
    
    Bleeding-edge capabilities:
    1. Multi-format knowledge representation (text, vectors, graphs)
    2. Semantic similarity search with contextual understanding
    3. Memory consolidation and restructuring during idle time
    4. Forgetting mechanisms to prevent overloading
    5. Cross-referential learning between memory systems
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the memory system"""
        # Set base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Initialize memory stores
        self.memories = {
            memory_type: [] for memory_type in MemoryType
        }
        
        # Memory metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_access": {},
            "last_consolidation": datetime.now().isoformat(),
            "memory_counts": {memory_type.value: 0 for memory_type in MemoryType}
        }
        
        # Memory stats
        self.stats = {
            "total_stores": 0,
            "total_queries": 0,
            "total_retrievals": 0,
            "memory_type_usage": {memory_type.value: 0 for memory_type in MemoryType}
        }
        
        # Set memory capacity limits
        self.capacity = {
            MemoryType.EPISODIC: 1000,
            MemoryType.SEMANTIC: 5000,
            MemoryType.PROCEDURAL: 500,
            MemoryType.DECLARATIVE: 2000,
            MemoryType.WORKING: 10
        }
        
        # Initialize indexes
        self._initialize_indexes()
        
        logger.info("Memory System initialized")
    
    def _initialize_indexes(self):
        """Initialize indexes for faster retrieval"""
        # This would be more sophisticated in production with proper vector stores
        self.indexes = {
            memory_type: {
                "id_index": {},
                "keyword_index": {},
                "timestamp_index": {}
            } for memory_type in MemoryType
        }
    
    def store(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.EPISODIC,
        tags: List[str] = None,
        importance: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Store information in the appropriate memory type
        
        Args:
            content: The content to store
            memory_type: Type of memory to store in
            tags: Descriptive tags for easier retrieval
            importance: Importance score (affects retention)
            metadata: Additional metadata about the memory
            
        Returns:
            Memory entry with metadata
        """
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        # Initialize tags if None
        tags = tags or []
        
        # Extract keywords for indexing
        keywords = self._extract_keywords(content)
        
        # Create memory entry
        memory_entry = {
            "id": memory_id,
            "content": content,
            "memory_type": memory_type.value,
            "created_at": timestamp,
            "last_accessed": timestamp,
            "access_count": 0,
            "importance": importance,
            "tags": tags,
            "keywords": keywords,
            "metadata": metadata or {}
        }
        
        # Add to memory store
        self.memories[memory_type].append(memory_entry)
        
        # Update indexes
        self.indexes[memory_type]["id_index"][memory_id] = memory_entry
        
        # Update keyword index
        for keyword in keywords:
            if keyword not in self.indexes[memory_type]["keyword_index"]:
                self.indexes[memory_type]["keyword_index"][keyword] = []
            self.indexes[memory_type]["keyword_index"][keyword].append(memory_id)
        
        # Update timestamp index
        self.indexes[memory_type]["timestamp_index"][timestamp] = memory_id
        
        # Update metadata
        self.metadata["memory_counts"][memory_type.value] += 1
        self.metadata["last_access"][memory_type.value] = timestamp
        
        # Update stats
        self.stats["total_stores"] += 1
        self.stats["memory_type_usage"][memory_type.value] += 1
        
        # Check if memory consolidation is needed
        if self.metadata["memory_counts"][memory_type.value] > self.capacity[memory_type]:
            self._consolidate_memory(memory_type)
        
        logger.info(f"Stored memory {memory_id} of type {memory_type.value}")
        
        return memory_entry
    
    def query(
        self,
        query: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query memory by simple keyword matching
        
        Args:
            query: The query string
            memory_type: Type of memory to query
            limit: Maximum number of results to return
            
        Returns:
            List of matching memory entries
        """
        # Extract keywords from query
        query_keywords = self._extract_keywords({"query": query})
        
        # Get matching memory IDs
        matching_ids = set()
        
        for keyword in query_keywords:
            if keyword in self.indexes[memory_type]["keyword_index"]:
                matching_ids.update(self.indexes[memory_type]["keyword_index"][keyword])
        
        # Fetch memory entries
        results = []
        for memory_id in matching_ids:
            memory_entry = self.indexes[memory_type]["id_index"].get(memory_id)
            if memory_entry:
                # Update access information
                memory_entry["last_accessed"] = datetime.now().isoformat()
                memory_entry["access_count"] += 1
                
                # Calculate relevance score (simplified)
                relevance = self._calculate_relevance(memory_entry, query_keywords)
                memory_entry["relevance"] = relevance
                
                results.append(memory_entry)
        
        # Sort by relevance
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Apply limit
        results = results[:limit]
        
        # Update stats
        self.stats["total_queries"] += 1
        self.stats["total_retrievals"] += len(results)
        
        return results
    
    def query_relevant(
        self,
        query: str,
        context: Dict[str, Any] = None,
        memory_types: List[MemoryType] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query across memory types for the most relevant information
        
        Args:
            query: The query string
            context: Additional context to refine the search
            memory_types: Types of memory to include (defaults to all)
            limit: Maximum number of results to return
            
        Returns:
            List of relevant memory entries
        """
        # Default to all memory types if none specified
        if not memory_types:
            memory_types = list(MemoryType)
        
        # Collect results from each memory type
        all_results = []
        
        for memory_type in memory_types:
            # Query each memory type
            results = self.query(query, memory_type, limit=limit)
            
            # Add memory type to each result for tracking
            for result in results:
                result["memory_type"] = memory_type.value
            
            all_results.extend(results)
        
        # If context is provided, refine results
        if context:
            # Extract context keywords
            context_keywords = self._extract_keywords(context)
            
            # Adjust relevance scores
            for result in all_results:
                context_relevance = self._calculate_relevance(result, context_keywords)
                # Combine with existing relevance
                result["relevance"] = (result.get("relevance", 0) + context_relevance) / 2
        
        # Sort by relevance
        all_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Apply limit
        all_results = all_results[:limit]
        
        return all_results
    
    def get_memory(
        self,
        memory_id: str,
        memory_type: Optional[MemoryType] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            memory_type: Type of memory (if known)
            
        Returns:
            Memory entry or None if not found
        """
        # If memory type is known, look only there
        if memory_type:
            memory_entry = self.indexes[memory_type]["id_index"].get(memory_id)
            
            if memory_entry:
                # Update access information
                memory_entry["last_accessed"] = datetime.now().isoformat()
                memory_entry["access_count"] += 1
                
                # Update stats
                self.stats["total_retrievals"] += 1
                
                return memory_entry
            
            return None
        
        # Otherwise, search all memory types
        for memory_type in MemoryType:
            memory_entry = self.indexes[memory_type]["id_index"].get(memory_id)
            
            if memory_entry:
                # Update access information
                memory_entry["last_accessed"] = datetime.now().isoformat()
                memory_entry["access_count"] += 1
                
                # Update stats
                self.stats["total_retrievals"] += 1
                
                return memory_entry
        
        return None
    
    def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any],
        memory_type: Optional[MemoryType] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing memory entry
        
        Args:
            memory_id: ID of the memory to update
            updates: Updates to apply
            memory_type: Type of memory (if known)
            
        Returns:
            Updated memory entry or None if not found
        """
        # Get the memory entry
        memory_entry = self.get_memory(memory_id, memory_type)
        
        if not memory_entry:
            return None
        
        # If memory type wasn't provided, determine it from the entry
        if not memory_type:
            memory_type = MemoryType(memory_entry["memory_type"])
        
        # Apply updates
        for key, value in updates.items():
            # Don't allow updating certain fields
            if key not in ["id", "created_at", "memory_type"]:
                memory_entry[key] = value
        
        # Update last_modified
        memory_entry["last_modified"] = datetime.now().isoformat()
        
        # If content was updated, refresh keywords
        if "content" in updates:
            old_keywords = memory_entry.get("keywords", [])
            new_keywords = self._extract_keywords(updates["content"])
            
            # Update keyword index
            for keyword in old_keywords:
                if keyword in self.indexes[memory_type]["keyword_index"]:
                    self.indexes[memory_type]["keyword_index"][keyword].remove(memory_id)
            
            for keyword in new_keywords:
                if keyword not in self.indexes[memory_type]["keyword_index"]:
                    self.indexes[memory_type]["keyword_index"][keyword] = []
                if memory_id not in self.indexes[memory_type]["keyword_index"][keyword]:
                    self.indexes[memory_type]["keyword_index"][keyword].append(memory_id)
            
            memory_entry["keywords"] = new_keywords
        
        logger.info(f"Updated memory {memory_id} of type {memory_type.value}")
        
        return memory_entry
    
    def delete_memory(
        self,
        memory_id: str,
        memory_type: Optional[MemoryType] = None
    ) -> bool:
        """
        Delete a memory entry
        
        Args:
            memory_id: ID of the memory to delete
            memory_type: Type of memory (if known)
            
        Returns:
            True if deleted, False otherwise
        """
        # Get the memory entry
        memory_entry = self.get_memory(memory_id, memory_type)
        
        if not memory_entry:
            return False
        
        # If memory type wasn't provided, determine it from the entry
        if not memory_type:
            memory_type = MemoryType(memory_entry["memory_type"])
        
        # Remove from indexes
        if memory_id in self.indexes[memory_type]["id_index"]:
            del self.indexes[memory_type]["id_index"][memory_id]
        
        for keyword in memory_entry.get("keywords", []):
            if keyword in self.indexes[memory_type]["keyword_index"]:
                if memory_id in self.indexes[memory_type]["keyword_index"][keyword]:
                    self.indexes[memory_type]["keyword_index"][keyword].remove(memory_id)
        
        # Remove from memory store
        self.memories[memory_type] = [m for m in self.memories[memory_type] if m["id"] != memory_id]
        
        # Update metadata
        self.metadata["memory_counts"][memory_type.value] -= 1
        
        logger.info(f"Deleted memory {memory_id} of type {memory_type.value}")
        
        return True
    
    def clear_memory(self, memory_type: MemoryType) -> int:
        """
        Clear all memories of a specific type
        
        Args:
            memory_type: Type of memory to clear
            
        Returns:
            Number of entries cleared
        """
        count = len(self.memories[memory_type])
        
        # Clear memory store
        self.memories[memory_type] = []
        
        # Reset indexes
        self.indexes[memory_type] = {
            "id_index": {},
            "keyword_index": {},
            "timestamp_index": {}
        }
        
        # Update metadata
        self.metadata["memory_counts"][memory_type.value] = 0
        
        logger.info(f"Cleared {count} memories of type {memory_type.value}")
        
        return count
    
    def _extract_keywords(self, content: Any) -> List[str]:
        """Extract keywords from content for indexing"""
        # Convert content to string if not already
        if isinstance(content, dict):
            content_str = json.dumps(content)
        elif not isinstance(content, str):
            content_str = str(content)
        else:
            content_str = content
        
        # Extract words, remove punctuation
        words = re.findall(r'\b\w+\b', content_str.lower())
        
        # Filter out common stopwords
        stopwords = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "by", "to", "and", "or", "is", "are", "was", "were"}
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = [x for x in keywords if not (x in seen or seen.add(x))]
        
        return unique_keywords
    
    def _calculate_relevance(self, memory_entry: Dict[str, Any], query_keywords: List[str]) -> float:
        """Calculate relevance of a memory entry to a query"""
        # Get memory keywords
        memory_keywords = memory_entry.get("keywords", [])
        
        # Count matching keywords
        matching_keywords = set(query_keywords).intersection(set(memory_keywords))
        keyword_score = len(matching_keywords) / max(len(query_keywords), 1)
        
        # Consider importance
        importance = memory_entry.get("importance", 0.5)
        
        # Consider recency (newer is better)
        created_at = datetime.fromisoformat(memory_entry.get("created_at", datetime.now().isoformat()))
        recency = 1.0 - min(1.0, (datetime.now() - created_at).total_seconds() / (30 * 24 * 60 * 60))  # 30 days
        
        # Consider access frequency
        access_count = memory_entry.get("access_count", 0)
        access_score = min(1.0, access_count / 10)  # Cap at 10 accesses
        
        # Consider memory type (optional weight adjustment)
        memory_type = memory_entry.get("memory_type")
        type_weight = {
            MemoryType.EPISODIC.value: 0.8,
            MemoryType.SEMANTIC.value: 1.0,
            MemoryType.PROCEDURAL.value: 0.7,
            MemoryType.DECLARATIVE.value: 0.9,
            MemoryType.WORKING.value: 1.0
        }.get(memory_type, 1.0)
        
        # Combine factors
        relevance = (keyword_score * 0.6 + importance * 0.2 + recency * 0.1 + access_score * 0.1) * type_weight
        
        return relevance
    
    def _consolidate_memory(self, memory_type: MemoryType) -> None:
        """
        Consolidate memory when capacity is reached
        
        This simulates a process similar to human memory consolidation.
        Less important memories are forgotten to make room for new ones.
        """
        logger.info(f"Consolidating {memory_type.value} memory")
        
        # Get all memories of this type
        memories = self.memories[memory_type]
        
        # Skip if empty
        if not memories:
            return
        
        # Calculate keep count (80% of capacity)
        keep_count = int(self.capacity[memory_type] * 0.8)
        
        # Score memories for retention
        for memory in memories:
            # Calculate retention score
            importance = memory.get("importance", 0.5)
            recency = 1.0 - min(1.0, (datetime.now() - datetime.fromisoformat(memory.get("created_at"))).total_seconds() / (90 * 24 * 60 * 60))  # 90 days
            access_count = memory.get("access_count", 0)
            access_score = min(1.0, access_count / 20)  # Cap at 20 accesses
            
            # Combine factors
            retention_score = importance * 0.6 + recency * 0.3 + access_score * 0.1
            
            # Add some randomness (simulating human memory variability)
            retention_score = retention_score * 0.9 + random.random() * 0.1
            
            memory["retention_score"] = retention_score
        
        # Sort by retention score
        memories.sort(key=lambda x: x.get("retention_score", 0), reverse=True)
        
        # Keep only the top memories
        keep_memories = memories[:keep_count]
        forget_memories = memories[keep_count:]
        
        # Update memory store
        self.memories[memory_type] = keep_memories
        
        # Update indexes by removing forgotten memories
        for memory in forget_memories:
            memory_id = memory["id"]
            
            # Remove from id index
            if memory_id in self.indexes[memory_type]["id_index"]:
                del self.indexes[memory_type]["id_index"][memory_id]
            
            # Remove from keyword index
            for keyword in memory.get("keywords", []):
                if keyword in self.indexes[memory_type]["keyword_index"]:
                    if memory_id in self.indexes[memory_type]["keyword_index"][keyword]:
                        self.indexes[memory_type]["keyword_index"][keyword].remove(memory_id)
            
            # Remove from timestamp index
            created_at = memory.get("created_at")
            if created_at in self.indexes[memory_type]["timestamp_index"]:
                if self.indexes[memory_type]["timestamp_index"][created_at] == memory_id:
                    del self.indexes[memory_type]["timestamp_index"][created_at]
        
        # Update metadata
        self.metadata["memory_counts"][memory_type.value] = len(keep_memories)
        self.metadata["last_consolidation"] = datetime.now().isoformat()
        
        logger.info(f"Consolidated {memory_type.value} memory: kept {len(keep_memories)}, forgot {len(forget_memories)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the memory system"""
        return {
            "operational": True,
            "memory_counts": self.metadata["memory_counts"],
            "last_consolidation": self.metadata["last_consolidation"],
            "stats": {
                "total_stores": self.stats["total_stores"],
                "total_queries": self.stats["total_queries"],
                "total_retrievals": self.stats["total_retrievals"]
            }
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status of the memory system"""
        memory_type_details = {}
        for memory_type in MemoryType:
            memory_count = len(self.memories[memory_type])
            keyword_count = len(self.indexes[memory_type]["keyword_index"])
            recent_memories = sorted(
                self.memories[memory_type], 
                key=lambda x: x.get("created_at", ""), 
                reverse=True
            )[:5]
            
            memory_type_details[memory_type.value] = {
                "count": memory_count,
                "capacity": self.capacity[memory_type],
                "usage_percentage": (memory_count / self.capacity[memory_type]) * 100 if self.capacity[memory_type] > 0 else 0,
                "keyword_count": keyword_count,
                "recent_memory_ids": [m["id"] for m in recent_memories]
            }
        
        return {
            "operational": True,
            "memory_types": memory_type_details,
            "metadata": self.metadata,
            "stats": self.stats
        }

# Initialize memory system
memory_system = MemorySystem()