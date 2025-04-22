"""
AI Memory System

Provides persistent, searchable memory for the AI system using vector databases.
Enables semantic search, episodic memory, and knowledge persistence.
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

class MemoryType(str, enum.Enum):
    """Types of AI memory"""
    EPISODIC = "episodic"  # Memory of events and interactions
    SEMANTIC = "semantic"  # Factual knowledge
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"  # Temporary, active memory
    ASSOCIATIVE = "associative"  # Connections between concepts

class MemorySystem:
    """
    AI Memory System
    
    Manages different types of memory (episodic, semantic, procedural)
    using vector databases for efficient storage and retrieval.
    
    Key capabilities:
    1. Store memories with metadata and context
    2. Retrieve memories by semantic similarity
    3. Track memory importance and decay
    4. Organize knowledge hierarchically
    5. Establish connections between related memories
    """
    
    def __init__(self):
        """Initialize the memory system"""
        # Initialize memory stores
        self.memories = {
            MemoryType.EPISODIC: [],
            MemoryType.SEMANTIC: [],
            MemoryType.PROCEDURAL: [],
            MemoryType.WORKING: [],
            MemoryType.ASSOCIATIVE: []
        }
        
        # Vector DB clients would be initialized here in a real implementation
        self.vector_db = None
        self.use_vector_db = False
        
        # Memory statistics
        self.stats = {
            "total_memories": 0,
            "memories_by_type": {memory_type: 0 for memory_type in MemoryType},
            "retrievals": 0,
            "last_retrieval": None,
            "last_storage": None
        }
        
        # Initialize memory indexes
        self.memory_indexes = {
            "id_to_memory": {},
            "keyword_to_ids": {},
            "entity_to_ids": {},
            "time_to_ids": {}
        }
        
        try:
            # In a real implementation, this would connect to a vector database
            # self._initialize_vector_db()
            logger.info("Memory system initialized with local storage (vector DB not available)")
        except Exception as e:
            logger.warning(f"Failed to initialize vector database: {str(e)}")
            logger.info("Falling back to in-memory storage")
    
    def _initialize_vector_db(self):
        """Initialize connection to vector database"""
        # In a real implementation, this would connect to Weaviate, ChromaDB, etc.
        # For now, we'll use in-memory storage only
        try:
            # Try to import vector DB libraries
            # import weaviate
            # import chromadb
            
            logger.info("Vector database initialized successfully")
            self.use_vector_db = True
        except ImportError:
            logger.warning("Vector database libraries not available")
            self.use_vector_db = False
    
    def store(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a new memory
        
        Args:
            content: The content to store
            memory_type: The type of memory
            importance: Importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            ID of the stored memory
        """
        # Create memory object
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        memory = {
            "id": memory_id,
            "content": content,
            "type": memory_type,
            "created_at": timestamp,
            "last_accessed": timestamp,
            "importance": importance,
            "access_count": 0,
            "metadata": metadata or {}
        }
        
        # Store in the appropriate memory store
        self.memories[memory_type].append(memory)
        
        # Update indexes
        self.memory_indexes["id_to_memory"][memory_id] = memory
        
        # Extract and index keywords
        keywords = self._extract_keywords(content)
        for keyword in keywords:
            if keyword not in self.memory_indexes["keyword_to_ids"]:
                self.memory_indexes["keyword_to_ids"][keyword] = set()
            self.memory_indexes["keyword_to_ids"][keyword].add(memory_id)
        
        # Extract and index entities
        entities = self._extract_entities(content)
        for entity in entities:
            if entity not in self.memory_indexes["entity_to_ids"]:
                self.memory_indexes["entity_to_ids"][entity] = set()
            self.memory_indexes["entity_to_ids"][entity].add(memory_id)
        
        # Index by time
        date_str = timestamp.split("T")[0]
        if date_str not in self.memory_indexes["time_to_ids"]:
            self.memory_indexes["time_to_ids"][date_str] = set()
        self.memory_indexes["time_to_ids"][date_str].add(memory_id)
        
        # Update stats
        self.stats["total_memories"] += 1
        self.stats["memories_by_type"][memory_type] += 1
        self.stats["last_storage"] = timestamp
        
        if self.use_vector_db:
            # In a real implementation, store in vector database
            pass
        
        logger.info(f"Stored new {memory_type} memory with ID {memory_id}")
        return memory_id
    
    def _extract_keywords(self, content: Dict[str, Any]) -> List[str]:
        """Extract keywords from content for indexing"""
        # In a real implementation, this would use NLP techniques
        # For now, use a simple approach
        keywords = []
        
        if isinstance(content, dict):
            # Extract from common fields
            for field in ["query", "topic", "subject", "title", "category", "tags"]:
                if field in content and isinstance(content[field], str):
                    words = content[field].lower().split()
                    keywords.extend([word for word in words if len(word) > 3])
            
            # Extract from text content
            for field in ["content", "text", "description", "body"]:
                if field in content and isinstance(content[field], str):
                    words = content[field].lower().split()
                    # Take only substantive words
                    keywords.extend([word for word in words if len(word) > 4])
        
        # Remove duplicates
        return list(set(keywords))
    
    def _extract_entities(self, content: Dict[str, Any]) -> List[str]:
        """Extract entities from content for indexing"""
        # In a real implementation, this would use named entity recognition
        # For now, use a simple approach
        entities = []
        
        if isinstance(content, dict):
            # Extract from common entity fields
            for field in ["name", "person", "organization", "location", "company", "product"]:
                if field in content and isinstance(content[field], str):
                    entities.append(content[field].lower())
            
            # Check for entities in metadata
            if "entities" in content and isinstance(content["entities"], list):
                entities.extend([e.lower() for e in content["entities"] if isinstance(e, str)])
        
        # Remove duplicates
        return list(set(entities))
    
    def query(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Query memories by semantic search
        
        Args:
            query: The search query
            memory_type: Optional filter by memory type
            limit: Maximum number of results
            threshold: Similarity threshold
            
        Returns:
            List of matching memories
        """
        logger.info(f"Querying memories: '{query}'")
        self.stats["retrievals"] += 1
        self.stats["last_retrieval"] = datetime.now().isoformat()
        
        if self.use_vector_db:
            # In a real implementation, this would use vector search
            # return self._vector_search(query, memory_type, limit, threshold)
            pass
        
        # Fallback to keyword search
        return self._keyword_search(query, memory_type, limit)
    
    def _keyword_search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        query_keywords = query.lower().split()
        
        # Filter to substantive words
        query_keywords = [word for word in query_keywords if len(word) > 3]
        
        # Find matching memories by keyword overlap
        matches = {}
        
        for keyword in query_keywords:
            if keyword in self.memory_indexes["keyword_to_ids"]:
                for memory_id in self.memory_indexes["keyword_to_ids"][keyword]:
                    memory = self.memory_indexes["id_to_memory"][memory_id]
                    
                    # Apply memory type filter if specified
                    if memory_type and memory["type"] != memory_type:
                        continue
                    
                    # Compute match score based on keyword overlap
                    if memory_id not in matches:
                        matches[memory_id] = {
                            "memory": memory,
                            "score": 0
                        }
                    
                    matches[memory_id]["score"] += 1
        
        # Sort by score
        scored_matches = sorted(
            matches.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Update access metadata for returned memories
        results = []
        for match in scored_matches[:limit]:
            memory = match["memory"]
            memory["last_accessed"] = datetime.now().isoformat()
            memory["access_count"] += 1
            
            # Create a copy with score
            result = dict(memory)
            result["match_score"] = match["score"] / len(query_keywords)
            results.append(result)
        
        return results
    
    def _vector_search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Semantic search using vector database"""
        # In a real implementation, this would use vector search
        # For now, just return a placeholder
        raise NotImplementedError("Vector search not available in this implementation")
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory object or None if not found
        """
        memory = self.memory_indexes["id_to_memory"].get(memory_id)
        
        if memory:
            # Update access metadata
            memory["last_accessed"] = datetime.now().isoformat()
            memory["access_count"] += 1
            self.stats["retrievals"] += 1
            self.stats["last_retrieval"] = memory["last_accessed"]
            
            return memory
        
        return None
    
    def query_relevant(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query memories relevant to the current context
        
        Args:
            query: The current query
            context: Additional context information
            memory_types: Types of memories to include
            limit: Maximum number of results
            
        Returns:
            List of relevant memories
        """
        logger.info(f"Finding relevant memories for query: '{query[:50]}...'")
        
        # Default to all memory types if not specified
        memory_types = memory_types or list(MemoryType)
        
        results = []
        
        # Search each memory type
        for memory_type in memory_types:
            # Adjust limit to ensure diverse memory types
            type_limit = max(1, limit // len(memory_types))
            
            # Query this memory type
            type_results = self.query(
                query=query,
                memory_type=memory_type,
                limit=type_limit
            )
            
            results.extend(type_results)
        
        # Re-rank results based on relevance to context
        if context:
            results = self._rerank_by_context(results, context)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        # Limit to requested number
        return results[:limit]
    
    def _rerank_by_context(
        self,
        results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Re-rank results based on relevance to context"""
        # In a real implementation, this would use more sophisticated methods
        # For now, use a simple approach
        
        # Extract context keywords
        context_keywords = set()
        for key, value in context.items():
            if isinstance(value, str):
                words = value.lower().split()
                context_keywords.update([word for word in words if len(word) > 3])
        
        # If no context keywords, return original results
        if not context_keywords:
            return results
        
        # Re-score based on context relevance
        for result in results:
            memory_keywords = set(self._extract_keywords(result["content"]))
            
            # Calculate overlap with context keywords
            overlap = len(memory_keywords.intersection(context_keywords))
            context_boost = overlap / max(1, len(context_keywords))
            
            # Apply boost to match score
            original_score = result.get("match_score", 0.5)
            result["match_score"] = original_score * (1 + context_boost)
        
        return results
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        memory = self.memory_indexes["id_to_memory"].get(memory_id)
        
        if not memory:
            return False
        
        # Remove from type-specific store
        memory_type = memory["type"]
        self.memories[memory_type] = [m for m in self.memories[memory_type] if m["id"] != memory_id]
        
        # Remove from indexes
        del self.memory_indexes["id_to_memory"][memory_id]
        
        # Remove from keyword index
        keywords = self._extract_keywords(memory["content"])
        for keyword in keywords:
            if keyword in self.memory_indexes["keyword_to_ids"]:
                self.memory_indexes["keyword_to_ids"][keyword].discard(memory_id)
        
        # Remove from entity index
        entities = self._extract_entities(memory["content"])
        for entity in entities:
            if entity in self.memory_indexes["entity_to_ids"]:
                self.memory_indexes["entity_to_ids"][entity].discard(memory_id)
        
        # Remove from time index
        date_str = memory["created_at"].split("T")[0]
        if date_str in self.memory_indexes["time_to_ids"]:
            self.memory_indexes["time_to_ids"][date_str].discard(memory_id)
        
        # Update stats
        self.stats["total_memories"] -= 1
        self.stats["memories_by_type"][memory_type] -= 1
        
        if self.use_vector_db:
            # In a real implementation, delete from vector database
            pass
        
        logger.info(f"Deleted memory with ID {memory_id}")
        return True
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory by ID
        
        Args:
            memory_id: ID of the memory to update
            updates: Fields to update
            
        Returns:
            True if updated, False if not found
        """
        memory = self.memory_indexes["id_to_memory"].get(memory_id)
        
        if not memory:
            return False
        
        # Handle content updates specially (need to update indexes)
        if "content" in updates:
            old_content = memory["content"]
            new_content = updates["content"]
            
            # Remove from keyword index
            old_keywords = self._extract_keywords(old_content)
            for keyword in old_keywords:
                if keyword in self.memory_indexes["keyword_to_ids"]:
                    self.memory_indexes["keyword_to_ids"][keyword].discard(memory_id)
            
            # Add to keyword index with new content
            new_keywords = self._extract_keywords(new_content)
            for keyword in new_keywords:
                if keyword not in self.memory_indexes["keyword_to_ids"]:
                    self.memory_indexes["keyword_to_ids"][keyword] = set()
                self.memory_indexes["keyword_to_ids"][keyword].add(memory_id)
            
            # Update entity index similarly
            old_entities = self._extract_entities(old_content)
            for entity in old_entities:
                if entity in self.memory_indexes["entity_to_ids"]:
                    self.memory_indexes["entity_to_ids"][entity].discard(memory_id)
            
            new_entities = self._extract_entities(new_content)
            for entity in new_entities:
                if entity not in self.memory_indexes["entity_to_ids"]:
                    self.memory_indexes["entity_to_ids"][entity] = set()
                self.memory_indexes["entity_to_ids"][entity].add(memory_id)
        
        # Update fields
        memory.update(updates)
        
        if self.use_vector_db:
            # In a real implementation, update in vector database
            pass
        
        logger.info(f"Updated memory with ID {memory_id}")
        return True
    
    def consolidate(self):
        """
        Consolidate memories to improve retrieval efficiency
        
        This process:
        1. Merges similar memories
        2. Updates importance based on access patterns
        3. Removes or archives old, unimportant memories
        """
        logger.info("Starting memory consolidation")
        
        # Update importance scores based on access patterns
        self._update_importance_scores()
        
        # Identify candidates for merging (similar memories)
        merge_candidates = self._identify_merge_candidates()
        
        # Perform merges
        merged_count = 0
        for primary_id, secondary_ids in merge_candidates.items():
            success = self._merge_memories(primary_id, secondary_ids)
            if success:
                merged_count += len(secondary_ids)
        
        # Identify low-importance memories for archival
        archived_count = self._archive_low_importance_memories()
        
        logger.info(f"Memory consolidation complete: merged {merged_count} memories, archived {archived_count} memories")
    
    def _update_importance_scores(self):
        """Update importance scores based on access patterns"""
        # In a real implementation, this would be more sophisticated
        # For now, use a simple approach
        
        # For each memory, adjust importance based on:
        # - Recency of access
        # - Frequency of access
        # - Initial importance
        
        now = datetime.now()
        
        for memory_type in MemoryType:
            for memory in self.memories[memory_type]:
                # Calculate recency factor (0-1, higher for more recent access)
                last_accessed = datetime.fromisoformat(memory["last_accessed"])
                days_since_access = (now - last_accessed).days
                recency_factor = max(0, 1 - (days_since_access / 30))  # Decay over 30 days
                
                # Calculate frequency factor (0-1, higher for more frequent access)
                frequency_factor = min(1, memory["access_count"] / 10)  # Saturate at 10 accesses
                
                # Combine factors with original importance
                original_importance = memory["importance"]
                new_importance = (
                    original_importance * 0.5 +
                    recency_factor * 0.3 +
                    frequency_factor * 0.2
                )
                
                # Update importance
                memory["importance"] = new_importance
    
    def _identify_merge_candidates(self) -> Dict[str, List[str]]:
        """Identify candidates for memory merging"""
        # In a real implementation, this would use clustering or similarity analysis
        # For now, use a simple approach based on keyword overlap
        
        # Map primary memories to similar secondary memories
        merge_candidates = {}
        
        # Only consider episodic memories for merging
        episodic_memories = self.memories[MemoryType.EPISODIC]
        
        # Skip if too few memories
        if len(episodic_memories) < 5:
            return {}
        
        # For each memory, find similar memories based on keyword overlap
        for i, memory in enumerate(episodic_memories):
            memory_id = memory["id"]
            memory_keywords = set(self._extract_keywords(memory["content"]))
            
            # Skip if too few keywords
            if len(memory_keywords) < 3:
                continue
            
            similar_memories = []
            
            # Compare with other memories
            for other_memory in episodic_memories[i+1:]:
                other_id = other_memory["id"]
                other_keywords = set(self._extract_keywords(other_memory["content"]))
                
                # Skip if too few keywords
                if len(other_keywords) < 3:
                    continue
                
                # Calculate similarity based on keyword overlap
                overlap = len(memory_keywords.intersection(other_keywords))
                similarity = overlap / max(1, len(memory_keywords.union(other_keywords)))
                
                # If similar enough, add to candidates
                if similarity > 0.7:  # Threshold for similarity
                    similar_memories.append(other_id)
            
            # If we found similar memories, add as merge candidate
            if similar_memories:
                merge_candidates[memory_id] = similar_memories
        
        return merge_candidates
    
    def _merge_memories(self, primary_id: str, secondary_ids: List[str]) -> bool:
        """Merge similar memories"""
        # Get primary memory
        primary_memory = self.memory_indexes["id_to_memory"].get(primary_id)
        if not primary_memory:
            return False
        
        # Get secondary memories
        secondary_memories = []
        for memory_id in secondary_ids:
            memory = self.memory_indexes["id_to_memory"].get(memory_id)
            if memory:
                secondary_memories.append(memory)
        
        # Skip if no secondary memories found
        if not secondary_memories:
            return False
        
        # Merge content
        merged_content = dict(primary_memory["content"])
        
        # For each field in the content, merge values
        for memory in secondary_memories:
            for key, value in memory["content"].items():
                if key not in merged_content:
                    # Add new field
                    merged_content[key] = value
                elif isinstance(merged_content[key], list) and isinstance(value, list):
                    # Merge lists
                    merged_content[key].extend(value)
                    # Remove duplicates if strings
                    if all(isinstance(item, str) for item in merged_content[key]):
                        merged_content[key] = list(set(merged_content[key]))
                elif isinstance(merged_content[key], dict) and isinstance(value, dict):
                    # Merge dicts
                    merged_content[key].update(value)
                # For other types, keep primary value
        
        # Update primary memory
        updates = {
            "content": merged_content,
            "importance": primary_memory["importance"],  # Will be recalculated
            "metadata": {
                **primary_memory["metadata"],
                "merged_from": secondary_ids,
                "merged_at": datetime.now().isoformat()
            }
        }
        
        # Apply updates
        self.update(primary_id, updates)
        
        # Delete secondary memories
        for memory_id in secondary_ids:
            self.delete(memory_id)
        
        return True
    
    def _archive_low_importance_memories(self) -> int:
        """Archive low-importance memories"""
        # In a real implementation, this would move to cold storage
        # For now, just identify candidates
        
        archived_count = 0
        importance_threshold = 0.2  # Arbitrary threshold
        
        for memory_type in MemoryType:
            low_importance_memories = []
            
            for memory in self.memories[memory_type]:
                if memory["importance"] < importance_threshold:
                    low_importance_memories.append(memory["id"])
            
            # For demonstration, don't actually delete but log
            archived_count += len(low_importance_memories)
            logger.info(f"Identified {len(low_importance_memories)} {memory_type} memories for archival")
        
        return archived_count
    
    def reset(self):
        """Reset the memory system (for testing or admin purposes)"""
        # Clear all memories
        for memory_type in MemoryType:
            self.memories[memory_type] = []
        
        # Clear indexes
        self.memory_indexes = {
            "id_to_memory": {},
            "keyword_to_ids": {},
            "entity_to_ids": {},
            "time_to_ids": {}
        }
        
        # Reset stats
        self.stats = {
            "total_memories": 0,
            "memories_by_type": {memory_type: 0 for memory_type in MemoryType},
            "retrievals": 0,
            "last_retrieval": None,
            "last_storage": None
        }
        
        logger.info("Memory system reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get basic status information"""
        return {
            "total_memories": self.stats["total_memories"],
            "retrieval_count": self.stats["retrievals"],
            "vector_db_enabled": self.use_vector_db
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            "stats": self.stats,
            "memory_counts": {str(memory_type): len(memories) for memory_type, memories in self.memories.items()},
            "index_stats": {
                "id_index_size": len(self.memory_indexes["id_to_memory"]),
                "keyword_index_size": len(self.memory_indexes["keyword_to_ids"]),
                "entity_index_size": len(self.memory_indexes["entity_to_ids"]),
                "time_index_size": len(self.memory_indexes["time_to_ids"])
            },
            "vector_db_enabled": self.use_vector_db
        }