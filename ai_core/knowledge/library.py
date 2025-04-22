"""
Knowledge Library for Seren

Manages a shared repository of knowledge that can be accessed
and updated by the AI models.
"""

import os
import sys
import json
import logging
import time
import hashlib
import uuid
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# For vector embeddings
try:
    import numpy as np
    from numpy.linalg import norm
    has_numpy = True
except ImportError:
    has_numpy = False
    logging.warning("NumPy not available. Vector embeddings will be disabled.")

# For text processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    nltk.download('punkt', quiet=True)
    has_nltk = True
except ImportError:
    has_nltk = False
    logging.warning("NLTK not available. Advanced text processing will be limited.")

# For vector storage
try:
    import faiss
    has_faiss = True
except ImportError:
    has_faiss = False
    logging.warning("FAISS not available. Vector search will be disabled.")

# For embedding generation
try:
    from ai_core.model_manager import model_manager, ModelType
    can_generate_embeddings = True
except ImportError:
    can_generate_embeddings = False
    logging.warning("Model manager not available. Embedding generation will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeSource(object):
    """Knowledge source types"""
    TEXT_FILE = "text_file"        # Plain text file
    TEXT_CONTENT = "text_content"  # Directly provided text content
    URL = "url"                    # Web URL
    PDF = "pdf"                    # PDF document
    API = "api"                    # External API
    DATABASE = "database"          # Database query result
    CONVERSATION = "conversation"  # Conversation history

class KnowledgeEntry(object):
    """A single entry in the knowledge library"""
    
    def __init__(
        self, 
        content: str,
        source_type: str,
        source_reference: str,
        metadata: Dict[str, Any] = None,
        embedding: Optional[List[float]] = None,
        categories: Optional[List[str]] = None,
        id: Optional[str] = None,
        created_by: Optional[str] = None
    ):
        """
        Initialize a knowledge entry
        
        Args:
            content: The knowledge content text
            source_type: Type of knowledge source
            source_reference: Reference to the source (e.g., filename, URL)
            metadata: Additional metadata about the knowledge
            embedding: Vector embedding of the content
            categories: List of categories this knowledge belongs to
            id: Unique identifier (will be generated if not provided)
            created_by: Identifier of the model that created this entry
        """
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.source_type = source_type
        self.source_reference = source_reference
        self.metadata = metadata or {}
        self.embedding = embedding
        self.categories = categories or ["general"]
        self.created_by = created_by
        
        # Auto-fill some metadata
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.datetime.now().isoformat()
        if "word_count" not in self.metadata:
            self.metadata["word_count"] = len(content.split())
        if created_by and "created_by" not in self.metadata:
            self.metadata["created_by"] = created_by
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "content": self.content,
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "categories": self.categories,
            "created_by": self.created_by,
            "metadata": self.metadata,
            # We don't store embeddings in the JSON representation
            # as they can be very large
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        """Create from dictionary representation"""
        return cls(
            content=data.get("content", ""),
            source_type=data.get("source_type", ""),
            source_reference=data.get("source_reference", ""),
            metadata=data.get("metadata", {}),
            categories=data.get("categories", ["general"]),
            created_by=data.get("created_by"),
            id=data.get("id")
        )
        
    def to_plain_text(self) -> str:
        """
        Convert entry to plain text format for easy sharing
        
        Returns:
            Plain text representation
        """
        text = f"# Knowledge Entry: {self.id}\n\n"
        text += f"## Source: {self.source_reference} ({self.source_type})\n"
        text += f"## Categories: {', '.join(self.categories)}\n"
        if self.created_by:
            text += f"## Created by: {self.created_by}\n"
        text += f"## Created at: {self.metadata.get('created_at', 'Unknown')}\n\n"
        text += f"---\n\n"
        text += self.content
        text += f"\n\n---\n"
        return text
    
    @classmethod
    def from_plain_text(cls, text: str) -> Optional['KnowledgeEntry']:
        """
        Create entry from plain text format
        
        Args:
            text: Plain text representation
            
        Returns:
            KnowledgeEntry or None if parsing failed
        """
        try:
            lines = text.split("\n")
            id_line = lines[0].strip()
            id_match = re.search(r"Knowledge Entry: (.+)$", id_line)
            if not id_match:
                return None
            
            entry_id = id_match.group(1)
            
            # Extract metadata
            source_line = next((l for l in lines if l.strip().startswith("## Source:")), "")
            source_match = re.search(r"## Source: (.+) \((.+)\)", source_line)
            if source_match:
                source_reference = source_match.group(1)
                source_type = source_match.group(2)
            else:
                source_reference = "unknown"
                source_type = "text"
            
            # Extract categories
            categories_line = next((l for l in lines if l.strip().startswith("## Categories:")), "")
            categories_match = re.search(r"## Categories: (.+)$", categories_line)
            categories = categories_match.group(1).split(", ") if categories_match else ["general"]
            
            # Extract created_by
            created_by_line = next((l for l in lines if l.strip().startswith("## Created by:")), "")
            created_by_match = re.search(r"## Created by: (.+)$", created_by_line) 
            created_by = created_by_match.group(1) if created_by_match else None
            
            # Extract created_at
            created_at_line = next((l for l in lines if l.strip().startswith("## Created at:")), "")
            created_at_match = re.search(r"## Created at: (.+)$", created_at_line)
            created_at = created_at_match.group(1) if created_at_match else None
            
            # Extract content
            content_start = next((i for i, l in enumerate(lines) if l.strip() == "---"), 0) + 1
            content_end = next((i for i, l in enumerate(lines[content_start:], content_start) if l.strip() == "---"), len(lines))
            content = "\n".join(lines[content_start:content_end]).strip()
            
            # Create metadata
            metadata = {}
            if created_at:
                metadata["created_at"] = created_at
            
            return cls(
                content=content,
                source_type=source_type,
                source_reference=source_reference,
                metadata=metadata,
                categories=categories,
                created_by=created_by,
                id=entry_id
            )
        
        except Exception as e:
            logger.error(f"Error parsing plain text knowledge entry: {str(e)}")
            return None

class KnowledgeContext(object):
    """
    A context-specific collection of knowledge entries
    
    This represents a subset of the knowledge library that's
    relevant to a specific context or task.
    """
    
    def __init__(self, name: str, entries: List[KnowledgeEntry] = None, metadata: Dict[str, Any] = None):
        """Initialize a knowledge context"""
        self.name = name
        self.entries = entries or []
        self.metadata = metadata or {}
    
    def add_entry(self, entry: KnowledgeEntry) -> None:
        """Add an entry to this context"""
        self.entries.append(entry)
        # Update metadata
        self.metadata["last_updated"] = datetime.datetime.now().isoformat()
        self.metadata["entry_count"] = len(self.entries)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "entries": [entry.to_dict() for entry in self.entries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeContext':
        """Create from dictionary representation"""
        entries = [KnowledgeEntry.from_dict(entry_data) for entry_data in data.get("entries", [])]
        return cls(
            name=data.get("name", ""),
            entries=entries,
            metadata=data.get("metadata", {})
        )

class KnowledgeLibrary(object):
    """
    Knowledge Library for Seren
    
    Manages a shared repository of knowledge that can be accessed
    and updated by the AI models. Features include:
    - Storage and retrieval of text-based knowledge
    - Semantic search using vector embeddings
    - Extraction of relevant context for queries
    - Reading and processing of text files
    - Retrieval of online content (when enabled)
    - Context-based knowledge organization
    - Shared knowledge between all models
    """
    
    def __init__(self, base_dir: str = None, vector_dim: int = 768, enable_web_access: bool = False):
        """
        Initialize the knowledge library
        
        Args:
            base_dir: Base directory for storing knowledge files
            vector_dim: Dimension of vector embeddings
            enable_web_access: Whether to enable web access for knowledge retrieval
        """
        # Set base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Create directories
        self.knowledge_dir = os.path.join(self.base_dir, "data", "knowledge")
        self.index_dir = os.path.join(self.knowledge_dir, "index")
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Knowledge entries
        self.entries = {}  # id -> KnowledgeEntry
        
        # Knowledge contexts
        self.contexts = {}  # name -> KnowledgeContext
        
        # Vector settings
        self.vector_dim = vector_dim
        self.use_vectors = has_numpy and has_faiss and can_generate_embeddings
        
        # Vector index
        self.vector_index = None
        if self.use_vectors:
            self._init_vector_index()
        
        # Web access settings
        self.enable_web_access = enable_web_access
        
        # Load existing knowledge
        self._load_knowledge()
        
        logger.info(f"Knowledge Library initialized with {len(self.entries)} entries")
    
    def _init_vector_index(self) -> None:
        """Initialize the vector index"""
        if not has_faiss:
            return
        
        try:
            # Create or load FAISS index
            index_path = os.path.join(self.index_dir, "vectors.index")
            if os.path.exists(index_path):
                # Load existing index
                self.vector_index = faiss.read_index(index_path)
                logger.info(f"Loaded existing vector index with {self.vector_index.ntotal} vectors")
            else:
                # Create new index
                self.vector_index = faiss.IndexFlatL2(self.vector_dim)
                logger.info(f"Created new vector index with dimension {self.vector_dim}")
        except Exception as e:
            logger.error(f"Error initializing vector index: {str(e)}")
            self.vector_index = None
    
    def _save_vector_index(self) -> bool:
        """Save the vector index"""
        if not has_faiss or self.vector_index is None:
            return False
        
        try:
            index_path = os.path.join(self.index_dir, "vectors.index")
            faiss.write_index(self.vector_index, index_path)
            logger.info(f"Saved vector index with {self.vector_index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error saving vector index: {str(e)}")
            return False
    
    def _load_knowledge(self) -> None:
        """Load existing knowledge entries and contexts"""
        # Load entries
        entries_file = os.path.join(self.knowledge_dir, "entries.json")
        if os.path.exists(entries_file):
            try:
                with open(entries_file, 'r', encoding='utf-8') as f:
                    entries_data = json.load(f)
                
                for entry_data in entries_data:
                    entry = KnowledgeEntry.from_dict(entry_data)
                    self.entries[entry.id] = entry
                
                logger.info(f"Loaded {len(self.entries)} knowledge entries")
            except Exception as e:
                logger.error(f"Error loading knowledge entries: {str(e)}")
        
        # Load contexts
        contexts_file = os.path.join(self.knowledge_dir, "contexts.json")
        if os.path.exists(contexts_file):
            try:
                with open(contexts_file, 'r', encoding='utf-8') as f:
                    contexts_data = json.load(f)
                
                for context_data in contexts_data:
                    context = KnowledgeContext.from_dict(context_data)
                    self.contexts[context.name] = context
                
                logger.info(f"Loaded {len(self.contexts)} knowledge contexts")
            except Exception as e:
                logger.error(f"Error loading knowledge contexts: {str(e)}")
        
        # Load vector embeddings
        embeddings_file = os.path.join(self.knowledge_dir, "embeddings.json")
        if os.path.exists(embeddings_file) and self.use_vectors:
            try:
                with open(embeddings_file, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)
                
                # Map entry IDs to embeddings
                id_to_index = {}
                embeddings = []
                
                for entry_id, embedding in embeddings_data.items():
                    if entry_id in self.entries:
                        # Store the embedding in the entry
                        self.entries[entry_id].embedding = embedding
                        
                        # Add to the list for FAISS
                        id_to_index[entry_id] = len(embeddings)
                        embeddings.append(embedding)
                
                # Add embeddings to FAISS index
                if embeddings and self.vector_index is not None:
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    self.vector_index.add(embeddings_array)
                    
                    # Store the mapping for retrieval
                    self.id_to_index = id_to_index
                    self.index_to_id = {v: k for k, v in id_to_index.items()}
                    
                    logger.info(f"Loaded {len(embeddings)} vector embeddings")
            except Exception as e:
                logger.error(f"Error loading vector embeddings: {str(e)}")
    
    def _save_knowledge(self) -> bool:
        """Save knowledge entries and contexts"""
        try:
            # Save entries
            entries_file = os.path.join(self.knowledge_dir, "entries.json")
            entries_data = [entry.to_dict() for entry in self.entries.values()]
            with open(entries_file, 'w', encoding='utf-8') as f:
                json.dump(entries_data, f, indent=2, ensure_ascii=False)
            
            # Save contexts
            contexts_file = os.path.join(self.knowledge_dir, "contexts.json")
            contexts_data = [context.to_dict() for context in self.contexts.values()]
            with open(contexts_file, 'w', encoding='utf-8') as f:
                json.dump(contexts_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if self.use_vectors:
                embeddings_file = os.path.join(self.knowledge_dir, "embeddings.json")
                embeddings_data = {
                    entry_id: entry.embedding 
                    for entry_id, entry in self.entries.items() 
                    if entry.embedding is not None
                }
                with open(embeddings_file, 'w', encoding='utf-8') as f:
                    json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
            
            # Save vector index
            if self.use_vectors:
                self._save_vector_index()
            
            logger.info(f"Saved knowledge library with {len(self.entries)} entries and {len(self.contexts)} contexts")
            return True
        
        except Exception as e:
            logger.error(f"Error saving knowledge: {str(e)}")
            return False
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate vector embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding or None if not available
        """
        if not self.use_vectors:
            return None
        
        try:
            # Use Qwen for embeddings
            if can_generate_embeddings:
                # Generate embedding using the model
                embedding_text = model_manager.generate_text(
                    model_type=ModelType.QWEN,
                    prompt=f"Generate a vector embedding for this text: {text}",
                    max_length=0,  # We don't need output text
                    temperature=0.0,
                    _embedding=True  # Special flag to get embedding instead of text
                )
                
                # If embedding_text is actually a list of floats (embedding)
                if isinstance(embedding_text, list) and all(isinstance(x, float) for x in embedding_text):
                    return embedding_text
            
            # Fallback to simple method if model embedding fails
            return self._generate_simple_embedding(text)
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return self._generate_simple_embedding(text)
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding for text without using ML
        
        This is a fallback when ML embeddings are not available.
        It's not as good as proper embeddings but better than nothing.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple vector embedding
        """
        if not has_numpy:
            # If we don't have numpy, return None
            return None
        
        # Convert text to lowercase and normalize
        text = text.lower()
        
        # Define simple feature extraction
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        
        # Use word hashing to fill the vector
        words = text.split()
        for i, word in enumerate(words):
            # Hash the word to get a position in the vector
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            pos = hash_val % self.vector_dim
            
            # Set the value at that position
            vector[pos] += 1.0
        
        # Normalize the vector
        if np.sum(vector) > 0:
            vector = vector / np.sqrt(np.sum(vector * vector))
        
        return vector.tolist()
    
    def add_knowledge_from_text(
        self, 
        text: str, 
        source_reference: str, 
        chunk_size: int = 1000,
        overlap: int = 100,
        context_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """
        Add knowledge from a text string
        
        Args:
            text: Text content to add
            source_reference: Reference to the source
            chunk_size: Size of text chunks (in characters)
            overlap: Overlap between chunks
            context_name: Optional context to add to
            metadata: Additional metadata
            
        Returns:
            List of entry IDs added
        """
        if not text:
            logger.warning("Empty text provided, no knowledge added")
            return []
        
        # Add basic metadata
        metadata = metadata or {}
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
        if "full_text_length" not in metadata:
            metadata["full_text_length"] = len(text)
        
        # Chunk the text
        chunks = self._chunk_text(text, chunk_size, overlap)
        
        # Process each chunk
        entry_ids = []
        for i, chunk in enumerate(chunks):
            # Create a new entry
            metadata_copy = metadata.copy()
            metadata_copy["chunk_index"] = i
            metadata_copy["chunk_count"] = len(chunks)
            
            entry = KnowledgeEntry(
                content=chunk,
                source_type=KnowledgeSource.TEXT_CONTENT,
                source_reference=source_reference,
                metadata=metadata_copy
            )
            
            # Generate embedding
            if self.use_vectors:
                entry.embedding = self.generate_embedding(chunk)
                
                # Add to vector index if available
                if entry.embedding and self.vector_index is not None:
                    try:
                        # Add the embedding to the index
                        embedding_array = np.array([entry.embedding], dtype=np.float32)
                        index = self.vector_index.ntotal
                        self.vector_index.add(embedding_array)
                        
                        # Update the ID mapping
                        if not hasattr(self, 'id_to_index'):
                            self.id_to_index = {}
                            self.index_to_id = {}
                        
                        self.id_to_index[entry.id] = index
                        self.index_to_id[index] = entry.id
                    except Exception as e:
                        logger.error(f"Error adding embedding to index: {str(e)}")
            
            # Add to entries
            self.entries[entry.id] = entry
            entry_ids.append(entry.id)
            
            # Add to context if specified
            if context_name:
                self.add_to_context(entry.id, context_name)
        
        # Save changes
        self._save_knowledge()
        
        logger.info(f"Added {len(entry_ids)} knowledge entries from text")
        return entry_ids
    
    def add_knowledge_from_file(
        self, 
        file_path: str, 
        chunk_size: int = 1000,
        overlap: int = 100,
        context_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """
        Add knowledge from a text file
        
        Args:
            file_path: Path to the text file
            chunk_size: Size of text chunks (in characters)
            overlap: Overlap between chunks
            context_name: Optional context to add to
            metadata: Additional metadata
            
        Returns:
            List of entry IDs added
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return []
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Add metadata about the file
            file_metadata = metadata or {}
            file_metadata["filename"] = os.path.basename(file_path)
            file_metadata["file_size"] = os.path.getsize(file_path)
            file_metadata["file_created"] = datetime.datetime.fromtimestamp(
                os.path.getctime(file_path)).isoformat()
            file_metadata["file_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(file_path)).isoformat()
            
            # Add the knowledge
            return self.add_knowledge_from_text(
                text=text,
                source_reference=file_path,
                chunk_size=chunk_size,
                overlap=overlap,
                context_name=context_name,
                metadata=file_metadata
            )
        
        except Exception as e:
            logger.error(f"Error adding knowledge from file: {str(e)}")
            return []
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Chunk text into smaller pieces
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk (in characters)
            overlap: Overlap between chunks (in characters)
            
        Returns:
            List of text chunks
        """
        # Use NLTK for sentence tokenization if available
        if has_nltk:
            try:
                sentences = sent_tokenize(text)
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    # If adding this sentence would exceed the chunk size
                    if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                        # Add the current chunk to the list
                        chunks.append(current_chunk)
                        # Start a new chunk with overlap
                        words = current_chunk.split()
                        overlap_word_count = min(len(words), overlap // 5)  # Approx 5 chars per word
                        overlap_text = " ".join(words[-overlap_word_count:])
                        current_chunk = overlap_text + " " + sentence
                    else:
                        # Add the sentence to the current chunk
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                
                # Add the final chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                return chunks
            
            except Exception as e:
                logger.error(f"Error chunking text with NLTK: {str(e)}")
                # Fall back to simple chunking
        
        # Simple chunking by characters
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            # Ensure we don't go beyond the text length
            end = min(i + chunk_size, len(text))
            chunks.append(text[i:end])
            
            # If we've reached the end, break
            if end == len(text):
                break
        
        return chunks
    
    def create_context(self, name: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Create a new knowledge context
        
        Args:
            name: Name of the context
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if name in self.contexts:
            logger.warning(f"Context already exists: {name}")
            return False
        
        # Create the context
        context = KnowledgeContext(name=name, metadata=metadata)
        self.contexts[name] = context
        
        # Save changes
        self._save_knowledge()
        
        logger.info(f"Created knowledge context: {name}")
        return True
    
    def add_to_context(self, entry_id: str, context_name: str) -> bool:
        """
        Add a knowledge entry to a context
        
        Args:
            entry_id: ID of the entry to add
            context_name: Name of the context
            
        Returns:
            Success status
        """
        # Check if entry exists
        if entry_id not in self.entries:
            logger.error(f"Entry not found: {entry_id}")
            return False
        
        # Check if context exists, create if not
        if context_name not in self.contexts:
            self.create_context(context_name)
        
        # Get the entry and context
        entry = self.entries[entry_id]
        context = self.contexts[context_name]
        
        # Check if already in context
        if any(e.id == entry_id for e in context.entries):
            logger.warning(f"Entry already in context: {entry_id} in {context_name}")
            return False
        
        # Add to context
        context.add_entry(entry)
        
        # Save changes
        self._save_knowledge()
        
        logger.info(f"Added entry {entry_id} to context {context_name}")
        return True
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Get a knowledge entry by ID
        
        Args:
            entry_id: ID of the entry
            
        Returns:
            Knowledge entry or None if not found
        """
        return self.entries.get(entry_id)
    
    def get_context(self, context_name: str) -> Optional[KnowledgeContext]:
        """
        Get a knowledge context by name
        
        Args:
            context_name: Name of the context
            
        Returns:
            Knowledge context or None if not found
        """
        return self.contexts.get(context_name)
    
    def search_knowledge(self, query: str, limit: int = 5, categories: List[str] = None) -> List[KnowledgeEntry]:
        """
        Search knowledge using semantic similarity
        
        Args:
            query: Search query
            limit: Maximum number of results
            categories: Optional list of categories to filter by
            
        Returns:
            List of relevant knowledge entries
        """
        if not self.use_vectors or not self.vector_index or self.vector_index.ntotal == 0:
            # Fall back to keyword search if vectors not available
            return self._keyword_search(query, limit, categories)
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return self._keyword_search(query, limit, categories)
            
            # Convert to the right format
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search the index - get more results than needed to allow for category filtering
            search_limit = min(limit * 3, self.vector_index.ntotal) if categories else min(limit, self.vector_index.ntotal)
            distances, indices = self.vector_index.search(query_vector, search_limit)
            
            # Convert indices to entry IDs
            results = []
            for i, index in enumerate(indices[0]):
                if index < 0:
                    continue  # Skip invalid indices
                
                entry_id = self.index_to_id.get(int(index))
                if entry_id and entry_id in self.entries:
                    entry = self.entries[entry_id]
                    
                    # Apply category filter if specified
                    if categories and not any(cat in entry.categories for cat in categories):
                        continue
                    
                    # Add the distance as metadata
                    entry.metadata["_search_distance"] = float(distances[0][i])
                    results.append(entry)
                    
                    # Stop once we have enough results
                    if len(results) >= limit:
                        break
            
            # If not enough results, supplement with keyword search
            if len(results) < limit:
                keyword_results = self._keyword_search(query, limit - len(results), categories)
                # Filter out duplicates
                for entry in keyword_results:
                    if not any(r.id == entry.id for r in results):
                        results.append(entry)
            
            return results
        
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return self._keyword_search(query, limit, categories)
    
    def _keyword_search(self, query: str, limit: int = 5, categories: List[str] = None) -> List[KnowledgeEntry]:
        """
        Search knowledge using keyword matching
        
        Args:
            query: Search query
            limit: Maximum number of results
            categories: Optional list of categories to filter by
            
        Returns:
            List of relevant knowledge entries
        """
        # Normalize query
        query = query.lower()
        query_words = set(query.split())
        
        # Score each entry
        scored_entries = []
        for entry in self.entries.values():
            # Apply category filter if specified
            if categories and not any(cat in entry.categories for cat in categories):
                continue
                
            content = entry.content.lower()
            
            # Calculate match score
            word_matches = sum(1 for word in query_words if word in content)
            exact_phrase = query in content
            
            # Calculate a score
            score = word_matches * 0.5
            if exact_phrase:
                score += 2
            
            # Add to results if there's any match
            if score > 0:
                entry_copy = KnowledgeEntry(
                    content=entry.content,
                    source_type=entry.source_type,
                    source_reference=entry.source_reference,
                    metadata=entry.metadata.copy(),
                    categories=entry.categories,
                    created_by=entry.created_by,
                    id=entry.id
                )
                entry_copy.metadata["_search_score"] = score
                scored_entries.append((score, entry_copy))
        
        # Sort by score and take top results
        scored_entries.sort(reverse=True, key=lambda x: x[0])
        return [entry for _, entry in scored_entries[:limit]]
    
    def get_knowledge_for_query(self, query: str, limit: int = 5) -> List[KnowledgeEntry]:
        """
        Get knowledge relevant to a query
        
        Args:
            query: The user's query
            limit: Maximum number of knowledge entries to return
            
        Returns:
            List of relevant knowledge entries
        """
        return self.search_knowledge(query, limit)
    
    def extract_context_for_query(self, query: str, limit: int = 5) -> str:
        """
        Extract a context string for a query
        
        Args:
            query: The user's query
            limit: Maximum number of knowledge entries to include
            
        Returns:
            Context string
        """
        entries = self.get_knowledge_for_query(query, limit)
        if not entries:
            return ""
        
        # Combine the entries into a context string
        context_parts = []
        for i, entry in enumerate(entries):
            source = entry.source_reference
            context_parts.append(f"[{i+1}] From {source}:\n{entry.content}\n")
        
        return "\n".join(context_parts)
    
    def knowledge_enhancement_service(self, query: str, response: str) -> str:
        """
        Enhance a response with knowledge
        
        Args:
            query: User query
            response: Initial response
            
        Returns:
            Knowledge-enhanced response
        """
        # Get relevant knowledge
        entries = self.get_knowledge_for_query(query, limit=3)
        if not entries:
            return response
        
        # Add knowledge to the response
        knowledge_text = "\n\nAdditional information from knowledge library:\n"
        for i, entry in enumerate(entries):
            source = entry.source_reference
            categories = ", ".join(entry.categories) if entry.categories else "general"
            knowledge_text += f"[From {source} (Categories: {categories})]: {entry.content[:300]}...\n"
        
        return response + knowledge_text
    
    def get_categories(self) -> List[str]:
        """
        Get all unique categories in the knowledge library
        
        Returns:
            List of unique category names
        """
        categories = set()
        for entry in self.entries.values():
            for category in entry.categories:
                categories.add(category)
        return sorted(list(categories))
    
    def get_entries_by_category(self, category: str) -> List[KnowledgeEntry]:
        """
        Get all entries that belong to a specific category
        
        Args:
            category: Category name
            
        Returns:
            List of entries in the category
        """
        return [entry for entry in self.entries.values() if category in entry.categories]
    
    def get_entries_by_model(self, model_id: str) -> List[KnowledgeEntry]:
        """
        Get all entries created by a specific model
        
        Args:
            model_id: Identifier of the model
            
        Returns:
            List of entries created by the model
        """
        return [entry for entry in self.entries.values() if entry.created_by == model_id]
    
    def export_entries_to_plain_text(self, entries: List[KnowledgeEntry], output_file: str) -> bool:
        """
        Export knowledge entries to a plain text file
        
        Args:
            entries: List of entries to export
            output_file: Path to output file
            
        Returns:
            Success status
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, entry in enumerate(entries):
                    if i > 0:
                        f.write("\n\n" + "="*80 + "\n\n")
                    f.write(entry.to_plain_text())
            
            logger.info(f"Exported {len(entries)} entries to {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting entries to plain text: {str(e)}")
            return False
    
    def export_category_to_plain_text(self, category: str, output_file: str) -> bool:
        """
        Export all entries in a category to a plain text file
        
        Args:
            category: Category name
            output_file: Path to output file
            
        Returns:
            Success status
        """
        entries = self.get_entries_by_category(category)
        return self.export_entries_to_plain_text(entries, output_file)
    
    def export_entries_to_json(self, entries: List[KnowledgeEntry], output_file: str) -> bool:
        """
        Export knowledge entries to a JSON file
        
        Args:
            entries: List of entries to export
            output_file: Path to output file
            
        Returns:
            Success status
        """
        try:
            data = [entry.to_dict() for entry in entries]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(entries)} entries to {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting entries to JSON: {str(e)}")
            return False
    
    def import_entries_from_plain_text(self, input_file: str) -> List[str]:
        """
        Import knowledge entries from a plain text file
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of entry IDs added
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by separator
            entry_texts = content.split("=" * 80)
            
            # Parse each entry
            entry_ids = []
            for text in entry_texts:
                text = text.strip()
                if not text:
                    continue
                
                entry = KnowledgeEntry.from_plain_text(text)
                if entry:
                    # Add to library
                    self.entries[entry.id] = entry
                    entry_ids.append(entry.id)
            
            # Save changes
            self._save_knowledge()
            
            logger.info(f"Imported {len(entry_ids)} entries from {input_file}")
            return entry_ids
        
        except Exception as e:
            logger.error(f"Error importing entries from plain text: {str(e)}")
            return []
    
    def import_entries_from_json(self, input_file: str) -> List[str]:
        """
        Import knowledge entries from a JSON file
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of entry IDs added
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse each entry
            entry_ids = []
            for entry_data in data:
                entry = KnowledgeEntry.from_dict(entry_data)
                
                # Add to library
                self.entries[entry.id] = entry
                entry_ids.append(entry.id)
            
            # Save changes
            self._save_knowledge()
            
            logger.info(f"Imported {len(entry_ids)} entries from {input_file}")
            return entry_ids
        
        except Exception as e:
            logger.error(f"Error importing entries from JSON: {str(e)}")
            return []
    
    def add_knowledge_entry(
        self,
        content: str,
        source_type: str,
        source_reference: str,
        categories: List[str] = None,
        created_by: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Add a single knowledge entry directly
        
        Args:
            content: The knowledge content
            source_type: Type of knowledge source
            source_reference: Reference to the source
            categories: List of categories
            created_by: Identifier of the model that created this entry
            metadata: Additional metadata
            
        Returns:
            Entry ID or None if failed
        """
        try:
            # Create entry
            entry = KnowledgeEntry(
                content=content,
                source_type=source_type,
                source_reference=source_reference,
                categories=categories,
                created_by=created_by,
                metadata=metadata
            )
            
            # Generate embedding
            if self.use_vectors:
                entry.embedding = self.generate_embedding(content)
                
                # Add to vector index if available
                if entry.embedding and self.vector_index is not None:
                    try:
                        # Add the embedding to the index
                        embedding_array = np.array([entry.embedding], dtype=np.float32)
                        index = self.vector_index.ntotal
                        self.vector_index.add(embedding_array)
                        
                        # Update the ID mapping
                        if not hasattr(self, 'id_to_index'):
                            self.id_to_index = {}
                            self.index_to_id = {}
                        
                        self.id_to_index[entry.id] = index
                        self.index_to_id[index] = entry.id
                    except Exception as e:
                        logger.error(f"Error adding embedding to index: {str(e)}")
            
            # Add to entries
            self.entries[entry.id] = entry
            
            # Save changes
            self._save_knowledge()
            
            logger.info(f"Added knowledge entry {entry.id}")
            return entry.id
        
        except Exception as e:
            logger.error(f"Error adding knowledge entry: {str(e)}")
            return None

# Initialize the knowledge library
knowledge_library = KnowledgeLibrary()