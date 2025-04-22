"""
Self-Learning System for Seren

Enables models to learn from interactions and update their knowledge
based on new information.
"""

import os
import sys
import json
import logging
import time
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import threading
import queue

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local imports
try:
    from ai_core.knowledge.library import knowledge_library, KnowledgeEntry, KnowledgeSource
    from ai_core.model_manager import model_manager, ModelType
    from ai_core.model_communication import communication_system
except ImportError as e:
    logging.error(f"Error importing dependencies: {str(e)}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class LearningPriority:
    """Learning priority levels"""
    HIGH = "high"            # Critical new information
    MEDIUM = "medium"        # Important but not critical
    LOW = "low"              # Nice-to-have knowledge

class LearningTask:
    """A task for the self-learning system"""
    
    def __init__(
        self, 
        content: str,
        source: str, 
        priority: str = LearningPriority.MEDIUM,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a learning task
        
        Args:
            content: The content to learn
            source: Source of the content
            priority: Priority level
            metadata: Additional metadata
        """
        self.content = content
        self.source = source
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "content": self.content,
            "source": self.source,
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningTask':
        """Create from dictionary representation"""
        task = cls(
            content=data.get("content", ""),
            source=data.get("source", ""),
            priority=data.get("priority", LearningPriority.MEDIUM),
            metadata=data.get("metadata", {})
        )
        task.created_at = data.get("created_at", task.created_at)
        return task

class SelfLearningSystem:
    """
    Self-Learning System for Seren
    
    Enables models to learn from interactions and improve over time:
    - Extract new knowledge from conversations
    - Prioritize and process learning tasks
    - Update knowledge library with new information
    - Generate insights from existing knowledge
    - Share learnings between models
    """
    
    def __init__(self, base_dir: str = None, auto_learn: bool = True):
        """
        Initialize the self-learning system
        
        Args:
            base_dir: Base directory for storing learning data
            auto_learn: Whether to enable automatic learning
        """
        # Set base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Create directories
        self.learning_dir = os.path.join(self.base_dir, "data", "learning")
        os.makedirs(self.learning_dir, exist_ok=True)
        
        # Learning settings
        self.auto_learn = auto_learn
        
        # Learning task queue
        self.task_queue = queue.PriorityQueue()
        
        # Task history
        self.task_history = []
        
        # Learning thread
        self.learning_thread = None
        self.stop_learning = threading.Event()
        
        # Load existing tasks
        self._load_tasks()
        
        # Start learning thread if auto-learn enabled
        if self.auto_learn:
            self._start_learning_thread()
        
        logger.info(f"Self-Learning System initialized. Auto-learn: {self.auto_learn}")
    
    def _load_tasks(self) -> None:
        """Load existing learning tasks"""
        tasks_file = os.path.join(self.learning_dir, "tasks.json")
        if not os.path.exists(tasks_file):
            return
        
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load task history
            for task_data in data.get("history", []):
                task = LearningTask.from_dict(task_data)
                self.task_history.append(task)
            
            # Load pending tasks
            for task_data in data.get("pending", []):
                task = LearningTask.from_dict(task_data)
                # Convert priority to numeric value for the queue
                priority_value = self._get_priority_value(task.priority)
                self.task_queue.put((priority_value, task))
            
            logger.info(f"Loaded {len(self.task_history)} completed tasks and {self.task_queue.qsize()} pending tasks")
        
        except Exception as e:
            logger.error(f"Error loading learning tasks: {str(e)}")
    
    def _save_tasks(self) -> None:
        """Save learning tasks"""
        tasks_file = os.path.join(self.learning_dir, "tasks.json")
        
        try:
            # Get pending tasks from queue
            pending_tasks = []
            temp_queue = queue.PriorityQueue()
            
            # Extract all items
            while not self.task_queue.empty():
                priority, task = self.task_queue.get()
                pending_tasks.append(task)
                temp_queue.put((priority, task))
            
            # Restore the queue
            self.task_queue = temp_queue
            
            # Prepare data
            data = {
                "history": [task.to_dict() for task in self.task_history],
                "pending": [task.to_dict() for task in pending_tasks]
            }
            
            # Save to file
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"Error saving learning tasks: {str(e)}")
    
    def _get_priority_value(self, priority: str) -> int:
        """Convert priority string to numeric value"""
        if priority == LearningPriority.HIGH:
            return 1
        elif priority == LearningPriority.MEDIUM:
            return 2
        else:  # LearningPriority.LOW
            return 3
    
    def _start_learning_thread(self) -> None:
        """Start the learning thread"""
        if self.learning_thread is not None and self.learning_thread.is_alive():
            logger.warning("Learning thread already running")
            return
        
        # Reset stop event
        self.stop_learning.clear()
        
        # Create and start thread
        self.learning_thread = threading.Thread(target=self._learning_worker, daemon=True)
        self.learning_thread.start()
        
        logger.info("Learning thread started")
    
    def _stop_learning_thread(self) -> None:
        """Stop the learning thread"""
        if self.learning_thread is None or not self.learning_thread.is_alive():
            logger.warning("Learning thread not running")
            return
        
        # Set stop event
        self.stop_learning.set()
        
        # Wait for thread to stop
        self.learning_thread.join(timeout=5)
        
        # Check if thread stopped
        if self.learning_thread.is_alive():
            logger.warning("Learning thread did not stop gracefully")
        else:
            logger.info("Learning thread stopped")
    
    def _learning_worker(self) -> None:
        """Learning worker thread"""
        logger.info("Learning worker started")
        
        while not self.stop_learning.is_set():
            try:
                # Get a task from the queue with timeout
                try:
                    priority, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    # No task available, sleep and continue
                    time.sleep(1)
                    continue
                
                # Process the task
                logger.info(f"Processing learning task: {task.source}")
                self._process_learning_task(task)
                
                # Mark as done
                self.task_queue.task_done()
                
                # Add to history
                self.task_history.append(task)
                
                # Save tasks
                self._save_tasks()
                
            except Exception as e:
                logger.error(f"Error in learning worker: {str(e)}")
                time.sleep(5)
        
        logger.info("Learning worker stopped")
    
    def _process_learning_task(self, task: LearningTask) -> None:
        """
        Process a learning task
        
        Args:
            task: The task to process
        """
        try:
            # Extract key information from the content
            insights = self._extract_insights(task.content)
            
            # Add to knowledge library
            context_name = task.metadata.get("context", "learned_knowledge")
            
            # Add the original content
            entry_ids = knowledge_library.add_knowledge_from_text(
                text=task.content,
                source_reference=task.source,
                context_name=context_name,
                metadata={
                    "learning_priority": task.priority,
                    "learning_time": datetime.datetime.now().isoformat(),
                    "original_metadata": task.metadata
                }
            )
            
            # Add the insights if available
            if insights:
                insights_text = "\n\n".join(insights)
                knowledge_library.add_knowledge_from_text(
                    text=insights_text,
                    source_reference=f"Insights from {task.source}",
                    context_name=context_name,
                    metadata={
                        "insight_source": task.source,
                        "insight_type": "extracted",
                        "learning_time": datetime.datetime.now().isoformat()
                    }
                )
            
            # Update task metadata
            task.metadata["processed"] = True
            task.metadata["processed_time"] = datetime.datetime.now().isoformat()
            task.metadata["entry_ids"] = entry_ids
            
            logger.info(f"Successfully processed learning task from {task.source}")
        
        except Exception as e:
            logger.error(f"Error processing learning task: {str(e)}")
            task.metadata["error"] = str(e)
    
    def _extract_insights(self, content: str) -> List[str]:
        """
        Extract insights from content
        
        Args:
            content: Content to extract insights from
            
        Returns:
            List of insights
        """
        # In a real system, this would use ML models to extract
        # key insights. For now, we'll use a simple approach.
        
        insights = []
        
        # Split into paragraphs
        paragraphs = content.split("\n\n")
        
        # Process each paragraph
        for i, paragraph in enumerate(paragraphs):
            # Skip short paragraphs
            if len(paragraph.strip()) < 50:
                continue
            
            # Add to insights (in a real system, would be more sophisticated)
            if i < 3:  # Just take first few paragraphs as "insights"
                insights.append(f"Insight {i+1}: {paragraph.strip()}")
        
        return insights
    
    def add_learning_task(
        self, 
        content: str,
        source: str,
        priority: str = LearningPriority.MEDIUM,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a learning task
        
        Args:
            content: Content to learn
            source: Source of the content
            priority: Priority level
            metadata: Additional metadata
        """
        # Create task
        task = LearningTask(
            content=content,
            source=source,
            priority=priority,
            metadata=metadata
        )
        
        # Add to queue
        priority_value = self._get_priority_value(priority)
        self.task_queue.put((priority_value, task))
        
        # Save tasks
        self._save_tasks()
        
        logger.info(f"Added learning task from {source} with priority {priority}")
    
    def learn_from_file(
        self, 
        file_path: str,
        priority: str = LearningPriority.MEDIUM,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Learn from a text file
        
        Args:
            file_path: Path to text file
            priority: Priority level
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add metadata about the file
            meta = metadata or {}
            meta["filename"] = os.path.basename(file_path)
            meta["file_size"] = os.path.getsize(file_path)
            
            # Add learning task
            self.add_learning_task(
                content=content,
                source=file_path,
                priority=priority,
                metadata=meta
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error learning from file: {str(e)}")
            return False
    
    def learn_from_conversation(
        self,
        conversation_id: str,
        priority: str = LearningPriority.MEDIUM
    ) -> bool:
        """
        Learn from a conversation
        
        Args:
            conversation_id: ID of the conversation
            priority: Priority level
            
        Returns:
            Success status
        """
        try:
            # Get conversation from communication system
            conversation = communication_system.get_conversation(conversation_id)
            if not conversation:
                logger.error(f"Conversation not found: {conversation_id}")
                return False
            
            # Extract messages
            messages = conversation.get("messages", [])
            if not messages:
                logger.warning(f"No messages in conversation: {conversation_id}")
                return False
            
            # Combine messages into content
            content = f"Conversation {conversation_id} ({conversation.get('topic', 'Untitled')})\n\n"
            
            for msg in messages:
                role = msg.get("from_model", "unknown")
                message_type = msg.get("message_type", "message")
                message_content = msg.get("content", "")
                
                content += f"[{role} - {message_type}]\n{message_content}\n\n"
            
            # Add learning task
            self.add_learning_task(
                content=content,
                source=f"conversation:{conversation_id}",
                priority=priority,
                metadata={
                    "conversation_id": conversation_id,
                    "topic": conversation.get("topic"),
                    "participants": conversation.get("participants", []),
                    "message_count": len(messages)
                }
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error learning from conversation: {str(e)}")
            return False
    
    def get_learning_status(self) -> Dict[str, Any]:
        """
        Get learning system status
        
        Returns:
            Status information
        """
        return {
            "auto_learn": self.auto_learn,
            "pending_tasks": self.task_queue.qsize(),
            "completed_tasks": len(self.task_history),
            "learning_thread_active": (
                self.learning_thread is not None and 
                self.learning_thread.is_alive()
            )
        }

# Initialize self-learning system
self_learning_system = SelfLearningSystem()