#!/usr/bin/env python3
"""
Agent System for Seren AI (OpenManus Integration)

This module implements an agentic system inspired by OpenManus architecture,
providing autonomous task planning, execution, and debugging capabilities.
It integrates with our existing model infrastructure to create a truly autonomous
software development system.
"""

import os
import sys
import json
import time
import uuid
import logging
import traceback
import datetime
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Agent] %(message)s"
)
logger = logging.getLogger("agent_system")

# Import conditional dependencies
try:
    # Try to import from ai_core modules
    from model_server import ModelType, MessageType
    from neurosymbolic_reasoning import ReasoningSystem, ReasoningStrategy
    from liquid_neural_network import LiquidNeuralNetwork
    from metacognition import MetacognitiveSystem
    from knowledge_library import KnowledgeLibrary
    
    HAS_CORE_MODULES = True
    logger.info("Successfully imported core AI modules")
except ImportError as e:
    logger.warning(f"Could not import some core modules: {e}")
    HAS_CORE_MODULES = False

# =============================================================================
# Agent System Components
# =============================================================================

class AgentRole:
    """Possible roles for agents in the system"""
    PLANNER = "planner"         # High-level planning and task decomposition
    CODER = "coder"             # Writing and modifying code
    TESTER = "tester"           # Testing and validation
    REVIEWER = "reviewer"       # Code review and quality assurance
    FIXER = "fixer"             # Bug fixing and debugging
    RESEARCHER = "researcher"   # Research and information gathering
    INTEGRATOR = "integrator"   # Integrating components and managing dependencies
    
class AgentState:
    """Possible states for agents"""
    IDLE = "idle"               # Agent is not active
    PLANNING = "planning"       # Agent is planning next steps
    EXECUTING = "executing"     # Agent is executing a task
    WAITING = "waiting"         # Agent is waiting for input or resources
    REVIEWING = "reviewing"     # Agent is reviewing results
    DEBUGGING = "debugging"     # Agent is debugging issues
    LEARNING = "learning"       # Agent is updating its knowledge

class TaskStatus:
    """Possible statuses for tasks"""
    PENDING = "pending"         # Task is created but not started
    IN_PROGRESS = "in_progress" # Task is currently being executed
    BLOCKED = "blocked"         # Task is blocked by dependencies or resources
    COMPLETED = "completed"     # Task is successfully completed
    FAILED = "failed"           # Task failed to complete

class TaskPriority:
    """Priority levels for tasks"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MemoryType:
    """Types of agent memory"""
    SHORT_TERM = "short_term"   # Recent actions and observations
    WORKING = "working"         # Current context and active information
    LONG_TERM = "long_term"     # Persistent knowledge and experience
    EPISODIC = "episodic"       # Specific sequences of events or experiences

# =============================================================================
# Core Data Structures
# =============================================================================

class Task:
    """Represents a task to be executed by an agent"""
    
    def __init__(self, 
                 task_id: str = None,
                 name: str = "",
                 description: str = "",
                 role: str = AgentRole.PLANNER,
                 dependencies: List[str] = None,
                 priority: str = TaskPriority.MEDIUM,
                 parent_id: str = None,
                 context: Dict[str, Any] = None):
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.role = role
        self.dependencies = dependencies or []
        self.status = TaskStatus.PENDING
        self.priority = priority
        self.parent_id = parent_id
        self.context = context or {}
        self.created_at = datetime.datetime.now().isoformat()
        self.started_at = None
        self.completed_at = None
        self.assigned_to = None
        self.result = None
        self.error = None
        self.subtasks = []
        self.logs = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "dependencies": self.dependencies,
            "status": self.status,
            "priority": self.priority,
            "parent_id": self.parent_id,
            "context": self.context,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "assigned_to": self.assigned_to,
            "result": self.result,
            "error": self.error,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks] if self.subtasks else [],
            "logs": self.logs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary"""
        task = cls(
            task_id=data.get("task_id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            role=data.get("role", AgentRole.PLANNER),
            dependencies=data.get("dependencies", []),
            priority=data.get("priority", TaskPriority.MEDIUM),
            parent_id=data.get("parent_id"),
            context=data.get("context", {})
        )
        task.status = data.get("status", TaskStatus.PENDING)
        task.created_at = data.get("created_at", datetime.datetime.now().isoformat())
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.assigned_to = data.get("assigned_to")
        task.result = data.get("result")
        task.error = data.get("error")
        task.logs = data.get("logs", [])
        
        # Recursively create subtasks
        if "subtasks" in data and data["subtasks"]:
            task.subtasks = [cls.from_dict(subtask) for subtask in data["subtasks"]]
        
        return task
    
    def add_log(self, message: str, level: str = "info"):
        """Add a log entry to the task"""
        self.logs.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level,
            "message": message
        })
    
    def add_subtask(self, subtask: 'Task'):
        """Add a subtask to this task"""
        subtask.parent_id = self.task_id
        self.subtasks.append(subtask)
        return subtask
    
    def mark_started(self, agent_id: str = None):
        """Mark task as started"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.datetime.now().isoformat()
        self.assigned_to = agent_id
        self.add_log(f"Task started by agent {agent_id}")
    
    def mark_completed(self, result: Any = None):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.datetime.now().isoformat()
        self.result = result
        self.add_log("Task completed successfully")
    
    def mark_failed(self, error: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.datetime.now().isoformat()
        self.error = error
        self.add_log(f"Task failed: {error}", level="error")
    
    def is_ready(self, completed_tasks: List[str]) -> bool:
        """Check if task is ready to execute (all dependencies satisfied)"""
        return all(dep in completed_tasks for dep in self.dependencies)

class Agent:
    """Autonomous agent that can execute tasks"""
    
    def __init__(self, 
                 agent_id: str = None,
                 name: str = "",
                 role: str = AgentRole.PLANNER,
                 model_type: str = "hybrid",
                 capabilities: List[str] = None,
                 config: Dict[str, Any] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"{role.capitalize()} Agent"
        self.role = role
        self.model_type = model_type
        self.capabilities = capabilities or []
        self.config = config or {}
        self.state = AgentState.IDLE
        self.current_task_id = None
        self.memory = {
            MemoryType.SHORT_TERM: [],
            MemoryType.WORKING: {},
            MemoryType.LONG_TERM: {},
            MemoryType.EPISODIC: []
        }
        self.created_at = datetime.datetime.now().isoformat()
        self.task_history = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_completion_time": 0,
            "success_rate": 1.0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "model_type": self.model_type,
            "capabilities": self.capabilities,
            "config": self.config,
            "state": self.state,
            "current_task_id": self.current_task_id,
            "memory": {
                "short_term": self.memory[MemoryType.SHORT_TERM][-10:] if self.memory[MemoryType.SHORT_TERM] else [],
                "working": self.memory[MemoryType.WORKING],
                # Exclude long-term memory as it can be large
                "episodic_count": len(self.memory[MemoryType.EPISODIC])
            },
            "created_at": self.created_at,
            "task_history": self.task_history[-10:],  # Only include recent history
            "performance_metrics": self.performance_metrics
        }
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle a specific task"""
        return task.role == self.role or task.role in self.capabilities
    
    def assign_task(self, task: Task):
        """Assign a task to this agent"""
        self.state = AgentState.PLANNING
        self.current_task_id = task.task_id
        task.mark_started(self.agent_id)
        self.remember(MemoryType.SHORT_TERM, f"Assigned task: {task.name}")
        self.remember(MemoryType.WORKING, "current_task", task.to_dict())
    
    def remember(self, memory_type: str, key_or_value: Any, value: Any = None):
        """Store information in agent's memory"""
        if memory_type == MemoryType.SHORT_TERM or memory_type == MemoryType.EPISODIC:
            # For list-based memories
            if value is None:
                self.memory[memory_type].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "content": key_or_value
                })
            else:
                self.memory[memory_type].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": key_or_value,
                    "content": value
                })
        else:
            # For dictionary-based memories
            if value is None:
                raise ValueError("Value must be provided for working/long-term memory")
            self.memory[memory_type][key_or_value] = value
    
    def recall(self, memory_type: str, key: Any = None, limit: int = 10) -> Any:
        """Retrieve information from agent's memory"""
        if memory_type == MemoryType.SHORT_TERM or memory_type == MemoryType.EPISODIC:
            if key is None:
                # Return recent memories
                return self.memory[memory_type][-limit:]
            else:
                # Filter by type
                return [m for m in self.memory[memory_type] if m.get("type") == key][-limit:]
        else:
            # For dictionary-based memories
            if key is None:
                return self.memory[memory_type]
            return self.memory[memory_type].get(key)
    
    def update_performance(self, task: Task):
        """Update agent performance metrics based on task"""
        if task.status == TaskStatus.COMPLETED:
            self.performance_metrics["tasks_completed"] += 1
        elif task.status == TaskStatus.FAILED:
            self.performance_metrics["tasks_failed"] += 1
        
        total_tasks = (self.performance_metrics["tasks_completed"] + 
                      self.performance_metrics["tasks_failed"])
        
        if total_tasks > 0:
            self.performance_metrics["success_rate"] = (
                self.performance_metrics["tasks_completed"] / total_tasks
            )
        
        # Calculate average completion time if available
        if task.started_at and task.completed_at:
            start_time = datetime.datetime.fromisoformat(task.started_at)
            end_time = datetime.datetime.fromisoformat(task.completed_at)
            duration = (end_time - start_time).total_seconds()
            
            # Update average
            old_avg = self.performance_metrics["avg_completion_time"]
            old_count = total_tasks - 1  # Exclude current task
            
            if old_count == 0:
                self.performance_metrics["avg_completion_time"] = duration
            else:
                self.performance_metrics["avg_completion_time"] = (
                    (old_avg * old_count + duration) / total_tasks
                )

class AgentSystem:
    """Main agent system class coordinating all agents and tasks"""
    
    def __init__(self, system_id: str = None, config: Dict[str, Any] = None):
        self.system_id = system_id or str(uuid.uuid4())
        self.config = config or {}
        self.agents = {}  # agent_id -> Agent
        self.tasks = {}   # task_id -> Task
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.task_queue = []  # List of task_ids prioritized
        self.created_at = datetime.datetime.now().isoformat()
        self.logs = []
        self.active = True
        
        # Used for integration with existing models
        self.model_processors = {}
        
        # Initialize reasoning and metacognitive systems if available
        self.reasoning_system = None
        self.metacognitive_system = None
        self.knowledge_library = None
        
        if HAS_CORE_MODULES:
            try:
                self.reasoning_system = ReasoningSystem()
                self.metacognitive_system = MetacognitiveSystem()
                self.knowledge_library = KnowledgeLibrary()
                logger.info("Initialized reasoning and metacognitive systems")
            except Exception as e:
                logger.error(f"Failed to initialize AI systems: {e}")
    
    def register_agent(self, agent: Agent) -> str:
        """Register an agent with the system"""
        self.agents[agent.agent_id] = agent
        self.log(f"Registered agent: {agent.name} (ID: {agent.agent_id})")
        return agent.agent_id
    
    def create_task(self, 
                   name: str,
                   description: str,
                   role: str = AgentRole.PLANNER,
                   dependencies: List[str] = None,
                   priority: str = TaskPriority.MEDIUM,
                   parent_id: str = None,
                   context: Dict[str, Any] = None) -> Task:
        """Create a new task"""
        task = Task(
            name=name,
            description=description,
            role=role,
            dependencies=dependencies,
            priority=priority,
            parent_id=parent_id,
            context=context
        )
        self.tasks[task.task_id] = task
        self.task_queue.append(task.task_id)
        
        # Sort task queue by priority
        self._prioritize_tasks()
        
        self.log(f"Created task: {task.name} (ID: {task.task_id})")
        return task
    
    def add_existing_task(self, task: Task) -> str:
        """Add an existing task to the system"""
        self.tasks[task.task_id] = task
        if task.status == TaskStatus.PENDING:
            self.task_queue.append(task.task_id)
            self._prioritize_tasks()
        elif task.status == TaskStatus.COMPLETED:
            self.completed_tasks.add(task.task_id)
        elif task.status == TaskStatus.FAILED:
            self.failed_tasks.add(task.task_id)
        
        self.log(f"Added existing task: {task.name} (ID: {task.task_id}, Status: {task.status})")
        return task.task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def _prioritize_tasks(self):
        """Sort task queue by priority"""
        priority_values = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3
        }
        
        def task_priority(task_id):
            task = self.tasks.get(task_id)
            if not task:
                return 999  # Tasks that don't exist go to the end
            return priority_values.get(task.priority, 999)
        
        self.task_queue.sort(key=task_priority)
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next available task from the queue"""
        while self.task_queue:
            task_id = self.task_queue[0]
            task = self.tasks.get(task_id)
            
            if not task:
                # Task doesn't exist, remove from queue
                self.task_queue.pop(0)
                continue
                
            if task.status != TaskStatus.PENDING:
                # Task is already being processed or completed
                self.task_queue.pop(0)
                continue
                
            if not task.is_ready(self.completed_tasks):
                # Dependencies not satisfied, skip for now
                # Move to end of queue
                self.task_queue.pop(0)
                self.task_queue.append(task_id)
                continue
                
            # Found a valid task
            return task
        
        return None
    
    def assign_next_task(self) -> Tuple[Optional[Agent], Optional[Task]]:
        """Find the next task and assign it to the most suitable agent"""
        task = self.get_next_task()
        if not task:
            return None, None
        
        # Find the most suitable agent based on role and performance
        suitable_agents = [
            agent for agent in self.agents.values()
            if agent.can_handle_task(task) and agent.state == AgentState.IDLE
        ]
        
        if not suitable_agents:
            return None, None
        
        # Select agent with best performance for this role
        selected_agent = max(
            suitable_agents,
            key=lambda a: a.performance_metrics["success_rate"]
        )
        
        # Assign task to agent
        selected_agent.assign_task(task)
        
        # Remove task from queue
        self.task_queue.remove(task.task_id)
        
        self.log(f"Assigned task '{task.name}' to agent '{selected_agent.name}'")
        
        return selected_agent, task
    
    def complete_task(self, task_id: str, result: Any = None):
        """Mark a task as completed"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        task.mark_completed(result)
        self.completed_tasks.add(task_id)
        
        # Update agent metrics
        if task.assigned_to and task.assigned_to in self.agents:
            agent = self.agents[task.assigned_to]
            agent.state = AgentState.IDLE
            agent.current_task_id = None
            agent.task_history.append(task_id)
            agent.update_performance(task)
        
        self.log(f"Completed task: {task.name} (ID: {task_id})")
        
        # Process any waiting tasks that depended on this one
        self._check_dependent_tasks()
        
        return task
    
    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        task.mark_failed(error)
        self.failed_tasks.add(task_id)
        
        # Update agent metrics
        if task.assigned_to and task.assigned_to in self.agents:
            agent = self.agents[task.assigned_to]
            agent.state = AgentState.IDLE
            agent.current_task_id = None
            agent.task_history.append(task_id)
            agent.update_performance(task)
        
        self.log(f"Failed task: {task.name} (ID: {task_id}). Error: {error}", level="error")
        
        return task
    
    def _check_dependent_tasks(self):
        """Check if any tasks are now ready due to completed dependencies"""
        # This helps the system to immediately identify when tasks become available
        task_queue_changed = False
        
        for task_id in list(self.task_queue):  # Use a copy to avoid modification during iteration
            task = self.tasks.get(task_id)
            if not task:
                continue
                
            if task.is_ready(self.completed_tasks) and task.status == TaskStatus.BLOCKED:
                # Task was blocked but is now ready
                task.status = TaskStatus.PENDING
                task_queue_changed = True
                self.log(f"Task now ready: {task.name} (ID: {task_id})")
        
        if task_queue_changed:
            # Re-prioritize if anything changed
            self._prioritize_tasks()
    
    def log(self, message: str, level: str = "info"):
        """Add a log entry to the system"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.debug(message)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the agent system"""
        return {
            "system_id": self.system_id,
            "created_at": self.created_at,
            "active": self.active,
            "agents": {
                agent_id: agent.to_dict() 
                for agent_id, agent in self.agents.items()
            },
            "task_counts": {
                "total": len(self.tasks),
                "pending": len(self.task_queue),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "in_progress": sum(1 for t in self.tasks.values() 
                                  if t.status == TaskStatus.IN_PROGRESS)
            },
            "recent_logs": self.logs[-10:] if self.logs else []
        }
    
    def register_model_processor(self, model_type: str, processor_fn: Callable):
        """Register a processor function for a specific model type"""
        self.model_processors[model_type] = processor_fn
        self.log(f"Registered model processor for {model_type}")
    
    def execute_with_model(self, 
                          model_type: str, 
                          prompt: str, 
                          role: str = None, 
                          options: Dict[str, Any] = None) -> Any:
        """Execute a prompt using a specific model type"""
        processor = self.model_processors.get(model_type)
        if not processor:
            raise ValueError(f"No processor registered for model: {model_type}")
        
        options = options or {}
        if role:
            options["role"] = role
            
        return processor(prompt, options)

# =============================================================================
# OpenManus Integration with Existing Models
# =============================================================================

class IntegratedAgentSystem:
    """Integrates OpenManus agent system with existing model infrastructure"""
    
    def __init__(self, model_server_interface=None):
        self.agent_system = AgentSystem()
        self.model_server = model_server_interface
        
        # Initialize default agents
        self._initialize_default_agents()
        
        # Register model processors
        self._register_model_processors()
        
        logger.info("Initialized integrated agent system with OpenManus architecture")
    
    def _initialize_default_agents(self):
        """Initialize default agents for the system"""
        # Planner agent (uses Qwen for planning and reasoning)
        planner = Agent(
            name="Task Planner",
            role=AgentRole.PLANNER,
            model_type="qwen2.5-7b-omni",
            capabilities=[AgentRole.RESEARCHER]
        )
        self.agent_system.register_agent(planner)
        
        # Coder agent (uses OlympicCoder for implementation)
        coder = Agent(
            name="Code Generator",
            role=AgentRole.CODER,
            model_type="olympiccoder-7b",
            capabilities=[AgentRole.INTEGRATOR]
        )
        self.agent_system.register_agent(coder)
        
        # Tester agent (uses hybrid approach)
        tester = Agent(
            name="Code Tester",
            role=AgentRole.TESTER,
            model_type="hybrid",
            capabilities=[AgentRole.FIXER]
        )
        self.agent_system.register_agent(tester)
        
        # Reviewer agent (uses Qwen for high-level reviewing)
        reviewer = Agent(
            name="Code Reviewer",
            role=AgentRole.REVIEWER,
            model_type="qwen2.5-7b-omni",
            capabilities=[]
        )
        self.agent_system.register_agent(reviewer)
        
        # Fixer agent (uses OlympicCoder for debugging)
        fixer = Agent(
            name="Bug Fixer",
            role=AgentRole.FIXER,
            model_type="olympiccoder-7b",
            capabilities=[]
        )
        self.agent_system.register_agent(fixer)
    
    def _register_model_processors(self):
        """Register model processors for different model types"""
        # Register processor for Qwen
        self.agent_system.register_model_processor(
            "qwen2.5-7b-omni", 
            self._process_with_qwen
        )
        
        # Register processor for OlympicCoder
        self.agent_system.register_model_processor(
            "olympiccoder-7b",
            self._process_with_olympic
        )
        
        # Register processor for hybrid approach
        self.agent_system.register_model_processor(
            "hybrid",
            self._process_with_hybrid
        )
    
    def _process_with_qwen(self, prompt: str, options: Dict[str, Any] = None) -> Any:
        """Process a prompt with the Qwen model"""
        options = options or {}
        
        # If we have a model server interface, use it
        if self.model_server:
            try:
                return self.model_server.process_message({
                    "type": "request",
                    "model": "qwen2.5-7b-omni",
                    "role": options.get("role", "architect"),
                    "content": {
                        "prompt": prompt,
                        **options
                    },
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.error(f"Error processing with Qwen model: {e}")
                logger.error(traceback.format_exc())
        
        # Fallback to simple text generation
        logger.warning("Using fallback text generation for Qwen")
        return self._fallback_text_generation(prompt, "qwen2.5-7b-omni", options)
    
    def _process_with_olympic(self, prompt: str, options: Dict[str, Any] = None) -> Any:
        """Process a prompt with the OlympicCoder model"""
        options = options or {}
        
        # If we have a model server interface, use it
        if self.model_server:
            try:
                return self.model_server.process_message({
                    "type": "request",
                    "model": "olympiccoder-7b",
                    "role": options.get("role", "builder"),
                    "content": {
                        "prompt": prompt,
                        **options
                    },
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.error(f"Error processing with OlympicCoder model: {e}")
                logger.error(traceback.format_exc())
        
        # Fallback to simple text generation
        logger.warning("Using fallback text generation for OlympicCoder")
        return self._fallback_text_generation(prompt, "olympiccoder-7b", options)
    
    def _process_with_hybrid(self, prompt: str, options: Dict[str, Any] = None) -> Any:
        """Process a prompt with both models and combine results"""
        options = options or {}
        
        # Try processing with both models
        qwen_result = self._process_with_qwen(prompt, options)
        olympic_result = self._process_with_olympic(prompt, options)
        
        # Combine results based on role
        role = options.get("role", "planner")
        if role == AgentRole.PLANNER or role == AgentRole.REVIEWER:
            # For planning and reviewing, favor Qwen
            primary_result = qwen_result
            secondary_result = olympic_result
            primary_weight = 0.7
        elif role == AgentRole.CODER or role == AgentRole.FIXER:
            # For coding and fixing, favor OlympicCoder
            primary_result = olympic_result
            secondary_result = qwen_result
            primary_weight = 0.7
        else:
            # For other roles, balance equally
            primary_result = qwen_result
            secondary_result = olympic_result
            primary_weight = 0.5
        
        # If results are just strings, combine them
        if isinstance(primary_result, str) and isinstance(secondary_result, str):
            # Simple weighted combination for demonstration
            # In a real system, this would be more sophisticated
            combined_result = f"{primary_result}\n\n"
            combined_result += f"Additional insights:\n{secondary_result}"
            return combined_result
        
        # Otherwise return the primary result
        return primary_result
    
    def _fallback_text_generation(self, 
                                 prompt: str, 
                                 model_type: str, 
                                 options: Dict[str, Any] = None) -> str:
        """Fallback text generation when models are not available"""
        options = options or {}
        role = options.get("role", "assistant")
        
        # Return a simple response based on the role
        if role == "architect" or role == AgentRole.PLANNER:
            return "I'll help you plan this software project. [Fallback response]"
        elif role == "builder" or role == AgentRole.CODER:
            return "Here's some example code to get started: [Fallback code]"
        elif role == "tester" or role == AgentRole.TESTER:
            return "I've analyzed the code and found the following test cases: [Fallback tests]"
        elif role == "reviewer" or role == AgentRole.REVIEWER:
            return "My review of the code suggests the following improvements: [Fallback review]"
        else:
            return f"I'll help you with your request using {model_type}. [Fallback response]"
    
    def create_software_project(self, 
                              project_name: str,
                              requirements: str,
                              language: str = None,
                              framework: str = None) -> str:
        """Create a new software project using the agent system"""
        # Create the main project task
        project_task = self.agent_system.create_task(
            name=f"Create {project_name}",
            description=f"Build a complete software project for {project_name} with the following requirements: {requirements}",
            role=AgentRole.PLANNER,
            priority=TaskPriority.HIGH,
            context={
                "project_name": project_name,
                "requirements": requirements,
                "language": language,
                "framework": framework
            }
        )
        
        # Log the creation
        self.agent_system.log(f"Created new software project task: {project_name}")
        
        return project_task.task_id
    
    def get_project_status(self, project_task_id: str) -> Dict[str, Any]:
        """Get the status of a software project"""
        task = self.agent_system.get_task(project_task_id)
        if not task:
            return {
                "error": f"Project task not found: {project_task_id}",
                "status": "not_found"
            }
        
        # Count subtasks by status
        subtask_counts = {
            "total": len(task.subtasks),
            "pending": len([s for s in task.subtasks if s.status == TaskStatus.PENDING]),
            "in_progress": len([s for s in task.subtasks if s.status == TaskStatus.IN_PROGRESS]),
            "completed": len([s for s in task.subtasks if s.status == TaskStatus.COMPLETED]),
            "failed": len([s for s in task.subtasks if s.status == TaskStatus.FAILED])
        }
        
        # Calculate progress percentage
        progress = 0
        if subtask_counts["total"] > 0:
            progress = (subtask_counts["completed"] / subtask_counts["total"]) * 100
        
        return {
            "project_id": project_task_id,
            "project_name": task.context.get("project_name", task.name),
            "status": task.status,
            "progress": progress,
            "subtask_counts": subtask_counts,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "requirements": task.context.get("requirements", ""),
            "language": task.context.get("language", ""),
            "framework": task.context.get("framework", ""),
            "recent_logs": task.logs[-10:] if task.logs else []
        }
    
    def run_continuous_execution(self, max_steps: int = 100) -> Dict[str, Any]:
        """Run the agent system for a fixed number of steps"""
        step_count = 0
        results = {
            "steps_executed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "events": []
        }
        
        while step_count < max_steps and self.agent_system.active:
            # Get and assign the next task
            agent, task = self.agent_system.assign_next_task()
            if not agent or not task:
                # No tasks available or no suitable agents
                break
            
            # Record the assignment
            results["events"].append({
                "step": step_count,
                "type": "task_assigned",
                "task_id": task.task_id,
                "task_name": task.name,
                "agent_id": agent.agent_id,
                "agent_name": agent.name
            })
            
            # Simulate task execution
            success, result = self._execute_task(agent, task)
            
            if success:
                # Complete the task
                self.agent_system.complete_task(task.task_id, result)
                results["tasks_completed"] += 1
                results["events"].append({
                    "step": step_count,
                    "type": "task_completed",
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name
                })
            else:
                # Fail the task
                self.agent_system.fail_task(task.task_id, result)
                results["tasks_failed"] += 1
                results["events"].append({
                    "step": step_count,
                    "type": "task_failed",
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "error": result
                })
            
            step_count += 1
        
        results["steps_executed"] = step_count
        return results
    
    def _execute_task(self, agent: Agent, task: Task) -> Tuple[bool, Any]:
        """Execute a task using the appropriate model for the agent"""
        try:
            # Get the prompt based on task and agent role
            prompt = self._generate_task_prompt(task, agent)
            
            # Execute with the appropriate model
            result = self.agent_system.execute_with_model(
                agent.model_type,
                prompt,
                role=agent.role,
                options={"task": task.to_dict()}
            )
            
            # For demonstration, just simulate success most of the time
            # In a real system, we would evaluate the result
            return True, result
            
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg
    
    def _generate_task_prompt(self, task: Task, agent: Agent) -> str:
        """Generate a prompt for a task based on the agent's role"""
        if agent.role == AgentRole.PLANNER:
            return self._generate_planner_prompt(task)
        elif agent.role == AgentRole.CODER:
            return self._generate_coder_prompt(task)
        elif agent.role == AgentRole.TESTER:
            return self._generate_tester_prompt(task)
        elif agent.role == AgentRole.REVIEWER:
            return self._generate_reviewer_prompt(task)
        elif agent.role == AgentRole.FIXER:
            return self._generate_fixer_prompt(task)
        else:
            return f"""
            Task: {task.name}
            Description: {task.description}
            
            Your role is: {agent.role}
            
            Please complete this task based on the following context:
            {json.dumps(task.context, indent=2)}
            """
    
    def _generate_planner_prompt(self, task: Task) -> str:
        """Generate a prompt for a planning task"""
        return f"""
        You are an expert software architect and planner.
        
        Task: {task.name}
        Description: {task.description}
        
        Project name: {task.context.get('project_name', 'Software Project')}
        
        Requirements:
        {task.context.get('requirements', 'No specific requirements provided.')}
        
        Language: {task.context.get('language', 'Not specified')}
        Framework: {task.context.get('framework', 'Not specified')}
        
        Your job is to:
        1. Analyze the requirements thoroughly
        2. Design a high-level architecture for the software
        3. Break down the project into specific, actionable tasks
        4. Prioritize these tasks in a logical order
        5. Provide a detailed technical specification
        
        Please provide a comprehensive project plan and architecture design.
        """
    
    def _generate_coder_prompt(self, task: Task) -> str:
        """Generate a prompt for a coding task"""
        return f"""
        You are an expert software developer.
        
        Task: {task.name}
        Description: {task.description}
        
        Project context:
        {json.dumps(task.context, indent=2)}
        
        Your job is to:
        1. Implement the code according to the specification
        2. Follow best practices for {task.context.get('language', 'the specified language')}
        3. Include appropriate comments and documentation
        4. Ensure the code is efficient, secure, and maintainable
        5. Handle potential edge cases and errors
        
        Please provide the implementation code.
        """
    
    def _generate_tester_prompt(self, task: Task) -> str:
        """Generate a prompt for a testing task"""
        return f"""
        You are an expert software tester.
        
        Task: {task.name}
        Description: {task.description}
        
        Project context:
        {json.dumps(task.context, indent=2)}
        
        Your job is to:
        1. Create comprehensive test cases for the code
        2. Test for correctness, performance, and security
        3. Identify any bugs or issues
        4. Ensure the code meets all requirements
        5. Verify edge cases and error handling
        
        Please provide the test cases and results.
        """
    
    def _generate_reviewer_prompt(self, task: Task) -> str:
        """Generate a prompt for a code review task"""
        return f"""
        You are an expert code reviewer.
        
        Task: {task.name}
        Description: {task.description}
        
        Project context:
        {json.dumps(task.context, indent=2)}
        
        Your job is to:
        1. Review the code for clarity, efficiency, and correctness
        2. Identify any potential bugs, security issues, or performance problems
        3. Suggest improvements to the code structure and organization
        4. Ensure the code follows industry best practices
        5. Verify that the code meets all requirements
        
        Please provide a detailed code review.
        """
    
    def _generate_fixer_prompt(self, task: Task) -> str:
        """Generate a prompt for a bug fixing task"""
        return f"""
        You are an expert debugger and bug fixer.
        
        Task: {task.name}
        Description: {task.description}
        
        Project context:
        {json.dumps(task.context, indent=2)}
        
        Your job is to:
        1. Identify the root cause of the bug or issue
        2. Fix the problem without introducing new issues
        3. Ensure the solution is compatible with the existing codebase
        4. Verify that the fix resolves the issue completely
        5. Document the fix and the reason for the bug
        
        Please provide the fixed code and an explanation of the issue.
        """

# =============================================================================
# Model Server Interface
# =============================================================================

class ModelServerInterface:
    """Interface to communicate with the model server"""
    
    def __init__(self):
        self.model_servers = {
            "qwen2.5-7b-omni": None,
            "olympiccoder-7b": None
        }
        self.initialized = False
    
    def initialize(self):
        """Initialize the interface to the model server"""
        try:
            # In a real implementation, this would establish 
            # a connection to the model server processes
            
            # For now, just set a flag indicating we're initialized
            self.initialized = True
            logger.info("ModelServerInterface initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ModelServerInterface: {e}")
            return False
    
    def process_message(self, message: Dict[str, Any]) -> Any:
        """Process a message using the model server"""
        if not self.initialized:
            self.initialize()
        
        model_type = message.get("model", "hybrid")
        message_type = message.get("type", "request")
        content = message.get("content", {})
        
        # Log the request
        logger.info(f"Processing {message_type} for {model_type}")
        
        # In a real implementation, this would send the message to 
        # the appropriate model server and wait for a response
        
        # For now, just return a fallback response
        if message_type == "request":
            prompt = content.get("prompt", "")
            
            # Generate a response based on the role
            role = message.get("role", "assistant")
            
            if "code" in prompt.lower() or role == "builder" or role == "coder":
                return "```python\ndef hello_world():\n    print('Hello, World!')\n\nhello_world()\n```"
            elif "architecture" in prompt.lower() or role == "architect" or role == "planner":
                return "# Architecture Design\n\n1. Frontend: React\n2. Backend: Flask\n3. Database: PostgreSQL\n\n## API Endpoints\n\n- GET /api/items\n- POST /api/items\n- GET /api/items/:id"
            elif "test" in prompt.lower() or role == "tester":
                return "```python\nimport unittest\n\nclass TestHello(unittest.TestCase):\n    def test_hello(self):\n        self.assertEqual(hello_world(), None)\n```"
            else:
                return "I'll help you with that task. Here's a response based on your requirements..."
        else:
            return {"status": "ok", "message": "Message processed"}
    
    def shutdown(self):
        """Shut down the interface"""
        logger.info("Shutting down ModelServerInterface")
        self.initialized = False

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function for testing the agent system"""
    logger.info("Starting Agent System with OpenManus architecture")
    
    # Create the model server interface
    model_server = ModelServerInterface()
    model_server.initialize()
    
    # Create the integrated agent system
    agent_system = IntegratedAgentSystem(model_server)
    
    # Example: Create a software project
    project_id = agent_system.create_software_project(
        project_name="Todo List App",
        requirements="""
        Create a web-based todo list application with the following features:
        - User registration and login
        - Ability to create, edit, and delete tasks
        - Mark tasks as complete
        - Filter tasks by status
        - Responsive design for mobile and desktop
        """,
        language="Python",
        framework="Flask"
    )
    
    # Run continuous execution for a few steps
    execution_results = agent_system.run_continuous_execution(max_steps=5)
    
    # Check the project status
    project_status = agent_system.get_project_status(project_id)
    
    # Print results
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Project Status: {project_status['status']}")
    logger.info(f"Progress: {project_status['progress']}%")
    logger.info(f"Execution Steps: {execution_results['steps_executed']}")
    logger.info(f"Tasks Completed: {execution_results['tasks_completed']}")
    
    # Clean up
    model_server.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())