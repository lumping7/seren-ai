"""
Continuous Execution System for Seren

Implements an advanced continuous execution framework allowing the
AI to run in a persistent, self-directing mode with autonomous
goal pursuit, monitoring, and adaptation.
"""

import os
import sys
import json
import logging
import time
import threading
import uuid
import queue
import datetime
import random
from enum import Enum, auto
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local imports
try:
    from ai_core.neurosymbolic_reasoning import neurosymbolic_reasoning, ReasoningStrategy
    has_reasoning = True
except ImportError:
    has_reasoning = False
    logging.warning("Neurosymbolic reasoning not available. Continuous execution will operate with limited reasoning.")

try:
    from ai_core.metacognition import metacognitive_system, MetacognitiveLevel, CognitiveOperation
    has_metacognition = True
except ImportError:
    has_metacognition = False
    logging.warning("Metacognition not available. Continuous execution will operate with limited self-awareness.")

try:
    from ai_core.knowledge.library import knowledge_library
    has_knowledge_lib = True
except ImportError:
    has_knowledge_lib = False
    logging.warning("Knowledge library not available. Continuous execution will operate with limited knowledge.")

try:
    from ai_core.liquid_neural_network import continuous_learning_system
    has_learning = True
except ImportError:
    has_learning = False
    logging.warning("Liquid neural network not available. Continuous execution will operate with limited learning.")

try:
    from ai_core.model_communication import communication_system, MessageType
    has_communication = True
except ImportError:
    has_communication = False
    logging.warning("Model communication not available. Continuous execution will operate with limited communication.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ========================= Continuous Execution Components =========================

class ExecutionPhase(Enum):
    """Phases of continuous execution"""
    INITIALIZATION = auto()    # Setting up, loading context
    PERCEPTION = auto()        # Processing inputs and context
    PLANNING = auto()          # Planning actions and goals
    ACTION = auto()            # Executing actions
    REFLECTION = auto()        # Reflecting on results
    ADAPTATION = auto()        # Adapting based on reflection
    IDLE = auto()              # Waiting for new inputs

class ExecutionMode(Enum):
    """Modes of continuous execution"""
    AUTONOMOUS = auto()        # Full self-directing operation
    SEMI_AUTONOMOUS = auto()   # Partial self-direction with oversight
    INTERACTIVE = auto()       # Interactive operation with user
    FOCUSED = auto()           # Focus on specific task/domain
    LOW_POWER = auto()         # Reduced resource usage
    EMERGENCY = auto()         # Critical operation mode

class GoalStatus(Enum):
    """Status of execution goals"""
    PENDING = auto()           # Not yet started
    ACTIVE = auto()            # Currently being pursued
    COMPLETED = auto()         # Successfully completed
    FAILED = auto()            # Failed to complete
    SUSPENDED = auto()         # Temporarily suspended
    DELEGATED = auto()         # Delegated to another agent/component

class Goal:
    """Representation of an execution goal"""
    
    def __init__(
        self,
        name: str,
        description: str,
        priority: float = 0.5,
        deadline: Optional[float] = None,
        dependencies: List[str] = None,
        criteria: Dict[str, Any] = None
    ):
        """
        Initialize a goal
        
        Args:
            name: Goal name
            description: Goal description
            priority: Goal priority (0-1)
            deadline: Deadline timestamp or None
            dependencies: IDs of goals this depends on
            criteria: Success criteria
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.priority = priority
        self.deadline = deadline
        self.dependencies = dependencies or []
        self.criteria = criteria or {}
        
        self.status = GoalStatus.PENDING
        self.progress = 0.0  # 0-1
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.result = None
        
        # Sub-goals
        self.sub_goals = []  # List of Goals
    
    def update_progress(self, progress: float, status: GoalStatus = None):
        """
        Update goal progress
        
        Args:
            progress: Progress value (0-1)
            status: New status or None to keep current
        """
        self.progress = max(0.0, min(1.0, progress))
        
        if status:
            self.status = status
            
            if status == GoalStatus.ACTIVE and not self.started_at:
                self.started_at = time.time()
            elif status in [GoalStatus.COMPLETED, GoalStatus.FAILED] and not self.completed_at:
                self.completed_at = time.time()
    
    def time_remaining(self) -> Optional[float]:
        """Get time remaining until deadline in seconds"""
        if not self.deadline:
            return None
        
        return max(0.0, self.deadline - time.time())
    
    def urgency(self) -> float:
        """Calculate urgency based on deadline and priority"""
        if not self.deadline:
            return self.priority
        
        time_left = self.time_remaining()
        if time_left <= 0:
            return 1.0  # Maximum urgency if past deadline
        
        # Urgency increases as deadline approaches
        deadline_factor = 1.0 / (1.0 + time_left / 3600)  # Normalize with 1 hour scale
        return 0.5 * self.priority + 0.5 * deadline_factor
    
    def is_blocked(self, completed_goal_ids: Set[str]) -> bool:
        """
        Check if goal is blocked by dependencies
        
        Args:
            completed_goal_ids: Set of completed goal IDs
            
        Returns:
            Whether the goal is blocked
        """
        return any(dep not in completed_goal_ids for dep in self.dependencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "deadline": self.deadline,
            "dependencies": self.dependencies,
            "criteria": self.criteria,
            "status": self.status.name,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "sub_goals": [sg.to_dict() for sg in self.sub_goals]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """Create from dictionary"""
        goal = cls(
            name=data.get("name", "Unknown"),
            description=data.get("description", ""),
            priority=data.get("priority", 0.5),
            deadline=data.get("deadline"),
            dependencies=data.get("dependencies", []),
            criteria=data.get("criteria", {})
        )
        
        goal.id = data.get("id", goal.id)
        goal.status = GoalStatus[data.get("status", "PENDING")]
        goal.progress = data.get("progress", 0.0)
        goal.created_at = data.get("created_at", goal.created_at)
        goal.started_at = data.get("started_at")
        goal.completed_at = data.get("completed_at")
        goal.result = data.get("result")
        
        # Recursively create sub-goals
        for sg_data in data.get("sub_goals", []):
            goal.sub_goals.append(Goal.from_dict(sg_data))
        
        return goal
    
    def __str__(self) -> str:
        status_str = f"{self.status.name}, {self.progress:.1%}"
        deadline_str = f", deadline in {(self.deadline - time.time()) / 3600:.1f}h" if self.deadline else ""
        return f"Goal({self.name}, priority={self.priority:.1f}, status={status_str}{deadline_str})"

class Action:
    """Representation of an execution action"""
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        goal_id: Optional[str] = None
    ):
        """
        Initialize an action
        
        Args:
            name: Action name
            description: Action description
            function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            goal_id: Associated goal ID
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}
        self.goal_id = goal_id
        
        self.status = "pending"  # pending, running, completed, failed
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
    
    def execute(self) -> Any:
        """
        Execute the action
        
        Returns:
            Action result
        """
        self.status = "running"
        self.started_at = time.time()
        
        try:
            self.result = self.function(*self.args, **self.kwargs)
            self.status = "completed"
        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            logger.error(f"Action '{self.name}' failed: {str(e)}")
        
        self.completed_at = time.time()
        return self.result
    
    def execution_time(self) -> Optional[float]:
        """Get execution time in seconds"""
        if not self.started_at or not self.completed_at:
            return None
        
        return self.completed_at - self.started_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal_id": self.goal_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error
        }
    
    def __str__(self) -> str:
        return f"Action({self.name}, status={self.status})"

class ExecutionContext:
    """Context for continuous execution"""
    
    def __init__(
        self,
        name: str = "main",
        mode: ExecutionMode = ExecutionMode.SEMI_AUTONOMOUS,
        max_history: int = 100
    ):
        """
        Initialize execution context
        
        Args:
            name: Context name
            mode: Execution mode
            max_history: Maximum history items to keep
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.mode = mode
        self.max_history = max_history
        
        self.created_at = time.time()
        self.last_active = time.time()
        
        # Runtime data
        self.variables = {}  # name -> value
        self.state = {}      # name -> value
        
        # History
        self.action_history = []  # List of executed Actions
        self.phase_history = []   # List of {phase, timestamp, duration}
        self.event_history = []   # List of {type, description, timestamp}
    
    def set_variable(self, name: str, value: Any):
        """Set runtime variable"""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get runtime variable"""
        return self.variables.get(name, default)
    
    def set_state(self, name: str, value: Any):
        """Set state value"""
        self.state[name] = value
    
    def get_state(self, name: str, default: Any = None) -> Any:
        """Get state value"""
        return self.state.get(name, default)
    
    def record_action(self, action: Action):
        """Record executed action"""
        self.action_history.append(action)
        self.last_active = time.time()
        
        # Limit history size
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history:]
    
    def record_phase(self, phase: ExecutionPhase, duration: float):
        """Record execution phase"""
        entry = {
            "phase": phase,
            "timestamp": time.time(),
            "duration": duration
        }
        
        self.phase_history.append(entry)
        self.last_active = time.time()
        
        # Limit history size
        if len(self.phase_history) > self.max_history:
            self.phase_history = self.phase_history[-self.max_history:]
    
    def record_event(self, event_type: str, description: str):
        """Record execution event"""
        entry = {
            "type": event_type,
            "description": description,
            "timestamp": time.time()
        }
        
        self.event_history.append(entry)
        self.last_active = time.time()
        
        # Limit history size
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
    
    def idle_time(self) -> float:
        """Get idle time in seconds"""
        return time.time() - self.last_active
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode.name,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "variables": {k: str(v) for k, v in self.variables.items()},
            "state": self.state,
            "action_history": [a.to_dict() for a in self.action_history[-10:]],  # Last 10
            "phase_history": self.phase_history[-10:],  # Last 10
            "event_history": self.event_history[-10:]  # Last 10
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionContext':
        """Create from dictionary"""
        context = cls(
            name=data.get("name", "main"),
            mode=ExecutionMode[data.get("mode", "SEMI_AUTONOMOUS")]
        )
        
        context.id = data.get("id", context.id)
        context.created_at = data.get("created_at", context.created_at)
        context.last_active = data.get("last_active", context.last_active)
        
        # Load state
        context.state = data.get("state", {})
        
        return context
    
    def __str__(self) -> str:
        return f"ExecutionContext({self.name}, mode={self.mode.name}, vars={len(self.variables)})"

class ExecutionMonitor:
    """Monitoring system for continuous execution"""
    
    def __init__(
        self,
        sampling_interval: float = 1.0,  # seconds
        history_size: int = 100
    ):
        """
        Initialize execution monitor
        
        Args:
            sampling_interval: Sampling interval in seconds
            history_size: Maximum history items to keep
        """
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        
        # Resource usage history
        self.resource_history = []  # List of {timestamp, cpu, memory, actions_per_second}
        
        # Performance metrics
        self.metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "actions_per_second": 0.0,
            "success_rate": 1.0,
            "avg_action_time": 0.0,
            "goal_completion_rate": 1.0
        }
        
        # Warnings and alerts
        self.active_warnings = []  # List of {type, message, timestamp}
        self.alert_thresholds = {
            "cpu_usage": 0.9,           # 90% CPU usage
            "memory_usage": 0.9,        # 90% memory usage
            "idle_time": 300,           # 5 minutes idle
            "action_failure_rate": 0.2  # 20% action failures
        }
        
        # Monitor state
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start execution monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started execution monitoring")
    
    def stop_monitoring(self):
        """Stop execution monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        
        logger.info("Stopped execution monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Sample resource usage
                self._sample_resources()
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep for the sampling interval
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in execution monitoring: {str(e)}")
                time.sleep(self.sampling_interval * 2)  # Longer sleep on error
    
    def _sample_resources(self):
        """Sample current resource usage"""
        try:
            # This is a simplified implementation
            # Real implementation would use psutil or similar
            
            # Estimate CPU usage
            cpu_usage = 0.3 + random.random() * 0.2  # Random between 30-50%
            
            # Estimate memory usage
            memory_usage = 0.4 + random.random() * 0.1  # Random between 40-50%
            
            # Estimate actions per second
            actions_per_second = 5 + random.random() * 5  # Random between 5-10
            
            # Update metrics
            self.metrics["cpu_usage"] = cpu_usage
            self.metrics["memory_usage"] = memory_usage
            self.metrics["actions_per_second"] = actions_per_second
            
            # Record history
            sample = {
                "timestamp": time.time(),
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "actions_per_second": actions_per_second
            }
            
            self.resource_history.append(sample)
            
            # Limit history size
            if len(self.resource_history) > self.history_size:
                self.resource_history = self.resource_history[-self.history_size:]
        
        except Exception as e:
            logger.error(f"Error sampling resources: {str(e)}")
    
    def _check_alerts(self):
        """Check for alert conditions"""
        # Check CPU usage
        if self.metrics["cpu_usage"] > self.alert_thresholds["cpu_usage"]:
            self._add_warning("high_cpu_usage", f"CPU usage is high: {self.metrics['cpu_usage']:.1%}")
        
        # Check memory usage
        if self.metrics["memory_usage"] > self.alert_thresholds["memory_usage"]:
            self._add_warning("high_memory_usage", f"Memory usage is high: {self.metrics['memory_usage']:.1%}")
        
        # Check action failure rate
        if 1.0 - self.metrics["success_rate"] > self.alert_thresholds["action_failure_rate"]:
            failure_rate = 1.0 - self.metrics["success_rate"]
            self._add_warning("high_failure_rate", f"Action failure rate is high: {failure_rate:.1%}")
        
        # Remove expired warnings
        current_time = time.time()
        self.active_warnings = [w for w in self.active_warnings 
                               if current_time - w["timestamp"] < 300]  # 5 min expiry
    
    def _add_warning(self, warning_type: str, message: str):
        """Add new warning"""
        # Check if this type already exists
        for warning in self.active_warnings:
            if warning["type"] == warning_type:
                # Update existing warning
                warning["message"] = message
                warning["timestamp"] = time.time()
                warning["count"] = warning.get("count", 1) + 1
                return
        
        # Add new warning
        self.active_warnings.append({
            "type": warning_type,
            "message": message,
            "timestamp": time.time(),
            "count": 1
        })
        
        logger.warning(f"Execution warning: {message}")
    
    def update_metrics(
        self,
        action_success: Optional[bool] = None,
        action_time: Optional[float] = None,
        goal_completed: Optional[bool] = None
    ):
        """
        Update performance metrics
        
        Args:
            action_success: Whether an action succeeded
            action_time: Execution time of an action
            goal_completed: Whether a goal was completed
        """
        # Update success rate
        if action_success is not None:
            # Use exponential moving average
            alpha = 0.1  # Weight for new observation
            success_value = 1.0 if action_success else 0.0
            self.metrics["success_rate"] = (1 - alpha) * self.metrics["success_rate"] + alpha * success_value
        
        # Update average action time
        if action_time is not None and action_time > 0:
            if self.metrics["avg_action_time"] == 0:
                self.metrics["avg_action_time"] = action_time
            else:
                # Use exponential moving average
                alpha = 0.1  # Weight for new observation
                self.metrics["avg_action_time"] = (1 - alpha) * self.metrics["avg_action_time"] + alpha * action_time
        
        # Update goal completion rate
        if goal_completed is not None:
            # Use exponential moving average
            alpha = 0.05  # Lower weight for goals (less frequent)
            completion_value = 1.0 if goal_completed else 0.0
            self.metrics["goal_completion_rate"] = (1 - alpha) * self.metrics["goal_completion_rate"] + alpha * completion_value
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance report
        
        Returns:
            Performance report dictionary
        """
        report = {
            "metrics": self.metrics.copy(),
            "warnings": self.active_warnings.copy(),
            "resource_summary": self._summarize_resources(),
            "timestamp": time.time()
        }
        
        return report
    
    def _summarize_resources(self) -> Dict[str, Any]:
        """Summarize resource usage history"""
        if not self.resource_history:
            return {
                "avg_cpu": 0.0,
                "avg_memory": 0.0,
                "avg_actions_per_second": 0.0,
                "peak_cpu": 0.0,
                "peak_memory": 0.0,
                "samples": 0
            }
        
        # Calculate averages and peaks
        cpu_values = [r["cpu_usage"] for r in self.resource_history]
        memory_values = [r["memory_usage"] for r in self.resource_history]
        aps_values = [r["actions_per_second"] for r in self.resource_history]
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        avg_aps = sum(aps_values) / len(aps_values)
        
        peak_cpu = max(cpu_values)
        peak_memory = max(memory_values)
        
        return {
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
            "avg_actions_per_second": avg_aps,
            "peak_cpu": peak_cpu,
            "peak_memory": peak_memory,
            "samples": len(self.resource_history)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metrics": self.metrics,
            "warnings": self.active_warnings,
            "resource_history": self.resource_history[-10:],  # Last 10 samples
            "alert_thresholds": self.alert_thresholds
        }

# ========================= Continuous Execution Engine =========================

class ContinuousExecutionEngine:
    """
    Continuous Execution Engine for Seren
    
    Implements an advanced execution framework allowing the AI to run
    in a persistent, self-directing mode with autonomous goal pursuit,
    monitoring, and adaptation.
    """
    
    def __init__(
        self,
        name: str = "Seren Execution Engine",
        storage_path: str = None,
        max_goals: int = 10,
        max_contexts: int = 5,
        execute_in_thread: bool = True
    ):
        """
        Initialize continuous execution engine
        
        Args:
            name: Engine name
            storage_path: Path to store execution data
            max_goals: Maximum active goals
            max_contexts: Maximum execution contexts
            execute_in_thread: Whether to execute actions in separate threads
        """
        self.name = name
        self.max_goals = max_goals
        self.max_contexts = max_contexts
        self.execute_in_thread = execute_in_thread
        
        # Set storage path
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(
                parent_dir, "data", "execution"
            )
            os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize execution components
        self.contexts = {}  # id -> ExecutionContext
        self.goals = {}     # id -> Goal
        self.action_queue = queue.Queue()  # Queue of pending Actions
        
        # Execution monitoring
        self.monitor = ExecutionMonitor()
        
        # Execution state
        self.current_phase = ExecutionPhase.IDLE
        self.running = False
        self.execution_thread = None
        self.phase_lock = threading.Lock()
        
        # Create main context
        self.create_execution_context("main", ExecutionMode.SEMI_AUTONOMOUS)
        
        # Set default goals
        self._initialize_default_goals()
        
        logger.info(f"Continuous execution engine '{name}' initialized")
    
    def _initialize_default_goals(self):
        """Initialize with some default goals"""
        self.add_goal(Goal(
            name="system_monitoring",
            description="Monitor system health and performance",
            priority=0.8,
            criteria={"duration": "continuous"}
        ))
        
        self.add_goal(Goal(
            name="knowledge_maintenance",
            description="Maintain and organize knowledge library",
            priority=0.6,
            criteria={"frequency": "daily"}
        ))
        
        self.add_goal(Goal(
            name="self_improvement",
            description="Continuously improve reasoning and capabilities",
            priority=0.7,
            criteria={"continuous_learning": True}
        ))
    
    def create_execution_context(
        self,
        name: str,
        mode: ExecutionMode
    ) -> str:
        """
        Create new execution context
        
        Args:
            name: Context name
            mode: Execution mode
            
        Returns:
            Context ID
        """
        # Check if we have too many contexts
        if len(self.contexts) >= self.max_contexts:
            # Find oldest idle context to replace
            oldest_id = None
            oldest_time = float('inf')
            
            for ctx_id, ctx in self.contexts.items():
                if ctx.name != "main" and ctx.idle_time() > oldest_time:
                    oldest_id = ctx_id
                    oldest_time = ctx.idle_time()
            
            # Remove oldest if found
            if oldest_id:
                del self.contexts[oldest_id]
        
        # Create new context
        context = ExecutionContext(name=name, mode=mode)
        context_id = context.id
        
        # Add to contexts
        self.contexts[context_id] = context
        
        logger.info(f"Created execution context: {name} ({context_id})")
        return context_id
    
    def get_context(self, context_id: str) -> Optional[ExecutionContext]:
        """
        Get execution context by ID
        
        Args:
            context_id: Context ID
            
        Returns:
            Execution context or None if not found
        """
        return self.contexts.get(context_id)
    
    def get_main_context(self) -> ExecutionContext:
        """
        Get main execution context
        
        Returns:
            Main execution context
        """
        for ctx in self.contexts.values():
            if ctx.name == "main":
                return ctx
        
        # Create main context if not found
        context_id = self.create_execution_context("main", ExecutionMode.SEMI_AUTONOMOUS)
        return self.contexts[context_id]
    
    def add_goal(self, goal: Goal) -> str:
        """
        Add execution goal
        
        Args:
            goal: Goal to add
            
        Returns:
            Goal ID
        """
        # Check if we have too many goals
        if len(self.goals) >= self.max_goals:
            # Find lowest priority completed goal to replace
            lowest_id = None
            lowest_priority = float('inf')
            
            for goal_id, g in self.goals.items():
                if g.status == GoalStatus.COMPLETED and g.priority < lowest_priority:
                    lowest_id = goal_id
                    lowest_priority = g.priority
            
            # Remove lowest if found
            if lowest_id:
                del self.goals[lowest_id]
        
        # Add to goals
        goal_id = goal.id
        self.goals[goal_id] = goal
        
        logger.info(f"Added goal: {goal.name} ({goal_id})")
        return goal_id
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Get goal by ID
        
        Args:
            goal_id: Goal ID
            
        Returns:
            Goal or None if not found
        """
        return self.goals.get(goal_id)
    
    def update_goal(
        self,
        goal_id: str,
        progress: Optional[float] = None,
        status: Optional[GoalStatus] = None,
        result: Any = None
    ) -> bool:
        """
        Update goal status
        
        Args:
            goal_id: Goal ID
            progress: New progress or None to keep current
            status: New status or None to keep current
            result: Goal result
            
        Returns:
            Success status
        """
        goal = self.get_goal(goal_id)
        if not goal:
            return False
        
        # Update progress
        if progress is not None:
            goal.progress = max(0.0, min(1.0, progress))
        
        # Update status
        if status:
            old_status = goal.status
            goal.status = status
            
            # Set timestamps
            if status == GoalStatus.ACTIVE and not goal.started_at:
                goal.started_at = time.time()
            elif status in [GoalStatus.COMPLETED, GoalStatus.FAILED] and not goal.completed_at:
                goal.completed_at = time.time()
            
            # Log status change
            logger.info(f"Goal '{goal.name}' changed from {old_status.name} to {status.name}")
            
            # Update monitor metrics
            if status == GoalStatus.COMPLETED:
                self.monitor.update_metrics(goal_completed=True)
            elif status == GoalStatus.FAILED:
                self.monitor.update_metrics(goal_completed=False)
        
        # Update result
        if result is not None:
            goal.result = result
        
        return True
    
    def queue_action(
        self,
        action: Action,
        execute_immediately: bool = False
    ) -> bool:
        """
        Queue action for execution
        
        Args:
            action: Action to queue
            execute_immediately: Whether to execute immediately
            
        Returns:
            Success status
        """
        try:
            # Add to queue
            self.action_queue.put(action)
            
            # Execute immediately if requested
            if execute_immediately:
                self._execute_next_action()
            
            return True
        except Exception as e:
            logger.error(f"Error queueing action: {str(e)}")
            return False
    
    def create_and_queue_action(
        self,
        name: str,
        description: str,
        function: Callable,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        goal_id: Optional[str] = None,
        execute_immediately: bool = False
    ) -> Optional[str]:
        """
        Create and queue action for execution
        
        Args:
            name: Action name
            description: Action description
            function: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            goal_id: Associated goal ID
            execute_immediately: Whether to execute immediately
            
        Returns:
            Action ID or None on failure
        """
        try:
            # Create action
            action = Action(
                name=name,
                description=description,
                function=function,
                args=args,
                kwargs=kwargs,
                goal_id=goal_id
            )
            
            # Queue action
            success = self.queue_action(action, execute_immediately)
            
            if success:
                return action.id
            else:
                return None
        except Exception as e:
            logger.error(f"Error creating action: {str(e)}")
            return None
    
    def _execute_next_action(self) -> Optional[Action]:
        """
        Execute next action in queue
        
        Returns:
            Executed action or None if queue is empty
        """
        try:
            # Get next action from queue
            if self.action_queue.empty():
                return None
            
            action = self.action_queue.get(block=False)
            
            # Check if we need to execute in thread
            if self.execute_in_thread:
                threading.Thread(
                    target=self._execute_action,
                    args=(action,),
                    daemon=True
                ).start()
                return action
            else:
                return self._execute_action(action)
        
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error executing next action: {str(e)}")
            return None
    
    def _execute_action(self, action: Action) -> Action:
        """
        Execute a specific action
        
        Args:
            action: Action to execute
            
        Returns:
            Executed action
        """
        try:
            # Get main context
            context = self.get_main_context()
            
            # Execute action
            logger.info(f"Executing action: {action.name}")
            result = action.execute()
            
            # Record in context
            context.record_action(action)
            
            # Record in monitor
            self.monitor.update_metrics(
                action_success=(action.status == "completed"),
                action_time=action.execution_time()
            )
            
            # Update associated goal if any
            if action.goal_id:
                goal = self.get_goal(action.goal_id)
                if goal:
                    # Update goal progress based on action success
                    progress_increment = 0.1 if action.status == "completed" else 0.0
                    new_progress = min(1.0, goal.progress + progress_increment)
                    
                    # Check if goal is now complete
                    new_status = None
                    if new_progress >= 1.0:
                        new_status = GoalStatus.COMPLETED
                    
                    self.update_goal(
                        goal_id=action.goal_id,
                        progress=new_progress,
                        status=new_status
                    )
            
            return action
        
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")
            
            # Update action status
            action.status = "failed"
            action.error = str(e)
            action.completed_at = time.time()
            
            return action
    
    def start(self):
        """Start continuous execution"""
        if self.running:
            return
        
        self.running = True
        self.current_phase = ExecutionPhase.INITIALIZATION
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start execution thread
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        
        logger.info("Started continuous execution")
    
    def stop(self):
        """Stop continuous execution"""
        self.running = False
        
        if self.execution_thread:
            self.execution_thread.join(timeout=1.0)
            self.execution_thread = None
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        logger.info("Stopped continuous execution")
    
    def _execution_loop(self):
        """Main execution loop"""
        while self.running:
            try:
                # Get main context
                context = self.get_main_context()
                
                # Execute current phase
                self._execute_phase(context)
                
                # Check for actions to execute
                self._execute_next_action()
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.01)
            
            except Exception as e:
                logger.error(f"Error in execution loop: {str(e)}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _execute_phase(self, context: ExecutionContext):
        """
        Execute current phase
        
        Args:
            context: Execution context
        """
        # Acquire phase lock to prevent concurrent phase execution
        if not self.phase_lock.acquire(blocking=False):
            return
        
        try:
            # Record phase start time
            phase_start = time.time()
            
            # Execute phase
            if self.current_phase == ExecutionPhase.INITIALIZATION:
                self._execute_initialization_phase(context)
                self._advance_phase(ExecutionPhase.PERCEPTION)
            
            elif self.current_phase == ExecutionPhase.PERCEPTION:
                self._execute_perception_phase(context)
                self._advance_phase(ExecutionPhase.PLANNING)
            
            elif self.current_phase == ExecutionPhase.PLANNING:
                self._execute_planning_phase(context)
                self._advance_phase(ExecutionPhase.ACTION)
            
            elif self.current_phase == ExecutionPhase.ACTION:
                self._execute_action_phase(context)
                self._advance_phase(ExecutionPhase.REFLECTION)
            
            elif self.current_phase == ExecutionPhase.REFLECTION:
                self._execute_reflection_phase(context)
                self._advance_phase(ExecutionPhase.ADAPTATION)
            
            elif self.current_phase == ExecutionPhase.ADAPTATION:
                self._execute_adaptation_phase(context)
                self._advance_phase(ExecutionPhase.PERCEPTION)
            
            elif self.current_phase == ExecutionPhase.IDLE:
                self._execute_idle_phase(context)
                
                # Stay in IDLE until activated
                if context.get_state("activation_pending", False):
                    context.set_state("activation_pending", False)
                    self._advance_phase(ExecutionPhase.PERCEPTION)
            
            # Calculate phase duration
            phase_duration = time.time() - phase_start
            
            # Record phase information
            context.record_phase(self.current_phase, phase_duration)
        
        finally:
            # Release phase lock
            self.phase_lock.release()
    
    def _advance_phase(self, next_phase: ExecutionPhase):
        """
        Advance to next execution phase
        
        Args:
            next_phase: Next execution phase
        """
        logger.debug(f"Advancing from {self.current_phase.name} to {next_phase.name}")
        self.current_phase = next_phase
    
    def _execute_initialization_phase(self, context: ExecutionContext):
        """
        Execute initialization phase
        
        Args:
            context: Execution context
        """
        # Set initialization time
        context.set_state("initialization_time", time.time())
        
        # Set execution flags
        context.set_state("pause_requested", False)
        context.set_state("emergency_mode", False)
        
        # Record initialization event
        context.record_event(
            event_type="initialization",
            description=f"Initialized execution with mode {context.mode.name}"
        )
        
        # Check if we have neurosymbolic reasoning
        if has_reasoning:
            context.set_state("has_reasoning", True)
        else:
            context.set_state("has_reasoning", False)
        
        # Check if we have metacognition
        if has_metacognition:
            context.set_state("has_metacognition", True)
        else:
            context.set_state("has_metacognition", False)
        
        # Check if we have knowledge library
        if has_knowledge_lib:
            context.set_state("has_knowledge_lib", True)
        else:
            context.set_state("has_knowledge_lib", False)
        
        # Check if we have continuous learning
        if has_learning:
            context.set_state("has_learning", True)
        else:
            context.set_state("has_learning", False)
    
    def _execute_perception_phase(self, context: ExecutionContext):
        """
        Execute perception phase
        
        Args:
            context: Execution context
        """
        # Get inputs (in real deployment, this would receive inputs from outside)
        context.set_variable("inputs", [])
        
        # Check for activation requests
        activation_source = context.get_variable("activation_source")
        if activation_source:
            context.record_event(
                event_type="activation",
                description=f"Activated by {activation_source}"
            )
            context.set_variable("activation_source", None)
        
        # Get current goals
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        pending_goals = [g for g in self.goals.values() if g.status == GoalStatus.PENDING]
        
        context.set_variable("active_goals", active_goals)
        context.set_variable("pending_goals", pending_goals)
        
        # Create goal summary
        goal_summary = {
            "active": len(active_goals),
            "pending": len(pending_goals),
            "completed": len([g for g in self.goals.values() if g.status == GoalStatus.COMPLETED]),
            "failed": len([g for g in self.goals.values() if g.status == GoalStatus.FAILED])
        }
        
        context.set_variable("goal_summary", goal_summary)
        
        # Get performance report
        performance_report = self.monitor.get_performance_report()
        context.set_variable("performance_report", performance_report)
        
        # Check emergencies
        if performance_report["warnings"]:
            # We have active warnings
            high_severity_warnings = [
                w for w in performance_report["warnings"]
                if w.get("count", 0) >= 3  # Repeated warnings
            ]
            
            if high_severity_warnings:
                context.set_state("emergency_mode", True)
                context.record_event(
                    event_type="emergency",
                    description=f"Entering emergency mode due to: {high_severity_warnings[0]['message']}"
                )
        
        # Get knowledge context (if available)
        if has_knowledge_lib and "current_query" in context.variables:
            current_query = context.get_variable("current_query")
            knowledge_context = knowledge_library.extract_context_for_query(current_query)
            context.set_variable("knowledge_context", knowledge_context)
        
        # Get cognitive state (if available)
        if has_metacognition:
            cognitive_state = metacognitive_system.cognitive_model.current_state
            context.set_variable("cognitive_state", cognitive_state)
    
    def _execute_planning_phase(self, context: ExecutionContext):
        """
        Execute planning phase
        
        Args:
            context: Execution context
        """
        # Check for emergency mode
        if context.get_state("emergency_mode", False):
            self._execute_emergency_planning(context)
            return
        
        # Get active and pending goals
        active_goals = context.get_variable("active_goals", [])
        pending_goals = context.get_variable("pending_goals", [])
        
        # Determine which pending goals to activate
        completed_goal_ids = {g.id for g in self.goals.values() if g.status == GoalStatus.COMPLETED}
        
        goals_to_activate = []
        for goal in pending_goals:
            # Check if we have capacity for more active goals
            if len(active_goals) + len(goals_to_activate) >= self.max_goals:
                break
            
            # Check if goal is blocked by dependencies
            if goal.is_blocked(completed_goal_ids):
                continue
            
            # Add to activation list
            goals_to_activate.append(goal)
        
        # Sort by urgency (highest first)
        goals_to_activate.sort(key=lambda g: g.urgency(), reverse=True)
        
        # Activate goals
        for goal in goals_to_activate[:3]:  # Activate at most 3 at once
            self.update_goal(goal.id, status=GoalStatus.ACTIVE)
            
            context.record_event(
                event_type="goal_activation",
                description=f"Activated goal: {goal.name}"
            )
        
        # Update list of active goals
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        context.set_variable("active_goals", active_goals)
        
        # Create actions for active goals
        for goal in active_goals:
            self._create_actions_for_goal(context, goal)
    
    def _execute_emergency_planning(self, context: ExecutionContext):
        """
        Execute emergency planning
        
        Args:
            context: Execution context
        """
        # Get performance report
        performance_report = context.get_variable("performance_report", {})
        
        # Identify critical issues
        critical_issues = []
        
        if "warnings" in performance_report:
            for warning in performance_report["warnings"]:
                critical_issues.append(warning["message"])
        
        if not critical_issues:
            critical_issues.append("Unknown emergency condition")
        
        # Create emergency response actions
        for issue in critical_issues:
            if "high_cpu_usage" in issue:
                # Handle high CPU usage
                self.create_and_queue_action(
                    name="reduce_cpu_usage",
                    description="Emergency action to reduce CPU usage",
                    function=self._emergency_reduce_cpu,
                    kwargs={"context_id": context.id}
                )
            
            elif "high_memory_usage" in issue:
                # Handle high memory usage
                self.create_and_queue_action(
                    name="reduce_memory_usage",
                    description="Emergency action to reduce memory usage",
                    function=self._emergency_reduce_memory,
                    kwargs={"context_id": context.id}
                )
            
            elif "high_failure_rate" in issue:
                # Handle high failure rate
                self.create_and_queue_action(
                    name="handle_failure_rate",
                    description="Emergency action to address high failure rate",
                    function=self._emergency_handle_failures,
                    kwargs={"context_id": context.id}
                )
        
        # Create event
        context.record_event(
            event_type="emergency_planning",
            description=f"Created emergency response plan for {len(critical_issues)} issues"
        )
    
    def _create_actions_for_goal(self, context: ExecutionContext, goal: Goal):
        """
        Create actions for a goal
        
        Args:
            context: Execution context
            goal: Goal to create actions for
        """
        # Different handling based on goal name
        if goal.name == "system_monitoring":
            # System monitoring goal
            if random.random() < 0.2:  # 20% chance to create monitoring action
                self.create_and_queue_action(
                    name="system_health_check",
                    description="Check system health and performance",
                    function=self._action_system_health_check,
                    goal_id=goal.id
                )
        
        elif goal.name == "knowledge_maintenance":
            # Knowledge maintenance goal
            if has_knowledge_lib and random.random() < 0.1:  # 10% chance
                self.create_and_queue_action(
                    name="organize_knowledge",
                    description="Organize knowledge library",
                    function=self._action_organize_knowledge,
                    goal_id=goal.id
                )
        
        elif goal.name == "self_improvement":
            # Self improvement goal
            if has_metacognition and random.random() < 0.1:  # 10% chance
                self.create_and_queue_action(
                    name="self_reflection",
                    description="Perform self-reflection for improvement",
                    function=self._action_self_reflection,
                    goal_id=goal.id
                )
    
    def _execute_action_phase(self, context: ExecutionContext):
        """
        Execute action phase
        
        Args:
            context: Execution context
        """
        # Check if there are actions in the queue
        if self.action_queue.empty():
            return
        
        # Process a batch of actions
        max_actions = 5  # Maximum actions to process in one phase
        actions_processed = 0
        
        while not self.action_queue.empty() and actions_processed < max_actions:
            # Execute next action
            action = self._execute_next_action()
            
            if action:
                actions_processed += 1
        
        if actions_processed > 0:
            context.record_event(
                event_type="action_execution",
                description=f"Executed {actions_processed} actions"
            )
    
    def _execute_reflection_phase(self, context: ExecutionContext):
        """
        Execute reflection phase
        
        Args:
            context: Execution context
        """
        # Check for completed goals
        completed_goals = []
        failed_goals = []
        
        for goal in self.goals.values():
            if goal.status == GoalStatus.COMPLETED and not goal.completed_at:
                completed_goals.append(goal)
                goal.completed_at = time.time()
            
            elif goal.status == GoalStatus.FAILED and not goal.completed_at:
                failed_goals.append(goal)
                goal.completed_at = time.time()
        
        # Reflect on goal progress
        if completed_goals or failed_goals:
            context.record_event(
                event_type="goal_reflection",
                description=f"Reflected on {len(completed_goals)} completed and {len(failed_goals)} failed goals"
            )
        
        # Perform metacognitive reflection if available
        if has_metacognition and random.random() < 0.1:  # 10% chance
            self.create_and_queue_action(
                name="metacognitive_reflection",
                description="Perform metacognitive reflection",
                function=self._action_metacognitive_reflection,
                kwargs={"context_id": context.id}
            )
        
        # Get performance report
        performance_report = context.get_variable("performance_report", {})
        
        # Check for performance issues
        if "metrics" in performance_report:
            metrics = performance_report["metrics"]
            
            if metrics.get("success_rate", 1.0) < 0.7:
                context.record_event(
                    event_type="performance_issue",
                    description=f"Low success rate: {metrics.get('success_rate', 0):.1%}"
                )
            
            if metrics.get("goal_completion_rate", 1.0) < 0.7:
                context.record_event(
                    event_type="performance_issue",
                    description=f"Low goal completion rate: {metrics.get('goal_completion_rate', 0):.1%}"
                )
    
    def _execute_adaptation_phase(self, context: ExecutionContext):
        """
        Execute adaptation phase
        
        Args:
            context: Execution context
        """
        # Check for adaptation needs
        adaptation_needed = False
        adaptation_reason = ""
        
        # Check performance for adaptation need
        performance_report = context.get_variable("performance_report", {})
        
        if "metrics" in performance_report:
            metrics = performance_report["metrics"]
            
            # Check success rate
            if metrics.get("success_rate", 1.0) < 0.7:
                adaptation_needed = True
                adaptation_reason = "low_success_rate"
            
            # Check goal completion rate
            elif metrics.get("goal_completion_rate", 1.0) < 0.7:
                adaptation_needed = True
                adaptation_reason = "low_goal_completion"
            
            # Check resource usage
            elif metrics.get("cpu_usage", 0.0) > 0.8 or metrics.get("memory_usage", 0.0) > 0.8:
                adaptation_needed = True
                adaptation_reason = "high_resource_usage"
        
        # Perform adaptation if needed
        if adaptation_needed:
            if adaptation_reason == "low_success_rate":
                self._adapt_for_success_rate(context)
            
            elif adaptation_reason == "low_goal_completion":
                self._adapt_for_goal_completion(context)
            
            elif adaptation_reason == "high_resource_usage":
                self._adapt_for_resource_usage(context)
            
            context.record_event(
                event_type="adaptation",
                description=f"Adapted execution for {adaptation_reason}"
            )
        
        # Perform self-learning if available
        if has_learning and random.random() < 0.05:  # 5% chance
            self.create_and_queue_action(
                name="self_learning",
                description="Perform self-learning",
                function=self._action_self_learning,
                kwargs={"context_id": context.id}
            )
    
    def _execute_idle_phase(self, context: ExecutionContext):
        """
        Execute idle phase
        
        Args:
            context: Execution context
        """
        # Check if we should exit idle mode
        if context.get_variable("activation_source"):
            context.set_state("activation_pending", True)
        
        # Perform background maintenance
        if random.random() < 0.01:  # 1% chance during idle
            self.create_and_queue_action(
                name="background_maintenance",
                description="Perform background system maintenance",
                function=self._action_background_maintenance,
                kwargs={"context_id": context.id}
            )
    
    def _adapt_for_success_rate(self, context: ExecutionContext):
        """
        Adapt execution for low success rate
        
        Args:
            context: Execution context
        """
        # Simplify active goals
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        
        if len(active_goals) > 2:
            # Suspend some goals to focus
            active_goals.sort(key=lambda g: g.priority)
            
            for goal in active_goals[:-2]:  # Keep top 2
                self.update_goal(goal.id, status=GoalStatus.SUSPENDED)
        
        # Adjust execution mode
        if context.mode == ExecutionMode.AUTONOMOUS:
            context.mode = ExecutionMode.SEMI_AUTONOMOUS
        
        # Create evaluation action
        self.create_and_queue_action(
            name="success_rate_evaluation",
            description="Evaluate causes of low success rate",
            function=self._action_evaluate_success_rate,
            kwargs={"context_id": context.id}
        )
    
    def _adapt_for_goal_completion(self, context: ExecutionContext):
        """
        Adapt execution for low goal completion rate
        
        Args:
            context: Execution context
        """
        # Review and reset stalled goals
        for goal in self.goals.values():
            if goal.status == GoalStatus.ACTIVE:
                # Check if goal is stalled (no progress)
                if goal.started_at and (time.time() - goal.started_at) > 300 and goal.progress < 0.1:
                    # Reset the goal
                    self.update_goal(goal.id, status=GoalStatus.PENDING, progress=0.0)
        
        # Create goal review action
        self.create_and_queue_action(
            name="goal_review",
            description="Review goals for obstacles",
            function=self._action_review_goals,
            kwargs={"context_id": context.id}
        )
    
    def _adapt_for_resource_usage(self, context: ExecutionContext):
        """
        Adapt execution for high resource usage
        
        Args:
            context: Execution context
        """
        # Switch to low power mode
        if context.mode != ExecutionMode.LOW_POWER:
            context.mode = ExecutionMode.LOW_POWER
        
        # Reduce active goals
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        
        if len(active_goals) > 1:
            # Keep only highest priority goal
            active_goals.sort(key=lambda g: g.priority, reverse=True)
            
            for goal in active_goals[1:]:  # Suspend all except highest priority
                self.update_goal(goal.id, status=GoalStatus.SUSPENDED)
        
        # Clear action queue
        while not self.action_queue.empty():
            try:
                self.action_queue.get(block=False)
            except queue.Empty:
                break
    
    # ==================== Action Implementations ====================
    
    def _action_system_health_check(self) -> Dict[str, Any]:
        """
        Action: System health check
        
        Returns:
            Health check results
        """
        # Get performance report
        performance_report = self.monitor.get_performance_report()
        
        # Check for issues
        issues = []
        
        if "metrics" in performance_report:
            metrics = performance_report["metrics"]
            
            if metrics.get("cpu_usage", 0.0) > 0.7:
                issues.append("High CPU usage")
            
            if metrics.get("memory_usage", 0.0) > 0.7:
                issues.append("High memory usage")
            
            if metrics.get("success_rate", 1.0) < 0.8:
                issues.append("Below target success rate")
        
        # Get queue status
        queue_size = self.action_queue.qsize()
        
        # Create health report
        health_report = {
            "issues": issues,
            "issue_count": len(issues),
            "status": "healthy" if not issues else "issues_detected",
            "queue_size": queue_size,
            "active_goals": len([g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]),
            "performance": performance_report.get("metrics", {})
        }
        
        logger.info(f"System health check: {health_report['status']} with {len(issues)} issues")
        return health_report
    
    def _action_organize_knowledge(self) -> Dict[str, Any]:
        """
        Action: Organize knowledge library
        
        Returns:
            Organization results
        """
        if not has_knowledge_lib:
            return {"status": "knowledge_library_unavailable"}
        
        # Get categories
        categories = knowledge_library.get_categories()
        
        # Gather statistics
        category_stats = {}
        
        for category in categories:
            entries = knowledge_library.get_entries_by_category(category)
            category_stats[category] = len(entries)
        
        # Identify large categories that might need splitting
        large_categories = []
        
        for category, count in category_stats.items():
            if count > 100:  # More than 100 entries in a category
                large_categories.append(category)
        
        logger.info(f"Knowledge organization: reviewed {len(categories)} categories")
        
        return {
            "status": "completed",
            "categories": len(categories),
            "large_categories": large_categories,
            "category_stats": category_stats
        }
    
    def _action_self_reflection(self) -> Dict[str, Any]:
        """
        Action: Self-reflection for improvement
        
        Returns:
            Reflection results
        """
        if not has_metacognition:
            return {"status": "metacognition_unavailable"}
        
        # Perform self-reflection
        reflection_result = metacognitive_system.self_reflection(
            query="How can I improve my reasoning and capabilities?",
            reflection_depth=2
        )
        
        # Create summary
        summary = {
            "cognitive_efficiency": reflection_result.get("cognitive_efficiency", {}),
            "capabilities": list(reflection_result.get("capabilities", {}).keys()),
            "active_patterns": list(reflection_result.get("active_patterns", {}).keys())
        }
        
        logger.info(f"Self-reflection: analyzed {len(summary['capabilities'])} capabilities")
        
        return {
            "status": "completed",
            "summary": summary,
            "timestamp": time.time()
        }
    
    def _action_metacognitive_reflection(self, context_id: str) -> Dict[str, Any]:
        """
        Action: Metacognitive reflection
        
        Args:
            context_id: Execution context ID
            
        Returns:
            Reflection results
        """
        if not has_metacognition:
            return {"status": "metacognition_unavailable"}
        
        # Get context
        context = self.get_context(context_id)
        if not context:
            return {"status": "context_not_found"}
        
        # Perform introspection
        introspection = metacognitive_system.introspection("general")
        
        # Record insights in context
        if "cognitive" in introspection:
            active_patterns = introspection["cognitive"].get("active_patterns", {})
            context.set_variable("active_cognitive_patterns", active_patterns)
        
        if "metrics" in introspection:
            metrics = introspection["metrics"]
            context.set_variable("metacognitive_metrics", metrics)
        
        # Create reflection result
        reflection = {
            "status": "completed",
            "active_patterns": introspection.get("cognitive", {}).get("active_patterns", {}),
            "metrics": introspection.get("metrics", {})
        }
        
        logger.info("Performed metacognitive reflection")
        return reflection
    
    def _action_self_learning(self, context_id: str) -> Dict[str, Any]:
        """
        Action: Self-learning
        
        Args:
            context_id: Execution context ID
            
        Returns:
            Learning results
        """
        if not has_learning:
            return {"status": "learning_system_unavailable"}
        
        # Get context
        context = self.get_context(context_id)
        if not context:
            return {"status": "context_not_found"}
        
        # Identify learning opportunities
        # In a real system, this would generate learning data
        # For demonstration, we'll just return a status
        
        return {
            "status": "completed",
            "learning_opportunities": ["reasoning", "knowledge_organization"],
            "timestamp": time.time()
        }
    
    def _action_background_maintenance(self, context_id: str) -> Dict[str, Any]:
        """
        Action: Background maintenance
        
        Args:
            context_id: Execution context ID
            
        Returns:
            Maintenance results
        """
        # Get context
        context = self.get_context(context_id)
        if not context:
            return {"status": "context_not_found"}
        
        # Check for stalled goals
        stalled_goals = []
        
        for goal in self.goals.values():
            if goal.status == GoalStatus.ACTIVE:
                # Check if goal is stalled (no progress for 10+ minutes)
                if goal.started_at and (time.time() - goal.started_at) > 600 and goal.progress < 0.2:
                    stalled_goals.append(goal.id)
        
        # Reset stalled goals
        for goal_id in stalled_goals:
            self.update_goal(goal_id, status=GoalStatus.PENDING, progress=0.0)
        
        # Clean up old completed goals
        completed_goals = [g for g in self.goals.values() 
                         if g.status == GoalStatus.COMPLETED and g.completed_at]
        
        old_completed = []
        current_time = time.time()
        
        for goal in completed_goals:
            # Goals completed more than 1 hour ago
            if (current_time - goal.completed_at) > 3600:
                old_completed.append(goal.id)
        
        # Remove old completed goals (up to 5)
        for goal_id in old_completed[:5]:
            if goal_id in self.goals:
                del self.goals[goal_id]
        
        logger.info(f"Background maintenance: reset {len(stalled_goals)} stalled goals, removed {len(old_completed[:5])} old goals")
        
        return {
            "status": "completed",
            "stalled_goals_reset": len(stalled_goals),
            "old_goals_removed": len(old_completed[:5])
        }
    
    def _action_evaluate_success_rate(self, context_id: str) -> Dict[str, Any]:
        """
        Action: Evaluate success rate
        
        Args:
            context_id: Execution context ID
            
        Returns:
            Evaluation results
        """
        # Get context
        context = self.get_context(context_id)
        if not context:
            return {"status": "context_not_found"}
        
        # Get recent actions
        recent_actions = context.action_history[-20:]
        
        # Calculate success rate
        successful = sum(1 for a in recent_actions if a.status == "completed")
        total = len(recent_actions)
        success_rate = successful / total if total > 0 else 0
        
        # Analyze failures
        failures = [a for a in recent_actions if a.status == "failed"]
        
        # Group failures by error
        error_counts = {}
        for action in failures:
            error = action.error or "unknown_error"
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        recommendations = []
        
        if sorted_errors:
            top_error, _ = sorted_errors[0]
            
            if "timeout" in top_error.lower():
                recommendations.append("Increase action timeout limits")
            elif "memory" in top_error.lower():
                recommendations.append("Optimize memory usage")
            elif "resource" in top_error.lower():
                recommendations.append("Reduce resource consumption")
            else:
                recommendations.append("Review error handling logic")
        
        logger.info(f"Success rate evaluation: {success_rate:.1%} with {len(sorted_errors)} error types")
        
        return {
            "status": "completed",
            "success_rate": success_rate,
            "error_types": len(sorted_errors),
            "top_errors": sorted_errors[:3],
            "recommendations": recommendations
        }
    
    def _action_review_goals(self, context_id: str) -> Dict[str, Any]:
        """
        Action: Review goals
        
        Args:
            context_id: Execution context ID
            
        Returns:
            Review results
        """
        # Get context
        context = self.get_context(context_id)
        if not context:
            return {"status": "context_not_found"}
        
        # Analyze goals
        blocked_goals = []
        stalled_goals = []
        high_priority_goals = []
        
        completed_goal_ids = {g.id for g in self.goals.values() if g.status == GoalStatus.COMPLETED}
        current_time = time.time()
        
        for goal in self.goals.values():
            # Check if goal is blocked
            if goal.status == GoalStatus.PENDING and goal.is_blocked(completed_goal_ids):
                blocked_goals.append(goal.id)
            
            # Check if goal is stalled
            if goal.status == GoalStatus.ACTIVE and goal.started_at:
                time_active = current_time - goal.started_at
                if time_active > 300 and goal.progress < 0.3:  # 5 min active with < 30% progress
                    stalled_goals.append(goal.id)
            
            # Check if goal is high priority
            if goal.priority > 0.8:
                high_priority_goals.append(goal.id)
        
        # Update stalled goals
        for goal_id in stalled_goals:
            # Reset the goal
            self.update_goal(goal_id, status=GoalStatus.PENDING, progress=0.0)
        
        logger.info(f"Goal review: {len(blocked_goals)} blocked, {len(stalled_goals)} stalled, {len(high_priority_goals)} high priority")
        
        return {
            "status": "completed",
            "blocked_goals": len(blocked_goals),
            "stalled_goals": len(stalled_goals),
            "high_priority_goals": len(high_priority_goals),
            "stalled_goals_reset": len(stalled_goals)
        }
    
    def _emergency_reduce_cpu(self, context_id: str) -> Dict[str, Any]:
        """
        Emergency action: Reduce CPU usage
        
        Args:
            context_id: Execution context ID
            
        Returns:
            Action results
        """
        # Get context
        context = self.get_context(context_id)
        if not context:
            return {"status": "context_not_found"}
        
        # Switch to low power mode
        context.mode = ExecutionMode.LOW_POWER
        
        # Clear action queue
        queue_size = self.action_queue.qsize()
        while not self.action_queue.empty():
            try:
                self.action_queue.get(block=False)
            except queue.Empty:
                break
        
        # Suspend all active goals
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        for goal in active_goals:
            self.update_goal(goal.id, status=GoalStatus.SUSPENDED)
        
        logger.warning(f"Emergency CPU reduction: cleared {queue_size} actions, suspended {len(active_goals)} goals")
        
        return {
            "status": "completed",
            "actions_cleared": queue_size,
            "goals_suspended": len(active_goals)
        }
    
    def _emergency_reduce_memory(self, context_id: str) -> Dict[str, Any]:
        """
        Emergency action: Reduce memory usage
        
        Args:
            context_id: Execution context ID
            
        Returns:
            Action results
        """
        # Get context
        context = self.get_context(context_id)
        if not context:
            return {"status": "context_not_found"}
        
        # Clear context variables
        variable_count = len(context.variables)
        context.variables = {}
        
        # Clear old history
        action_history_count = len(context.action_history)
        phase_history_count = len(context.phase_history)
        event_history_count = len(context.event_history)
        
        context.action_history = []
        context.phase_history = []
        context.event_history = []
        
        logger.warning(f"Emergency memory reduction: cleared {variable_count} variables, {action_history_count} actions history")
        
        return {
            "status": "completed",
            "variables_cleared": variable_count,
            "action_history_cleared": action_history_count,
            "phase_history_cleared": phase_history_count,
            "event_history_cleared": event_history_count
        }
    
    def _emergency_handle_failures(self, context_id: str) -> Dict[str, Any]:
        """
        Emergency action: Handle high failure rate
        
        Args:
            context_id: Execution context ID
            
        Returns:
            Action results
        """
        # Get context
        context = self.get_context(context_id)
        if not context:
            return {"status": "context_not_found"}
        
        # Reset all active goals
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        for goal in active_goals:
            self.update_goal(goal.id, status=GoalStatus.PENDING, progress=0.0)
        
        # Clear action queue
        queue_size = self.action_queue.qsize()
        while not self.action_queue.empty():
            try:
                self.action_queue.get(block=False)
            except queue.Empty:
                break
        
        logger.warning(f"Emergency failure handling: reset {len(active_goals)} goals, cleared {queue_size} actions")
        
        return {
            "status": "completed",
            "goals_reset": len(active_goals),
            "actions_cleared": queue_size
        }
    
    def save_state(self, filename: str = None) -> bool:
        """
        Save execution state
        
        Args:
            filename: Filename or None for default
            
        Returns:
            Success status
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"execution_state_{timestamp}.json"
        
        filepath = os.path.join(self.storage_path, filename)
        
        try:
            # Convert contexts to dict
            context_dicts = {ctx_id: ctx.to_dict() for ctx_id, ctx in self.contexts.items()}
            
            # Convert goals to dict
            goal_dicts = {goal_id: goal.to_dict() for goal_id, goal in self.goals.items()}
            
            # Create state object
            state = {
                "name": self.name,
                "current_phase": self.current_phase.name,
                "contexts": context_dicts,
                "goals": goal_dicts,
                "monitor": self.monitor.to_dict(),
                "timestamp": time.time()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved execution state to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving execution state: {str(e)}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """
        Load execution state
        
        Args:
            filename: Filename to load
            
        Returns:
            Success status
        """
        filepath = os.path.join(self.storage_path, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Load contexts
            if "contexts" in state:
                self.contexts = {}
                for ctx_id, ctx_dict in state["contexts"].items():
                    self.contexts[ctx_id] = ExecutionContext.from_dict(ctx_dict)
            
            # Load goals
            if "goals" in state:
                self.goals = {}
                for goal_id, goal_dict in state["goals"].items():
                    self.goals[goal_id] = Goal.from_dict(goal_dict)
            
            # Load current phase
            if "current_phase" in state:
                self.current_phase = ExecutionPhase[state["current_phase"]]
            
            logger.info(f"Loaded execution state from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading execution state: {str(e)}")
            return False
    
    def activate(self, source: str = "user"):
        """
        Activate execution engine
        
        Args:
            source: Activation source
        """
        # Get main context
        context = self.get_main_context()
        
        # Set activation source
        context.set_variable("activation_source", source)
        
        # Set activation pending
        context.set_state("activation_pending", True)
        
        # Start if not running
        if not self.running:
            self.start()
        
        logger.info(f"Activated execution engine from source: {source}")
    
    def deactivate(self):
        """Deactivate execution engine"""
        # Set idle phase
        self.current_phase = ExecutionPhase.IDLE
        
        logger.info("Deactivated execution engine")

# Initialize the continuous execution engine
continuous_execution_engine = ContinuousExecutionEngine(
    name="Seren Continuous Execution Engine",
    execute_in_thread=True
)