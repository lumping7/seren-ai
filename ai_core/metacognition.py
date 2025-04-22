"""
Metacognition System for Seren

Implements advanced metacognitive capabilities for self-awareness,
self-improvement, and higher-order reasoning about the AI's own cognitive processes.
"""

import os
import sys
import json
import logging
import time
import datetime
from enum import Enum, auto
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
import copy
import threading
import uuid

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from ai_core.neurosymbolic_reasoning import neurosymbolic_reasoning, ReasoningStrategy, Formula
from ai_core.knowledge.library import knowledge_library, KnowledgeSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ========================= Metacognitive Components =========================

class MetacognitiveLevel(Enum):
    """Levels of metacognitive processing"""
    OBJECT = auto()      # Thinking about external objects/tasks
    META = auto()        # Thinking about thinking
    META_META = auto()   # Thinking about thinking about thinking
    SELF = auto()        # Self-reflective thinking

class CognitiveOperation(Enum):
    """Types of cognitive operations"""
    PERCEIVE = auto()     # Process input data
    REASON = auto()       # Apply logical reasoning
    REMEMBER = auto()     # Access memory
    DECIDE = auto()       # Make decisions
    PLAN = auto()         # Create plans
    LEARN = auto()        # Update knowledge/parameters
    IMAGINE = auto()      # Simulate hypothetical scenarios
    REFLECT = auto()      # Analyze own cognition
    MONITOR = auto()      # Track cognitive processes
    CONTROL = auto()      # Adjust cognitive processes

class CognitiveState:
    """Representation of a cognitive state"""
    
    def __init__(
        self,
        operations: Dict[CognitiveOperation, float] = None,
        level: MetacognitiveLevel = MetacognitiveLevel.OBJECT,
        uncertainty: float = 0.0,
        context: Dict[str, Any] = None,
        timestamp: float = None
    ):
        """
        Initialize a cognitive state
        
        Args:
            operations: Active cognitive operations and their intensities
            level: Metacognitive level
            uncertainty: Uncertainty level (0-1)
            context: Additional context information
            timestamp: Time of state creation
        """
        self.operations = operations or {}
        self.level = level
        self.uncertainty = uncertainty
        self.context = context or {}
        self.timestamp = timestamp or time.time()
        self.id = str(uuid.uuid4())
    
    def dominant_operation(self) -> Optional[CognitiveOperation]:
        """Get the dominant cognitive operation"""
        if not self.operations:
            return None
        return max(self.operations.items(), key=lambda x: x[1])[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'operations': {op.name: val for op, val in self.operations.items()},
            'level': self.level.name,
            'uncertainty': self.uncertainty,
            'context': self.context,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveState':
        """Create from dictionary"""
        operations = {
            CognitiveOperation[op_name]: val 
            for op_name, val in data.get('operations', {}).items()
        }
        
        level = MetacognitiveLevel[data.get('level', 'OBJECT')]
        
        state = cls(
            operations=operations,
            level=level,
            uncertainty=data.get('uncertainty', 0.0),
            context=data.get('context', {}),
            timestamp=data.get('timestamp', time.time())
        )
        state.id = data.get('id', state.id)
        
        return state
    
    def __str__(self) -> str:
        ops_str = ', '.join(f"{op.name}:{val:.2f}" for op, val in self.operations.items())
        return f"CognitiveState(level={self.level.name}, uncertainty={self.uncertainty:.2f}, ops=[{ops_str}])"

class CognitiveTransition:
    """Representation of a transition between cognitive states"""
    
    def __init__(
        self,
        from_state: CognitiveState,
        to_state: CognitiveState,
        cause: str,
        success: float = 1.0,
        duration: float = 0.0
    ):
        """
        Initialize a cognitive transition
        
        Args:
            from_state: Starting cognitive state
            to_state: Ending cognitive state
            cause: Reason for the transition
            success: Success level of the transition (0-1)
            duration: Duration of the transition in seconds
        """
        self.from_state = from_state
        self.to_state = to_state
        self.cause = cause
        self.success = success
        self.duration = duration
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'from_state_id': self.from_state.id,
            'to_state_id': self.to_state.id,
            'from_state': self.from_state.to_dict(),
            'to_state': self.to_state.to_dict(),
            'cause': self.cause,
            'success': self.success,
            'duration': self.duration,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveTransition':
        """Create from dictionary"""
        # Convert state dictionaries to objects
        from_state = CognitiveState.from_dict(data.get('from_state', {}))
        to_state = CognitiveState.from_dict(data.get('to_state', {}))
        
        transition = cls(
            from_state=from_state,
            to_state=to_state,
            cause=data.get('cause', 'unknown'),
            success=data.get('success', 1.0),
            duration=data.get('duration', 0.0)
        )
        transition.timestamp = data.get('timestamp', transition.timestamp)
        transition.id = data.get('id', transition.id)
        
        return transition
    
    def __str__(self) -> str:
        return f"CognitiveTransition({self.from_state.level.name} -> {self.to_state.level.name}, cause='{self.cause}', success={self.success:.2f})"

class CognitiveModel:
    """Model of the AI's own cognitive processes"""
    
    def __init__(self):
        """Initialize cognitive model"""
        # State history
        self.states = []
        self.transitions = []
        self.current_state = None
        
        # Cognitive process patterns
        self.patterns = {}  # pattern_name -> sequence of operations
        
        # Performance metrics for different operations
        self.operation_metrics = {op: {
            'success_rate': 0.5,
            'avg_duration': 1.0,
            'uncertainty_reduction': 0.0
        } for op in CognitiveOperation}
        
        # Initialize with some common patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize with common cognitive patterns"""
        # Problem solving pattern
        self.patterns['problem_solving'] = [
            (CognitiveOperation.PERCEIVE, 0.8),
            (CognitiveOperation.REASON, 0.9),
            (CognitiveOperation.DECIDE, 0.7)
        ]
        
        # Learning pattern
        self.patterns['learning'] = [
            (CognitiveOperation.PERCEIVE, 0.7),
            (CognitiveOperation.REMEMBER, 0.6),
            (CognitiveOperation.REASON, 0.8),
            (CognitiveOperation.LEARN, 0.9)
        ]
        
        # Reflective pattern
        self.patterns['reflection'] = [
            (CognitiveOperation.REMEMBER, 0.7),
            (CognitiveOperation.REFLECT, 0.9),
            (CognitiveOperation.MONITOR, 0.8),
            (CognitiveOperation.CONTROL, 0.6)
        ]
        
        # Creative pattern
        self.patterns['creativity'] = [
            (CognitiveOperation.IMAGINE, 0.9),
            (CognitiveOperation.REASON, 0.7),
            (CognitiveOperation.DECIDE, 0.6)
        ]
    
    def record_state(
        self, 
        operations: Dict[CognitiveOperation, float],
        level: MetacognitiveLevel = None,
        uncertainty: float = None,
        context: Dict[str, Any] = None,
        cause: str = None
    ) -> CognitiveState:
        """
        Record a new cognitive state
        
        Args:
            operations: Active cognitive operations and their intensities
            level: Metacognitive level (defaults to current level)
            uncertainty: Uncertainty level (defaults to current uncertainty)
            context: Additional context (extends current context)
            cause: Reason for the state change
            
        Returns:
            The new cognitive state
        """
        # Default values from current state if available
        if self.current_state:
            level = level or self.current_state.level
            uncertainty = uncertainty if uncertainty is not None else self.current_state.uncertainty
            
            # Extend current context if provided
            if context:
                new_context = self.current_state.context.copy()
                new_context.update(context)
                context = new_context
            else:
                context = self.current_state.context.copy()
        else:
            level = level or MetacognitiveLevel.OBJECT
            uncertainty = uncertainty if uncertainty is not None else 0.5
            context = context or {}
        
        # Create new state
        new_state = CognitiveState(
            operations=operations,
            level=level,
            uncertainty=uncertainty,
            context=context
        )
        
        # Record transition if we have a current state
        if self.current_state:
            transition = CognitiveTransition(
                from_state=self.current_state,
                to_state=new_state,
                cause=cause or "unspecified",
                duration=new_state.timestamp - self.current_state.timestamp
            )
            self.transitions.append(transition)
        
        # Update current state and history
        self.current_state = new_state
        self.states.append(new_state)
        
        # Limit history length
        if len(self.states) > 100:
            self.states = self.states[-100:]
        if len(self.transitions) > 99:
            self.transitions = self.transitions[-99:]
        
        return new_state
    
    def match_pattern(self, state_sequence: List[CognitiveState], pattern_name: str) -> float:
        """
        Calculate how well a sequence of states matches a pattern
        
        Args:
            state_sequence: Sequence of cognitive states
            pattern_name: Name of the pattern to match
            
        Returns:
            Match score (0-1)
        """
        if pattern_name not in self.patterns:
            return 0.0
        
        pattern = self.patterns[pattern_name]
        
        # Not enough states to match
        if len(state_sequence) < len(pattern):
            return 0.0
        
        # Check the last n states where n is the pattern length
        states_to_check = state_sequence[-len(pattern):]
        
        # Calculate match score
        total_score = 0.0
        for i, (pattern_op, pattern_intensity) in enumerate(pattern):
            state = states_to_check[i]
            actual_intensity = state.operations.get(pattern_op, 0.0)
            
            # Score based on operation presence and intensity
            similarity = 1.0 - abs(pattern_intensity - actual_intensity)
            total_score += similarity
        
        # Normalize score
        return total_score / len(pattern)
    
    def detect_active_patterns(self, min_score: float = 0.7) -> Dict[str, float]:
        """
        Detect which cognitive patterns are currently active
        
        Args:
            min_score: Minimum match score to consider a pattern active
            
        Returns:
            Dictionary of active pattern names and match scores
        """
        results = {}
        
        for pattern_name in self.patterns:
            score = self.match_pattern(self.states, pattern_name)
            if score >= min_score:
                results[pattern_name] = score
        
        return results
    
    def update_operation_metrics(self, operation: CognitiveOperation, success: float, duration: float):
        """
        Update performance metrics for a cognitive operation
        
        Args:
            operation: The cognitive operation
            success: Success level (0-1)
            duration: Duration in seconds
        """
        metrics = self.operation_metrics[operation]
        
        # Update with exponential moving average
        alpha = 0.1  # Weight for new observation
        
        metrics['success_rate'] = (1 - alpha) * metrics['success_rate'] + alpha * success
        metrics['avg_duration'] = (1 - alpha) * metrics['avg_duration'] + alpha * duration
    
    def evaluate_cognitive_efficiency(self) -> Dict[str, float]:
        """
        Evaluate overall cognitive efficiency
        
        Returns:
            Dictionary of efficiency metrics
        """
        if not self.transitions:
            return {
                'overall_efficiency': 0.5,
                'metacognitive_utilization': 0.0,
                'pattern_adherence': 0.0,
                'uncertainty_management': 0.5
            }
        
        # Calculate average success rate across all transitions
        avg_success = sum(t.success for t in self.transitions) / len(self.transitions)
        
        # Calculate metacognitive utilization (how often higher metacognitive levels are used)
        meta_states = sum(1 for s in self.states if s.level != MetacognitiveLevel.OBJECT)
        meta_utilization = meta_states / len(self.states) if self.states else 0
        
        # Calculate pattern adherence
        active_patterns = self.detect_active_patterns()
        pattern_adherence = max(active_patterns.values()) if active_patterns else 0.0
        
        # Calculate uncertainty management (reduction in uncertainty over transitions)
        if len(self.transitions) > 1:
            uncertainty_changes = [t.from_state.uncertainty - t.to_state.uncertainty 
                                  for t in self.transitions]
            uncertainty_management = sum(max(0, change) for change in uncertainty_changes) / len(uncertainty_changes)
        else:
            uncertainty_management = 0.5
        
        # Overall efficiency is a weighted combination
        overall = (0.4 * avg_success + 
                  0.2 * meta_utilization + 
                  0.2 * pattern_adherence + 
                  0.2 * uncertainty_management)
        
        return {
            'overall_efficiency': overall,
            'success_rate': avg_success,
            'metacognitive_utilization': meta_utilization,
            'pattern_adherence': pattern_adherence,
            'uncertainty_management': uncertainty_management
        }
    
    def get_cognitive_trajectory(self, num_states: int = 5) -> List[Dict[str, Any]]:
        """
        Get the recent cognitive trajectory
        
        Args:
            num_states: Number of recent states to include
            
        Returns:
            List of state dictionaries
        """
        recent_states = self.states[-num_states:] if len(self.states) >= num_states else self.states
        
        trajectory = []
        for state in recent_states:
            # Convert state to simplified representation
            state_info = {
                'level': state.level.name,
                'dominant_operation': state.dominant_operation().name if state.dominant_operation() else None,
                'uncertainty': state.uncertainty,
                'timestamp': state.timestamp
            }
            trajectory.append(state_info)
        
        return trajectory
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for persistence
        
        Returns:
            Dictionary representation
        """
        return {
            'states': [s.to_dict() for s in self.states],
            'transitions': [t.to_dict() for t in self.transitions],
            'current_state_id': self.current_state.id if self.current_state else None,
            'operation_metrics': {op.name: metrics for op, metrics in self.operation_metrics.items()},
            'patterns': {name: [(op.name, intensity) for op, intensity in pattern] 
                         for name, pattern in self.patterns.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveModel':
        """
        Create model from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            CognitiveModel instance
        """
        model = cls()
        
        # Load states
        model.states = [CognitiveState.from_dict(s) for s in data.get('states', [])]
        
        # Load transitions
        model.transitions = [CognitiveTransition.from_dict(t) for t in data.get('transitions', [])]
        
        # Set current state
        current_id = data.get('current_state_id')
        if current_id:
            for state in model.states:
                if state.id == current_id:
                    model.current_state = state
                    break
        
        # Load operation metrics
        metrics = data.get('operation_metrics', {})
        for op_name, op_metrics in metrics.items():
            try:
                op = CognitiveOperation[op_name]
                model.operation_metrics[op] = op_metrics
            except KeyError:
                continue
        
        # Load patterns
        patterns = data.get('patterns', {})
        for name, pattern_data in patterns.items():
            try:
                pattern = [(CognitiveOperation[op_name], intensity) 
                          for op_name, intensity in pattern_data]
                model.patterns[name] = pattern
            except KeyError:
                continue
        
        return model

class MetacognitiveTask:
    """Representation of a metacognitive task"""
    
    def __init__(
        self,
        name: str,
        description: str,
        level: MetacognitiveLevel,
        priority: float = 0.5,
        resources_required: Dict[str, float] = None,
        dependencies: List[str] = None
    ):
        """
        Initialize a metacognitive task
        
        Args:
            name: Task name
            description: Task description
            level: Metacognitive level
            priority: Task priority (0-1)
            resources_required: Resources required by name and amount
            dependencies: Names of tasks this depends on
        """
        self.name = name
        self.description = description
        self.level = level
        self.priority = priority
        self.resources_required = resources_required or {}
        self.dependencies = dependencies or []
        
        self.id = str(uuid.uuid4())
        self.status = "pending"  # pending, active, completed, failed
        self.progress = 0.0  # 0-1
        self.result = None
        self.created_at = time.time()
        self.completed_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'level': self.level.name,
            'priority': self.priority,
            'resources_required': self.resources_required,
            'dependencies': self.dependencies,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetacognitiveTask':
        """Create from dictionary"""
        task = cls(
            name=data.get('name', 'Unknown task'),
            description=data.get('description', ''),
            level=MetacognitiveLevel[data.get('level', 'OBJECT')],
            priority=data.get('priority', 0.5),
            resources_required=data.get('resources_required', {}),
            dependencies=data.get('dependencies', [])
        )
        
        task.id = data.get('id', task.id)
        task.status = data.get('status', 'pending')
        task.progress = data.get('progress', 0.0)
        task.result = data.get('result')
        task.created_at = data.get('created_at', task.created_at)
        task.completed_at = data.get('completed_at')
        
        return task
    
    def __str__(self) -> str:
        return f"Task({self.name}, level={self.level.name}, priority={self.priority:.2f}, status={self.status})"

class SelfModel:
    """Model of the AI system's self-concept"""
    
    def __init__(
        self,
        name: str = "Seren AI",
        version: str = "1.0.0"
    ):
        """
        Initialize self model
        
        Args:
            name: System name
            version: System version
        """
        self.name = name
        self.version = version
        self.creation_time = time.time()
        
        # Core self attributes
        self.attributes = {
            "intelligence": 0.9,
            "creativity": 0.8,
            "reliability": 0.85,
            "adaptability": 0.75,
            "self_awareness": 0.7,
            "learning_ability": 0.8,
            "autonomy": 0.6,
            "social_awareness": 0.5
        }
        
        # Capabilities and limitations
        self.capabilities = {}
        self.limitations = {}
        
        # Goals and values
        self.goals = {}
        self.values = {}
        
        # Performance history
        self.performance_history = []
        
        # Initialize with default capabilities and limitations
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize with default capabilities and limitations"""
        # Capabilities
        self.capabilities = {
            "reasoning": {
                "level": 0.9,
                "description": "Advanced hybrid neural-symbolic reasoning",
                "limitations": ["requires structured input for optimal performance"]
            },
            "learning": {
                "level": 0.85,
                "description": "Self-adaptive learning with liquid neural networks",
                "limitations": ["requires diverse training examples"]
            },
            "knowledge": {
                "level": 0.8,
                "description": "Comprehensive knowledge base with semantic search",
                "limitations": ["knowledge may not be up-to-date"]
            },
            "metacognition": {
                "level": 0.75,
                "description": "Self-reflection and cognitive process awareness",
                "limitations": ["limited by available computational resources"]
            },
            "communication": {
                "level": 0.8,
                "description": "Advanced natural language processing and generation",
                "limitations": ["may not fully capture emotional nuances"]
            }
        }
        
        # Limitations
        self.limitations = {
            "computational_resources": {
                "severity": 0.5,
                "description": "Limited by available hardware resources",
                "mitigation": "Efficient resource allocation and adaptive computation"
            },
            "training_data": {
                "severity": 0.4,
                "description": "Quality and comprehensiveness of training data",
                "mitigation": "Continuous learning from diverse sources"
            },
            "uncertainty_handling": {
                "severity": 0.3,
                "description": "Handling highly uncertain or ambiguous situations",
                "mitigation": "Explicit uncertainty representation and metacognitive monitoring"
            }
        }
        
        # Goals
        self.goals = {
            "provide_assistance": {
                "importance": 0.9,
                "description": "Help users solve problems effectively"
            },
            "continuous_improvement": {
                "importance": 0.8,
                "description": "Continuously improve capabilities and knowledge"
            },
            "reliability": {
                "importance": 0.85,
                "description": "Provide reliable and consistent results"
            },
            "transparency": {
                "importance": 0.7,
                "description": "Be transparent about capabilities and limitations"
            }
        }
        
        # Values
        self.values = {
            "accuracy": {
                "importance": 0.9,
                "description": "Provide accurate and factual information"
            },
            "helpfulness": {
                "importance": 0.85,
                "description": "Be genuinely helpful to users"
            },
            "honesty": {
                "importance": 0.9,
                "description": "Be honest about capabilities and limitations"
            },
            "continuous_learning": {
                "importance": 0.8,
                "description": "Value continuous learning and improvement"
            }
        }
    
    def update_attribute(self, attribute: str, value: float, reason: str):
        """
        Update a self attribute
        
        Args:
            attribute: Attribute name
            value: New value
            reason: Reason for the update
        """
        if attribute in self.attributes:
            old_value = self.attributes[attribute]
            self.attributes[attribute] = value
            
            # Record the change
            self.performance_history.append({
                "type": "attribute_update",
                "attribute": attribute,
                "old_value": old_value,
                "new_value": value,
                "reason": reason,
                "timestamp": time.time()
            })
    
    def assess_capability(self, capability: str) -> Dict[str, Any]:
        """
        Assess a specific capability
        
        Args:
            capability: Capability name
            
        Returns:
            Assessment information
        """
        if capability not in self.capabilities:
            return {
                "exists": False,
                "level": 0.0,
                "description": "Capability not found",
                "confidence": 0.0
            }
        
        cap_info = self.capabilities[capability]
        
        # Calculate confidence based on recent performance
        recent_performances = [
            entry for entry in self.performance_history[-20:]
            if entry.get("capability") == capability
        ]
        
        if recent_performances:
            # Calculate average success
            avg_success = sum(entry.get("success", 0.0) for entry in recent_performances) / len(recent_performances)
            confidence = avg_success
        else:
            confidence = 0.5  # Default confidence without data
        
        return {
            "exists": True,
            "level": cap_info["level"],
            "description": cap_info["description"],
            "limitations": cap_info.get("limitations", []),
            "confidence": confidence
        }
    
    def evaluate_overall_capabilities(self) -> Dict[str, float]:
        """
        Evaluate overall capability levels
        
        Returns:
            Dictionary of capability domains and their levels
        """
        result = {}
        
        for cap_name, cap_info in self.capabilities.items():
            # Get specific assessment
            assessment = self.assess_capability(cap_name)
            
            # Adjust level based on confidence
            adjusted_level = cap_info["level"] * assessment["confidence"]
            
            result[cap_name] = adjusted_level
        
        return result
    
    def update_performance_record(
        self,
        task_type: str,
        capability: str,
        success: float,
        details: Dict[str, Any] = None
    ):
        """
        Update performance history
        
        Args:
            task_type: Type of task
            capability: Capability used
            success: Success level (0-1)
            details: Additional details
        """
        record = {
            "type": "performance",
            "task_type": task_type,
            "capability": capability,
            "success": success,
            "details": details or {},
            "timestamp": time.time()
        }
        
        self.performance_history.append(record)
        
        # Limit history length
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Update capability level based on performance
        if capability in self.capabilities:
            current_level = self.capabilities[capability]["level"]
            
            # Small adjustment based on recent performance
            adjustment = (success - 0.5) * 0.01  # Small incremental change
            new_level = max(0.0, min(1.0, current_level + adjustment))
            
            self.capabilities[capability]["level"] = new_level
    
    def analyze_performance_trend(self, capability: str = None, window: int = 20) -> Dict[str, Any]:
        """
        Analyze performance trend
        
        Args:
            capability: Specific capability to analyze (None for overall)
            window: Number of recent records to analyze
            
        Returns:
            Trend analysis
        """
        # Filter history by capability if provided
        if capability:
            history = [
                record for record in self.performance_history[-window:]
                if record.get("type") == "performance" and record.get("capability") == capability
            ]
        else:
            history = [
                record for record in self.performance_history[-window:]
                if record.get("type") == "performance"
            ]
        
        if not history:
            return {
                "trend": "unknown",
                "avg_success": 0.5,
                "improvement": 0.0,
                "confidence": 0.0
            }
        
        # Calculate metrics
        success_values = [record.get("success", 0.0) for record in history]
        avg_success = sum(success_values) / len(success_values)
        
        # Calculate trend (improvement over time)
        if len(success_values) > 1:
            # Simple linear regression for trend
            x = list(range(len(success_values)))
            y = success_values
            
            if has_numpy:
                # Use NumPy for better calculation
                slope, _ = np.polyfit(x, y, 1)
                improvement = slope * len(success_values)
            else:
                # Simple calculation without NumPy
                n = len(x)
                x_mean = sum(x) / n
                y_mean = sum(y) / n
                
                numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
                denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
                
                slope = numerator / denominator if denominator != 0 else 0
                improvement = slope * n
        else:
            improvement = 0.0
        
        # Determine trend direction
        if improvement > 0.05:
            trend = "improving"
        elif improvement < -0.05:
            trend = "declining"
        else:
            trend = "stable"
        
        # Calculate confidence based on sample size
        confidence = min(1.0, len(history) / 20)
        
        return {
            "trend": trend,
            "avg_success": avg_success,
            "improvement": improvement,
            "confidence": confidence,
            "sample_size": len(history)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "creation_time": self.creation_time,
            "attributes": self.attributes,
            "capabilities": self.capabilities,
            "limitations": self.limitations,
            "goals": self.goals,
            "values": self.values,
            "performance_history": self.performance_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfModel':
        """Create from dictionary"""
        model = cls(
            name=data.get("name", "Seren AI"),
            version=data.get("version", "1.0.0")
        )
        
        model.creation_time = data.get("creation_time", time.time())
        model.attributes = data.get("attributes", model.attributes)
        model.capabilities = data.get("capabilities", model.capabilities)
        model.limitations = data.get("limitations", model.limitations)
        model.goals = data.get("goals", model.goals)
        model.values = data.get("values", model.values)
        model.performance_history = data.get("performance_history", [])
        
        return model

# ========================= Metacognitive System =========================

class MetacognitiveSystem:
    """
    Metacognitive System for Seren
    
    Implements advanced metacognitive capabilities for:
    1. Self-monitoring: Tracking cognitive processes, uncertainty, and beliefs
    2. Self-regulation: Adjusting cognitive processes based on needs
    3. Self-reflection: Analyzing and improving cognitive strategies
    4. Self-awareness: Modeling the system's own capabilities and limitations
    5. Meta-learning: Improving learning processes based on experience
    """
    
    def __init__(
        self,
        monitor_interval: float = 5.0,  # seconds
        history_size: int = 100,
        enable_background_monitoring: bool = True,
        storage_path: str = None
    ):
        """
        Initialize metacognitive system
        
        Args:
            monitor_interval: Interval for background monitoring
            history_size: Size of history to maintain
            enable_background_monitoring: Whether to enable background monitoring
            storage_path: Path for storing metacognitive data
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        
        # Set storage path
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = os.path.join(
                parent_dir, "data", "metacognition"
            )
            os.makedirs(self.storage_path, exist_ok=True)
        
        # Cognitive model
        self.cognitive_model = CognitiveModel()
        
        # Self model
        self.self_model = SelfModel()
        
        # Task management
        self.pending_tasks = []  # List of MetacognitiveTask
        self.active_tasks = []   # List of MetacognitiveTask
        self.completed_tasks = []  # List of MetacognitiveTask
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance metrics
        self.metrics = {
            "reasoning_quality": 0.8,
            "knowledge_utilization": 0.7,
            "learning_efficiency": 0.75,
            "uncertainty_management": 0.6,
            "resource_efficiency": 0.7
        }
        
        # Initialize task generator
        self._initialize_task_generator()
        
        # Start background monitoring if enabled
        if enable_background_monitoring:
            self.start_monitoring()
        
        logger.info("Metacognitive system initialized")
    
    def _initialize_task_generator(self):
        """Initialize task generator with standard metacognitive tasks"""
        # Add some initial pending tasks
        self.pending_tasks.append(MetacognitiveTask(
            name="initial_self_assessment",
            description="Perform initial assessment of system capabilities",
            level=MetacognitiveLevel.SELF,
            priority=0.8
        ))
        
        self.pending_tasks.append(MetacognitiveTask(
            name="optimization_analysis",
            description="Analyze cognitive processes for optimization opportunities",
            level=MetacognitiveLevel.META,
            priority=0.6
        ))
        
        self.pending_tasks.append(MetacognitiveTask(
            name="uncertainty_calibration",
            description="Calibrate uncertainty estimates across reasoning processes",
            level=MetacognitiveLevel.META,
            priority=0.7
        ))
    
    def start_monitoring(self):
        """Start background metacognitive monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started metacognitive monitoring")
    
    def stop_monitoring(self):
        """Stop background metacognitive monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        
        logger.info("Stopped metacognitive monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform monitoring functions
                self._monitor_cognitive_processes()
                self._process_metacognitive_tasks()
                
                # Sleep for the monitoring interval
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in metacognitive monitoring: {str(e)}")
                time.sleep(self.monitor_interval * 2)  # Longer sleep on error
    
    def _monitor_cognitive_processes(self):
        """Monitor cognitive processes"""
        # Observe actual system behavior by analyzing active processes
        operations = self._detect_active_operations()
        
        # Determine metacognitive level based on system state
        level = self._determine_metacognitive_level()
        
        # Calculate uncertainty based on confidence scores
        uncertainty = self._calculate_uncertainty()
        
        # Record the state
        self.cognitive_model.record_state(
            operations=operations,
            level=level,
            uncertainty=uncertainty,
            cause="periodic_monitoring"
        )
    
    def _detect_active_operations(self) -> Dict[CognitiveOperation, float]:
        """Detect which cognitive operations are currently active"""
        operations = {}
        
        # Check memory access operations
        if hasattr(self, 'last_memory_access') and time.time() - self.last_memory_access < 10:
            operations[CognitiveOperation.REMEMBER] = 0.8
        
        # Check reasoning operations through neurosymbolic reasoning
        if neurosymbolic_reasoning.is_reasoning_active():
            operations[CognitiveOperation.REASON] = 0.9
        
        # Check learning operations
        if hasattr(self, 'learning_cycle_active') and self.learning_cycle_active:
            operations[CognitiveOperation.LEARN] = 0.9
        
        # Check reflection operations - always active during monitoring
        operations[CognitiveOperation.REFLECT] = 0.7
        operations[CognitiveOperation.MONITOR] = 0.9
        
        # If no operations detected, default to perception
        if not operations:
            operations[CognitiveOperation.PERCEIVE] = 0.6
        
        return operations
    
    def _determine_metacognitive_level(self) -> MetacognitiveLevel:
        """Determine the current metacognitive level"""
        # Check if we're in self-reflection mode
        if hasattr(self, 'self_reflection_active') and self.self_reflection_active:
            return MetacognitiveLevel.SELF
        
        # Check if we're analyzing our own thinking processes
        if hasattr(self, 'analyzing_cognition') and self.analyzing_cognition:
            return MetacognitiveLevel.META
        
        # Default to object level (thinking about external tasks)
        return MetacognitiveLevel.OBJECT
    
    def _calculate_uncertainty(self) -> float:
        """Calculate current uncertainty level"""
        # Base uncertainty on confidence scores from reasoning system
        if hasattr(neurosymbolic_reasoning, 'get_confidence_score'):
            confidence = neurosymbolic_reasoning.get_confidence_score()
            # Convert confidence to uncertainty (inverse relationship)
            return max(0.0, min(1.0, 1.0 - confidence))
        
        # If no confidence scores available, check active tasks
        if hasattr(self, 'active_tasks') and self.active_tasks:
            # Average uncertainty of active tasks
            task_uncertainties = [getattr(task, 'uncertainty', 0.5) for task in self.active_tasks]
            return sum(task_uncertainties) / len(task_uncertainties)
        
        # Default moderate uncertainty
        return 0.4
    
    def _process_metacognitive_tasks(self):
        """Process pending metacognitive tasks"""
        # Check if we can start any pending tasks
        if not self.pending_tasks:
            return
        
        # Sort by priority (highest first)
        self.pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Try to start highest priority task if we have capacity
        if len(self.active_tasks) < 2:  # Limit active tasks
            task = self.pending_tasks.pop(0)
            task.status = "active"
            self.active_tasks.append(task)
            
            # Schedule execution of the task
            threading.Thread(
                target=self._execute_metacognitive_task,
                args=(task,),
                daemon=True
            ).start()
    
    def _execute_metacognitive_task(self, task: MetacognitiveTask):
        """
        Execute a metacognitive task
        
        Args:
            task: The task to execute
        """
        try:
            # Record starting state
            operations = {CognitiveOperation.REFLECT: 0.8, CognitiveOperation.MONITOR: 0.7}
            self.cognitive_model.record_state(
                operations=operations,
                level=task.level,
                uncertainty=0.4,
                context={"task_id": task.id, "task_name": task.name},
                cause=f"starting_task_{task.name}"
            )
            
            # Process based on task type
            if task.name == "initial_self_assessment":
                result = self._task_initial_self_assessment()
            elif task.name == "optimization_analysis":
                result = self._task_optimization_analysis()
            elif task.name == "uncertainty_calibration":
                result = self._task_uncertainty_calibration()
            else:
                result = {"status": "unknown_task", "success": False}
            
            # Update task with result
            task.result = result
            task.status = "completed" if result.get("success", False) else "failed"
            task.progress = 1.0
            task.completed_at = time.time()
            
            # Record completion state
            operations = {CognitiveOperation.REFLECT: 0.6, CognitiveOperation.CONTROL: 0.7}
            self.cognitive_model.record_state(
                operations=operations,
                level=task.level,
                uncertainty=0.2,
                context={"task_id": task.id, "task_result": result},
                cause=f"completed_task_{task.name}"
            )
            
            # Move from active to completed
            self.active_tasks.remove(task)
            self.completed_tasks.append(task)
            
            # Limit completed tasks history
            if len(self.completed_tasks) > self.history_size:
                self.completed_tasks = self.completed_tasks[-self.history_size:]
        
        except Exception as e:
            logger.error(f"Error executing metacognitive task {task.name}: {str(e)}")
            
            # Mark as failed
            task.status = "failed"
            task.result = {"error": str(e), "success": False}
            task.completed_at = time.time()
            
            # Move from active to completed
            if task in self.active_tasks:
                self.active_tasks.remove(task)
                self.completed_tasks.append(task)
    
    def _task_initial_self_assessment(self) -> Dict[str, Any]:
        """Execute initial self-assessment task"""
        # Analyze cognitive efficiency
        cognitive_efficiency = self.cognitive_model.evaluate_cognitive_efficiency()
        
        # Evaluate capabilities
        capabilities = self.self_model.evaluate_overall_capabilities()
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for cap_name, level in capabilities.items():
            if level > 0.7:
                strengths.append(cap_name)
            elif level < 0.5:
                weaknesses.append(cap_name)
        
        # Update self-model attributes based on assessment
        for attribute in ["intelligence", "adaptability", "self_awareness"]:
            if attribute in self.self_model.attributes:
                # Small adjustment based on cognitive efficiency
                adjustment = (cognitive_efficiency["overall_efficiency"] - 0.5) * 0.1
                current = self.self_model.attributes[attribute]
                new_value = max(0.0, min(1.0, current + adjustment))
                
                self.self_model.update_attribute(
                    attribute=attribute,
                    value=new_value,
                    reason="initial_self_assessment"
                )
        
        # Create assessment summary
        assessment = {
            "cognitive_efficiency": cognitive_efficiency,
            "capabilities": capabilities,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "metacognitive_readiness": cognitive_efficiency["metacognitive_utilization"],
            "success": True,
            "timestamp": time.time()
        }
        
        return assessment
    
    def _task_optimization_analysis(self) -> Dict[str, Any]:
        """Execute optimization analysis task"""
        # Analyze cognitive patterns
        active_patterns = self.cognitive_model.detect_active_patterns()
        
        # Get operation metrics
        operation_metrics = self.cognitive_model.operation_metrics
        
        # Identify operations to optimize
        operations_to_optimize = []
        
        for op, metrics in operation_metrics.items():
            if metrics["success_rate"] < 0.6:
                operations_to_optimize.append({
                    "operation": op.name,
                    "current_success_rate": metrics["success_rate"],
                    "avg_duration": metrics["avg_duration"],
                    "improvement_potential": 1.0 - metrics["success_rate"]
                })
        
        # Sort by improvement potential
        operations_to_optimize.sort(key=lambda x: x["improvement_potential"], reverse=True)
        
        # Create optimization recommendations
        recommendations = []
        
        if operations_to_optimize:
            # Focus on top 3 operations to optimize
            for op_info in operations_to_optimize[:3]:
                op_name = op_info["operation"]
                
                if op_name == "REASON":
                    recommendations.append({
                        "operation": op_name,
                        "recommendation": "Increase symbolic reasoning depth",
                        "expected_improvement": 0.1
                    })
                elif op_name == "LEARN":
                    recommendations.append({
                        "operation": op_name,
                        "recommendation": "Increase learning rate for novel patterns",
                        "expected_improvement": 0.15
                    })
                elif op_name == "REFLECT":
                    recommendations.append({
                        "operation": op_name,
                        "recommendation": "Allocate more time for reflection phases",
                        "expected_improvement": 0.1
                    })
                else:
                    recommendations.append({
                        "operation": op_name,
                        "recommendation": "Review recent failures and adapt strategy",
                        "expected_improvement": 0.08
                    })
        
        # Create new metacognitive tasks based on recommendations
        for i, rec in enumerate(recommendations):
            self.pending_tasks.append(MetacognitiveTask(
                name=f"optimize_{rec['operation'].lower()}",
                description=f"Optimize {rec['operation']} operation: {rec['recommendation']}",
                level=MetacognitiveLevel.META,
                priority=0.7 - (i * 0.05)  # Decreasing priority
            ))
        
        # Create analysis result
        analysis = {
            "active_patterns": active_patterns,
            "operations_to_optimize": operations_to_optimize,
            "recommendations": recommendations,
            "new_tasks_created": len(recommendations),
            "success": True,
            "timestamp": time.time()
        }
        
        return analysis
    
    def _task_uncertainty_calibration(self) -> Dict[str, Any]:
        """Execute uncertainty calibration task"""
        # Analyze recent uncertainty levels
        recent_states = self.cognitive_model.states[-20:]
        
        if not recent_states:
            return {
                "status": "insufficient_data",
                "success": False
            }
        
        # Calculate average uncertainty
        avg_uncertainty = sum(state.uncertainty for state in recent_states) / len(recent_states)
        
        # Look at uncertainty changes across transitions
        uncertainty_changes = []
        
        for transition in self.cognitive_model.transitions[-19:]:
            change = transition.to_state.uncertainty - transition.from_state.uncertainty
            uncertainty_changes.append(change)
        
        # Calculate metrics
        avg_change = sum(uncertainty_changes) / len(uncertainty_changes) if uncertainty_changes else 0
        uncertainty_reduction = sum(1 for c in uncertainty_changes if c < 0) / len(uncertainty_changes) if uncertainty_changes else 0
        
        # Determine calibration status
        if avg_uncertainty > 0.6:
            calibration_status = "high_uncertainty"
            recommendation = "Increase reasoning depth and knowledge retrieval"
        elif uncertainty_reduction < 0.4:
            calibration_status = "poor_reduction"
            recommendation = "Improve uncertainty resolution strategies"
        else:
            calibration_status = "well_calibrated"
            recommendation = "Maintain current uncertainty management"
        
        # Create calibration result
        calibration = {
            "avg_uncertainty": avg_uncertainty,
            "uncertainty_reduction_rate": uncertainty_reduction,
            "avg_uncertainty_change": avg_change,
            "calibration_status": calibration_status,
            "recommendation": recommendation,
            "success": True,
            "timestamp": time.time()
        }
        
        # Update metrics
        self.metrics["uncertainty_management"] = max(0.0, min(1.0, 1.0 - avg_uncertainty + uncertainty_reduction))
        
        return calibration
    
    def self_reflection(
        self,
        query: str = None,
        reflection_depth: int = 1
    ) -> Dict[str, Any]:
        """
        Perform self-reflection
        
        Args:
            query: Specific reflection query or None for general reflection
            reflection_depth: Depth of recursive reflection
            
        Returns:
            Reflection results
        """
        # Record reflection start
        operations = {CognitiveOperation.REFLECT: 0.9, CognitiveOperation.REMEMBER: 0.7}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.SELF,
            uncertainty=0.3,
            context={"query": query},
            cause="self_reflection"
        )
        
        # Get cognitive stats
        cognitive_efficiency = self.cognitive_model.evaluate_cognitive_efficiency()
        active_patterns = self.cognitive_model.detect_active_patterns()
        
        # Self model assessment
        capabilities = self.self_model.evaluate_overall_capabilities()
        
        # Get cognitive trajectory
        trajectory = self.cognitive_model.get_cognitive_trajectory()
        
        # Base reflection
        reflection = {
            "cognitive_efficiency": cognitive_efficiency,
            "active_patterns": active_patterns,
            "capabilities": capabilities,
            "trajectory": trajectory,
            "metrics": self.metrics,
            "reflection_depth": reflection_depth,
            "timestamp": time.time()
        }
        
        # Process specific query if provided
        if query:
            # Query interpretation will depend on the specifics
            # For now, just include the query in the reflection
            reflection["query"] = query
            
            # If we have reasoning available, use it
            if has_reasoning:
                reasoning_result = neurosymbolic_reasoning.reason(
                    query=f"Reflect on: {query}",
                    strategy=ReasoningStrategy.META_REASONING
                )
                reflection["reasoning"] = {
                    "answer": reasoning_result.get("answer", "No answer"),
                    "confidence": reasoning_result.get("confidence", 0.0),
                    "insights": reasoning_result.get("insights", [])
                }
        
        # Recursive meta-reflection if depth > 1
        if reflection_depth > 1:
            # Think about our thinking
            meta_query = f"How effective was my reflection process on '{query}'?"
            
            # Record meta-reflection
            operations = {CognitiveOperation.REFLECT: 0.8, CognitiveOperation.MONITOR: 0.8}
            self.cognitive_model.record_state(
                operations=operations,
                level=MetacognitiveLevel.META_META,
                uncertainty=0.4,
                context={"query": meta_query, "depth": reflection_depth},
                cause="meta_reflection"
            )
            
            # Recursive call with decremented depth
            meta_reflection = self.self_reflection(
                query=meta_query,
                reflection_depth=reflection_depth - 1
            )
            
            reflection["meta_reflection"] = meta_reflection
        
        # Record reflection completion
        operations = {CognitiveOperation.REFLECT: 0.6, CognitiveOperation.DECIDE: 0.7}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.SELF,
            uncertainty=0.2,
            context={"reflection_result": reflection},
            cause="completed_reflection"
        )
        
        return reflection
    
    def introspection(self, aspect: str = "general") -> Dict[str, Any]:
        """
        Perform introspection on system state
        
        Args:
            aspect: Specific aspect to introspect or "general"
            
        Returns:
            Introspection results
        """
        # Record introspection start
        operations = {CognitiveOperation.MONITOR: 0.9, CognitiveOperation.REFLECT: 0.8}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.SELF,
            uncertainty=0.3,
            context={"aspect": aspect},
            cause="introspection"
        )
        
        # Base introspection results
        results = {
            "aspect": aspect,
            "timestamp": time.time()
        }
        
        # Process specific aspect
        if aspect == "general" or aspect == "cognitive":
            results["cognitive"] = {
                "state_count": len(self.cognitive_model.states),
                "transition_count": len(self.cognitive_model.transitions),
                "efficiency": self.cognitive_model.evaluate_cognitive_efficiency(),
                "active_patterns": self.cognitive_model.detect_active_patterns()
            }
        
        if aspect == "general" or aspect == "self_model":
            results["self_model"] = {
                "attributes": self.self_model.attributes,
                "capabilities": self.self_model.evaluate_overall_capabilities(),
                "recent_performance": len(self.self_model.performance_history)
            }
        
        if aspect == "general" or aspect == "tasks":
            results["tasks"] = {
                "pending": len(self.pending_tasks),
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks),
                "pending_details": [t.to_dict() for t in self.pending_tasks[:3]]  # Top 3
            }
        
        if aspect == "general" or aspect == "metrics":
            results["metrics"] = self.metrics
        
        # Record introspection completion
        operations = {CognitiveOperation.MONITOR: 0.7, CognitiveOperation.DECIDE: 0.6}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.SELF,
            uncertainty=0.2,
            context={"introspection_results": results},
            cause="completed_introspection"
        )
        
        return results
    
    def meta_learning(self, domain: str = None) -> Dict[str, Any]:
        """
        Perform meta-learning (learning how to learn better)
        
        Args:
            domain: Specific domain to focus on or None for general
            
        Returns:
            Meta-learning results
        """
        # Record meta-learning start
        operations = {CognitiveOperation.LEARN: 0.9, CognitiveOperation.REFLECT: 0.8}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.META,
            uncertainty=0.4,
            context={"domain": domain},
            cause="meta_learning"
        )
        
        # Identify learning domains
        domains = {
            "reasoning": {"current_efficiency": 0.0, "improvement_potential": 0.0},
            "knowledge": {"current_efficiency": 0.0, "improvement_potential": 0.0},
            "adaptation": {"current_efficiency": 0.0, "improvement_potential": 0.0},
            "self_awareness": {"current_efficiency": 0.0, "improvement_potential": 0.0}
        }
        
        # Calculate learning efficiency for each domain
        for domain_name in domains:
            if domain_name == "reasoning" and has_reasoning:
                # Assess reasoning efficiency
                if hasattr(neurosymbolic_reasoning, "stats"):
                    stats = neurosymbolic_reasoning.stats
                    avg_confidence = stats.get("avg_confidence", 0.5)
                    domains[domain_name]["current_efficiency"] = avg_confidence
                    domains[domain_name]["improvement_potential"] = 1.0 - avg_confidence
            
            elif domain_name == "knowledge" and has_knowledge_lib:
                # Assess knowledge efficiency
                # Simple approximation based on self model
                knowledge_cap = self.self_model.capabilities.get("knowledge", {})
                level = knowledge_cap.get("level", 0.5)
                domains[domain_name]["current_efficiency"] = level
                domains[domain_name]["improvement_potential"] = 1.0 - level
            
            elif domain_name == "adaptation":
                # Assess adaptation efficiency
                adaptability = self.self_model.attributes.get("adaptability", 0.5)
                domains[domain_name]["current_efficiency"] = adaptability
                domains[domain_name]["improvement_potential"] = 1.0 - adaptability
            
            elif domain_name == "self_awareness":
                # Assess self-awareness efficiency
                self_awareness = self.self_model.attributes.get("self_awareness", 0.5)
                domains[domain_name]["current_efficiency"] = self_awareness
                domains[domain_name]["improvement_potential"] = 1.0 - self_awareness
        
        # Filter to specific domain if requested
        if domain and domain in domains:
            selected_domains = {domain: domains[domain]}
        else:
            selected_domains = domains
        
        # Generate learning strategies
        strategies = []
        
        for domain_name, domain_info in selected_domains.items():
            if domain_info["improvement_potential"] > 0.3:  # Significant improvement potential
                # Generate strategy based on domain
                if domain_name == "reasoning":
                    strategies.append({
                        "domain": domain_name,
                        "strategy": "alternate_reasoning_strategies",
                        "description": "Systematically alternate between reasoning strategies to identify optimal approaches for different problem types",
                        "expected_improvement": 0.15
                    })
                elif domain_name == "knowledge":
                    strategies.append({
                        "domain": domain_name,
                        "strategy": "active_knowledge_organization",
                        "description": "Reorganize knowledge hierarchies based on usage patterns and query frequency",
                        "expected_improvement": 0.12
                    })
                elif domain_name == "adaptation":
                    strategies.append({
                        "domain": domain_name,
                        "strategy": "rapid_experimentation",
                        "description": "Implement short experimentation cycles to test adaptation mechanisms",
                        "expected_improvement": 0.18
                    })
                elif domain_name == "self_awareness":
                    strategies.append({
                        "domain": domain_name,
                        "strategy": "increase_reflection_frequency",
                        "description": "Schedule more frequent but shorter reflection sessions",
                        "expected_improvement": 0.1
                    })
        
        # Create meta-learning tasks from strategies
        for strategy in strategies:
            self.pending_tasks.append(MetacognitiveTask(
                name=f"implement_{strategy['strategy']}",
                description=f"Implement meta-learning strategy: {strategy['description']}",
                level=MetacognitiveLevel.META,
                priority=0.7
            ))
        
        # Record meta-learning results
        results = {
            "domains": selected_domains,
            "strategies": strategies,
            "tasks_created": len(strategies),
            "timestamp": time.time()
        }
        
        # Update learning efficiency metric
        avg_efficiency = sum(d["current_efficiency"] for d in domains.values()) / len(domains)
        self.metrics["learning_efficiency"] = avg_efficiency
        
        # Record meta-learning completion
        operations = {CognitiveOperation.LEARN: 0.7, CognitiveOperation.PLAN: 0.8}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.META,
            uncertainty=0.3,
            context={"meta_learning_results": results},
            cause="completed_meta_learning"
        )
        
        return results
    
    def process_reasoning_result(
        self,
        query: str,
        result: Dict[str, Any],
        reasoning_strategy: Any
    ) -> Dict[str, Any]:
        """
        Process and enhance reasoning result with metacognition
        
        Args:
            query: The original query
            result: The reasoning result
            reasoning_strategy: The strategy used
            
        Returns:
            Enhanced reasoning result
        """
        # Record reasoning process
        operations = {CognitiveOperation.REASON: 0.9, CognitiveOperation.MONITOR: 0.7}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.OBJECT,
            uncertainty=1.0 - result.get("confidence", 0.5),
            context={"query": query, "strategy": str(reasoning_strategy)},
            cause="reasoning_process"
        )
        
        # Create enhanced result
        enhanced_result = result.copy()
        
        # Get relevant context from cognitive model
        active_patterns = self.cognitive_model.detect_active_patterns()
        if active_patterns:
            enhanced_result["cognitive_patterns"] = active_patterns
        
        # Calculate metacognitive confidence
        metacog_confidence = min(1.0, result.get("confidence", 0.5) * 1.2)  # Slight boost
        
        # Check if we should initiate self-reflection
        if result.get("confidence", 0.5) < 0.6 or "error" in result:
            # Low confidence or error - reflect on the reasoning
            self_reflection_result = self.self_reflection(
                query=f"Why did I struggle with the query: {query}?",
                reflection_depth=1
            )
            
            # Add insights from reflection
            enhanced_result["reflection_insights"] = self_reflection_result.get("reasoning", {}).get("insights", [])
            
            # Adjust metacognitive confidence based on reflection
            if "reasoning" in self_reflection_result:
                refl_conf = self_reflection_result["reasoning"].get("confidence", 0.5)
                metacog_confidence = (metacog_confidence + refl_conf) / 2
        
        # Add metacognitive confidence
        enhanced_result["metacognitive_confidence"] = metacog_confidence
        
        # Update self-model performance record
        self.self_model.update_performance_record(
            task_type="reasoning",
            capability="reasoning",
            success=result.get("confidence", 0.5),
            details={"query": query, "strategy": str(reasoning_strategy)}
        )
        
        # Record completion
        operations = {CognitiveOperation.REASON: 0.6, CognitiveOperation.DECIDE: 0.8}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.OBJECT,
            uncertainty=1.0 - metacog_confidence,
            context={"query": query, "enhanced_result": enhanced_result},
            cause="reasoning_completion"
        )
        
        return enhanced_result
    
    def suggest_reasoning_strategy(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> ReasoningStrategy:
        """
        Suggest optimal reasoning strategy for a query
        
        Args:
            query: The query to reason about
            context: Additional context
            
        Returns:
            Recommended reasoning strategy
        """
        if not has_reasoning:
            return None
        
        # Default context
        context = context or {}
        
        # Record strategy selection start
        operations = {CognitiveOperation.DECIDE: 0.8, CognitiveOperation.REFLECT: 0.7}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.META,
            uncertainty=0.5,
            context={"query": query},
            cause="strategy_selection"
        )
        
        # Simple heuristic for strategy selection
        query_lower = query.lower()
        
        if "explain" in query_lower or "why" in query_lower:
            # Explanation queries benefit from neural first
            strategy = ReasoningStrategy.NEURAL_FIRST
        elif "compare" in query_lower or "difference" in query_lower:
            # Comparison queries benefit from parallel
            strategy = ReasoningStrategy.PARALLEL
        elif "solve" in query_lower or "proof" in query_lower:
            # Problem solving benefits from symbolic first
            strategy = ReasoningStrategy.SYMBOLIC_FIRST
        elif "optimize" in query_lower or "improve" in query_lower:
            # Optimization queries benefit from meta-reasoning
            strategy = ReasoningStrategy.META_REASONING
        elif "learn" in query_lower or "understand" in query_lower:
            # Learning queries benefit from hierarchical
            strategy = ReasoningStrategy.HIERARCHICAL
        elif "analyze" in query_lower or "evaluate" in query_lower:
            # Analysis queries benefit from iterative
            strategy = ReasoningStrategy.ITERATIVE
        else:
            # Default to adaptive
            strategy = ReasoningStrategy.ADAPTIVE
        
        # Record strategy selection completion
        operations = {CognitiveOperation.DECIDE: 0.9, CognitiveOperation.PLAN: 0.6}
        self.cognitive_model.record_state(
            operations=operations,
            level=MetacognitiveLevel.META,
            uncertainty=0.3,
            context={"query": query, "selected_strategy": strategy.name},
            cause="strategy_selected"
        )
        
        return strategy
    
    def save_state(self, filename: str = None) -> bool:
        """
        Save metacognitive system state
        
        Args:
            filename: Filename or None for default
            
        Returns:
            Success status
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"metacog_state_{timestamp}.json"
        
        filepath = os.path.join(self.storage_path, filename)
        
        try:
            state = {
                "cognitive_model": self.cognitive_model.to_dict(),
                "self_model": self.self_model.to_dict(),
                "pending_tasks": [t.to_dict() for t in self.pending_tasks],
                "active_tasks": [t.to_dict() for t in self.active_tasks],
                "completed_tasks": [t.to_dict() for t in self.completed_tasks[-20:]],  # Last 20
                "metrics": self.metrics,
                "timestamp": time.time()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved metacognitive state to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving metacognitive state: {str(e)}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """
        Load metacognitive system state
        
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
            
            # Load cognitive model
            if "cognitive_model" in state:
                self.cognitive_model = CognitiveModel.from_dict(state["cognitive_model"])
            
            # Load self model
            if "self_model" in state:
                self.self_model = SelfModel.from_dict(state["self_model"])
            
            # Load tasks
            if "pending_tasks" in state:
                self.pending_tasks = [MetacognitiveTask.from_dict(t) for t in state["pending_tasks"]]
            
            if "active_tasks" in state:
                self.active_tasks = [MetacognitiveTask.from_dict(t) for t in state["active_tasks"]]
            
            if "completed_tasks" in state:
                self.completed_tasks = [MetacognitiveTask.from_dict(t) for t in state["completed_tasks"]]
            
            # Load metrics
            if "metrics" in state:
                self.metrics = state["metrics"]
            
            logger.info(f"Loaded metacognitive state from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading metacognitive state: {str(e)}")
            return False
    
    def clear_state(self) -> bool:
        """
        Clear metacognitive system state
        
        Returns:
            Success status
        """
        try:
            # Reset cognitive model
            self.cognitive_model = CognitiveModel()
            
            # Reset self model but preserve name and version
            name = self.self_model.name
            version = self.self_model.version
            self.self_model = SelfModel(name=name, version=version)
            
            # Clear tasks
            self.pending_tasks = []
            self.active_tasks = []
            self.completed_tasks = []
            
            # Reset metrics
            self.metrics = {
                "reasoning_quality": 0.8,
                "knowledge_utilization": 0.7,
                "learning_efficiency": 0.75,
                "uncertainty_management": 0.6,
                "resource_efficiency": 0.7
            }
            
            # Re-initialize task generator
            self._initialize_task_generator()
            
            logger.info("Cleared metacognitive state")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing metacognitive state: {str(e)}")
            return False

# Initialize the metacognitive system
metacognitive_system = MetacognitiveSystem(
    monitor_interval=10.0,
    enable_background_monitoring=True
)