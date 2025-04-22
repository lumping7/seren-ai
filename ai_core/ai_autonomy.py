"""
AI Autonomy Engine

Enables AI system to take actions and make decisions autonomously
with appropriate safeguards and human oversight controls.
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

class AutonomyLevel(str, enum.Enum):
    """Levels of AI autonomy"""
    NONE = "none"  # No autonomous actions, requires explicit approval
    GUIDED = "guided"  # Some autonomous actions, but follows strict guidelines
    SUPERVISED = "supervised"  # Can act autonomously but notifies human
    FULL = "full"  # Can act autonomously with minimal oversight

class ActionCategory(str, enum.Enum):
    """Categories of autonomous actions"""
    INFORMATION = "information"  # Retrieving or analyzing information
    COMMUNICATION = "communication"  # Communicating with users or systems
    RESOURCE = "resource"  # Managing computational resources
    CODE = "code"  # Generating or modifying code
    DATA = "data"  # Processing or transforming data
    SYSTEM = "system"  # System-level operations (restricted)

class AutonomyEngine:
    """
    AI Autonomy Engine
    
    Manages autonomous actions by the AI system with appropriate
    oversight, approval, and safety constraints.
    
    Key capabilities:
    1. Determine when autonomous action is appropriate
    2. Manage approval workflows for actions
    3. Monitor action outcomes and impacts
    4. Implement safety guardrails
    5. Learn from past actions and outcomes
    """
    
    def __init__(self):
        """Initialize the autonomy engine"""
        # Action history
        self.action_history = []
        self.pending_actions = []
        
        # Configuration
        self.autonomy_config = {
            AutonomyLevel.NONE: {
                "requires_approval": True,
                "requires_notification": True,
                "allowed_categories": [ActionCategory.INFORMATION]
            },
            AutonomyLevel.GUIDED: {
                "requires_approval": True,
                "requires_notification": True,
                "allowed_categories": [
                    ActionCategory.INFORMATION,
                    ActionCategory.COMMUNICATION,
                    ActionCategory.DATA
                ]
            },
            AutonomyLevel.SUPERVISED: {
                "requires_approval": False,
                "requires_notification": True,
                "allowed_categories": [
                    ActionCategory.INFORMATION,
                    ActionCategory.COMMUNICATION,
                    ActionCategory.RESOURCE,
                    ActionCategory.DATA,
                    ActionCategory.CODE
                ]
            },
            AutonomyLevel.FULL: {
                "requires_approval": False,
                "requires_notification": False,
                "allowed_categories": [category for category in ActionCategory]
            }
        }
        
        # Current autonomy level (default to guided)
        self.current_autonomy_level = AutonomyLevel.GUIDED
        
        # Admin-configured constraints
        self.admin_constraints = {
            "max_actions_per_minute": 10,
            "max_resource_utilization": 0.7,  # 70% of available resources
            "restricted_actions": [
                "system_update",
                "security_change",
                "network_configuration",
                "user_management"
            ],
            "allowed_domains": [
                "api.openai.com",
                "huggingface.co",
                "github.com"
            ]
        }
        
        # Rate-limiting state
        self.action_timestamps = []
        
        logger.info(f"Autonomy Engine initialized with {self.current_autonomy_level} autonomy level")
    
    def plan_action(
        self,
        category: ActionCategory,
        description: str,
        parameters: Dict[str, Any],
        urgency: float = 0.5,
        impact: float = 0.5,
        requires_approval: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Plan an autonomous action
        
        Args:
            category: Type of action
            description: Human-readable description
            parameters: Action parameters
            urgency: How urgent the action is (0-1)
            impact: Potential impact of the action (0-1)
            requires_approval: Override default approval requirement
            
        Returns:
            Action plan details
        """
        # Generate action ID
        action_id = str(uuid.uuid4())
        
        # Get autonomy settings for current level
        autonomy_settings = self.autonomy_config[self.current_autonomy_level]
        
        # Determine if action is allowed at current autonomy level
        if category not in autonomy_settings["allowed_categories"]:
            logger.warning(f"Action {description} of category {category} not allowed at {self.current_autonomy_level} autonomy level")
            return {
                "id": action_id,
                "status": "rejected",
                "reason": f"Action category {category} not allowed at current autonomy level",
                "category": category,
                "description": description
            }
        
        # Apply rate limiting
        current_time = time.time()
        
        # Remove timestamps older than 60 seconds
        self.action_timestamps = [ts for ts in self.action_timestamps if current_time - ts <= 60]
        
        # Check if rate limit exceeded
        if len(self.action_timestamps) >= self.admin_constraints["max_actions_per_minute"]:
            logger.warning(f"Action {description} rejected due to rate limiting")
            return {
                "id": action_id,
                "status": "rejected",
                "reason": f"Rate limit of {self.admin_constraints['max_actions_per_minute']} actions per minute exceeded",
                "category": category,
                "description": description
            }
        
        # Determine if approval is required
        needs_approval = requires_approval if requires_approval is not None else autonomy_settings["requires_approval"]
        
        # High-impact actions always require approval
        if impact > 0.7:
            needs_approval = True
        
        # Check for restricted actions
        for restricted_action in self.admin_constraints["restricted_actions"]:
            if restricted_action.lower() in description.lower():
                needs_approval = True
                break
        
        # Create action plan
        action_plan = {
            "id": action_id,
            "category": category,
            "description": description,
            "parameters": parameters,
            "urgency": urgency,
            "impact": impact,
            "status": "pending_approval" if needs_approval else "approved",
            "requires_approval": needs_approval,
            "requires_notification": autonomy_settings["requires_notification"],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Add to pending actions if approval needed
        if needs_approval:
            self.pending_actions.append(action_plan)
            logger.info(f"Action {action_id} ({description}) planned and awaiting approval")
        else:
            # Add to action timestamps for rate limiting
            self.action_timestamps.append(current_time)
            logger.info(f"Action {action_id} ({description}) planned and auto-approved")
        
        return action_plan
    
    def approve_action(self, action_id: str) -> Dict[str, Any]:
        """
        Approve a pending action
        
        Args:
            action_id: ID of action to approve
            
        Returns:
            Updated action details
        """
        # Find the action
        for i, action in enumerate(self.pending_actions):
            if action["id"] == action_id:
                # Update status
                action["status"] = "approved"
                action["updated_at"] = datetime.now().isoformat()
                action["approved_at"] = datetime.now().isoformat()
                
                # Remove from pending actions
                self.pending_actions.pop(i)
                
                # Add to action timestamps for rate limiting
                self.action_timestamps.append(time.time())
                
                logger.info(f"Action {action_id} ({action['description']}) approved")
                return action
        
        logger.warning(f"Action {action_id} not found in pending actions")
        return {
            "id": action_id,
            "status": "error",
            "reason": "Action not found in pending actions"
        }
    
    def reject_action(self, action_id: str, reason: str = "") -> Dict[str, Any]:
        """
        Reject a pending action
        
        Args:
            action_id: ID of action to reject
            reason: Reason for rejection
            
        Returns:
            Updated action details
        """
        # Find the action
        for i, action in enumerate(self.pending_actions):
            if action["id"] == action_id:
                # Update status
                action["status"] = "rejected"
                action["updated_at"] = datetime.now().isoformat()
                action["rejected_at"] = datetime.now().isoformat()
                action["rejection_reason"] = reason
                
                # Remove from pending actions
                self.pending_actions.pop(i)
                
                logger.info(f"Action {action_id} ({action['description']}) rejected: {reason}")
                return action
        
        logger.warning(f"Action {action_id} not found in pending actions")
        return {
            "id": action_id,
            "status": "error",
            "reason": "Action not found in pending actions"
        }
    
    def execute_action(
        self,
        action_id: str,
        execution_engine: Any = None
    ) -> Dict[str, Any]:
        """
        Execute an approved action
        
        Args:
            action_id: ID of approved action to execute
            execution_engine: Optional execution engine for code actions
            
        Returns:
            Execution results
        """
        # Find the action
        action = None
        
        # Check if action is in pending actions (should be approved first)
        for i, pending_action in enumerate(self.pending_actions):
            if pending_action["id"] == action_id:
                if pending_action["status"] != "approved":
                    logger.warning(f"Cannot execute action {action_id} because it is not approved")
                    return {
                        "id": action_id,
                        "status": "error",
                        "reason": f"Action is in '{pending_action['status']}' state, not 'approved'"
                    }
                
                action = pending_action
                self.pending_actions.pop(i)
                break
        
        # If not found in pending actions, check action history
        if action is None:
            for past_action in self.action_history:
                if past_action["id"] == action_id:
                    if past_action["status"] != "approved":
                        logger.warning(f"Cannot execute action {action_id} because it is in {past_action['status']} state")
                        return {
                            "id": action_id,
                            "status": "error",
                            "reason": f"Action is in '{past_action['status']}' state, not 'approved'"
                        }
                    
                    if "executed_at" in past_action:
                        logger.warning(f"Action {action_id} has already been executed")
                        return {
                            "id": action_id,
                            "status": "error",
                            "reason": "Action has already been executed"
                        }
                    
                    action = past_action
                    break
        
        if action is None:
            logger.warning(f"Action {action_id} not found")
            return {
                "id": action_id,
                "status": "error",
                "reason": "Action not found"
            }
        
        # Prepare result structure
        execution_result = {
            "id": action_id,
            "category": action["category"],
            "description": action["description"],
            "started_at": datetime.now().isoformat(),
            "success": False,
            "output": None,
            "error": None
        }
        
        try:
            # Execute action based on category
            if action["category"] == ActionCategory.INFORMATION:
                execution_result.update(self._execute_information_action(action))
            
            elif action["category"] == ActionCategory.COMMUNICATION:
                execution_result.update(self._execute_communication_action(action))
            
            elif action["category"] == ActionCategory.RESOURCE:
                execution_result.update(self._execute_resource_action(action))
            
            elif action["category"] == ActionCategory.CODE:
                if execution_engine:
                    execution_result.update(self._execute_code_action(action, execution_engine))
                else:
                    raise ValueError("Execution engine required for code actions")
            
            elif action["category"] == ActionCategory.DATA:
                execution_result.update(self._execute_data_action(action))
            
            elif action["category"] == ActionCategory.SYSTEM:
                # System actions are highly restricted
                if self.current_autonomy_level != AutonomyLevel.FULL:
                    raise ValueError("System actions only allowed at FULL autonomy level")
                
                execution_result.update(self._execute_system_action(action))
            
            else:
                raise ValueError(f"Unknown action category: {action['category']}")
            
            # If we got here without an exception, mark as successful
            execution_result["success"] = True
            
        except Exception as e:
            logger.error(f"Error executing action {action_id}: {str(e)}")
            execution_result["success"] = False
            execution_result["error"] = str(e)
        
        # Finalize execution
        execution_result["completed_at"] = datetime.now().isoformat()
        execution_result["execution_time"] = (
            datetime.fromisoformat(execution_result["completed_at"]) -
            datetime.fromisoformat(execution_result["started_at"])
        ).total_seconds()
        
        # Update action with execution results
        action["status"] = "completed" if execution_result["success"] else "failed"
        action["executed_at"] = execution_result["started_at"]
        action["execution_result"] = execution_result
        action["updated_at"] = datetime.now().isoformat()
        
        # Add to action history
        self.action_history.append(action)
        
        # Limit history size
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        logger.info(f"Action {action_id} ({action['description']}) executed with status {action['status']}")
        
        return execution_result
    
    def _execute_information_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an information retrieval or analysis action"""
        # In a real implementation, this would connect to information sources
        parameters = action["parameters"]
        
        # Simulate information retrieval
        if "query" in parameters:
            return {
                "output": f"Simulated information retrieval for query: {parameters['query']}",
                "metadata": {
                    "type": "information_retrieval",
                    "query": parameters["query"]
                }
            }
        
        elif "analyze" in parameters:
            return {
                "output": f"Simulated analysis of: {parameters['analyze']}",
                "metadata": {
                    "type": "information_analysis",
                    "subject": parameters["analyze"]
                }
            }
        
        else:
            return {
                "output": "Simulated generic information action",
                "metadata": {
                    "type": "generic_information"
                }
            }
    
    def _execute_communication_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a communication action"""
        # In a real implementation, this would handle communication channels
        parameters = action["parameters"]
        
        # Simulate communication
        if "message" in parameters and "recipient" in parameters:
            return {
                "output": f"Simulated message to {parameters['recipient']}: {parameters['message']}",
                "metadata": {
                    "type": "direct_message",
                    "recipient": parameters["recipient"]
                }
            }
        
        elif "notification" in parameters:
            return {
                "output": f"Simulated notification: {parameters['notification']}",
                "metadata": {
                    "type": "notification"
                }
            }
        
        else:
            return {
                "output": "Simulated generic communication action",
                "metadata": {
                    "type": "generic_communication"
                }
            }
    
    def _execute_resource_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a resource management action"""
        # In a real implementation, this would manage computational resources
        parameters = action["parameters"]
        
        # Check resource utilization constraints
        if "allocation" in parameters:
            allocation = float(parameters["allocation"])
            
            if allocation > self.admin_constraints["max_resource_utilization"]:
                raise ValueError(f"Resource allocation {allocation} exceeds maximum allowed {self.admin_constraints['max_resource_utilization']}")
            
            return {
                "output": f"Simulated resource allocation of {allocation}",
                "metadata": {
                    "type": "resource_allocation",
                    "allocation": allocation
                }
            }
        
        elif "release" in parameters:
            return {
                "output": f"Simulated resource release of {parameters['release']}",
                "metadata": {
                    "type": "resource_release"
                }
            }
        
        else:
            return {
                "output": "Simulated generic resource action",
                "metadata": {
                    "type": "generic_resource"
                }
            }
    
    def _execute_code_action(
        self,
        action: Dict[str, Any],
        execution_engine: Any
    ) -> Dict[str, Any]:
        """Execute a code generation or modification action"""
        # This would use the provided execution engine
        parameters = action["parameters"]
        
        if "code" in parameters and "language" in parameters:
            # Execute the code using the provided engine
            result = execution_engine.execute_code(
                code=parameters["code"],
                language=parameters["language"],
                context=parameters.get("context"),
                security_level=parameters.get("security_level", "standard")
            )
            
            return {
                "output": result["output"],
                "error": result["error"] if not result["success"] else None,
                "metadata": {
                    "type": "code_execution",
                    "language": parameters["language"],
                    "success": result["success"],
                    "execution_time": result["execution_time"]
                }
            }
        
        elif "generate" in parameters:
            return {
                "output": f"Simulated code generation for {parameters['generate']}",
                "metadata": {
                    "type": "code_generation"
                }
            }
        
        else:
            return {
                "output": "Simulated generic code action",
                "metadata": {
                    "type": "generic_code"
                }
            }
    
    def _execute_data_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data processing or transformation action"""
        # In a real implementation, this would handle data operations
        parameters = action["parameters"]
        
        if "transform" in parameters and "data" in parameters:
            return {
                "output": f"Simulated data transformation of {len(parameters['data'])} items using {parameters['transform']} method",
                "metadata": {
                    "type": "data_transformation",
                    "method": parameters["transform"],
                    "data_size": len(parameters["data"])
                }
            }
        
        elif "analyze" in parameters and "data" in parameters:
            return {
                "output": f"Simulated data analysis of {len(parameters['data'])} items",
                "metadata": {
                    "type": "data_analysis",
                    "data_size": len(parameters["data"])
                }
            }
        
        else:
            return {
                "output": "Simulated generic data action",
                "metadata": {
                    "type": "generic_data"
                }
            }
    
    def _execute_system_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a system-level action (highly restricted)"""
        # In a real implementation, this would be very restricted
        parameters = action["parameters"]
        
        # Additional security checks for system actions
        if not self._validate_system_action(action):
            raise ValueError("System action failed additional security validation")
        
        # Simulate system action
        if "operation" in parameters:
            return {
                "output": f"Simulated system operation: {parameters['operation']}",
                "metadata": {
                    "type": "system_operation",
                    "operation": parameters["operation"]
                }
            }
        
        else:
            return {
                "output": "Simulated generic system action",
                "metadata": {
                    "type": "generic_system"
                }
            }
    
    def _validate_system_action(self, action: Dict[str, Any]) -> bool:
        """Perform additional validation for system actions"""
        # System actions require additional security checks
        parameters = action["parameters"]
        
        # Check for restricted operations
        for restricted_action in self.admin_constraints["restricted_actions"]:
            if ("operation" in parameters and
                    restricted_action.lower() in parameters["operation"].lower()):
                logger.warning(f"System action contains restricted operation: {restricted_action}")
                return False
        
        # Check for network domain restrictions
        if "domain" in parameters:
            domain = parameters["domain"]
            if domain not in self.admin_constraints["allowed_domains"]:
                logger.warning(f"System action targets non-allowed domain: {domain}")
                return False
        
        # Additional checks could be implemented here
        
        return True
    
    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get list of actions pending approval"""
        return self.pending_actions
    
    def get_action_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent action history"""
        return self.action_history[-limit:]
    
    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific action"""
        # Check pending actions
        for action in self.pending_actions:
            if action["id"] == action_id:
                return action
        
        # Check action history
        for action in self.action_history:
            if action["id"] == action_id:
                return action
        
        return None
    
    def set_autonomy_level(self, level: AutonomyLevel) -> Dict[str, Any]:
        """Change the current autonomy level"""
        if level not in AutonomyLevel:
            return {
                "success": False,
                "error": f"Invalid autonomy level: {level}",
                "current_level": self.current_autonomy_level
            }
        
        old_level = self.current_autonomy_level
        self.current_autonomy_level = level
        
        logger.info(f"Autonomy level changed from {old_level} to {level}")
        
        return {
            "success": True,
            "previous_level": old_level,
            "current_level": level,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Update admin-defined constraints"""
        # Update only the provided constraints
        for key, value in constraints.items():
            if key in self.admin_constraints:
                self.admin_constraints[key] = value
        
        logger.info(f"Admin constraints updated: {constraints.keys()}")
        
        return {
            "success": True,
            "updated_constraints": list(constraints.keys()),
            "current_constraints": self.admin_constraints
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get basic status information"""
        return {
            "autonomy_level": self.current_autonomy_level,
            "pending_actions": len(self.pending_actions),
            "action_history_size": len(self.action_history)
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        # Count actions by category
        category_counts = {}
        for action in self.action_history:
            category = action["category"]
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        # Count actions by status
        status_counts = {}
        for action in self.action_history:
            status = action["status"]
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
        
        return {
            "autonomy_level": self.current_autonomy_level,
            "autonomy_config": {
                level.value: config for level, config in self.autonomy_config.items()
            },
            "admin_constraints": self.admin_constraints,
            "pending_actions": len(self.pending_actions),
            "action_categories": category_counts,
            "action_statuses": status_counts,
            "rate_limit_state": {
                "recent_actions": len(self.action_timestamps),
                "max_actions_per_minute": self.admin_constraints["max_actions_per_minute"]
            }
        }