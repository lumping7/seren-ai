"""
System Integration for Seren

Provides comprehensive integration between all system components
and the OpenManus structure for a unified, production-ready AI system.
"""

import os
import sys
import json
import logging
import time
import uuid
import importlib
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from datetime import datetime
import threading
import queue

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import system components
from ai_core.integration_framework import integration_framework, ComponentType, EventType
from ai_core.ai_engine import ai_engine
from ai_core.model_communication import communication_system, ModelType, MessageType
from ai_core.neurosymbolic_reasoning import reasoning_engine
from ai_core.ai_memory import memory_system
from ai_core.ai_execution import execution_engine
from ai_core.ai_autonomy import autonomy_engine, ActionType
from ai_evolution.ai_upgrader import ai_upgrader, UpgradeType
from ai_evolution.ai_extension_manager import extension_manager
from ai_evolution.ai_auto_training import auto_training
from ai_evolution.model_creator import model_creator
from security.quantum_encryption import quantum_encryption

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SystemIntegration:
    """
    System Integration for Seren
    
    Provides comprehensive integration between all system components
    and the OpenManus structure for a unified, production-ready AI system:
    - Component registration and initialization
    - Inter-component communication
    - Workflow management
    - Resource allocation
    - Error handling and recovery
    - Performance monitoring
    - Security enforcement
    
    Core Features:
    1. Seamless communication between AI models
    2. Efficient resource coordination
    3. Secure data flow
    4. Automated workflow management
    5. Advanced error recovery
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the system integration"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Register all system components
        self._register_components()
        
        # Connect components according to architecture
        self._connect_components()
        
        # Subscribe to important events
        self._subscribe_to_events()
        
        # Initialize models
        self._initialize_models()
        
        logger.info("System Integration initialized successfully")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load system configuration"""
        # Default configuration
        default_config = {
            "name": "Seren System",
            "version": "1.0.0",
            "log_level": "INFO",
            "models": {
                "primary": "qwen",
                "secondary": "olympic",
                "collaboration_mode": "collaborative"
            },
            "security": {
                "encryption_level": "quantum",
                "authentication_required": True
            },
            "execution": {
                "parallelism": 4,
                "default_timeout": 30,  # seconds
                "security_level": "standard"
            },
            "memory": {
                "persistence": True,
                "encryption": True
            },
            "reasoning": {
                "default_strategy": "neurosymbolic",
                "confidence_threshold": 0.7
            },
            "autonomy": {
                "level": "semi_autonomous",
                "approval_threshold": 3  # Priority level requiring approval
            },
            "evolution": {
                "auto_upgrade": True,
                "upgrader_approval_required": True,
                "continuous_learning": True
            }
        }
        
        # Load configuration file if specified
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                
                # Merge with default configuration
                self._merge_config(default_config, loaded_config)
                
                logger.info(f"Loaded configuration from {config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        return default_config
    
    def _merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """Merge override configuration into base configuration"""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _register_components(self) -> None:
        """Register all system components with the integration framework"""
        # Register AI Engine
        integration_framework.register_component(
            component_id="ai_engine",
            component=ai_engine,
            component_type=ComponentType.ENGINE,
            description="Core AI Engine for Seren",
            capabilities=["inference", "code_generation", "analysis", "planning"]
        )
        
        # Register Communication System
        integration_framework.register_component(
            component_id="communication_system",
            component=communication_system,
            component_type=ComponentType.COMMUNICATION,
            description="Model Communication System",
            capabilities=["inter_model_communication", "conversation_management", "message_routing"]
        )
        
        # Register Reasoning Engine
        integration_framework.register_component(
            component_id="reasoning_engine",
            component=reasoning_engine,
            component_type=ComponentType.REASONING,
            description="Neurosymbolic Reasoning Engine",
            capabilities=["deductive_reasoning", "inductive_reasoning", "abductive_reasoning", "neurosymbolic_reasoning"]
        )
        
        # Register Memory System
        integration_framework.register_component(
            component_id="memory_system",
            component=memory_system,
            component_type=ComponentType.MEMORY,
            description="AI Memory System",
            capabilities=["short_term_memory", "long_term_memory", "episodic_memory", "semantic_memory", "procedural_memory"]
        )
        
        # Register Execution Engine
        integration_framework.register_component(
            component_id="execution_engine",
            component=execution_engine,
            component_type=ComponentType.EXECUTION,
            description="Code Execution Engine",
            capabilities=["code_execution", "sandboxing", "execution_monitoring"]
        )
        
        # Register Autonomy Engine
        integration_framework.register_component(
            component_id="autonomy_engine",
            component=autonomy_engine,
            component_type=ComponentType.AUTONOMY,
            description="System Autonomy Engine",
            capabilities=["self_monitoring", "action_planning", "autonomous_decision_making"]
        )
        
        # Register AI Upgrader
        integration_framework.register_component(
            component_id="ai_upgrader",
            component=ai_upgrader,
            component_type=ComponentType.UPGRADER,
            description="AI Upgrader System",
            capabilities=["model_upgrading", "capability_extension", "architecture_expansion"]
        )
        
        # Register Extension Manager
        integration_framework.register_component(
            component_id="extension_manager",
            component=extension_manager,
            component_type=ComponentType.EXTENSION,
            description="Extension Manager System",
            capabilities=["extension_management", "capability_discovery", "plugin_handling"]
        )
        
        # Register Auto-Training
        integration_framework.register_component(
            component_id="auto_training",
            component=auto_training,
            component_type=ComponentType.TRAINING,
            description="Auto-Training System",
            capabilities=["model_training", "dataset_management", "training_optimization"]
        )
        
        # Register Model Creator
        integration_framework.register_component(
            component_id="model_creator",
            component=model_creator,
            component_type=ComponentType.MODEL_CREATOR,
            description="Model Creator System",
            capabilities=["model_creation", "architecture_design", "optimization"]
        )
        
        # Register Quantum Encryption
        integration_framework.register_component(
            component_id="quantum_encryption",
            component=quantum_encryption,
            component_type=ComponentType.SECURITY,
            description="Quantum Encryption System",
            capabilities=["quantum_encryption", "key_management", "secure_communication"]
        )
    
    def _connect_components(self) -> None:
        """Connect components according to the system architecture"""
        # AI Engine connections (central hub)
        integration_framework.connect_components("ai_engine", "reasoning_engine", "reasoning")
        integration_framework.connect_components("ai_engine", "memory_system", "memory")
        integration_framework.connect_components("ai_engine", "execution_engine", "execution")
        integration_framework.connect_components("ai_engine", "communication_system", "communication")
        
        # Communication System connections
        integration_framework.connect_components("communication_system", "ai_engine", "model_interface")
        integration_framework.connect_components("communication_system", "reasoning_engine", "reasoning_queries")
        
        # Reasoning Engine connections
        integration_framework.connect_components("reasoning_engine", "memory_system", "knowledge_access")
        integration_framework.connect_components("reasoning_engine", "ai_engine", "reasoning_results")
        
        # Memory System connections
        integration_framework.connect_components("memory_system", "reasoning_engine", "memory_retrieval")
        integration_framework.connect_components("memory_system", "quantum_encryption", "secure_storage")
        
        # Execution Engine connections
        integration_framework.connect_components("execution_engine", "ai_engine", "execution_results")
        integration_framework.connect_components("execution_engine", "memory_system", "execution_history")
        
        # Autonomy Engine connections
        integration_framework.connect_components("autonomy_engine", "ai_engine", "system_monitoring")
        integration_framework.connect_components("autonomy_engine", "memory_system", "action_history")
        integration_framework.connect_components("autonomy_engine", "reasoning_engine", "decision_support")
        
        # AI Upgrader connections
        integration_framework.connect_components("ai_upgrader", "ai_engine", "model_upgrades")
        integration_framework.connect_components("ai_upgrader", "auto_training", "upgrade_training")
        integration_framework.connect_components("ai_upgrader", "model_creator", "new_architectures")
        
        # Extension Manager connections
        integration_framework.connect_components("extension_manager", "ai_engine", "capability_extension")
        integration_framework.connect_components("extension_manager", "reasoning_engine", "reasoning_extensions")
        
        # Auto-Training connections
        integration_framework.connect_components("auto_training", "ai_engine", "training_coordination")
        integration_framework.connect_components("auto_training", "memory_system", "training_data")
        
        # Model Creator connections
        integration_framework.connect_components("model_creator", "ai_engine", "model_integration")
        integration_framework.connect_components("model_creator", "auto_training", "model_training")
        
        # Quantum Encryption connections
        integration_framework.connect_components("quantum_encryption", "communication_system", "secure_messages")
        integration_framework.connect_components("quantum_encryption", "memory_system", "secure_storage")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to important system events"""
        # Component failure handling
        integration_framework.subscribe_to_events(
            EventType.COMPONENT_FAILED,
            self._handle_component_failure
        )
        
        # Request processing events
        integration_framework.subscribe_to_events(
            EventType.REQUEST_RECEIVED,
            self._handle_request_received
        )
        
        # Error handling
        integration_framework.subscribe_to_events(
            EventType.ERROR_OCCURRED,
            self._handle_error
        )
    
    def _handle_component_failure(self, event: Dict[str, Any]) -> None:
        """Handle component failure events"""
        component_id = event["data"].get("component_id")
        
        logger.warning(f"Component failure detected: {component_id}")
        
        # Create recovery action using Autonomy Engine
        autonomy_engine.propose_action(
            action_type=ActionType.RECOVERY,
            description=f"Recover from {component_id} failure",
            target_component=component_id,
            parameters={"event": event},
            priority=4  # High priority
        )
    
    def _handle_request_received(self, event: Dict[str, Any]) -> None:
        """Handle request received events"""
        request_id = event["data"].get("request_id")
        request_type = event["data"].get("request_type")
        
        logger.info(f"Request received: {request_id} ({request_type})")
        
        # Update memory with request information
        memory_system.add_memory(
            memory_type="episodic",
            content={
                "event": "request_received",
                "request_id": request_id,
                "request_type": request_type,
                "timestamp": event["timestamp"]
            }
        )
    
    def _handle_error(self, event: Dict[str, Any]) -> None:
        """Handle error events"""
        error = event["data"].get("error")
        request_id = event["data"].get("request_id")
        
        logger.error(f"Error in request {request_id}: {error}")
        
        # Create recovery action using Autonomy Engine
        autonomy_engine.propose_action(
            action_type=ActionType.RECOVERY,
            description=f"Recover from error in request {request_id}",
            target_component="ai_engine",
            parameters={"error": error, "request_id": request_id},
            priority=3
        )
    
    def _initialize_models(self) -> None:
        """Initialize the AI models"""
        primary_model = self.config["models"]["primary"]
        secondary_model = self.config["models"]["secondary"]
        collaboration_mode = self.config["models"]["collaboration_mode"]
        
        logger.info(f"Initializing models: {primary_model} and {secondary_model} in {collaboration_mode} mode")
        
        # Initialize AI Engine with models
        ai_engine.initialize_models(
            primary_model=primary_model,
            secondary_model=secondary_model,
            collaboration_mode=collaboration_mode
        )
        
        # Create model communication
        if collaboration_mode == "collaborative":
            # Create bi-directional communication channel
            conversation_id = communication_system.create_conversation(
                topic="Primary collaboration channel",
                participants=[ModelType.QWEN, ModelType.OLYMPIC]
            )
            
            # Store conversation ID
            self.primary_conversation_id = conversation_id
            
            logger.info(f"Created primary collaboration channel: {conversation_id}")
        
        elif collaboration_mode == "specialized":
            # Create specialized channels
            planning_conversation = communication_system.create_conversation(
                topic="Planning channel",
                participants=[ModelType.QWEN, ModelType.OLYMPIC]
            )
            
            implementation_conversation = communication_system.create_conversation(
                topic="Implementation channel",
                participants=[ModelType.QWEN, ModelType.OLYMPIC]
            )
            
            # Store conversation IDs
            self.planning_conversation_id = planning_conversation
            self.implementation_conversation_id = implementation_conversation
            
            logger.info(f"Created specialized channels: planning={planning_conversation}, implementation={implementation_conversation}")
        
        elif collaboration_mode == "competitive":
            # Create competitive evaluation channels
            qwen_conversation = communication_system.create_conversation(
                topic="Qwen evaluation channel",
                participants=[ModelType.QWEN, ModelType.SYSTEM]
            )
            
            olympic_conversation = communication_system.create_conversation(
                topic="Olympic evaluation channel",
                participants=[ModelType.OLYMPIC, ModelType.SYSTEM]
            )
            
            # Store conversation IDs
            self.qwen_conversation_id = qwen_conversation
            self.olympic_conversation_id = olympic_conversation
            
            logger.info(f"Created competitive channels: qwen={qwen_conversation}, olympic={olympic_conversation}")
    
    def process_user_query(
        self,
        query: str,
        mode: str = "default",
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the integrated system
        
        Args:
            query: User query text
            mode: Processing mode (default, code, analysis, etc.)
            context: Additional context information
            
        Returns:
            Response data
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        logger.info(f"Processing user query ({request_id}): {query[:50]}...")
        
        # Prepare request for integration framework
        request = {
            "type": "query",
            "payload": {
                "query": query,
                "mode": mode,
                "context": context or {}
            }
        }
        
        # Process through integration framework
        response = integration_framework.process_request("query", request)
        
        # Add to memory
        memory_system.add_memory(
            memory_type="episodic",
            content={
                "type": "user_query",
                "query": query,
                "mode": mode,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return response
    
    def generate_code(
        self,
        specification: str,
        language: str = "python",
        test_driven: bool = False,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate code based on a specification
        
        Args:
            specification: Code specification
            language: Programming language
            test_driven: Whether to use test-driven development
            context: Additional context information
            
        Returns:
            Generated code and metadata
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        logger.info(f"Generating code ({request_id}): {specification[:50]}...")
        
        # Prepare request for integration framework
        request = {
            "type": "generate",
            "payload": {
                "specification": specification,
                "language": language,
                "test_driven": test_driven,
                "context": context or {}
            }
        }
        
        # Process through integration framework
        response = integration_framework.process_request("generate", request)
        
        # Add to memory
        memory_system.add_memory(
            memory_type="episodic",
            content={
                "type": "code_generation",
                "specification": specification,
                "language": language,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Execute code if generated successfully and requested
        if response.get("success") and response.get("data", {}).get("generation", {}).get("code"):
            code = response["data"]["generation"]["code"]
            
            if context and context.get("execute_code", False):
                logger.info(f"Executing generated code for request {request_id}")
                
                execution_result = execution_engine.execute_code(
                    code=code,
                    language=language,
                    context={"request_id": request_id}
                )
                
                # Add execution result to response
                response["data"]["execution"] = execution_result
        
        return response
    
    def analyze_data(
        self,
        data: Any,
        analysis_type: str = "general",
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze data using the system
        
        Args:
            data: Data to analyze
            analysis_type: Type of analysis to perform
            context: Additional context information
            
        Returns:
            Analysis results
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        logger.info(f"Analyzing data ({request_id}): {analysis_type}")
        
        # Prepare request for integration framework
        request = {
            "type": "analyze",
            "payload": {
                "data": data,
                "analysis_type": analysis_type,
                "context": context or {}
            }
        }
        
        # Process through integration framework
        response = integration_framework.process_request("analyze", request)
        
        # Apply reasoning if needed
        if analysis_type in ["complex", "deep", "causal"]:
            logger.info(f"Applying additional reasoning for {analysis_type} analysis")
            
            reasoning_result = reasoning_engine.apply_reasoning(
                data=response["data"]["analysis"],
                strategy="neurosymbolic",
                context={"request_id": request_id}
            )
            
            # Add reasoning result to response
            response["data"]["reasoning"] = reasoning_result
        
        return response
    
    def train_model(
        self,
        model_id: str,
        dataset_ids: List[str],
        training_parameters: Dict[str, Any] = None,
        auto_deploy: bool = False
    ) -> Dict[str, Any]:
        """
        Train a model using the auto-training system
        
        Args:
            model_id: ID of the model to train
            dataset_ids: List of dataset IDs to use for training
            training_parameters: Training parameters
            auto_deploy: Whether to automatically deploy the trained model
            
        Returns:
            Training session information
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        logger.info(f"Training model ({request_id}): {model_id}")
        
        # Create training session
        session = auto_training.create_training_session(
            model_id=model_id,
            description=f"Training session {request_id}",
            strategy="supervised",  # Default strategy
            dataset_ids=dataset_ids,
            parameters=training_parameters or {},
            auto_start=True
        )
        
        # Wait for training to complete in production system
        # Here we just return the session info
        
        # If auto-deploy is requested, set up deployment when training completes
        if auto_deploy and "id" in session:
            # In production, this would set up a callback
            session["auto_deploy"] = True
            
            logger.info(f"Auto-deployment configured for training session {session['id']}")
        
        return {
            "success": True,
            "training_session": session
        }
    
    def upgrade_system(
        self,
        upgrade_type: str,
        target: str,
        description: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Upgrade a system component
        
        Args:
            upgrade_type: Type of upgrade to perform
            target: Target component or model
            description: Upgrade description
            parameters: Upgrade parameters
            
        Returns:
            Upgrade information
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        logger.info(f"Upgrading system ({request_id}): {upgrade_type} for {target}")
        
        # Plan the upgrade
        upgrade = ai_upgrader.plan_upgrade(
            upgrade_type=upgrade_type,
            target=target,
            description=description,
            parameters=parameters or {},
            expected_benefits=[f"Improvement from request {request_id}"],
            admin_approved=False  # Always require explicit approval
        )
        
        # Check if upgrade requires approval
        if "id" in upgrade:
            logger.info(f"Upgrade planned: {upgrade['id']} - Requires approval")
            
            # Create approval action
            autonomy_engine.propose_action(
                action_type=ActionType.OPTIMIZATION,
                description=f"Approve upgrade {upgrade['id']} for {target}",
                target_component=target,
                parameters={"upgrade_id": upgrade["id"]},
                priority=2,
                requires_approval=True
            )
        
        return {
            "success": "id" in upgrade,
            "upgrade": upgrade
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the overall system status"""
        # Get component statuses
        ai_engine_status = ai_engine.get_status()
        communication_status = communication_system.get_status()
        reasoning_status = reasoning_engine.get_status()
        memory_status = memory_system.get_status()
        execution_status = execution_engine.get_status()
        autonomy_status = autonomy_engine.get_status()
        upgrader_status = ai_upgrader.get_status()
        extension_status = extension_manager.get_status()
        training_status = auto_training.get_status()
        model_creator_status = model_creator.get_status()
        security_status = quantum_encryption.get_status()
        
        # Get framework status
        framework_status = integration_framework.get_system_status()
        
        # Determine overall system status
        components_operational = (
            ai_engine_status.get("operational", False) and
            communication_status.get("operational", False) and
            reasoning_status.get("operational", False) and
            memory_status.get("operational", False) and
            execution_status.get("operational", False) and
            autonomy_status.get("operational", False) and
            upgrader_status.get("operational", False) and
            extension_status.get("operational", False) and
            training_status.get("operational", False) and
            model_creator_status.get("operational", False) and
            security_status.get("operational", False)
        )
        
        # Compile comprehensive status report
        return {
            "system_name": self.config["name"],
            "version": self.config["version"],
            "status": "operational" if components_operational else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ai_engine": ai_engine_status,
                "communication": communication_status,
                "reasoning": reasoning_status,
                "memory": memory_status,
                "execution": execution_status,
                "autonomy": autonomy_status,
                "upgrader": upgrader_status,
                "extension": extension_status,
                "training": training_status,
                "model_creator": model_creator_status,
                "security": security_status
            },
            "framework": framework_status,
            "configuration": {
                "models": self.config["models"],
                "security": {
                    "encryption_level": self.config["security"]["encryption_level"]
                },
                "autonomy": {
                    "level": self.config["autonomy"]["level"]
                }
            }
        }

# Initialize system integration
system_integration = SystemIntegration()