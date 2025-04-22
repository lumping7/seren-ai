"""
Seren: Hyperintelligent AI Dev Team

Main API interface for the bleeding-edge AI development team, providing access to
all components including models, reasoning, memory, execution, and evolution.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="Hyperintelligent AI System",
    description="Bleeding-edge AI system with hybrid models, neuro-symbolic reasoning, and self-evolution",
    version="1.0.0"
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import AI components - with appropriate error handling for missing modules
try:
    # AI Engine components
    from ai_core.ai_engine import AIEngine, AIEngineMode, ModelType
    from ai_core.model_communication import CommunicationSystem, MessageType
    from ai_core.neurosymbolic_reasoning import NeuroSymbolicEngine, ReasoningStrategy
    from ai_core.ai_memory import MemorySystem, MemoryType
    from ai_core.ai_execution import ExecutionEngine, ExecutionSecurity
    from ai_core.ai_autonomy import AutonomyEngine, ActionCategory
    
    # AI Evolution components
    from ai_evolution.ai_upgrader import AIUpgrader, UpgradeType
    from ai_evolution.ai_extension_manager import ExtensionManager, ExtensionStatus
    from ai_evolution.ai_auto_training import AutoTrainingSystem, TrainingStrategy
    from ai_evolution.model_creator import ModelCreator, ModelSpecialization
    
    # Security components
    from security.quantum_encryption import QuantumEncryption, SecurityLevel
    
    # Initialize components
    ai_engine = AIEngine()
    communication_system = CommunicationSystem()
    reasoning_engine = NeuroSymbolicEngine()
    memory_system = MemorySystem()
    execution_engine = ExecutionEngine()
    autonomy_engine = AutonomyEngine()
    
    ai_upgrader = AIUpgrader()
    extension_manager = ExtensionManager()
    auto_training = AutoTrainingSystem()
    model_creator = ModelCreator()
    
    quantum_encryption = QuantumEncryption()
    
    # Track initialization success
    components_initialized = True
    missing_components = []
    
except ImportError as e:
    logger.warning(f"Some components could not be imported: {str(e)}")
    components_initialized = False
    missing_components = [str(e)]

# API models for requests and responses
class AIQuery(BaseModel):
    """Query for the AI system"""
    query: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    mode: Optional[str] = "collaborative"
    use_reasoning: Optional[bool] = True
    use_memory: Optional[bool] = True
    execute_code: Optional[bool] = False
    model_preference: Optional[str] = None
    conversation_id: Optional[str] = None

class AIResponse(BaseModel):
    """Response from the AI system"""
    response: str
    conversation_id: str
    reasoning_path: Optional[List[Dict[str, Any]]] = None
    execution_results: Optional[Dict[str, Any]] = None
    memory_accessed: Optional[List[Dict[str, Any]]] = None
    model_used: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CommunicationRequest(BaseModel):
    """Request for model-to-model communication"""
    from_model: str
    to_model: str
    content: str
    message_type: str
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ReasoningRequest(BaseModel):
    """Request for neuro-symbolic reasoning"""
    query: str
    context: Optional[Dict[str, Any]] = None
    strategy: Optional[str] = None

class ExecutionRequest(BaseModel):
    """Request for code execution"""
    code: str
    language: str
    context: Optional[Dict[str, Any]] = None
    security_level: Optional[str] = "standard"

class MemoryQueryRequest(BaseModel):
    """Request to query the memory system"""
    query: str
    memory_type: Optional[str] = "episodic"
    limit: Optional[int] = 10

class ModelCreationRequest(BaseModel):
    """Request to create a specialized model"""
    name: str
    specialization: str
    description: Optional[str] = ""
    scale: Optional[str] = None
    domain_knowledge: Optional[List[str]] = None
    custom_capabilities: Optional[List[str]] = None
    optimize_for: Optional[List[str]] = None

class UpgradeRequest(BaseModel):
    """Request to upgrade the AI system"""
    upgrade_type: str
    description: str
    files_to_modify: Optional[List[str]] = None
    required_capabilities: Optional[List[str]] = None
    priority: Optional[int] = 3
    admin_approved: Optional[bool] = False

class ExtensionRequest(BaseModel):
    """Request to manage extensions"""
    action: str  # install, uninstall, enable, disable
    extension_id: Optional[str] = None
    name: Optional[str] = None
    extension_type: Optional[str] = None
    description: Optional[str] = ""

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system overview"""
    if not components_initialized:
        return {
            "status": "initializing",
            "message": "Some components are still initializing or missing",
            "missing_components": missing_components
        }
    
    return {
        "name": "Hyperintelligent AI System",
        "status": "operational",
        "version": "1.0.0",
        "components": {
            "ai_engine": ai_engine.get_status(),
            "reasoning": reasoning_engine.get_status(),
            "memory": memory_system.get_status(),
            "execution": execution_engine.get_status(),
            "autonomy": autonomy_engine.get_status(),
            "evolution": {
                "upgrader": ai_upgrader.get_status(),
                "extension_manager": extension_manager.get_status(),
                "auto_training": auto_training.get_status(),
                "model_creator": model_creator.get_status()
            },
            "security": quantum_encryption.get_status()
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components_initialized": components_initialized
    }

# Main AI query endpoint
@app.post("/api/query", response_model=AIResponse)
async def query_ai(query: AIQuery, background_tasks: BackgroundTasks):
    """
    Main endpoint for querying the AI system
    
    This integrates all capabilities including:
    - Model selection and collaboration
    - Neuro-symbolic reasoning
    - Memory access
    - Code execution (if enabled)
    - Self-improvement monitoring
    """
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Process with reasoning if enabled
        reasoning_path = None
        if query.use_reasoning:
            reasoning_result = reasoning_engine.reason(
                query=query.query,
                context=query.context
            )
            reasoning_path = reasoning_result.get("reasoning_path")
        
        # Access memory if enabled
        memory_results = None
        if query.use_memory:
            memory_results = memory_system.query_relevant(
                query=query.query,
                context=query.context
            )
        
        # Determine the AI mode to use
        mode = AIEngineMode(query.mode) if query.mode else AIEngineMode.COLLABORATIVE
        
        # Generate AI response
        ai_response = ai_engine.generate_response(
            query=query.query,
            context=query.context,
            reasoning_path=reasoning_path,
            memory_results=memory_results,
            mode=mode
        )
        
        # Execute code if enabled and detected
        execution_results = None
        if query.execute_code and execution_engine.contains_executable_code(ai_response):
            code_blocks = execution_engine.extract_code_blocks(ai_response)
            execution_results = {}
            
            for i, code_block in enumerate(code_blocks):
                execution_result = execution_engine.execute_code(
                    code=code_block["code"],
                    language=code_block["language"],
                    context=query.context
                )
                execution_results[f"block_{i}"] = execution_result
            
            # Enhance response with execution results
            ai_response = ai_engine.enhance_response_with_execution(
                original_response=ai_response,
                execution_results=execution_results
            )
        
        # Store interaction in memory asynchronously
        background_tasks.add_task(
            memory_system.store,
            content={
                "query": query.query,
                "response": ai_response,
                "context": query.context,
                "timestamp": time.time(),
                "conversation_id": query.conversation_id
            },
            memory_type=MemoryType.EPISODIC
        )
        
        # Check if this query has training value
        if auto_training and _has_training_value(query.query, ai_response):
            background_tasks.add_task(
                auto_training.collect_training_data,
                data={
                    "query": query.query,
                    "response": ai_response,
                    "context": query.context,
                    "reasoning_path": reasoning_path,
                    "memory_results": memory_results
                },
                data_type="interaction",
                source="api_query"
            )
        
        # Prepare response model
        response = AIResponse(
            response=ai_response,
            conversation_id=query.conversation_id or str(time.time()),
            reasoning_path=reasoning_path,
            execution_results=execution_results,
            memory_accessed=memory_results,
            model_used=ai_engine.last_used_model.value if ai_engine.last_used_model else None,
            metadata={
                "timestamp": time.time(),
                "query_mode": query.mode,
                "reasoning_enabled": query.use_reasoning,
                "memory_enabled": query.use_memory,
                "execution_enabled": query.execute_code
            }
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

def _has_training_value(query: str, response: str) -> bool:
    """Determine if an interaction has value for training"""
    # This would be more sophisticated in a real implementation
    # For now, let's say 10% of interactions are valuable for training
    return len(query) > 50 and len(response) > 200

# Model communication endpoint
@app.post("/api/communication")
async def model_communication(request: CommunicationRequest):
    """Endpoint for model-to-model communication"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Convert string models to enum types
        from_model = getattr(ModelType, request.from_model.upper()) if hasattr(ModelType, request.from_model.upper()) else request.from_model
        to_model = getattr(ModelType, request.to_model.upper()) if hasattr(ModelType, request.to_model.upper()) else request.to_model
        message_type = getattr(MessageType, request.message_type.upper()) if hasattr(MessageType, request.message_type.upper()) else request.message_type
        
        # Send the message
        message = communication_system.ask_question(
            from_model=from_model,
            to_model=to_model,
            content=request.content,
            message_type=message_type,
            context=request.context,
            metadata=request.metadata
        )
        
        return message
    
    except Exception as e:
        logger.error(f"Error in model communication: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in model communication: {str(e)}"
        )

# Answer a model message
@app.post("/api/communication/{message_id}/answer")
async def answer_message(message_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
    """Endpoint to answer a message from another model"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Answer the message
        answer = communication_system.answer_question(
            question_id=message_id,
            content=content,
            metadata=metadata
        )
        
        return answer
    
    except Exception as e:
        logger.error(f"Error answering message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error answering message: {str(e)}"
        )

# Reasoning endpoint
@app.post("/api/reasoning")
async def perform_reasoning(request: ReasoningRequest):
    """Endpoint for neuro-symbolic reasoning"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Convert strategy to enum if specified
        strategy = None
        if request.strategy:
            strategy = getattr(ReasoningStrategy, request.strategy.upper()) if hasattr(ReasoningStrategy, request.strategy.upper()) else request.strategy
        
        # Perform reasoning
        result = reasoning_engine.reason(
            query=request.query,
            context=request.context,
            strategy=strategy
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error in reasoning: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in reasoning: {str(e)}"
        )

# Code execution endpoint
@app.post("/api/execute")
async def execute_code(request: ExecutionRequest):
    """Endpoint for code execution"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Execute the code
        result = execution_engine.execute_code(
            code=request.code,
            language=request.language,
            context=request.context,
            security_level=request.security_level
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing code: {str(e)}"
        )

# Memory query endpoint
@app.post("/api/memory")
async def query_memory(request: MemoryQueryRequest):
    """Endpoint to query the memory system"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Convert memory type to enum if possible
        memory_type = getattr(MemoryType, request.memory_type.upper()) if hasattr(MemoryType, request.memory_type.upper()) else request.memory_type
        
        # Query memory
        results = memory_system.query(
            query=request.query,
            memory_type=memory_type,
            limit=request.limit
        )
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error querying memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error querying memory: {str(e)}"
        )

# Create specialized model endpoint
@app.post("/api/models/create")
async def create_model(request: ModelCreationRequest):
    """Endpoint to create a specialized model"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Create the model
        result = model_creator.create_specialized_model(
            name=request.name,
            specialization=request.specialization,
            description=request.description,
            scale=request.scale,
            domain_knowledge=request.domain_knowledge,
            custom_capabilities=request.custom_capabilities,
            optimize_for=request.optimize_for,
            auto_start=True
        )
        
        # Check if result is a task ID or an error
        if isinstance(result, str):
            return {
                "success": True,
                "task_id": result,
                "message": f"Model creation started with task ID: {result}"
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "details": result
            }
    
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating model: {str(e)}"
        )

# List models endpoint
@app.get("/api/models")
async def list_models(
    specialization: Optional[str] = None,
    scale: Optional[str] = None,
    limit: int = 10
):
    """Endpoint to list available models"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # List models
        models = model_creator.list_models(
            specialization=specialization,
            scale=scale,
            limit=limit
        )
        
        return {"models": models}
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )

# Get model details endpoint
@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Endpoint to get model details"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Get model details
        model = model_creator.get_model_info(model_id)
        
        if not model or "error" in model:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}"
            )
        
        return model
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model: {str(e)}"
        )

# System upgrade endpoint
@app.post("/api/upgrade")
async def upgrade_system(request: UpgradeRequest):
    """Endpoint to upgrade the AI system"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Convert upgrade type to enum if possible
        upgrade_type = getattr(UpgradeType, request.upgrade_type.upper()) if hasattr(UpgradeType, request.upgrade_type.upper()) else request.upgrade_type
        
        # Plan the upgrade
        upgrade_plan = ai_upgrader.plan_upgrade(
            upgrade_type=upgrade_type,
            description=request.description,
            files_to_modify=request.files_to_modify,
            required_capabilities=request.required_capabilities,
            priority=request.priority,
            admin_approved=request.admin_approved
        )
        
        # Check if plan has errors
        if isinstance(upgrade_plan, dict) and upgrade_plan.get("status") == "failed":
            return {
                "success": False,
                "upgrade_id": upgrade_plan.get("id"),
                "errors": upgrade_plan.get("errors", []),
                "message": "Upgrade planning failed"
            }
        
        # Start analyzing code for upgrade
        analysis_result = ai_upgrader.analyze_code_for_upgrade(upgrade_plan.get("id"))
        
        return {
            "success": True,
            "upgrade_id": upgrade_plan.get("id"),
            "analysis": analysis_result,
            "message": "Upgrade planned and analyzed"
        }
    
    except Exception as e:
        logger.error(f"Error planning upgrade: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error planning upgrade: {str(e)}"
        )

# Apply upgrade endpoint
@app.post("/api/upgrade/{upgrade_id}/apply")
async def apply_upgrade(upgrade_id: str, admin_approved: bool = False):
    """Endpoint to apply a planned upgrade"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Apply the upgrade
        result = ai_upgrader.apply_upgrade(
            upgrade_id=upgrade_id,
            admin_approved=admin_approved
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error applying upgrade: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error applying upgrade: {str(e)}"
        )

# Extension management endpoint
@app.post("/api/extensions")
async def manage_extension(request: ExtensionRequest):
    """Endpoint to manage extensions"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        if request.action == "install":
            if request.extension_id:
                # Install existing extension
                result = extension_manager.install_extension(request.extension_id)
                return result
            elif request.name and request.extension_type:
                # Create and install new extension
                result = extension_manager.create_extension_template(
                    name=request.name,
                    extension_type=request.extension_type,
                    description=request.description
                )
                
                if isinstance(result, dict) and result.get("success"):
                    # Auto-install the new extension
                    install_result = extension_manager.install_extension(result["extension_id"])
                    result["install_result"] = install_result
                
                return result
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide either extension_id or (name and extension_type)"
                )
        
        elif request.action == "uninstall":
            if not request.extension_id:
                raise HTTPException(
                    status_code=400,
                    detail="extension_id is required for uninstall"
                )
            
            result = extension_manager.uninstall_extension(request.extension_id)
            return result
        
        elif request.action == "enable":
            if not request.extension_id:
                raise HTTPException(
                    status_code=400,
                    detail="extension_id is required for enable"
                )
            
            result = extension_manager.enable_extension(request.extension_id)
            return result
        
        elif request.action == "disable":
            if not request.extension_id:
                raise HTTPException(
                    status_code=400,
                    detail="extension_id is required for disable"
                )
            
            result = extension_manager.disable_extension(request.extension_id)
            return result
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error managing extension: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error managing extension: {str(e)}"
        )

# List extensions endpoint
@app.get("/api/extensions")
async def list_extensions(status: Optional[str] = None):
    """Endpoint to list extensions"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # List extensions
        extensions = extension_manager.list_extensions(status=status)
        
        return {"extensions": extensions}
    
    except Exception as e:
        logger.error(f"Error listing extensions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing extensions: {str(e)}"
        )

# Start auto-training endpoint
@app.post("/api/training/start")
async def start_training(
    model_name: str,
    description: Optional[str] = "Automatic training session",
    strategy: Optional[str] = None,
    dataset_ids: Optional[List[str]] = None
):
    """Endpoint to start an auto-training session"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Convert strategy to enum if possible
        if strategy:
            strategy = getattr(TrainingStrategy, strategy.upper()) if hasattr(TrainingStrategy, strategy.upper()) else strategy
        
        # Create training session
        session_id = auto_training.create_training_session(
            model_name=model_name,
            description=description,
            strategy=strategy,
            dataset_ids=dataset_ids,
            auto_start=True
        )
        
        if isinstance(session_id, str):
            return {
                "success": True,
                "session_id": session_id,
                "message": f"Training started with session ID: {session_id}"
            }
        else:
            return {
                "success": False,
                "error": session_id.get("error", "Unknown error"),
                "details": session_id
            }
    
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting training: {str(e)}"
        )

# Get detailed system status
@app.get("/api/status")
async def system_status():
    """Get detailed system status"""
    if not components_initialized:
        raise HTTPException(
            status_code=503,
            detail="AI system is still initializing components"
        )
    
    try:
        # Compile status from all components
        status = {
            "system": {
                "name": "Hyperintelligent AI System",
                "version": "1.0.0",
                "uptime": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0
            },
            "components": {
                "ai_engine": ai_engine.get_detailed_status(),
                "communication": communication_system.get_detailed_status(),
                "reasoning": reasoning_engine.get_detailed_status(),
                "memory": memory_system.get_detailed_status(),
                "execution": execution_engine.get_detailed_status(),
                "autonomy": autonomy_engine.get_detailed_status()
            },
            "evolution": {
                "upgrader": ai_upgrader.get_detailed_status(),
                "extension_manager": extension_manager.get_detailed_status(),
                "auto_training": auto_training.get_detailed_status(),
                "model_creator": model_creator.get_status()
            },
            "security": quantum_encryption.get_status()
        }
        
        return status
    
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system status: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Handle application startup events"""
    app.state.start_time = time.time()
    logger.info("Hyperintelligent AI API starting up")

# Run the server directly when this file is executed
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)