"""
Seren Server

Main server implementation for the Seren hyperintelligent AI dev team.
Provides RESTful API endpoints for interacting with the system.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

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

# Create the FastAPI app
app = FastAPI(
    title="Seren - Hyperintelligent AI Dev Team",
    description="Bleeding-edge AI development team with hybrid models, neuro-symbolic reasoning, and self-evolution",
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

# Import AI components
try:
    # Import core components
    from ai_core.ai_engine import AIEngine
    from ai_core.model_communication import CommunicationSystem, ModelType, MessageType
    from ai_core.neurosymbolic_reasoning import NeuroSymbolicEngine, ReasoningStrategy
    from ai_core.ai_memory import MemorySystem, MemoryType
    from ai_core.ai_execution import ExecutionEngine, ExecutionSecurity
    from ai_core.ai_autonomy import AutonomyEngine
    
    # Import evolution components
    from ai_evolution.ai_upgrader import AIUpgrader
    from ai_evolution.ai_extension_manager import ExtensionManager
    from ai_evolution.ai_auto_training import AutoTrainingSystem
    from ai_evolution.model_creator import ModelCreator
    
    # Import security components
    from security.quantum_encryption import QuantumEncryption
    
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
    
    # Record initialization success
    components_initialized = True
    missing_components = []

except ImportError as e:
    logger.warning(f"Failed to import all components: {str(e)}")
    components_initialized = False
    missing_components = [str(e)]

# Define API models
class Query(BaseModel):
    """Query for the AI system"""
    text: str
    mode: Optional[str] = "collaborative"  # collaborative, specialized, competitive
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict)

class Response(BaseModel):
    """Response from the AI system"""
    text: str
    execution_time: float
    reasoning_path: Optional[List[Dict[str, Any]]] = None
    models_used: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class CodeRequest(BaseModel):
    """Code generation/modification request"""
    description: str
    language: Optional[str] = "python"
    existing_code: Optional[str] = None
    requirements: Optional[List[str]] = None
    tests: Optional[bool] = True
    mode: Optional[str] = "collaborative"

class CodeResponse(BaseModel):
    """Code generation/modification response"""
    code: str
    explanation: str
    tests: Optional[str] = None
    reasoning_path: Optional[List[Dict[str, Any]]] = None
    models_used: Optional[List[str]] = None

class DebugRequest(BaseModel):
    """Debugging request"""
    code: str
    language: str
    error: Optional[str] = None
    expected_behavior: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DebugResponse(BaseModel):
    """Debugging response"""
    fixed_code: str
    explanation: str
    root_cause: str
    changes_made: List[str]
    reasoning_path: Optional[List[Dict[str, Any]]] = None

class ArchitectureRequest(BaseModel):
    """Architecture design request"""
    description: str
    requirements: List[str]
    constraints: Optional[List[str]] = None
    technologies: Optional[List[str]] = None
    scale: Optional[str] = "medium"  # small, medium, large, enterprise

class ArchitectureResponse(BaseModel):
    """Architecture design response"""
    design: str
    diagram_code: Optional[str] = None
    components: List[Dict[str, Any]]
    justification: str
    alternatives_considered: Optional[List[str]] = None

class ModelCommunicationRequest(BaseModel):
    """Model communication request"""
    from_model: str  # qwen, olympic
    to_model: str  # qwen, olympic
    message: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ModelCommunicationResponse(BaseModel):
    """Model communication response"""
    message_id: str
    response: Optional[str] = None
    status: str  # sent, delivered, awaiting_response, completed

@app.get("/")
async def root():
    """Root endpoint for the Seren system"""
    if not components_initialized:
        return {
            "status": "initializing",
            "name": "Seren",
            "version": "1.0.0",
            "missing_components": missing_components
        }
    
    return {
        "status": "operational",
        "name": "Seren - Hyperintelligent AI Dev Team",
        "version": "1.0.0",
        "components": {
            "ai_engine": "operational",
            "communication": "operational",
            "reasoning": "operational",
            "memory": "operational",
            "execution": "operational",
            "autonomy": "operational",
            "evolution": {
                "upgrader": "operational",
                "extension_manager": "operational",
                "auto_training": "operational",
                "model_creator": "operational"
            },
            "security": "operational"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components_initialized": components_initialized
    }

@app.get("/status")
async def detailed_status():
    """Get detailed system status"""
    if not components_initialized:
        return {
            "status": "initializing",
            "missing_components": missing_components
        }
    
    return {
        "status": "operational",
        "components": {
            "ai_engine": ai_engine.get_status(),
            "communication": communication_system.get_status(),
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

@app.post("/api/query", response_model=Response)
async def query(query: Query):
    """General query endpoint for the AI system"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    start_time = time.time()
    
    try:
        # Process query using the AI engine
        result = ai_engine.process_query(
            query=query.text,
            mode=query.mode,
            context=query.context,
            settings=query.settings
        )
        
        # Extract the response and metadata
        response_text = result.get("response", "")
        reasoning_path = result.get("reasoning_path")
        models_used = result.get("models_used", [])
        metadata = result.get("metadata", {})
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return Response(
            text=response_text,
            execution_time=execution_time,
            reasoning_path=reasoning_path,
            models_used=models_used,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/code/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """Generate code based on description"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Process code generation using the AI engine
        result = ai_engine.generate_code(
            description=request.description,
            language=request.language,
            existing_code=request.existing_code,
            requirements=request.requirements,
            generate_tests=request.tests,
            mode=request.mode
        )
        
        # Extract results
        code = result.get("code", "")
        explanation = result.get("explanation", "")
        tests = result.get("tests")
        reasoning_path = result.get("reasoning_path")
        models_used = result.get("models_used", [])
        
        return CodeResponse(
            code=code,
            explanation=explanation,
            tests=tests,
            reasoning_path=reasoning_path,
            models_used=models_used
        )
    
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")

@app.post("/api/code/debug", response_model=DebugResponse)
async def debug_code(request: DebugRequest):
    """Debug code and fix issues"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Process debugging using the AI engine
        result = ai_engine.debug_code(
            code=request.code,
            language=request.language,
            error=request.error,
            expected_behavior=request.expected_behavior,
            context=request.context
        )
        
        # Extract results
        fixed_code = result.get("fixed_code", "")
        explanation = result.get("explanation", "")
        root_cause = result.get("root_cause", "")
        changes_made = result.get("changes_made", [])
        reasoning_path = result.get("reasoning_path")
        
        return DebugResponse(
            fixed_code=fixed_code,
            explanation=explanation,
            root_cause=root_cause,
            changes_made=changes_made,
            reasoning_path=reasoning_path
        )
    
    except Exception as e:
        logger.error(f"Error debugging code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error debugging code: {str(e)}")

@app.post("/api/architecture/design", response_model=ArchitectureResponse)
async def design_architecture(request: ArchitectureRequest):
    """Design software architecture"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Process architecture design using the AI engine
        result = ai_engine.design_architecture(
            description=request.description,
            requirements=request.requirements,
            constraints=request.constraints,
            technologies=request.technologies,
            scale=request.scale
        )
        
        # Extract results
        design = result.get("design", "")
        diagram_code = result.get("diagram_code")
        components = result.get("components", [])
        justification = result.get("justification", "")
        alternatives = result.get("alternatives_considered", [])
        
        return ArchitectureResponse(
            design=design,
            diagram_code=diagram_code,
            components=components,
            justification=justification,
            alternatives_considered=alternatives
        )
    
    except Exception as e:
        logger.error(f"Error designing architecture: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error designing architecture: {str(e)}")

@app.post("/api/reasoning", response_model=Dict[str, Any])
async def perform_reasoning(query: str, context: Optional[Dict[str, Any]] = None, strategy: Optional[str] = None):
    """Perform neuro-symbolic reasoning"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Convert strategy string to enum if provided
        strategy_enum = None
        if strategy:
            try:
                strategy_enum = ReasoningStrategy(strategy)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid reasoning strategy: {strategy}")
        
        # Perform reasoning
        result = reasoning_engine.reason(
            query=query,
            context=context or {},
            strategy=strategy_enum
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error performing reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing reasoning: {str(e)}")

@app.post("/api/memory/store", response_model=Dict[str, Any])
async def store_memory(content: Dict[str, Any], memory_type: str = "episodic"):
    """Store information in memory"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Convert memory type string to enum
        try:
            memory_type_enum = MemoryType(memory_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid memory type: {memory_type}")
        
        # Store in memory
        result = memory_system.store(
            content=content,
            memory_type=memory_type_enum
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error storing in memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing in memory: {str(e)}")

@app.post("/api/memory/query", response_model=List[Dict[str, Any]])
async def query_memory(query: str, memory_type: str = "episodic", limit: int = 10):
    """Query the memory system"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Convert memory type string to enum
        try:
            memory_type_enum = MemoryType(memory_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid memory type: {memory_type}")
        
        # Query memory
        results = memory_system.query(
            query=query,
            memory_type=memory_type_enum,
            limit=limit
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error querying memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying memory: {str(e)}")

@app.post("/api/execute", response_model=Dict[str, Any])
async def execute_code(code: str, language: str, context: Optional[Dict[str, Any]] = None, security_level: str = "standard"):
    """Execute code with specified security level"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Execute code
        result = execution_engine.execute_code(
            code=code,
            language=language,
            context=context or {},
            security_level=security_level
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing code: {str(e)}")

@app.post("/api/models/communicate", response_model=ModelCommunicationResponse)
async def model_communication(request: ModelCommunicationRequest):
    """Enable communication between models"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Send message between models
        message = communication_system.ask_question(
            from_model=request.from_model,
            to_model=request.to_model,
            content=request.message,
            context=request.context
        )
        
        message_id = message.get("id", "unknown")
        status = message.get("state", "sent")
        
        return ModelCommunicationResponse(
            message_id=message_id,
            status=status
        )
    
    except Exception as e:
        logger.error(f"Error in model communication: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in model communication: {str(e)}")

@app.get("/api/models/message/{message_id}", response_model=Dict[str, Any])
async def get_message(message_id: str):
    """Get a specific message by ID"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Get message
        message = communication_system.get_message(message_id)
        
        if not message:
            raise HTTPException(status_code=404, detail=f"Message not found: {message_id}")
        
        return message
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting message: {str(e)}")

@app.post("/api/models/message/{message_id}/answer", response_model=Dict[str, Any])
async def answer_message(message_id: str, content: str):
    """Answer a message"""
    if not components_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    
    try:
        # Answer message
        answer = communication_system.answer_question(
            question_id=message_id,
            content=content
        )
        
        return answer
    
    except Exception as e:
        logger.error(f"Error answering message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering message: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Seren server starting up")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)