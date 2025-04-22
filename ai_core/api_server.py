"""
API Server for Seren

Provides a comprehensive API for external access to all system capabilities
with OpenManus integration for agentic, production-ready operation.
"""

import os
import sys
import json
import logging
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from datetime import datetime
import threading
import asyncio
import traceback

# FastAPI for API server
from fastapi import FastAPI, HTTPException, Depends, Header, Body, Query, Path, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import system components
from ai_core.system_integration import system_integration
from security.quantum_encryption import quantum_encryption, SecurityLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# API models

class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="User query text")
    mode: str = Field("default", description="Processing mode")
    context: Dict[str, Any] = Field({}, description="Additional context")

class CodeGenerationRequest(BaseModel):
    """Code generation request model"""
    specification: str = Field(..., description="Code specification")
    language: str = Field("python", description="Programming language")
    test_driven: bool = Field(False, description="Whether to use test-driven development")
    context: Dict[str, Any] = Field({}, description="Additional context")

class DataAnalysisRequest(BaseModel):
    """Data analysis request model"""
    data: Any = Field(..., description="Data to analyze")
    analysis_type: str = Field("general", description="Type of analysis")
    context: Dict[str, Any] = Field({}, description="Additional context")

class TrainingRequest(BaseModel):
    """Model training request model"""
    model_id: str = Field(..., description="ID of the model to train")
    dataset_ids: List[str] = Field(..., description="List of dataset IDs")
    training_parameters: Dict[str, Any] = Field({}, description="Training parameters")
    auto_deploy: bool = Field(False, description="Whether to auto-deploy after training")

class UpgradeRequest(BaseModel):
    """System upgrade request model"""
    upgrade_type: str = Field(..., description="Type of upgrade")
    target: str = Field(..., description="Target component or model")
    description: str = Field(..., description="Upgrade description")
    parameters: Dict[str, Any] = Field({}, description="Upgrade parameters")

class ConversationRequest(BaseModel):
    """Model conversation request model"""
    from_model: str = Field(..., description="Source model")
    to_model: str = Field(..., description="Target model")
    content: str = Field(..., description="Message content")
    context: Dict[str, Any] = Field({}, description="Additional context")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")

class ActionRequest(BaseModel):
    """Autonomy action request model"""
    action_id: str = Field(..., description="ID of the action to approve/reject")
    decision: str = Field(..., description="Decision (approve/reject)")
    reason: Optional[str] = Field(None, description="Reason for the decision")

class APIError(BaseModel):
    """API error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: str = Field(..., description="Request ID")

# Security

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Authenticate user from JWT token
    
    In a real implementation, this would validate the token against a database
    For simulation, we'll just check if it's present
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Decrypt and validate the token
    try:
        # Here we would decrypt and validate the token
        # For simulation, we'll just create a user record
        
        # In production, use quantum_encryption.decrypt_data() here
        user = {
            "id": "user-123",
            "username": "seren-user",
            "role": "admin"
        }
        
        return user
    
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )

# API Server

class APIServer:
    """
    API Server for Seren
    
    Provides a comprehensive API for external access to all system capabilities.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """Initialize the API server"""
        self.host = host
        self.port = port
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Seren API",
            description="API for the Seren AI system, following the OpenManus structure",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Add exception handlers
        self.app.add_exception_handler(Exception, self.handle_exception)
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"API Server initialized at {host}:{port}")
    
    def _setup_routes(self) -> None:
        """Setup API routes"""
        # Root endpoint
        @self.app.get("/", tags=["General"])
        async def root():
            """Root endpoint returning system information"""
            return {
                "name": "Seren API",
                "version": "1.0.0",
                "description": "API for the Seren AI system, following the OpenManus structure",
                "status": "operational"
            }
        
        # System status
        @self.app.get("/api/status", tags=["System"])
        async def get_status():
            """Get system status"""
            return system_integration.get_system_status()
        
        # Process query
        @self.app.post("/api/query", tags=["Core"])
        async def process_query(
            request: QueryRequest,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Process a user query"""
            try:
                # Add user context
                context = request.context.copy()
                context["user"] = user
                
                # Process query
                response = system_integration.process_user_query(
                    query=request.query,
                    mode=request.mode,
                    context=context
                )
                
                return response
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing query: {str(e)}"
                )
        
        # Generate code
        @self.app.post("/api/code/generate", tags=["Code"])
        async def generate_code(
            request: CodeGenerationRequest,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Generate code based on specification"""
            try:
                # Add user context
                context = request.context.copy()
                context["user"] = user
                
                # Generate code
                response = system_integration.generate_code(
                    specification=request.specification,
                    language=request.language,
                    test_driven=request.test_driven,
                    context=context
                )
                
                return response
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating code: {str(e)}"
                )
        
        # Execute code
        @self.app.post("/api/code/execute", tags=["Code"])
        async def execute_code(
            code: str = Body(..., description="Code to execute"),
            language: str = Body("python", description="Programming language"),
            context: Dict[str, Any] = Body({}, description="Execution context"),
            security_level: str = Body("standard", description="Security level"),
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Execute code"""
            try:
                # Add user context
                exec_context = context.copy()
                exec_context["user"] = user
                
                # Execute code
                from ai_core.ai_execution import execution_engine
                response = execution_engine.execute_code(
                    code=code,
                    language=language,
                    context=exec_context,
                    security_level=security_level
                )
                
                return response
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error executing code: {str(e)}"
                )
        
        # Analyze data
        @self.app.post("/api/analyze", tags=["Analysis"])
        async def analyze_data(
            request: DataAnalysisRequest,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Analyze data"""
            try:
                # Add user context
                context = request.context.copy()
                context["user"] = user
                
                # Analyze data
                response = system_integration.analyze_data(
                    data=request.data,
                    analysis_type=request.analysis_type,
                    context=context
                )
                
                return response
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error analyzing data: {str(e)}"
                )
        
        # Train model
        @self.app.post("/api/models/train", tags=["Training"])
        async def train_model(
            request: TrainingRequest,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Train a model"""
            try:
                # Check user permissions
                if user.get("role") != "admin":
                    raise HTTPException(
                        status_code=403,
                        detail="Training requires admin privileges"
                    )
                
                # Train model
                response = system_integration.train_model(
                    model_id=request.model_id,
                    dataset_ids=request.dataset_ids,
                    training_parameters=request.training_parameters,
                    auto_deploy=request.auto_deploy
                )
                
                return response
            
            except HTTPException:
                raise
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error training model: {str(e)}"
                )
        
        # Upgrade system
        @self.app.post("/api/system/upgrade", tags=["System"])
        async def upgrade_system(
            request: UpgradeRequest,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Upgrade system component"""
            try:
                # Check user permissions
                if user.get("role") != "admin":
                    raise HTTPException(
                        status_code=403,
                        detail="Upgrades require admin privileges"
                    )
                
                # Upgrade system
                response = system_integration.upgrade_system(
                    upgrade_type=request.upgrade_type,
                    target=request.target,
                    description=request.description,
                    parameters=request.parameters
                )
                
                return response
            
            except HTTPException:
                raise
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error upgrading system: {str(e)}"
                )
        
        # Model communication
        @self.app.post("/api/models/communicate", tags=["Communication"])
        async def model_communicate(
            request: ConversationRequest,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Facilitate communication between models"""
            try:
                # Import communication system
                from ai_core.model_communication import communication_system, ModelType, MessageType
                
                # Create or continue conversation
                if request.conversation_id:
                    conversation_id = request.conversation_id
                else:
                    conversation_id = communication_system.create_conversation(
                        topic=f"API-initiated conversation",
                        participants=[request.from_model, request.to_model],
                        context=request.context
                    )
                
                # Send message
                message = communication_system.ask_question(
                    from_model=request.from_model,
                    to_model=request.to_model,
                    content=request.content,
                    context=request.context,
                    conversation_id=conversation_id
                )
                
                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "message": message
                }
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error in model communication: {str(e)}"
                )
        
        # Get conversation
        @self.app.get("/api/models/conversation/{conversation_id}", tags=["Communication"])
        async def get_conversation(
            conversation_id: str,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Get a conversation by ID"""
            try:
                # Import communication system
                from ai_core.model_communication import communication_system
                
                # Get conversation
                conversation = communication_system.get_conversation(conversation_id)
                
                if not conversation:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Conversation not found: {conversation_id}"
                    )
                
                return conversation
            
            except HTTPException:
                raise
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving conversation: {str(e)}"
                )
        
        # Autonomy actions
        @self.app.post("/api/autonomy/actions/{action_id}", tags=["Autonomy"])
        async def handle_action(
            action_id: str,
            decision: str = Query(..., description="Decision (approve/reject)"),
            reason: str = Query(None, description="Reason for the decision"),
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Approve or reject an autonomy action"""
            try:
                # Check user permissions
                if user.get("role") != "admin":
                    raise HTTPException(
                        status_code=403,
                        detail="Action management requires admin privileges"
                    )
                
                # Import autonomy engine
                from ai_core.ai_autonomy import autonomy_engine
                
                # Handle decision
                if decision.lower() == "approve":
                    result = autonomy_engine.approve_action(action_id)
                    action = "approved"
                elif decision.lower() == "reject":
                    result = autonomy_engine.reject_action(action_id, reason)
                    action = "rejected"
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid decision: {decision}. Must be 'approve' or 'reject'."
                    )
                
                if not result:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Action not found or not in a state that can be {action}: {action_id}"
                    )
                
                return {
                    "success": True,
                    "action_id": action_id,
                    "decision": decision,
                    "reason": reason
                }
            
            except HTTPException:
                raise
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error handling action: {str(e)}"
                )
        
        # Get pending actions
        @self.app.get("/api/autonomy/actions", tags=["Autonomy"])
        async def get_pending_actions(
            component: str = Query(None, description="Filter by component"),
            action_type: str = Query(None, description="Filter by action type"),
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Get pending autonomy actions"""
            try:
                # Import autonomy engine
                from ai_core.ai_autonomy import autonomy_engine
                
                # Get pending actions
                actions = autonomy_engine.get_pending_actions(
                    target_component=component,
                    action_type=action_type
                )
                
                return {
                    "success": True,
                    "count": len(actions),
                    "actions": actions
                }
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving pending actions: {str(e)}"
                )
        
        # Memory management
        @self.app.get("/api/memory/{memory_type}", tags=["Memory"])
        async def get_memories(
            memory_type: str,
            query: str = Query(None, description="Query string for retrieval"),
            limit: int = Query(10, description="Maximum number of memories to return"),
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Retrieve memories by type"""
            try:
                # Import memory system
                from ai_core.ai_memory import memory_system, MemoryType
                
                # Validate memory type
                try:
                    memory_type_enum = MemoryType(memory_type)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid memory type: {memory_type}"
                    )
                
                # Retrieve memories
                if query:
                    memories = memory_system.retrieve_memories(
                        memory_type=memory_type_enum,
                        query=query,
                        limit=limit
                    )
                else:
                    memories = memory_system.get_memories(
                        memory_type=memory_type_enum,
                        limit=limit
                    )
                
                return {
                    "success": True,
                    "memory_type": memory_type,
                    "count": len(memories),
                    "memories": memories
                }
            
            except HTTPException:
                raise
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving memories: {str(e)}"
                )
        
        # Add memory
        @self.app.post("/api/memory/{memory_type}", tags=["Memory"])
        async def add_memory(
            memory_type: str,
            content: Dict[str, Any] = Body(..., description="Memory content"),
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Add a new memory"""
            try:
                # Check user permissions
                if user.get("role") != "admin":
                    raise HTTPException(
                        status_code=403,
                        detail="Memory creation requires admin privileges"
                    )
                
                # Import memory system
                from ai_core.ai_memory import memory_system, MemoryType
                
                # Validate memory type
                try:
                    memory_type_enum = MemoryType(memory_type)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid memory type: {memory_type}"
                    )
                
                # Add memory
                memory_id = memory_system.add_memory(
                    memory_type=memory_type_enum,
                    content=content
                )
                
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "memory_type": memory_type
                }
            
            except HTTPException:
                raise
            
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error adding memory: {str(e)}"
                )
    
    async def handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler"""
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log the error
        logger.error(f"API Error ({request_id}): {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Get status code
        status_code = 500
        if isinstance(exc, HTTPException):
            status_code = exc.status_code
        
        # Create error response
        error_response = APIError(
            error=str(exc),
            details=None,
            request_id=request_id
        )
        
        # Log error in system
        try:
            from ai_core.ai_memory import memory_system, MemoryType
            memory_system.add_memory(
                memory_type=MemoryType.EPISODIC,
                content={
                    "type": "error",
                    "request_id": request_id,
                    "path": str(request.url),
                    "method": request.method,
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error logging to memory system: {str(e)}")
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict()
        )
    
    def start(self) -> None:
        """Start the API server"""
        # In a real implementation, this would start the server
        # For demonstration, we'll just log that it would start
        logger.info(f"Starting API server at {self.host}:{self.port}")
        
        # The actual server would be started with:
        # uvicorn.run(self.app, host=self.host, port=self.port)
    
    def stop(self) -> None:
        """Stop the API server"""
        # In a real implementation, this would stop the server
        logger.info("Stopping API server")

# Initialize API server
api_server = APIServer(host="0.0.0.0", port=8000)