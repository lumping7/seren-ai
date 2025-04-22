"""
API Server for Seren

Implements a comprehensive API server to access the hyperintelligent system
capabilities, including reasoning, learning, reflection, collaboration,
execution, and knowledge management.
"""

import os
import sys
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Union

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import FastAPI
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Response, status, Query, Body
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    has_fastapi = True
except ImportError:
    has_fastapi = False
    logging.warning("FastAPI not available. API server cannot be started.")

# Import Hyperintelligence System
try:
    from ai_core.hyperintelligence import (
        hyperintelligence_system,
        IntelligenceMode
    )
    has_hyperintelligence = True
except ImportError:
    has_hyperintelligence = False
    logging.warning("Hyperintelligence system not available. API server will have limited functionality.")

# Import Neurosymbolic Reasoning
try:
    from ai_core.neurosymbolic_reasoning import (
        neurosymbolic_reasoning,
        ReasoningStrategy
    )
    has_reasoning = True
except ImportError:
    has_reasoning = False
    logging.warning("Neurosymbolic reasoning not available. Reasoning endpoints will be disabled.")

# Import Metacognition
try:
    from ai_core.metacognition import (
        metacognitive_system,
        MetacognitiveLevel,
        CognitiveOperation
    )
    has_metacognition = True
except ImportError:
    has_metacognition = False
    logging.warning("Metacognition not available. Reflection endpoints will be disabled.")

# Import Continuous Execution
try:
    from ai_core.continuous_execution import (
        continuous_execution_engine,
        ExecutionMode,
        Goal,
        GoalStatus
    )
    has_execution = True
except ImportError:
    has_execution = False
    logging.warning("Continuous execution not available. Execution endpoints will be disabled.")

# Import Knowledge Library
try:
    from ai_core.knowledge.library import knowledge_library
    has_knowledge_lib = True
except ImportError:
    has_knowledge_lib = False
    logging.warning("Knowledge library not available. Knowledge endpoints will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app if available
if has_fastapi:
    app = FastAPI(
        title="Seren API",
        description="API for Seren Hyperintelligent System",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ========================= API Models =========================

if has_fastapi:
    # Request Models
    class ReasoningRequest(BaseModel):
        query: str = Field(..., description="Query to reason about")
        strategy: Optional[str] = Field(None, description="Reasoning strategy")
        context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    class LearningRequest(BaseModel):
        domain: Optional[str] = Field(None, description="Knowledge domain")
        continuous: bool = Field(False, description="Whether to perform continuous learning")
        duration: float = Field(300.0, description="Maximum duration for continuous learning (seconds)")
        data: Optional[Dict[str, Any]] = Field(None, description="Data to learn from")
    
    class ReflectionRequest(BaseModel):
        query: Optional[str] = Field(None, description="Reflection query")
        depth: int = Field(2, description="Reflection depth")
        focus: Optional[str] = Field(None, description="Focus area")
    
    class CollaborationRequest(BaseModel):
        query: str = Field(..., description="Collaboration query")
        mode: Optional[str] = Field(None, description="Collaboration mode")
        models: Optional[List[str]] = Field(None, description="Models to collaborate with")
    
    class ExecutionRequest(BaseModel):
        goal_description: str = Field(..., description="Goal description")
        priority: float = Field(0.5, description="Goal priority (0-1)")
        deadline: Optional[float] = Field(None, description="Deadline timestamp")
        criteria: Optional[Dict[str, Any]] = Field(None, description="Success criteria")
    
    class KnowledgeRetrievalRequest(BaseModel):
        query: str = Field(..., description="Search query")
        categories: Optional[List[str]] = Field(None, description="Categories to search in")
        limit: int = Field(10, description="Maximum results")
    
    class KnowledgeAddRequest(BaseModel):
        content: str = Field(..., description="Knowledge content")
        categories: Optional[List[str]] = Field(None, description="Knowledge categories")
        source: Optional[str] = Field("api", description="Knowledge source")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class ModeChangeRequest(BaseModel):
        mode: str = Field(..., description="New intelligence mode")
    
    # Response Models
    class ApiResponse(BaseModel):
        status: str = Field("success", description="Response status")
        data: Optional[Dict[str, Any]] = Field(None, description="Response data")
        error: Optional[str] = Field(None, description="Error message if any")
        
        @validator('error')
        def check_error(cls, v, values):
            if v is not None:
                values['status'] = 'error'
            return v
    
    # Background Task Models
    class TaskStatus(BaseModel):
        id: str = Field(..., description="Task ID")
        status: str = Field(..., description="Task status")
        progress: float = Field(0.0, description="Task progress (0-1)")
        result: Optional[Dict[str, Any]] = Field(None, description="Task result if completed")
        error: Optional[str] = Field(None, description="Error message if failed")

# ========================= Background Task Management =========================

class TaskManager:
    """Manager for background tasks"""
    
    def __init__(self):
        """Initialize task manager"""
        self.tasks = {}  # task_id -> task_info
        self.task_lock = threading.Lock()
    
    def create_task(self, task_id: str = None) -> str:
        """
        Create a new task
        
        Args:
            task_id: Task ID or None to generate one
            
        Returns:
            Task ID
        """
        import uuid
        
        with self.task_lock:
            # Generate task ID if not provided
            if task_id is None:
                task_id = str(uuid.uuid4())
            
            # Create task info
            task_info = {
                "id": task_id,
                "status": "created",
                "progress": 0.0,
                "result": None,
                "error": None,
                "created_at": time.time()
            }
            
            # Store task
            self.tasks[task_id] = task_info
        
        return task_id
    
    def update_task(
        self,
        task_id: str,
        status: str = None,
        progress: float = None,
        result: Dict[str, Any] = None,
        error: str = None
    ) -> bool:
        """
        Update task information
        
        Args:
            task_id: Task ID
            status: New status
            progress: New progress
            result: Task result
            error: Error message
            
        Returns:
            Success status
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return False
            
            # Update task info
            if status is not None:
                self.tasks[task_id]["status"] = status
            
            if progress is not None:
                self.tasks[task_id]["progress"] = progress
            
            if result is not None:
                self.tasks[task_id]["result"] = result
            
            if error is not None:
                self.tasks[task_id]["error"] = error
            
            # Add completion time for final states
            if status in ["completed", "failed"]:
                self.tasks[task_id]["completed_at"] = time.time()
        
        return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task information
        
        Args:
            task_id: Task ID
            
        Returns:
            Task information or None if not found
        """
        with self.task_lock:
            return self.tasks.get(task_id)
    
    def clean_old_tasks(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old completed tasks
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of tasks removed
        """
        current_time = time.time()
        removed_count = 0
        
        with self.task_lock:
            to_remove = []
            
            for task_id, task_info in self.tasks.items():
                # Check if task is completed or failed
                if task_info["status"] in ["completed", "failed"]:
                    # Check age
                    completed_at = task_info.get("completed_at", current_time)
                    age = current_time - completed_at
                    
                    if age > max_age_seconds:
                        to_remove.append(task_id)
            
            # Remove old tasks
            for task_id in to_remove:
                del self.tasks[task_id]
                removed_count += 1
        
        return removed_count

# Initialize task manager
task_manager = TaskManager()

# ========================= Background Task Functions =========================

if has_hyperintelligence:
    async def continuous_learning_task(
        task_id: str,
        domain: Optional[str],
        duration: float,
        data: Optional[Dict[str, Any]]
    ):
        """
        Execute continuous learning task
        
        Args:
            task_id: Task ID
            domain: Knowledge domain
            duration: Maximum duration
            data: Learning data
        """
        try:
            # Update task status
            task_manager.update_task(task_id, status="running", progress=0.1)
            
            # Execute learning
            result = hyperintelligence_system.learn(
                data=data,
                domain=domain,
                continuous=True,
                duration=duration
            )
            
            # Update task with result
            task_manager.update_task(
                task_id=task_id,
                status="completed",
                progress=1.0,
                result=result
            )
        
        except Exception as e:
            logger.error(f"Error in continuous learning task: {str(e)}")
            
            # Update task with error
            task_manager.update_task(
                task_id=task_id,
                status="failed",
                error=str(e)
            )

if has_execution:
    async def execution_task(
        task_id: str,
        goal_id: str
    ):
        """
        Monitor execution task
        
        Args:
            task_id: Task ID
            goal_id: Goal ID
        """
        try:
            # Monitor goal progress
            max_duration = 3600  # 1 hour max
            start_time = time.time()
            completed = False
            
            while time.time() - start_time < max_duration:
                # Get goal status
                goal = continuous_execution_engine.get_goal(goal_id)
                
                if not goal:
                    # Goal not found
                    task_manager.update_task(
                        task_id=task_id,
                        status="failed",
                        error="Goal not found"
                    )
                    return
                
                # Update task progress
                task_manager.update_task(
                    task_id=task_id,
                    progress=goal.progress
                )
                
                # Check if goal is completed or failed
                if goal.status == GoalStatus.COMPLETED:
                    task_manager.update_task(
                        task_id=task_id,
                        status="completed",
                        progress=1.0,
                        result={"goal_status": goal.status.name, "result": goal.result}
                    )
                    completed = True
                    break
                
                elif goal.status == GoalStatus.FAILED:
                    task_manager.update_task(
                        task_id=task_id,
                        status="failed",
                        progress=goal.progress,
                        error="Goal execution failed"
                    )
                    completed = True
                    break
                
                # Sleep before checking again
                await asyncio.sleep(5)
            
            # Check if we timed out
            if not completed:
                task_manager.update_task(
                    task_id=task_id,
                    status="failed",
                    error="Execution timed out"
                )
        
        except Exception as e:
            logger.error(f"Error in execution task: {str(e)}")
            
            # Update task with error
            task_manager.update_task(
                task_id=task_id,
                status="failed",
                error=str(e)
            )

# ========================= API Endpoints =========================

if has_fastapi and has_hyperintelligence:
    @app.get("/api/status", response_model=ApiResponse)
    async def get_status():
        """Get system status"""
        try:
            status_info = hyperintelligence_system.get_system_status()
            
            return {
                "status": "success",
                "data": status_info
            }
        
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/activate", response_model=ApiResponse)
    async def activate_system():
        """Activate the system"""
        try:
            success = hyperintelligence_system.activate()
            
            if success:
                return {
                    "status": "success",
                    "data": {"activated": True}
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to activate system"
                }
        
        except Exception as e:
            logger.error(f"Error activating system: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/deactivate", response_model=ApiResponse)
    async def deactivate_system():
        """Deactivate the system"""
        try:
            success = hyperintelligence_system.deactivate()
            
            if success:
                return {
                    "status": "success",
                    "data": {"deactivated": True}
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to deactivate system"
                }
        
        except Exception as e:
            logger.error(f"Error deactivating system: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/mode", response_model=ApiResponse)
    async def change_mode(request: ModeChangeRequest):
        """Change intelligence mode"""
        try:
            # Convert mode string to enum
            try:
                mode = IntelligenceMode[request.mode.upper()]
            except (KeyError, ValueError):
                return {
                    "status": "error",
                    "error": f"Invalid mode: {request.mode}"
                }
            
            # Set mode
            success = hyperintelligence_system.set_mode(mode)
            
            if success:
                return {
                    "status": "success",
                    "data": {"mode": mode.name}
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to change mode"
                }
        
        except Exception as e:
            logger.error(f"Error changing mode: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/reason", response_model=ApiResponse)
    async def reason(request: ReasoningRequest):
        """Perform reasoning"""
        try:
            # Check if reasoning is available
            if not has_reasoning:
                return {
                    "status": "error",
                    "error": "Reasoning capability not available"
                }
            
            # Execute reasoning
            result = hyperintelligence_system.reason(
                query=request.query,
                strategy=request.strategy,
                context=request.context
            )
            
            return {
                "status": "success",
                "data": result
            }
        
        except Exception as e:
            logger.error(f"Error in reasoning: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/learn", response_model=ApiResponse)
    async def learn(request: LearningRequest, background_tasks: BackgroundTasks):
        """Perform learning"""
        try:
            # Create task
            task_id = task_manager.create_task()
            task_manager.update_task(task_id, status="pending")
            
            if request.continuous:
                # Execute in background for continuous learning
                background_tasks.add_task(
                    continuous_learning_task,
                    task_id=task_id,
                    domain=request.domain,
                    duration=request.duration,
                    data=request.data
                )
                
                return {
                    "status": "success",
                    "data": {"task_id": task_id}
                }
            else:
                # Execute immediately for single learning iteration
                result = hyperintelligence_system.learn(
                    data=request.data,
                    domain=request.domain,
                    continuous=False
                )
                
                # Update task
                task_manager.update_task(
                    task_id=task_id,
                    status="completed",
                    progress=1.0,
                    result=result
                )
                
                return {
                    "status": "success",
                    "data": result
                }
        
        except Exception as e:
            logger.error(f"Error in learning: {str(e)}")
            
            # Update task if it was created
            if 'task_id' in locals():
                task_manager.update_task(
                    task_id=task_id,
                    status="failed",
                    error=str(e)
                )
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/reflect", response_model=ApiResponse)
    async def reflect(request: ReflectionRequest):
        """Perform reflection"""
        try:
            # Check if metacognition is available
            if not has_metacognition:
                return {
                    "status": "error",
                    "error": "Metacognition capability not available"
                }
            
            # Execute reflection
            result = hyperintelligence_system.reflect(
                query=request.query,
                depth=request.depth,
                focus=request.focus
            )
            
            return {
                "status": "success",
                "data": result
            }
        
        except Exception as e:
            logger.error(f"Error in reflection: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/collaborate", response_model=ApiResponse)
    async def collaborate(request: CollaborationRequest):
        """Perform collaboration"""
        try:
            # Execute collaboration
            result = hyperintelligence_system.collaborate(
                query=request.query,
                mode=request.mode,
                models=request.models
            )
            
            return {
                "status": "success",
                "data": result
            }
        
        except Exception as e:
            logger.error(f"Error in collaboration: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/execute", response_model=ApiResponse)
    async def execute(request: ExecutionRequest, background_tasks: BackgroundTasks):
        """Execute goal"""
        try:
            # Check if execution is available
            if not has_execution:
                return {
                    "status": "error",
                    "error": "Execution capability not available"
                }
            
            # Execute goal
            result = hyperintelligence_system.execute(
                goal_description=request.goal_description,
                priority=request.priority,
                deadline=request.deadline,
                criteria=request.criteria
            )
            
            # Create monitoring task
            task_id = task_manager.create_task()
            task_manager.update_task(
                task_id=task_id,
                status="running",
                progress=0.0
            )
            
            # Add background task to monitor execution
            if "goal_id" in result:
                background_tasks.add_task(
                    execution_task,
                    task_id=task_id,
                    goal_id=result["goal_id"]
                )
                
                # Add task ID to result
                result["task_id"] = task_id
            
            return {
                "status": "success",
                "data": result
            }
        
        except Exception as e:
            logger.error(f"Error in execution: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/knowledge/retrieve", response_model=ApiResponse)
    async def retrieve_knowledge(request: KnowledgeRetrievalRequest):
        """Retrieve knowledge"""
        try:
            # Check if knowledge library is available
            if not has_knowledge_lib:
                return {
                    "status": "error",
                    "error": "Knowledge library not available"
                }
            
            # Retrieve knowledge
            result = hyperintelligence_system.retrieve_knowledge(
                query=request.query,
                categories=request.categories,
                limit=request.limit
            )
            
            return {
                "status": "success",
                "data": result
            }
        
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.post("/api/knowledge/add", response_model=ApiResponse)
    async def add_knowledge(request: KnowledgeAddRequest):
        """Add knowledge"""
        try:
            # Check if knowledge library is available
            if not has_knowledge_lib:
                return {
                    "status": "error",
                    "error": "Knowledge library not available"
                }
            
            # Add knowledge
            entry_id = knowledge_library.add_knowledge(
                content=request.content,
                categories=request.categories,
                source_reference=request.source,
                metadata=request.metadata
            )
            
            return {
                "status": "success",
                "data": {"entry_id": entry_id}
            }
        
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.get("/api/knowledge/categories", response_model=ApiResponse)
    async def get_knowledge_categories():
        """Get knowledge categories"""
        try:
            # Check if knowledge library is available
            if not has_knowledge_lib:
                return {
                    "status": "error",
                    "error": "Knowledge library not available"
                }
            
            # Get categories
            categories = knowledge_library.get_categories()
            
            return {
                "status": "success",
                "data": {"categories": categories}
            }
        
        except Exception as e:
            logger.error(f"Error getting knowledge categories: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.get("/api/tasks/{task_id}", response_model=ApiResponse)
    async def get_task_status(task_id: str):
        """Get task status"""
        try:
            # Get task
            task = task_manager.get_task(task_id)
            
            if task is None:
                return {
                    "status": "error",
                    "error": f"Task not found: {task_id}"
                }
            
            return {
                "status": "success",
                "data": task
            }
        
        except Exception as e:
            logger.error(f"Error getting task status: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.delete("/api/tasks/{task_id}", response_model=ApiResponse)
    async def cancel_task(task_id: str):
        """Cancel task"""
        try:
            # Get task
            task = task_manager.get_task(task_id)
            
            if task is None:
                return {
                    "status": "error",
                    "error": f"Task not found: {task_id}"
                }
            
            # Update task
            task_manager.update_task(
                task_id=task_id,
                status="cancelled"
            )
            
            return {
                "status": "success",
                "data": {"cancelled": True}
            }
        
        except Exception as e:
            logger.error(f"Error cancelling task: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e)
            }

# ========================= Server Functions =========================

def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info"
):
    """
    Start the API server
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to reload on file changes
        workers: Number of worker processes
        log_level: Logging level
    """
    if not has_fastapi:
        logger.error("FastAPI not available. Cannot start API server.")
        return
    
    if not has_hyperintelligence:
        logger.error("Hyperintelligence system not available. Cannot start API server.")
        return
    
    import asyncio
    
    logger.info(f"Starting API server on {host}:{port}")
    
    try:
        # Import uvicorn here to handle import errors
        import uvicorn
        
        # Start server
        uvicorn.run(
            "ai_core.api_server:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=log_level
        )
    
    except ImportError:
        logger.error("Uvicorn not available. Cannot start API server.")
    
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")

# For direct module execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the Seren API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Reload on file changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", help="Logging level")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )