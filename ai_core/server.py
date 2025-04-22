"""
Main FastAPI server to expose AI API

This is the main entry point for the superintelligent AI system, providing
HTTP endpoints for all AI functionality including:
- Neuro-symbolic reasoning
- Model communication
- AI execution
- Memory management
- Self-improvement
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any, Union
import time
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import core AI components
try:
    from .ai_engine import AIEngine, AIEngineMode
    from .ai_memory import MemorySystem, MemoryType
    from .neurosymbolic_reasoning import NeuroSymbolicEngine
    from .ai_execution import ExecutionEngine, ExecutionSecurity
    from .ai_autonomy import AutonomyEngine
except ImportError:
    # For local development
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ai_engine import AIEngine, AIEngineMode
    from ai_memory import MemorySystem, MemoryType
    from neurosymbolic_reasoning import NeuroSymbolicEngine
    from ai_execution import ExecutionEngine, ExecutionSecurity
    from ai_autonomy import AutonomyEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ai_server.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Superintelligent AI System",
    description="Next-generation AI system with neuro-symbolic reasoning, memory, and execution capabilities",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI components
ai_engine = AIEngine()
memory_system = MemorySystem()
reasoning_engine = NeuroSymbolicEngine()
execution_engine = ExecutionEngine()
autonomy_engine = AutonomyEngine()

# API Models
class AIQuery(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    mode: Optional[str] = "collaborative"
    use_reasoning: Optional[bool] = True
    use_memory: Optional[bool] = True
    execute_code: Optional[bool] = False
    conversation_id: Optional[str] = None

class AIResponse(BaseModel):
    response: str
    conversation_id: str
    reasoning_path: Optional[List[Dict[str, Any]]] = None
    execution_results: Optional[Dict[str, Any]] = None
    memory_accessed: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryQuery(BaseModel):
    query: str
    memory_type: Optional[str] = "episodic"
    limit: Optional[int] = 10

class ExecutionRequest(BaseModel):
    code: str
    language: str
    context: Optional[Dict[str, Any]] = None
    security_level: Optional[str] = "standard"

# Routes
@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "name": "Superintelligent AI System",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "ai_engine": ai_engine.get_status(),
            "memory": memory_system.get_status(),
            "reasoning": reasoning_engine.get_status(),
            "execution": execution_engine.get_status(),
            "autonomy": autonomy_engine.get_status()
        }
    }

@app.post("/api/query", response_model=AIResponse)
async def query_ai(query: AIQuery):
    """
    Main endpoint for querying the AI system
    
    This integrates all AI capabilities including:
    - Model selection and collaboration
    - Neuro-symbolic reasoning
    - Memory access
    - Code execution (if enabled)
    - Self-improvement
    """
    try:
        logger.info(f"Received query: {query.query[:100]}...")
        
        # Generate or use conversation ID
        conversation_id = query.conversation_id or str(uuid.uuid4())
        
        # Initialize response
        response_data = {
            "conversation_id": conversation_id,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query_mode": query.mode,
                "reasoning_enabled": query.use_reasoning,
                "memory_enabled": query.use_memory,
                "execution_enabled": query.execute_code
            }
        }
        
        # Process with reasoning if enabled
        reasoning_path = None
        if query.use_reasoning:
            reasoning_result = reasoning_engine.reason(
                query=query.query,
                context=query.context
            )
            reasoning_path = reasoning_result.get("reasoning_path")
            response_data["reasoning_path"] = reasoning_path
        
        # Access memory if enabled
        memory_results = None
        if query.use_memory:
            memory_results = memory_system.query_relevant(
                query=query.query,
                context=query.context
            )
            response_data["memory_accessed"] = memory_results
        
        # Generate AI response using the appropriate mode
        ai_response = ai_engine.generate_response(
            query=query.query,
            context=query.context,
            reasoning_path=reasoning_path,
            memory_results=memory_results,
            mode=AIEngineMode(query.mode)
        )
        
        # Execute code if enabled and detected
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
            
            response_data["execution_results"] = execution_results
            
            # Enhance response with execution results
            ai_response = ai_engine.enhance_response_with_execution(
                original_response=ai_response,
                execution_results=execution_results
            )
        
        # Store interaction in memory
        memory_system.store(
            content={
                "query": query.query,
                "response": ai_response,
                "context": query.context,
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id
            },
            memory_type=MemoryType.EPISODIC
        )
        
        # Complete response
        response_data["response"] = ai_response
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/api/memory/query")
async def query_memory(query: MemoryQuery):
    """Query the AI's memory system"""
    try:
        results = memory_system.query(
            query=query.query,
            memory_type=MemoryType(query.memory_type),
            limit=query.limit
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error querying memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error querying memory: {str(e)}"
        )

@app.post("/api/execute")
async def execute_code(request: ExecutionRequest):
    """Execute code with appropriate security measures"""
    try:
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

@app.post("/api/reasoning/analyze")
async def analyze_with_reasoning(query: AIQuery):
    """Analyze a query using neuro-symbolic reasoning"""
    try:
        result = reasoning_engine.reason(
            query=query.query,
            context=query.context
        )
        return result
    except Exception as e:
        logger.error(f"Error in reasoning: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in reasoning: {str(e)}"
        )

@app.get("/api/system/status")
async def system_status():
    """Get detailed system status"""
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "uptime": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0,
            "version": "1.0.0",
            "environment": os.environ.get("ENVIRONMENT", "development")
        },
        "components": {
            "ai_engine": ai_engine.get_detailed_status(),
            "memory": memory_system.get_detailed_status(),
            "reasoning": reasoning_engine.get_detailed_status(),
            "execution": execution_engine.get_detailed_status(),
            "autonomy": autonomy_engine.get_detailed_status()
        },
        "resources": {
            "memory_usage": execution_engine.get_memory_usage(),
            "cpu_usage": execution_engine.get_cpu_usage()
        }
    }

# Admin endpoints (should be protected in production)
@app.post("/api/admin/reset-memory")
async def admin_reset_memory():
    """Admin endpoint to reset the memory system"""
    memory_system.reset()
    return {"status": "success", "message": "Memory system reset successfully"}

@app.post("/api/admin/upgrade")
async def admin_trigger_upgrade(upgrade_data: Dict[str, Any]):
    """
    Admin endpoint to trigger AI self-upgrading
    
    This is used when requesting the AI to upgrade itself by adding
    new capabilities or improving existing ones.
    """
    # This would be implemented with the ai_evolution system
    return {"status": "not_implemented", "message": "AI upgrading system not yet available"}

# Run server when executed directly
if __name__ == "__main__":
    app.state.start_time = time.time()
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)