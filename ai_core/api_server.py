"""
API Server for Seren

Provides HTTP API access to Seren's AI functionality.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
import asyncio
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import FastAPI
try:
    from fastapi import FastAPI, Depends, HTTPException, Request, Form, File, UploadFile, BackgroundTasks
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import APIKeyHeader
    import pydantic
    from pydantic import BaseModel, Field
    import uvicorn
    has_fastapi = True
except ImportError:
    has_fastapi = False
    logging.warning("FastAPI not installed. API server will not be available.")

# Local imports
try:
    from ai_core.ai_engine import ai_engine
    from ai_core.model_manager import ModelType
    
    # Try to import knowledge components
    try:
        from ai_core.knowledge.library import knowledge_library, KnowledgeSource
        from ai_core.knowledge.web_retriever import web_retriever
        from ai_core.knowledge.self_learning import self_learning_system, LearningPriority
        has_knowledge_lib = True
    except ImportError:
        has_knowledge_lib = False
        logging.warning("Knowledge library not available. Related endpoints will be disabled.")
    
except ImportError as e:
    logging.error(f"Error importing AI components: {str(e)}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app if dependencies are available
if has_fastapi:
    app = FastAPI(
        title="Seren AI API",
        description="API for accessing Seren's AI capabilities",
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
    
    # API key security
    API_KEY_NAME = "Authorization"
    api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
    
    # Default API key for development
    DEFAULT_API_KEY = "dev-key"
    
    # Get API keys from environment or use default
    API_KEYS = os.environ.get("SEREN_API_KEYS", DEFAULT_API_KEY).split(",")
    
    # Model definitions
    class QueryRequest(BaseModel):
        query: str
        mode: Optional[str] = "default"
        conversation_id: Optional[str] = None
        settings: Optional[Dict[str, Any]] = {}
    
    class CodeGenRequest(BaseModel):
        specification: str
        language: Optional[str] = "python"
        generate_tests: Optional[bool] = False
        mode: Optional[str] = "standard"
    
    class CodeAnalysisRequest(BaseModel):
        code: str
        language: Optional[str] = "python"
        analysis_type: Optional[str] = "review"
    
    class CodeExplainRequest(BaseModel):
        code: str
        language: Optional[str] = "python"
        detail_level: Optional[str] = "standard"
    
    class CollaborativeRequest(BaseModel):
        query: str
        model_types: Optional[List[str]] = ["qwen", "olympic"]
        context: Optional[Dict[str, Any]] = {}
    
    # Knowledge library API models
    class AddTextKnowledgeRequest(BaseModel):
        content: str
        source_reference: str
        context_name: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = {}
    
    class SearchKnowledgeRequest(BaseModel):
        query: str
        limit: Optional[int] = 5
    
    class AddLearningTaskRequest(BaseModel):
        content: str
        source: str
        priority: Optional[str] = LearningPriority.MEDIUM if has_knowledge_lib else "medium"
        metadata: Optional[Dict[str, Any]] = {}
    
    class WebRetrieveRequest(BaseModel):
        url: str
        context_name: Optional[str] = None
    
    # API key verification
    async def verify_api_key(api_key_header: str = Depends(api_key_header)):
        if api_key_header is None:
            raise HTTPException(status_code=401, detail="API key is missing")
        
        # Extract the key from "Bearer <key>" format
        key = api_key_header
        if api_key_header.startswith("Bearer "):
            key = api_key_header[7:]
        
        if key not in API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return key
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    
    # API status endpoint
    @app.get("/api/status", dependencies=[Depends(verify_api_key)])
    async def api_status():
        status = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "engine_initialized": ai_engine.initialized,
            "stats": ai_engine.stats,
            "knowledge_library_available": has_knowledge_lib
        }
        
        if has_knowledge_lib:
            # Add knowledge library status
            status["knowledge_entries_count"] = len(knowledge_library.entries)
            status["knowledge_contexts_count"] = len(knowledge_library.contexts)
            status["web_retriever_enabled"] = web_retriever.enable_web_access
            status["self_learning_status"] = self_learning_system.get_learning_status()
        
        return status
    
    # General query endpoint
    @app.post("/api/query", dependencies=[Depends(verify_api_key)])
    async def process_query(request: QueryRequest):
        response = ai_engine.process_query(
            query=request.query,
            mode=request.mode,
            context={"conversation_id": request.conversation_id} if request.conversation_id else {},
            settings=request.settings
        )
        return response
    
    # Code generation endpoint
    @app.post("/api/code/generate", dependencies=[Depends(verify_api_key)])
    async def generate_code(request: CodeGenRequest):
        response = ai_engine.generate_code(
            specification=request.specification,
            language=request.language,
            generate_tests=request.generate_tests,
            mode=request.mode
        )
        return response
    
    # Code analysis endpoint
    @app.post("/api/code/analyze", dependencies=[Depends(verify_api_key)])
    async def analyze_code(request: CodeAnalysisRequest):
        response = ai_engine.analyze_code(
            code=request.code,
            language=request.language,
            analysis_type=request.analysis_type
        )
        return response
    
    # Code explanation endpoint
    @app.post("/api/code/explain", dependencies=[Depends(verify_api_key)])
    async def explain_code(request: CodeExplainRequest):
        response = ai_engine.explain_code(
            code=request.code,
            language=request.language,
            detail_level=request.detail_level
        )
        return response
    
    # Collaborative response endpoint
    @app.post("/api/collaborative", dependencies=[Depends(verify_api_key)])
    async def collaborative_response(request: CollaborativeRequest):
        response = ai_engine.collaborative_response(
            query=request.query,
            model_types=[ModelType(mt) for mt in request.model_types],
            context=request.context
        )
        return response
    
    # Specialized response endpoint
    @app.post("/api/specialized", dependencies=[Depends(verify_api_key)])
    async def specialized_response(request: CollaborativeRequest):
        response = ai_engine.specialized_response(
            query=request.query,
            model_types=[ModelType(mt) for mt in request.model_types],
            context=request.context
        )
        return response
    
    # Competitive response endpoint
    @app.post("/api/competitive", dependencies=[Depends(verify_api_key)])
    async def competitive_response(request: CollaborativeRequest):
        response = ai_engine.competitive_response(
            query=request.query,
            model_types=[ModelType(mt) for mt in request.model_types],
            context=request.context
        )
        return response
    
    # Knowledge library endpoints
    if has_knowledge_lib:
        # Add knowledge from text
        @app.post("/api/knowledge/add-text", dependencies=[Depends(verify_api_key)])
        async def add_knowledge_from_text(request: AddTextKnowledgeRequest):
            entry_ids = knowledge_library.add_knowledge_from_text(
                text=request.content,
                source_reference=request.source_reference,
                context_name=request.context_name,
                metadata=request.metadata
            )
            return {"entry_ids": entry_ids, "count": len(entry_ids)}
        
        # Add knowledge from file upload
        @app.post("/api/knowledge/add-file", dependencies=[Depends(verify_api_key)])
        async def add_knowledge_from_file(
            context_name: Optional[str] = Form(None),
            file: UploadFile = File(...),
            background_tasks: BackgroundTasks = None
        ):
            # Create a temporary file
            temp_file_path = f"temp_{int(time.time())}_{file.filename}"
            
            try:
                # Save the uploaded file
                with open(temp_file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Process the file
                entry_ids = knowledge_library.add_knowledge_from_file(
                    file_path=temp_file_path,
                    context_name=context_name,
                    metadata={"filename": file.filename, "content_type": file.content_type}
                )
                
                return {"entry_ids": entry_ids, "count": len(entry_ids), "filename": file.filename}
            
            finally:
                # Clean up the temporary file
                if background_tasks:
                    background_tasks.add_task(os.remove, temp_file_path)
                else:
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
        
        # Search knowledge
        @app.post("/api/knowledge/search", dependencies=[Depends(verify_api_key)])
        async def search_knowledge(request: SearchKnowledgeRequest):
            entries = knowledge_library.search_knowledge(
                query=request.query,
                limit=request.limit
            )
            
            # Convert entries to dict for JSON serialization
            results = []
            for entry in entries:
                results.append({
                    "id": entry.id,
                    "content": entry.content,
                    "source_type": entry.source_type,
                    "source_reference": entry.source_reference,
                    "metadata": entry.metadata
                })
            
            return {"results": results, "count": len(results)}
        
        # Get context for query
        @app.post("/api/knowledge/context-for-query", dependencies=[Depends(verify_api_key)])
        async def get_context_for_query(request: SearchKnowledgeRequest):
            context = knowledge_library.extract_context_for_query(
                query=request.query,
                limit=request.limit
            )
            
            return {"context": context, "length": len(context)}
        
        # Add learning task
        @app.post("/api/learning/add-task", dependencies=[Depends(verify_api_key)])
        async def add_learning_task(request: AddLearningTaskRequest):
            self_learning_system.add_learning_task(
                content=request.content,
                source=request.source,
                priority=request.priority,
                metadata=request.metadata
            )
            
            return {"status": "task_added", "priority": request.priority}
        
        # Get learning status
        @app.get("/api/learning/status", dependencies=[Depends(verify_api_key)])
        async def get_learning_status():
            status = self_learning_system.get_learning_status()
            return status
        
        # Retrieve from web (if enabled)
        @app.post("/api/knowledge/retrieve-from-web", dependencies=[Depends(verify_api_key)])
        async def retrieve_from_web(request: WebRetrieveRequest):
            if not web_retriever.enable_web_access:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Web access is disabled"}
                )
            
            entry_ids = web_retriever.retrieve_and_add_to_library(
                url=request.url,
                context_name=request.context_name
            )
            
            return {"entry_ids": entry_ids, "count": len(entry_ids), "url": request.url}
else:
    # If FastAPI is not available, create a dummy app
    app = None

def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    if not has_fastapi:
        logger.error("Cannot start API server: FastAPI not installed")
        return
    
    # Ensure AI engine is initialized
    if not ai_engine.initialized:
        logger.info("Initializing AI engine before starting API server")
        ai_engine.initialize_models()
    
    # Start the server
    uvicorn.run(app, host=host, port=port)