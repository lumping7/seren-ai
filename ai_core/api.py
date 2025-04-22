"""
API Interface for Seren

Main FastAPI implementation exposing the Seren AI system's capabilities
through a RESTful API interface.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import server implementation
from ai_core.server import app, Query, Response, CodeRequest, CodeResponse
from ai_core.server import DebugRequest, DebugResponse, ArchitectureRequest, ArchitectureResponse
from ai_core.server import ModelCommunicationRequest, ModelCommunicationResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Log startup
    logger.info("Starting Seren API server")
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)