"""
Central Server for Seren

Serves both HTTP API and WebSocket interfaces for Seren's AI system.
"""

import os
import sys
import json
import logging
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union
import uuid
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import FastAPI dependencies for API server
try:
    import fastapi
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import pydantic
    import uvicorn
    has_api_deps = True
except ImportError:
    logging.warning("FastAPI dependencies not installed. HTTP API will not be available.")
    has_api_deps = False

# For security
try:
    from security.quantum_encryption import encrypt_message, decrypt_message
except ImportError:
    # Fallback if security module not available
    def encrypt_message(message, recipient=None):
        return message

    def decrypt_message(encrypted_message, recipient=None):
        return encrypted_message

# Local imports
try:
    from ai_core.ai_engine import ai_engine
    from ai_core.model_manager import model_manager
    from ai_core.model_communication import communication_system
    from ai_core.api_server import app as api_app, start_api_server
except ImportError as e:
    logging.error(f"Error importing AI core components: {str(e)}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ServeMode(object):
    """Modes for serving the AI system"""
    API_ONLY = "api"           # Only API server
    COMBINED = "combined"      # API server and WebSocket server

class ServerOptions(object):
    """Options for the server"""
    def __init__(
        self, 
        host: str = "0.0.0.0", 
        api_port: int = 8000,
        ws_port: int = 8001,
        mode: str = ServeMode.COMBINED,
        debug: bool = False
    ):
        """Initialize server options"""
        self.host = host
        self.api_port = api_port
        self.ws_port = ws_port
        self.mode = mode
        self.debug = debug

class Server(object):
    """
    Central Server for Seren
    
    Provides access to the AI system through HTTP API and WebSocket interfaces:
    - HTTP API for standard request/response interactions
    - WebSocket for streaming and real-time interactions
    - Initialization and shutdown of AI components
    - Health monitoring and metrics
    """
    
    def __init__(self, options: ServerOptions = None):
        """Initialize the server"""
        self.options = options or ServerOptions()
        
        # Server state
        self.running = False
        self.api_thread = None
        self.ws_thread = None
        
        # Initialize AI components
        self._init_ai_components()
        
        logger.info("Server initialized")
    
    def _init_ai_components(self):
        """Initialize AI components"""
        try:
            # Initialize AI engine if not already initialized
            if not ai_engine.initialized:
                logger.info("Initializing AI engine...")
                success = ai_engine.initialize_models()
                if not success:
                    logger.error("Failed to initialize AI engine")
                else:
                    logger.info("AI engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI components: {str(e)}")
    
    def start(self):
        """Start the server"""
        if self.running:
            logger.warning("Server already running")
            return
        
        logger.info(f"Starting server in {self.options.mode} mode...")
        
        # Start API server if needed
        if self.options.mode in [ServeMode.API_ONLY, ServeMode.COMBINED]:
            if has_api_deps:
                # Start in a separate thread
                self.api_thread = threading.Thread(
                    target=start_api_server,
                    args=(self.options.host, self.options.api_port),
                    daemon=True
                )
                self.api_thread.start()
                
                logger.info(f"API server started on {self.options.host}:{self.options.api_port}")
            else:
                logger.error("Cannot start API server: FastAPI dependencies not installed")
        
        # Start WebSocket server if needed
        if self.options.mode == ServeMode.COMBINED:
            # For now, WebSocket server is not implemented
            logger.info("WebSocket server not yet implemented")
        
        self.running = True
        logger.info("Server started")
    
    def stop(self):
        """Stop the server"""
        if not self.running:
            logger.warning("Server not running")
            return
        
        logger.info("Stopping server...")
        
        # Server will stop when threads exit
        self.running = False
        
        logger.info("Server stopped")
    
    def wait(self):
        """Wait for server to stop"""
        if self.api_thread:
            self.api_thread.join()
        
        if self.ws_thread:
            self.ws_thread.join()

def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Seren server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--ws-port", type=int, default=8001, help="Port for WebSocket server")
    parser.add_argument("--mode", type=str, default=ServeMode.COMBINED, help="Server mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create server options
    options = ServerOptions(
        host=args.host,
        api_port=args.api_port,
        ws_port=args.ws_port,
        mode=args.mode,
        debug=args.debug
    )
    
    # Create and start server
    server = Server(options)
    server.start()
    
    try:
        # Wait for server to stop
        server.wait()
    except KeyboardInterrupt:
        # Stop server on Ctrl+C
        logger.info("Keyboard interrupt received. Stopping server...")
        server.stop()

if __name__ == "__main__":
    main()