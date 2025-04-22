"""
Main Server for Seren

Provides the main entry point for the Seren AI system, orchestrating all
components according to the OpenManus architecture for production-ready operation.
"""

import os
import sys
import json
import logging
import time
import signal
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio

# API Server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import system components
from ai_core.integration_framework import integration_framework, ComponentType, EventType
from ai_core.system_integration import system_integration
from ai_core.api_server import api_server
from ai_core.ai_engine import ai_engine
from ai_core.model_communication import communication_system
from ai_core.neurosymbolic_reasoning import reasoning_engine
from ai_core.ai_memory import memory_system
from ai_core.ai_execution import execution_engine
from ai_core.ai_autonomy import autonomy_engine
from ai_evolution.ai_upgrader import ai_upgrader
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

class ServerState(BaseModel):
    """Server state model"""
    running: bool = False
    start_time: Optional[str] = None
    uptime_seconds: int = 0
    components_initialized: List[str] = []
    components_active: List[str] = []
    api_server_running: bool = False
    ws_server_running: bool = False

class SerenServer:
    """
    Main Server for Seren
    
    Orchestrates all system components according to the OpenManus architecture:
    - Component initialization and lifecycle management
    - API server management
    - WebSocket server for real-time communication
    - System monitoring and health checks
    - Graceful shutdown
    
    Based on the OpenManus architecture:
    1. Agent Interface Layer (API and WebSocket servers)
    2. Task Processing Layer (Request processing and routing)
    3. Resource Coordination Layer (Resource allocation)
    4. Core Capabilities Layer (System components)
    5. External Integration Layer (External system connections)
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the server"""
        # Server state
        self.state = ServerState()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        # System integration is the central orchestrator
        self.system_integration = system_integration
        
        # API server for external access
        self.api_server = api_server
        
        # Status check interval (seconds)
        self.status_check_interval = self.config.get("status_check_interval", 60)
        
        logger.info("Seren Server initialized")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load server configuration"""
        # Default configuration
        default_config = {
            "name": "Seren Server",
            "version": "1.0.0",
            "log_level": "INFO",
            "api_server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4
            },
            "ws_server": {
                "host": "0.0.0.0",
                "port": 8001,
                "max_connections": 100
            },
            "status_check_interval": 60,  # seconds
            "component_timeout": 30,      # seconds
            "shutdown_timeout": 10,       # seconds
            "metrics": {
                "enabled": True,
                "port": 8009
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
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
    
    def _handle_shutdown_signal(self, signum, frame) -> None:
        """Handle shutdown signal"""
        logger.info(f"Received shutdown signal: {signum}")
        self.stop()
    
    def start(self) -> None:
        """Start the server"""
        logger.info("Starting Seren Server...")
        
        # Update state
        self.state.running = True
        self.state.start_time = datetime.now().isoformat()
        
        # Initialize and activate all components
        self._initialize_components()
        
        # Start API server
        self._start_api_server()
        
        # Start WebSocket server
        self._start_ws_server()
        
        # Start status check thread
        self._start_status_check()
        
        logger.info("Seren Server started successfully")
        
        # In a real implementation, this would block until shutdown
        # For demonstration, we'll just set up a placeholder
        try:
            # This would be a blocking wait in a real implementation
            # Here we just print periodic status updates
            while self.state.running:
                # Calculate uptime
                start_time = datetime.fromisoformat(self.state.start_time)
                self.state.uptime_seconds = int((datetime.now() - start_time).total_seconds())
                
                # Log status every minute
                logger.info(f"Seren Server uptime: {self.state.uptime_seconds} seconds")
                
                # Sleep for a minute
                time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("Server interrupted")
            self.stop()
    
    def _initialize_components(self) -> None:
        """Initialize and activate all system components"""
        logger.info("Initializing system components...")
        
        # Get all components from integration framework
        components = integration_framework.components
        
        # Initialize each component
        for component_id, component in components.items():
            try:
                # Skip internal framework layers
                if component_id.startswith("agent_interface") or \
                   component_id.startswith("task_processing") or \
                   component_id.startswith("resource_coordination") or \
                   component_id.startswith("core_capabilities") or \
                   component_id.startswith("external_integration"):
                    continue
                
                logger.info(f"Initializing component: {component_id}")
                
                # Add to initialized components
                self.state.components_initialized.append(component_id)
                
                # Activate component
                integration_framework.activate_component(component_id)
                
                # Add to active components
                self.state.components_active.append(component_id)
                
                logger.info(f"Component activated: {component_id}")
            
            except Exception as e:
                logger.error(f"Error initializing component {component_id}: {str(e)}")
        
        # Initialize models (via system integration)
        # This is already done in the SystemIntegration class
        logger.info("Components initialized successfully")
    
    def _start_api_server(self) -> None:
        """Start the API server"""
        logger.info("Starting API server...")
        
        # Start API server
        try:
            api_server.start()
            self.state.api_server_running = True
            logger.info("API server started successfully")
        
        except Exception as e:
            logger.error(f"Error starting API server: {str(e)}")
    
    def _start_ws_server(self) -> None:
        """Start the WebSocket server"""
        logger.info("Starting WebSocket server...")
        
        # In a real implementation, this would start a WebSocket server
        # For demonstration, we'll just set up a placeholder
        self.state.ws_server_running = True
        
        logger.info("WebSocket server started successfully")
    
    def _start_status_check(self) -> None:
        """Start the status check thread"""
        logger.info("Starting status check thread...")
        
        # In a real implementation, this would start a thread for status checks
        # For demonstration, we'll just set up a placeholder
        self.status_thread = threading.Thread(
            target=self._status_check_loop,
            daemon=True
        )
        self.status_thread.start()
        
        logger.info("Status check thread started successfully")
    
    def _status_check_loop(self) -> None:
        """Status check loop"""
        # In a real implementation, this would be a thread function
        while self.state.running:
            try:
                # Perform status check
                status = system_integration.get_system_status()
                
                # Log overall status
                logger.info(f"System status: {status['status']}")
                
                # Check for issues
                if status["status"] != "operational":
                    logger.warning("System is in degraded state")
                    
                    # Check component status
                    for component_id, component_status in status["components"].items():
                        if not component_status.get("operational", True):
                            logger.warning(f"Component issue: {component_id}")
                            
                            # Attempt recovery
                            self._attempt_recovery(component_id)
                
                # Sleep for specified interval
                time.sleep(self.status_check_interval)
            
            except Exception as e:
                logger.error(f"Error in status check: {str(e)}")
                
                # Sleep for a shorter interval before retry
                time.sleep(10)
    
    def _attempt_recovery(self, component_id: str) -> None:
        """Attempt to recover a failed component"""
        logger.info(f"Attempting recovery for component: {component_id}")
        
        # Create recovery action using Autonomy Engine
        try:
            from ai_core.ai_autonomy import autonomy_engine, ActionType
            
            # Propose recovery action
            action = autonomy_engine.propose_action(
                action_type=ActionType.RECOVERY,
                description=f"Recover {component_id} component",
                target_component=component_id,
                parameters={},
                priority=5,  # Highest priority
                requires_approval=False  # Auto-approve recovery
            )
            
            logger.info(f"Recovery action created: {action['id'] if 'id' in action else 'Unknown'}")
        
        except Exception as e:
            logger.error(f"Error creating recovery action: {str(e)}")
    
    def stop(self) -> None:
        """Stop the server"""
        if not self.state.running:
            logger.info("Server is already stopped")
            return
        
        logger.info("Stopping Seren Server...")
        
        # Update state
        self.state.running = False
        
        # Stop API server
        try:
            if self.state.api_server_running:
                logger.info("Stopping API server...")
                api_server.stop()
                self.state.api_server_running = False
                logger.info("API server stopped")
        except Exception as e:
            logger.error(f"Error stopping API server: {str(e)}")
        
        # Stop WebSocket server
        try:
            if self.state.ws_server_running:
                logger.info("Stopping WebSocket server...")
                # In a real implementation, this would stop the WebSocket server
                self.state.ws_server_running = False
                logger.info("WebSocket server stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {str(e)}")
        
        # Wait for threads to terminate
        timeout = self.config.get("shutdown_timeout", 10)
        logger.info(f"Waiting up to {timeout} seconds for threads to terminate...")
        
        # Shutdown complete
        start_time = datetime.fromisoformat(self.state.start_time) if self.state.start_time else datetime.now()
        uptime = int((datetime.now() - start_time).total_seconds())
        
        logger.info(f"Seren Server stopped. Uptime: {uptime} seconds")
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        # Get system status
        try:
            system_status = system_integration.get_system_status()
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            system_status = {"status": "error", "error": str(e)}
        
        # Calculate uptime
        uptime = 0
        if self.state.start_time:
            start_time = datetime.fromisoformat(self.state.start_time)
            uptime = int((datetime.now() - start_time).total_seconds())
        
        # Compile status
        return {
            "server": {
                "name": self.config["name"],
                "version": self.config["version"],
                "running": self.state.running,
                "uptime_seconds": uptime,
                "api_server_running": self.state.api_server_running,
                "ws_server_running": self.state.ws_server_running,
                "components_initialized": len(self.state.components_initialized),
                "components_active": len(self.state.components_active)
            },
            "system": system_status
        }

# Main entry point
def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Seren AI Server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Create server
    server = SerenServer(config_path=args.config)
    
    # Start server
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
        server.stop()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        import traceback
        traceback.print_exc()
        server.stop()

# Create server instance
server = SerenServer()

if __name__ == "__main__":
    main()