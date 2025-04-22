"""
AI Extension Manager

Allows the AI system to dynamically add new capabilities and extensions
based on admin commands or autonomous self-improvement initiatives.
"""

import os
import sys
import json
import logging
import time
import importlib
import importlib.util
import inspect
import shutil
import subprocess
import threading
import queue
import uuid
import re
import ast
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from datetime import datetime

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

class ExtensionStatus:
    """Status values for extensions"""
    AVAILABLE = "available"
    INSTALLING = "installing"
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    FAILED = "failed"
    UNINSTALLED = "uninstalled"

class ExtensionType:
    """Types of extensions"""
    CAPABILITY = "capability"
    MODEL = "model"
    INTEGRATION = "integration"
    TOOL = "tool"
    PROCESSOR = "processor"
    ANALYZER = "analyzer"
    UTILITY = "utility"

class ExtensionManager:
    """
    AI Extension Manager
    
    Manages dynamic extensions and capabilities for the AI system.
    
    Key capabilities:
    1. Discover available extensions
    2. Install/uninstall extensions
    3. Enable/disable extensions
    4. Manage extension dependencies
    5. Provide extension metadata and documentation
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the extension manager"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Extension directories
        self.extensions_dir = os.path.join(self.base_dir, "extensions")
        self.extension_registry_path = os.path.join(self.extensions_dir, "registry.json")
        
        # Create extensions directory if it doesn't exist
        os.makedirs(self.extensions_dir, exist_ok=True)
        
        # Extension registry
        self.registry = self._load_registry()
        
        # Extension module cache
        self.extension_modules = {}
        
        # Extension tasks queue and worker thread
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._extension_worker, daemon=True)
        self.worker_thread.start()
        
        # Configuration
        self.config = {
            "auto_enable": True,
            "extension_timeout": 30,  # seconds
            "max_parallel_installations": 1,
            "verify_dependencies": True,
            "extension_sources": [
                "local",
                "registry"
            ],
            "registry_url": "https://extensions.aicore.registry/api/v1/extensions",
            "allow_remote_sources": False
        }
        
        # Track active installations
        self.active_installations = 0
        
        logger.info(f"Extension Manager initialized with {len(self.registry)} registered extensions")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the extension registry from disk"""
        if os.path.exists(self.extension_registry_path):
            try:
                with open(self.extension_registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading extension registry: {str(e)}")
        
        # Create default registry
        registry = {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "extensions": {}
        }
        
        # Save it
        self._save_registry(registry)
        
        return registry
    
    def _save_registry(self, registry: Dict[str, Any] = None) -> None:
        """Save the extension registry to disk"""
        if registry is None:
            registry = self.registry
        
        # Update last_updated
        registry["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.extension_registry_path, "w") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving extension registry: {str(e)}")
    
    def discover_extensions(self, sources: List[str] = None) -> List[Dict[str, Any]]:
        """
        Discover available extensions
        
        Args:
            sources: List of sources to check (default: all configured sources)
            
        Returns:
            List of available extensions
        """
        sources = sources or self.config["extension_sources"]
        
        available_extensions = []
        
        # Check local extensions directory
        if "local" in sources:
            local_extensions = self._discover_local_extensions()
            available_extensions.extend(local_extensions)
            logger.info(f"Discovered {len(local_extensions)} local extensions")
        
        # Check online registry
        if "registry" in sources and self.config["allow_remote_sources"]:
            registry_extensions = self._discover_registry_extensions()
            available_extensions.extend(registry_extensions)
            logger.info(f"Discovered {len(registry_extensions)} registry extensions")
        
        # Update registry with newly discovered extensions
        self._update_registry_with_discoveries(available_extensions)
        
        return available_extensions
    
    def _discover_local_extensions(self) -> List[Dict[str, Any]]:
        """Discover extensions in the local extensions directory"""
        local_extensions = []
        
        # Look for extension directories
        for item in os.listdir(self.extensions_dir):
            item_path = os.path.join(self.extensions_dir, item)
            
            # Skip registry file and non-directories
            if item == "registry.json" or not os.path.isdir(item_path):
                continue
            
            # Check for extension.json
            metadata_path = os.path.join(item_path, "extension.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Add the extension
                    ext_id = metadata.get("id", item)
                    
                    extension = {
                        "id": ext_id,
                        "name": metadata.get("name", ext_id),
                        "version": metadata.get("version", "0.1.0"),
                        "description": metadata.get("description", ""),
                        "author": metadata.get("author", "Unknown"),
                        "type": metadata.get("type", ExtensionType.CAPABILITY),
                        "capabilities": metadata.get("capabilities", []),
                        "dependencies": metadata.get("dependencies", []),
                        "path": item_path,
                        "status": ExtensionStatus.AVAILABLE,
                        "source": "local"
                    }
                    
                    local_extensions.append(extension)
                    
                except Exception as e:
                    logger.warning(f"Error loading extension metadata from {metadata_path}: {str(e)}")
        
        return local_extensions
    
    def _discover_registry_extensions(self) -> List[Dict[str, Any]]:
        """Discover extensions from the online registry"""
        # This would make an HTTP request to the registry API
        # For now, return an empty list as a placeholder
        return []
    
    def _update_registry_with_discoveries(self, discoveries: List[Dict[str, Any]]) -> None:
        """Update the registry with newly discovered extensions"""
        for ext in discoveries:
            ext_id = ext["id"]
            
            if ext_id in self.registry["extensions"]:
                # Update existing entry with new information
                current = self.registry["extensions"][ext_id]
                
                # Only update if version is newer
                if self._compare_versions(ext["version"], current["version"]) > 0:
                    # Preserve status and installation info
                    status = current.get("status", ExtensionStatus.AVAILABLE)
                    enabled = current.get("enabled", False)
                    installed_at = current.get("installed_at", None)
                    
                    self.registry["extensions"][ext_id] = ext
                    self.registry["extensions"][ext_id]["status"] = status
                    self.registry["extensions"][ext_id]["enabled"] = enabled
                    self.registry["extensions"][ext_id]["installed_at"] = installed_at
                    
                    logger.info(f"Updated extension {ext_id} in registry (version {ext['version']})")
            else:
                # Add new extension
                self.registry["extensions"][ext_id] = ext
                logger.info(f"Added new extension {ext_id} to registry (version {ext['version']})")
        
        # Save the updated registry
        self._save_registry()
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings
        
        Returns:
            -1 if version1 < version2
             0 if version1 == version2
             1 if version1 > version2
        """
        try:
            v1_parts = [int(p) for p in version1.split(".")]
            v2_parts = [int(p) for p in version2.split(".")]
            
            # Pad with zeros if needed
            while len(v1_parts) < 3:
                v1_parts.append(0)
            while len(v2_parts) < 3:
                v2_parts.append(0)
            
            # Compare parts
            for i in range(3):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            # Versions are equal
            return 0
        except Exception:
            # If we can't parse as semantic versions, fall back to string comparison
            if version1 < version2:
                return -1
            elif version1 > version2:
                return 1
            else:
                return 0
    
    def install_extension(
        self,
        extension_id: str,
        version: str = None,
        force: bool = False,
        enable: bool = None
    ) -> Dict[str, Any]:
        """
        Install an extension
        
        Args:
            extension_id: ID of the extension
            version: Specific version to install (default: latest)
            force: Whether to force reinstallation if already installed
            enable: Whether to enable after installation (default: config setting)
            
        Returns:
            Installation status
        """
        # Check if extension exists in registry
        if extension_id not in self.registry["extensions"]:
            # Try to discover it
            self.discover_extensions()
            
            if extension_id not in self.registry["extensions"]:
                logger.warning(f"Extension {extension_id} not found in registry")
                return {
                    "success": False,
                    "extension_id": extension_id,
                    "error": "Extension not found in registry"
                }
        
        # Get extension info
        extension = self.registry["extensions"][extension_id]
        
        # Check if already installed
        if extension.get("status") == ExtensionStatus.INSTALLED and not force:
            logger.info(f"Extension {extension_id} is already installed")
            return {
                "success": True,
                "extension_id": extension_id,
                "status": extension.get("status"),
                "message": "Extension is already installed"
            }
        
        # Set up the installation task
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "type": "install",
            "extension_id": extension_id,
            "version": version,
            "force": force,
            "enable": enable if enable is not None else self.config["auto_enable"],
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        # Update extension status
        extension["status"] = ExtensionStatus.INSTALLING
        self._save_registry()
        
        # Add task to queue
        self.task_queue.put(task)
        
        logger.info(f"Queued installation task {task_id} for extension {extension_id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "extension_id": extension_id,
            "status": "queued"
        }
    
    def uninstall_extension(
        self,
        extension_id: str,
        remove_data: bool = False
    ) -> Dict[str, Any]:
        """
        Uninstall an extension
        
        Args:
            extension_id: ID of the extension
            remove_data: Whether to remove extension data
            
        Returns:
            Uninstallation status
        """
        # Check if extension exists in registry
        if extension_id not in self.registry["extensions"]:
            logger.warning(f"Extension {extension_id} not found in registry")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension not found in registry"
            }
        
        # Get extension info
        extension = self.registry["extensions"][extension_id]
        
        # Check if installed
        if extension.get("status") != ExtensionStatus.INSTALLED and extension.get("status") != ExtensionStatus.ENABLED:
            logger.warning(f"Extension {extension_id} is not installed")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension is not installed"
            }
        
        # Set up the uninstallation task
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "type": "uninstall",
            "extension_id": extension_id,
            "remove_data": remove_data,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        # Update extension status
        extension["status"] = ExtensionStatus.DISABLED
        self._save_registry()
        
        # Add task to queue
        self.task_queue.put(task)
        
        logger.info(f"Queued uninstallation task {task_id} for extension {extension_id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "extension_id": extension_id,
            "status": "queued"
        }
    
    def enable_extension(self, extension_id: str) -> Dict[str, Any]:
        """
        Enable an installed extension
        
        Args:
            extension_id: ID of the extension
            
        Returns:
            Enable status
        """
        # Check if extension exists in registry
        if extension_id not in self.registry["extensions"]:
            logger.warning(f"Extension {extension_id} not found in registry")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension not found in registry"
            }
        
        # Get extension info
        extension = self.registry["extensions"][extension_id]
        
        # Check if installed
        if extension.get("status") != ExtensionStatus.INSTALLED and extension.get("status") != ExtensionStatus.DISABLED:
            logger.warning(f"Extension {extension_id} is not installed")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension is not installed"
            }
        
        try:
            # Load the extension module
            module = self._load_extension_module(extension)
            
            if module is None:
                return {
                    "success": False,
                    "extension_id": extension_id,
                    "error": "Failed to load extension module"
                }
            
            # Call the enable function if it exists
            if hasattr(module, "enable"):
                enable_result = module.enable()
                
                if isinstance(enable_result, dict) and not enable_result.get("success", True):
                    return {
                        "success": False,
                        "extension_id": extension_id,
                        "error": enable_result.get("error", "Extension enable function returned an error")
                    }
            
            # Update extension status
            extension["status"] = ExtensionStatus.ENABLED
            extension["enabled"] = True
            extension["enabled_at"] = datetime.now().isoformat()
            self._save_registry()
            
            logger.info(f"Enabled extension {extension_id}")
            
            return {
                "success": True,
                "extension_id": extension_id,
                "status": extension["status"]
            }
        
        except Exception as e:
            logger.error(f"Error enabling extension {extension_id}: {str(e)}")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": f"Error enabling extension: {str(e)}"
            }
    
    def disable_extension(self, extension_id: str) -> Dict[str, Any]:
        """
        Disable an enabled extension
        
        Args:
            extension_id: ID of the extension
            
        Returns:
            Disable status
        """
        # Check if extension exists in registry
        if extension_id not in self.registry["extensions"]:
            logger.warning(f"Extension {extension_id} not found in registry")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension not found in registry"
            }
        
        # Get extension info
        extension = self.registry["extensions"][extension_id]
        
        # Check if enabled
        if extension.get("status") != ExtensionStatus.ENABLED:
            logger.warning(f"Extension {extension_id} is not enabled")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension is not enabled"
            }
        
        try:
            # Get the extension module
            module = self.extension_modules.get(extension_id)
            
            # Call the disable function if it exists
            if module is not None and hasattr(module, "disable"):
                disable_result = module.disable()
                
                if isinstance(disable_result, dict) and not disable_result.get("success", True):
                    return {
                        "success": False,
                        "extension_id": extension_id,
                        "error": disable_result.get("error", "Extension disable function returned an error")
                    }
            
            # Update extension status
            extension["status"] = ExtensionStatus.DISABLED
            extension["enabled"] = False
            extension["disabled_at"] = datetime.now().isoformat()
            self._save_registry()
            
            logger.info(f"Disabled extension {extension_id}")
            
            return {
                "success": True,
                "extension_id": extension_id,
                "status": extension["status"]
            }
        
        except Exception as e:
            logger.error(f"Error disabling extension {extension_id}: {str(e)}")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": f"Error disabling extension: {str(e)}"
            }
    
    def _extension_worker(self) -> None:
        """Worker thread to process extension tasks"""
        while True:
            try:
                # Get a task from the queue
                task = self.task_queue.get()
                
                # Process the task
                if task["type"] == "install":
                    self._process_install_task(task)
                elif task["type"] == "uninstall":
                    self._process_uninstall_task(task)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in extension worker: {str(e)}")
    
    def _process_install_task(self, task: Dict[str, Any]) -> None:
        """Process an extension installation task"""
        extension_id = task["extension_id"]
        
        # Check if we can start another installation
        while self.active_installations >= self.config["max_parallel_installations"]:
            time.sleep(1)
        
        # Increment active installations
        self.active_installations += 1
        
        try:
            # Get extension info
            extension = self.registry["extensions"].get(extension_id)
            
            if not extension:
                logger.error(f"Extension {extension_id} not found in registry")
                return
            
            logger.info(f"Installing extension {extension_id}...")
            
            # Check dependencies
            if self.config["verify_dependencies"] and extension.get("dependencies"):
                missing_deps = []
                
                for dep in extension["dependencies"]:
                    if isinstance(dep, str) and dep not in self.registry["extensions"]:
                        missing_deps.append(dep)
                    elif isinstance(dep, dict) and dep.get("id") not in self.registry["extensions"]:
                        missing_deps.append(dep.get("id"))
                
                if missing_deps:
                    logger.warning(f"Missing dependencies for extension {extension_id}: {missing_deps}")
                    
                    # Could automatically install dependencies here
                    extension["status"] = ExtensionStatus.FAILED
                    extension["error"] = f"Missing dependencies: {', '.join(missing_deps)}"
                    self._save_registry()
                    return
            
            # Get extension source path
            extension_path = extension.get("path")
            
            if not extension_path or not os.path.exists(extension_path):
                # Could download from registry here if it's a remote extension
                logger.error(f"Extension path {extension_path} not found")
                extension["status"] = ExtensionStatus.FAILED
                extension["error"] = "Extension path not found"
                self._save_registry()
                return
            
            # Run installation script if it exists
            install_script = os.path.join(extension_path, "install.py")
            if os.path.exists(install_script):
                logger.info(f"Running installation script for extension {extension_id}")
                
                try:
                    # Run the installation script
                    process = subprocess.Popen(
                        [sys.executable, install_script],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=extension_path,
                        text=True
                    )
                    
                    stdout, stderr = process.communicate(timeout=self.config["extension_timeout"])
                    exit_code = process.returncode
                    
                    if exit_code != 0:
                        logger.error(f"Installation script failed for extension {extension_id}: {stderr}")
                        extension["status"] = ExtensionStatus.FAILED
                        extension["error"] = f"Installation script failed: {stderr}"
                        self._save_registry()
                        return
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.error(f"Installation script timed out for extension {extension_id}")
                    extension["status"] = ExtensionStatus.FAILED
                    extension["error"] = "Installation script timed out"
                    self._save_registry()
                    return
                
                except Exception as e:
                    logger.error(f"Error running installation script for extension {extension_id}: {str(e)}")
                    extension["status"] = ExtensionStatus.FAILED
                    extension["error"] = f"Error running installation script: {str(e)}"
                    self._save_registry()
                    return
            
            # Update extension status
            extension["status"] = ExtensionStatus.INSTALLED
            extension["error"] = None
            extension["installed_at"] = datetime.now().isoformat()
            self._save_registry()
            
            logger.info(f"Extension {extension_id} installed successfully")
            
            # Enable if requested
            if task.get("enable", False):
                self.enable_extension(extension_id)
        
        except Exception as e:
            logger.error(f"Error installing extension {extension_id}: {str(e)}")
            
            # Update extension status
            extension = self.registry["extensions"].get(extension_id)
            if extension:
                extension["status"] = ExtensionStatus.FAILED
                extension["error"] = f"Installation error: {str(e)}"
                self._save_registry()
        
        finally:
            # Decrement active installations
            self.active_installations -= 1
    
    def _process_uninstall_task(self, task: Dict[str, Any]) -> None:
        """Process an extension uninstallation task"""
        extension_id = task["extension_id"]
        remove_data = task.get("remove_data", False)
        
        try:
            # Get extension info
            extension = self.registry["extensions"].get(extension_id)
            
            if not extension:
                logger.error(f"Extension {extension_id} not found in registry")
                return
            
            logger.info(f"Uninstalling extension {extension_id}...")
            
            # Disable first if enabled
            if extension.get("status") == ExtensionStatus.ENABLED:
                self.disable_extension(extension_id)
            
            # Get extension source path
            extension_path = extension.get("path")
            
            if not extension_path or not os.path.exists(extension_path):
                logger.warning(f"Extension path {extension_path} not found")
                # Continue with uninstallation anyway
            else:
                # Run uninstallation script if it exists
                uninstall_script = os.path.join(extension_path, "uninstall.py")
                if os.path.exists(uninstall_script):
                    logger.info(f"Running uninstallation script for extension {extension_id}")
                    
                    try:
                        # Run the uninstallation script
                        process = subprocess.Popen(
                            [sys.executable, uninstall_script],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=extension_path,
                            text=True
                        )
                        
                        stdout, stderr = process.communicate(timeout=self.config["extension_timeout"])
                        exit_code = process.returncode
                        
                        if exit_code != 0:
                            logger.error(f"Uninstallation script failed for extension {extension_id}: {stderr}")
                            # Continue with uninstallation anyway
                    
                    except Exception as e:
                        logger.error(f"Error running uninstallation script for extension {extension_id}: {str(e)}")
                        # Continue with uninstallation anyway
                
                # Remove extension directory if requested
                if remove_data:
                    try:
                        shutil.rmtree(extension_path)
                        logger.info(f"Removed extension directory for {extension_id}")
                    except Exception as e:
                        logger.error(f"Error removing extension directory for {extension_id}: {str(e)}")
            
            # Remove from module cache
            if extension_id in self.extension_modules:
                del self.extension_modules[extension_id]
            
            # Update extension status
            extension["status"] = ExtensionStatus.UNINSTALLED
            extension["enabled"] = False
            extension["uninstalled_at"] = datetime.now().isoformat()
            self._save_registry()
            
            logger.info(f"Extension {extension_id} uninstalled successfully")
        
        except Exception as e:
            logger.error(f"Error uninstalling extension {extension_id}: {str(e)}")
            
            # Update extension status
            extension = self.registry["extensions"].get(extension_id)
            if extension:
                extension["error"] = f"Uninstallation error: {str(e)}"
                self._save_registry()
    
    def _load_extension_module(self, extension: Dict[str, Any]) -> Any:
        """Load an extension module"""
        extension_id = extension["id"]
        
        # If already loaded, return it
        if extension_id in self.extension_modules:
            return self.extension_modules[extension_id]
        
        # Get extension path
        extension_path = extension.get("path")
        
        if not extension_path or not os.path.exists(extension_path):
            logger.error(f"Extension path {extension_path} not found for {extension_id}")
            return None
        
        # Find the main module
        main_module = extension.get("main", "__init__.py")
        main_path = os.path.join(extension_path, main_module)
        
        if not os.path.exists(main_path):
            # Try common alternatives
            alternatives = ["main.py", "extension.py", "index.py"]
            for alt in alternatives:
                alt_path = os.path.join(extension_path, alt)
                if os.path.exists(alt_path):
                    main_path = alt_path
                    break
        
        if not os.path.exists(main_path):
            logger.error(f"Main module not found for extension {extension_id}")
            return None
        
        try:
            # Load the module
            module_name = f"extension_{extension_id}"
            spec = importlib.util.spec_from_file_location(module_name, main_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Cache the module
            self.extension_modules[extension_id] = module
            
            return module
        
        except Exception as e:
            logger.error(f"Error loading extension module {extension_id}: {str(e)}")
            return None
    
    def call_extension(
        self,
        extension_id: str,
        function_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None
    ) -> Any:
        """
        Call a function in an extension
        
        Args:
            extension_id: ID of the extension
            function_name: Name of the function to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        # Check if extension exists in registry
        if extension_id not in self.registry["extensions"]:
            logger.warning(f"Extension {extension_id} not found in registry")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension not found in registry"
            }
        
        # Get extension info
        extension = self.registry["extensions"][extension_id]
        
        # Check if enabled
        if extension.get("status") != ExtensionStatus.ENABLED:
            logger.warning(f"Extension {extension_id} is not enabled")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension is not enabled"
            }
        
        # Get the extension module
        module = self.extension_modules.get(extension_id)
        
        if module is None:
            module = self._load_extension_module(extension)
            
            if module is None:
                return {
                    "success": False,
                    "extension_id": extension_id,
                    "error": "Failed to load extension module"
                }
        
        # Check if function exists
        if not hasattr(module, function_name):
            logger.warning(f"Function {function_name} not found in extension {extension_id}")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": f"Function {function_name} not found in extension"
            }
        
        # Call the function
        try:
            func = getattr(module, function_name)
            result = func(*(args or []), **(kwargs or {}))
            
            return {
                "success": True,
                "extension_id": extension_id,
                "function": function_name,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error calling function {function_name} in extension {extension_id}: {str(e)}")
            return {
                "success": False,
                "extension_id": extension_id,
                "function": function_name,
                "error": f"Error calling function: {str(e)}"
            }
    
    def get_extension_info(self, extension_id: str) -> Dict[str, Any]:
        """
        Get detailed information about an extension
        
        Args:
            extension_id: ID of the extension
            
        Returns:
            Extension details
        """
        # Check if extension exists in registry
        if extension_id not in self.registry["extensions"]:
            logger.warning(f"Extension {extension_id} not found in registry")
            return {
                "success": False,
                "extension_id": extension_id,
                "error": "Extension not found in registry"
            }
        
        # Get extension info
        extension = self.registry["extensions"][extension_id]
        
        # Get additional information for installed extensions
        if extension.get("status") in [ExtensionStatus.INSTALLED, ExtensionStatus.ENABLED]:
            # Get the extension module
            module = self.extension_modules.get(extension_id)
            
            if module is None:
                module = self._load_extension_module(extension)
            
            if module is not None:
                # Get functions
                functions = []
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and not name.startswith("_"):
                        try:
                            sig = inspect.signature(obj)
                            
                            functions.append({
                                "name": name,
                                "signature": str(sig),
                                "docstring": inspect.getdoc(obj) or "",
                                "parameters": [
                                    {"name": param_name, "kind": str(param.kind)}
                                    for param_name, param in sig.parameters.items()
                                ]
                            })
                        except Exception:
                            # Skip functions with issues
                            pass
                
                # Get classes
                classes = []
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and not name.startswith("_"):
                        classes.append({
                            "name": name,
                            "docstring": inspect.getdoc(obj) or "",
                            "methods": [
                                method_name for method_name, method_obj in inspect.getmembers(obj)
                                if inspect.isfunction(method_obj) and not method_name.startswith("_")
                            ]
                        })
                
                extension["api"] = {
                    "functions": functions,
                    "classes": classes
                }
        
        return {
            "success": True,
            "extension": extension
        }
    
    def list_extensions(
        self,
        status: Optional[str] = None,
        type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List extensions filtered by status and type
        
        Args:
            status: Filter by status (e.g., "installed", "enabled")
            type: Filter by extension type
            
        Returns:
            List of extensions
        """
        extensions = []
        
        for ext_id, ext in self.registry["extensions"].items():
            # Apply filters
            if status and ext.get("status") != status:
                continue
            
            if type and ext.get("type") != type:
                continue
            
            # Add to list
            extensions.append(ext)
        
        return extensions
    
    def create_extension_template(
        self,
        name: str,
        extension_type: str,
        description: str = "",
        author: str = "AI System"
    ) -> Dict[str, Any]:
        """
        Create a new extension template
        
        Args:
            name: Name of the extension
            extension_type: Type of extension
            description: Extension description
            author: Extension author
            
        Returns:
            Result of template creation
        """
        # Clean the name for use as ID
        extension_id = re.sub(r"[^a-zA-Z0-9_]", "", name.lower().replace(" ", "_"))
        
        # Check if ID already exists
        if extension_id in self.registry["extensions"]:
            return {
                "success": False,
                "error": f"Extension ID {extension_id} already exists"
            }
        
        # Create extension directory
        extension_dir = os.path.join(self.extensions_dir, extension_id)
        
        if os.path.exists(extension_dir):
            return {
                "success": False,
                "error": f"Extension directory already exists: {extension_dir}"
            }
        
        try:
            # Create directory
            os.makedirs(extension_dir, exist_ok=True)
            
            # Create extension.json
            metadata = {
                "id": extension_id,
                "name": name,
                "version": "0.1.0",
                "description": description,
                "author": author,
                "type": extension_type,
                "capabilities": [],
                "dependencies": []
            }
            
            with open(os.path.join(extension_dir, "extension.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create __init__.py
            with open(os.path.join(extension_dir, "__init__.py"), "w") as f:
                f.write(f'''"""
{name}

{description}
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

def enable() -> Dict[str, Any]:
    """
    Enable the extension
    
    Returns:
        Enable status
    """
    logger.info("Enabling {name} extension")
    return {{"success": True}}

def disable() -> Dict[str, Any]:
    """
    Disable the extension
    
    Returns:
        Disable status
    """
    logger.info("Disabling {name} extension")
    return {{"success": True}}

def get_capabilities() -> List[str]:
    """
    Get extension capabilities
    
    Returns:
        List of capabilities
    """
    return []

def get_info() -> Dict[str, Any]:
    """
    Get extension information
    
    Returns:
        Extension information
    """
    return {{
        "name": "{name}",
        "description": "{description}",
        "version": "0.1.0",
        "author": "{author}"
    }}
''')
            
            # Create README.md
            with open(os.path.join(extension_dir, "README.md"), "w") as f:
                f.write(f'''# {name}

{description}

## Installation

1. Place this directory in the `extensions` folder of the AI system.
2. Install the extension using the Extension Manager:
   ```python
   extension_manager.install_extension("{extension_id}")
   ```

## Usage

```python
# Example code showing how to use the extension
extension_manager.call_extension("{extension_id}", "get_info")
```

## API

The extension provides the following functions:

- `enable()`: Enable the extension
- `disable()`: Disable the extension
- `get_capabilities()`: Get extension capabilities
- `get_info()`: Get extension information

## Author

{author}
''')
            
            # Register the extension
            self.registry["extensions"][extension_id] = {
                **metadata,
                "path": extension_dir,
                "status": ExtensionStatus.AVAILABLE,
                "source": "local"
            }
            
            self._save_registry()
            
            logger.info(f"Created extension template: {extension_id}")
            
            return {
                "success": True,
                "extension_id": extension_id,
                "path": extension_dir
            }
        
        except Exception as e:
            logger.error(f"Error creating extension template: {str(e)}")
            
            # Clean up if needed
            if os.path.exists(extension_dir):
                try:
                    shutil.rmtree(extension_dir)
                except Exception:
                    pass
            
            return {
                "success": False,
                "error": f"Error creating extension template: {str(e)}"
            }
    
    def update_registry(self) -> Dict[str, Any]:
        """
        Update the extension registry
        
        Returns:
            Update status
        """
        try:
            # Discover local extensions
            self.discover_extensions(["local"])
            
            # If allowed, check remote registry
            if self.config["allow_remote_sources"]:
                self.discover_extensions(["registry"])
            
            return {
                "success": True,
                "extension_count": len(self.registry["extensions"])
            }
        
        except Exception as e:
            logger.error(f"Error updating registry: {str(e)}")
            return {
                "success": False,
                "error": f"Error updating registry: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get extension manager status"""
        return {
            "registry_version": self.registry.get("version", "unknown"),
            "extension_count": len(self.registry.get("extensions", {})),
            "installed_count": len([
                ext for ext in self.registry.get("extensions", {}).values()
                if ext.get("status") in [ExtensionStatus.INSTALLED, ExtensionStatus.ENABLED]
            ]),
            "enabled_count": len([
                ext for ext in self.registry.get("extensions", {}).values()
                if ext.get("status") == ExtensionStatus.ENABLED
            ]),
            "active_installations": self.active_installations,
            "pending_tasks": self.task_queue.qsize(),
            "config": self.config
        }

# Initialize the extension manager when module is imported
extension_manager = ExtensionManager()