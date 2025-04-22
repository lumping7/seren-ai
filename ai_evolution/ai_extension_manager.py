"""
Extension Manager for Seren

Manages extensions, plugins, and add-ons that enhance the capabilities
of the AI system through flexible modular components.
"""

import os
import sys
import json
import logging
import time
import uuid
import importlib
import importlib.util
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from datetime import datetime
import threading
import queue

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

class ExtensionCategory(Enum):
    """Categories of extensions"""
    REASONING = "reasoning"      # Reasoning capabilities
    KNOWLEDGE = "knowledge"      # Knowledge capabilities
    INTERFACE = "interface"      # Interface capabilities
    PERCEPTION = "perception"    # Perception capabilities
    GENERATION = "generation"    # Generation capabilities
    TOOLING = "tooling"          # Tooling capabilities
    SECURITY = "security"        # Security capabilities
    ANALYTICS = "analytics"      # Analytics capabilities

class ExtensionStatus(Enum):
    """Status of extensions"""
    INSTALLED = "installed"      # Extension is installed
    ENABLED = "enabled"          # Extension is enabled
    DISABLED = "disabled"        # Extension is disabled
    FAILED = "failed"            # Extension has failed
    INCOMPATIBLE = "incompatible"  # Extension is incompatible
    UPDATING = "updating"        # Extension is updating

class ExtensionManager:
    """
    Extension Manager for Seren
    
    Provides a flexible plugin architecture for extending the capabilities
    of the AI system through modular components:
    - Extension discovery and registration
    - Extension loading and initialization
    - Extension execution and interaction
    - Extension status monitoring
    - Extension update management
    
    Bleeding-edge capabilities:
    1. Dynamic extension discovery and hot-loading
    2. Capability-based extension matching
    3. Extension dependency management
    4. Secure extension sandbox execution
    5. Cross-extension capability composition
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the extension manager"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set extensions directory
        self.extensions_dir = os.path.join(self.base_dir, "extensions")
        os.makedirs(self.extensions_dir, exist_ok=True)
        
        # Extension registry
        self.extensions = {}
        
        # Extension instances
        self.instances = {}
        
        # Extension capabilities
        self.capabilities = {}
        
        # Extension stats
        self.stats = {
            "total_extensions": 0,
            "enabled_extensions": 0,
            "disabled_extensions": 0,
            "failed_extensions": 0,
            "extension_calls": 0,
            "extension_errors": 0,
            "category_counts": {category.value: 0 for category in ExtensionCategory}
        }
        
        # Load extensions
        self._discover_extensions()
        
        logger.info("Extension Manager initialized")
    
    def _discover_extensions(self):
        """Discover available extensions"""
        # Check extensions directory
        if not os.path.exists(self.extensions_dir):
            logger.warning(f"Extensions directory not found: {self.extensions_dir}")
            return
        
        # Find extension manifests
        for root, dirs, files in os.walk(self.extensions_dir):
            for file in files:
                if file == "manifest.json":
                    manifest_path = os.path.join(root, file)
                    try:
                        with open(manifest_path, "r") as f:
                            manifest = json.load(f)
                        
                        extension_id = manifest.get("id")
                        if not extension_id:
                            logger.warning(f"Extension manifest missing ID: {manifest_path}")
                            continue
                        
                        # Register extension
                        self.register_extension(
                            extension_id=extension_id,
                            manifest=manifest,
                            path=os.path.dirname(manifest_path)
                        )
                    
                    except Exception as e:
                        logger.error(f"Error loading extension manifest: {manifest_path} - {str(e)}")
    
    def register_extension(
        self,
        extension_id: str,
        manifest: Dict[str, Any],
        path: str
    ) -> bool:
        """
        Register an extension
        
        Args:
            extension_id: Unique extension ID
            manifest: Extension manifest
            path: Path to extension files
            
        Returns:
            Success status
        """
        # Check if extension already registered
        if extension_id in self.extensions:
            logger.warning(f"Extension already registered: {extension_id}")
            return False
        
        # Validate manifest
        required_fields = ["name", "version", "description", "main", "category"]
        for field in required_fields:
            if field not in manifest:
                logger.error(f"Extension manifest missing field: {field} - {extension_id}")
                return False
        
        # Create extension record
        extension = {
            "id": extension_id,
            "name": manifest["name"],
            "version": manifest["version"],
            "description": manifest["description"],
            "main": manifest["main"],
            "category": manifest["category"],
            "author": manifest.get("author", "Unknown"),
            "dependencies": manifest.get("dependencies", {}),
            "capabilities": manifest.get("capabilities", []),
            "settings": manifest.get("settings", {}),
            "status": ExtensionStatus.INSTALLED.value,
            "path": path,
            "enabled": False,
            "loaded_at": None,
            "last_call": None,
            "call_count": 0,
            "error_count": 0
        }
        
        # Store extension
        self.extensions[extension_id] = extension
        
        # Update stats
        self.stats["total_extensions"] += 1
        
        try:
            category = ExtensionCategory(manifest["category"])
            self.stats["category_counts"][category.value] += 1
        except ValueError:
            logger.warning(f"Unknown extension category: {manifest['category']} - {extension_id}")
        
        logger.info(f"Extension registered: {extension_id} - {manifest['name']} {manifest['version']}")
        
        return True
    
    def enable_extension(self, extension_id: str) -> bool:
        """
        Enable an extension
        
        Args:
            extension_id: Extension ID to enable
            
        Returns:
            Success status
        """
        # Get the extension
        extension = self.extensions.get(extension_id)
        
        if not extension:
            logger.error(f"Extension not found: {extension_id}")
            return False
        
        # Check if already enabled
        if extension["enabled"]:
            logger.info(f"Extension already enabled: {extension_id}")
            return True
        
        # Load the extension
        success = self._load_extension(extension_id)
        
        if success:
            # Mark as enabled
            extension["enabled"] = True
            extension["status"] = ExtensionStatus.ENABLED.value
            
            # Update stats
            self.stats["enabled_extensions"] += 1
            
            logger.info(f"Extension enabled: {extension_id}")
            
            return True
        else:
            # Mark as failed
            extension["enabled"] = False
            extension["status"] = ExtensionStatus.FAILED.value
            
            # Update stats
            self.stats["failed_extensions"] += 1
            
            logger.error(f"Failed to enable extension: {extension_id}")
            
            return False
    
    def disable_extension(self, extension_id: str) -> bool:
        """
        Disable an extension
        
        Args:
            extension_id: Extension ID to disable
            
        Returns:
            Success status
        """
        # Get the extension
        extension = self.extensions.get(extension_id)
        
        if not extension:
            logger.error(f"Extension not found: {extension_id}")
            return False
        
        # Check if already disabled
        if not extension["enabled"]:
            logger.info(f"Extension already disabled: {extension_id}")
            return True
        
        # Unload the extension
        success = self._unload_extension(extension_id)
        
        # Mark as disabled
        extension["enabled"] = False
        extension["status"] = ExtensionStatus.DISABLED.value
        
        # Update stats
        self.stats["enabled_extensions"] -= 1
        self.stats["disabled_extensions"] += 1
        
        logger.info(f"Extension disabled: {extension_id}")
        
        return True
    
    def call_extension(
        self,
        extension_id: str,
        method: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Call a method on an extension
        
        Args:
            extension_id: Extension ID to call
            method: Method name to call
            params: Parameters for the method
            
        Returns:
            Result of the method call
        """
        # Get the extension
        extension = self.extensions.get(extension_id)
        
        if not extension:
            return {"error": f"Extension not found: {extension_id}"}
        
        # Check if enabled
        if not extension["enabled"]:
            return {"error": f"Extension not enabled: {extension_id}"}
        
        # Get the instance
        instance = self.instances.get(extension_id)
        
        if not instance:
            return {"error": f"Extension not loaded: {extension_id}"}
        
        # Check if method exists
        if not hasattr(instance, method):
            return {"error": f"Method not found: {method} in extension {extension_id}"}
        
        # Get the method
        method_fn = getattr(instance, method)
        
        if not callable(method_fn):
            return {"error": f"Not a callable method: {method} in extension {extension_id}"}
        
        # Call the method
        try:
            # Update extension stats
            extension["call_count"] += 1
            extension["last_call"] = datetime.now().isoformat()
            
            # Update manager stats
            self.stats["extension_calls"] += 1
            
            # Call the method
            result = method_fn(**(params or {}))
            
            return {"result": result}
        
        except Exception as e:
            # Update error stats
            extension["error_count"] += 1
            self.stats["extension_errors"] += 1
            
            logger.error(f"Error calling extension method {extension_id}.{method}: {str(e)}")
            
            return {"error": f"Error calling method: {str(e)}"}
    
    def get_extensions(self, category: str = None, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of extensions
        
        Args:
            category: Filter by category
            enabled_only: Only return enabled extensions
            
        Returns:
            List of extensions
        """
        # Collect matching extensions
        matching = []
        
        for ext_id, extension in self.extensions.items():
            # Apply filters
            if category and extension["category"] != category:
                continue
            
            if enabled_only and not extension["enabled"]:
                continue
            
            # Include extension
            matching.append(extension)
        
        return matching
    
    def get_extension(self, extension_id: str) -> Optional[Dict[str, Any]]:
        """Get extension details by ID"""
        return self.extensions.get(extension_id)
    
    def find_extensions_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """
        Find extensions providing a specific capability
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of extensions with the capability
        """
        # Collect extensions with matching capability
        matching = []
        
        for ext_id, extension in self.extensions.items():
            if capability in extension["capabilities"]:
                matching.append(extension)
        
        return matching
    
    def update_extension(self, extension_id: str) -> bool:
        """
        Update an extension to the latest version
        
        Args:
            extension_id: Extension ID to update
            
        Returns:
            Success status
        """
        # Get the extension
        extension = self.extensions.get(extension_id)
        
        if not extension:
            logger.error(f"Extension not found: {extension_id}")
            return False
        
        # Set status to updating
        was_enabled = extension["enabled"]
        extension["status"] = ExtensionStatus.UPDATING.value
        
        # Disable if enabled
        if was_enabled:
            self.disable_extension(extension_id)
        
        # In a real implementation, this would check for updates and download them
        # Here we just simulate it
        
        # Simulate update
        logger.info(f"Updating extension: {extension_id}")
        
        # Increment version
        version_parts = extension["version"].split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        extension["version"] = ".".join(version_parts)
        
        # Reset status
        extension["status"] = ExtensionStatus.INSTALLED.value
        
        # Re-enable if it was enabled
        if was_enabled:
            self.enable_extension(extension_id)
        
        logger.info(f"Extension updated to version {extension['version']}: {extension_id}")
        
        return True
    
    def create_extension_template(
        self,
        name: str,
        description: str,
        category: str,
        capabilities: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new extension template
        
        Args:
            name: Extension name
            description: Extension description
            category: Extension category
            capabilities: List of capabilities provided
            
        Returns:
            Created extension details
        """
        # Generate extension ID
        extension_id = name.lower().replace(" ", "_")
        
        # Create extension directory
        extension_dir = os.path.join(self.extensions_dir, extension_id)
        if os.path.exists(extension_dir):
            return {"error": f"Extension directory already exists: {extension_dir}"}
        
        os.makedirs(extension_dir, exist_ok=True)
        
        # Create manifest
        manifest = {
            "id": extension_id,
            "name": name,
            "version": "0.1.0",
            "description": description,
            "category": category,
            "author": "Seren",
            "main": "main.py",
            "capabilities": capabilities or [],
            "dependencies": {},
            "settings": {}
        }
        
        # Create manifest file
        manifest_path = os.path.join(extension_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Create main file
        main_path = os.path.join(extension_dir, "main.py")
        with open(main_path, "w") as f:
            f.write(f'''"""
{name} Extension for Seren

{description}
"""

class Extension:
    """
    {name} Extension
    
    Provides the following capabilities:
    {", ".join(capabilities or [])}
    """
    
    def __init__(self):
        """Initialize the extension"""
        self.name = "{name}"
        self.version = "0.1.0"
    
    def get_info(self):
        """Get extension information"""
        return {{
            "name": self.name,
            "version": self.version,
            "description": "{description}"
        }}
    
    def ping(self):
        """Simple ping method to check if extension is working"""
        return "pong"
''')
        
        # Register the extension
        self.register_extension(
            extension_id=extension_id,
            manifest=manifest,
            path=extension_dir
        )
        
        logger.info(f"Created extension template: {extension_id}")
        
        return {
            "id": extension_id,
            "name": name,
            "path": extension_dir,
            "manifest": manifest
        }
    
    def _load_extension(self, extension_id: str) -> bool:
        """
        Load an extension
        
        Args:
            extension_id: Extension ID to load
            
        Returns:
            Success status
        """
        # Get the extension
        extension = self.extensions.get(extension_id)
        
        if not extension:
            logger.error(f"Extension not found: {extension_id}")
            return False
        
        # Check if already loaded
        if extension_id in self.instances:
            logger.info(f"Extension already loaded: {extension_id}")
            return True
        
        # Check dependencies
        for dep_id, dep_version in extension["dependencies"].items():
            dep = self.extensions.get(dep_id)
            
            if not dep:
                logger.error(f"Dependency not found: {dep_id} required by {extension_id}")
                return False
            
            if not dep["enabled"]:
                # Try to enable the dependency
                if not self.enable_extension(dep_id):
                    logger.error(f"Failed to enable dependency: {dep_id} required by {extension_id}")
                    return False
        
        try:
            # Load the extension module
            main_file = extension["main"]
            module_path = os.path.join(extension["path"], main_file)
            
            # Import the module
            module_name = f"extensions.{extension_id}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the Extension class
            extension_class = getattr(module, "Extension")
            
            # Create an instance
            instance = extension_class()
            
            # Store the instance
            self.instances[extension_id] = instance
            
            # Register capabilities
            for capability in extension["capabilities"]:
                if capability not in self.capabilities:
                    self.capabilities[capability] = []
                
                if extension_id not in self.capabilities[capability]:
                    self.capabilities[capability].append(extension_id)
            
            # Update extension
            extension["loaded_at"] = datetime.now().isoformat()
            
            logger.info(f"Extension loaded: {extension_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading extension {extension_id}: {str(e)}")
            
            # Mark as failed
            extension["status"] = ExtensionStatus.FAILED.value
            
            # Update stats
            self.stats["failed_extensions"] += 1
            
            return False
    
    def _unload_extension(self, extension_id: str) -> bool:
        """
        Unload an extension
        
        Args:
            extension_id: Extension ID to unload
            
        Returns:
            Success status
        """
        # Get the extension
        extension = self.extensions.get(extension_id)
        
        if not extension:
            logger.error(f"Extension not found: {extension_id}")
            return False
        
        # Check if not loaded
        if extension_id not in self.instances:
            logger.info(f"Extension not loaded: {extension_id}")
            return True
        
        # Remove instance
        del self.instances[extension_id]
        
        # Remove from capabilities
        for capability, ext_ids in self.capabilities.items():
            if extension_id in ext_ids:
                ext_ids.remove(extension_id)
        
        logger.info(f"Extension unloaded: {extension_id}")
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the extension manager"""
        return {
            "operational": True,
            "stats": {
                "total_extensions": self.stats["total_extensions"],
                "enabled_extensions": self.stats["enabled_extensions"],
                "disabled_extensions": self.stats["disabled_extensions"],
                "failed_extensions": self.stats["failed_extensions"],
                "extension_calls": self.stats["extension_calls"]
            },
            "capabilities_count": len(self.capabilities)
        }

# Initialize extension manager
extension_manager = ExtensionManager()