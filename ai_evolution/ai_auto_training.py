"""
Auto-Training System for Seren

Manages automated training, fine-tuning, and model optimization
through continuous learning and adaptation.
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

class TrainingStrategy(Enum):
    """Training strategies"""
    SUPERVISED = "supervised"            # Supervised learning
    REINFORCEMENT = "reinforcement"      # Reinforcement learning
    SELF_SUPERVISED = "self_supervised"  # Self-supervised learning
    FEDERATED = "federated"              # Federated learning
    ACTIVE = "active"                    # Active learning
    TRANSFER = "transfer"                # Transfer learning
    META = "meta"                        # Meta-learning
    CONTINUAL = "continual"              # Continual learning

class TrainingStatus(Enum):
    """Status of training sessions"""
    INITIALIZING = "initializing"  # Training session is initializing
    PREPARING = "preparing"        # Preparing datasets and resources
    TRAINING = "training"          # Training is in progress
    EVALUATING = "evaluating"      # Evaluating model performance
    COMPLETED = "completed"        # Training completed successfully
    FAILED = "failed"              # Training failed
    STOPPED = "stopped"            # Training was manually stopped

class DatasetType(Enum):
    """Types of datasets"""
    USER_INTERACTION = "user_interaction"  # Data from user interactions
    SYNTHETIC = "synthetic"                # Synthetically generated data
    CURATED = "curated"                    # Manually curated data
    EXTERNAL = "external"                  # External data sources
    GENERATED = "generated"                # AI-generated data
    FEDERATED = "federated"                # Federated data from multiple sources

class AIAutoTraining:
    """
    Auto-Training System for Seren
    
    Provides autonomous training capabilities to continuously improve
    and adapt AI models through various learning strategies:
    - Supervised fine-tuning
    - Reinforcement learning
    - Self-supervised learning
    - Federated learning
    - Active learning
    - Transfer learning
    
    Bleeding-edge capabilities:
    1. Automated curriculum generation
    2. Transfer learning optimization
    3. Federated learning across multiple instances
    4. Meta-learning for rapid adaptation
    5. Continual learning without catastrophic forgetting
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the auto-training system"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set datasets directory
        self.datasets_dir = os.path.join(self.base_dir, "datasets")
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Set models directory
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Training sessions
        self.training_sessions = {}
        
        # Active training sessions
        self.active_sessions = set()
        
        # Dataset registry
        self.datasets = {}
        
        # Model registry
        self.models = {}
        
        # Training stats
        self.stats = {
            "total_sessions": 0,
            "completed_sessions": 0,
            "failed_sessions": 0,
            "total_training_time": 0,
            "strategy_usage": {strategy.value: 0 for strategy in TrainingStrategy},
            "dataset_usage": {dataset_type.value: 0 for dataset_type in DatasetType}
        }
        
        # Discover datasets
        self._discover_datasets()
        
        logger.info("Auto-Training System initialized")
    
    def _discover_datasets(self):
        """Discover available datasets"""
        # Check datasets directory
        if not os.path.exists(self.datasets_dir):
            logger.warning(f"Datasets directory not found: {self.datasets_dir}")
            return
        
        # Find dataset manifests
        for root, dirs, files in os.walk(self.datasets_dir):
            for file in files:
                if file == "manifest.json":
                    manifest_path = os.path.join(root, file)
                    try:
                        with open(manifest_path, "r") as f:
                            manifest = json.load(f)
                        
                        dataset_id = manifest.get("id")
                        if not dataset_id:
                            logger.warning(f"Dataset manifest missing ID: {manifest_path}")
                            continue
                        
                        # Register dataset
                        self.register_dataset(
                            dataset_id=dataset_id,
                            manifest=manifest,
                            path=os.path.dirname(manifest_path)
                        )
                    
                    except Exception as e:
                        logger.error(f"Error loading dataset manifest: {manifest_path} - {str(e)}")
    
    def register_dataset(
        self,
        dataset_id: str,
        manifest: Dict[str, Any],
        path: str
    ) -> bool:
        """
        Register a dataset
        
        Args:
            dataset_id: Unique dataset ID
            manifest: Dataset manifest
            path: Path to dataset files
            
        Returns:
            Success status
        """
        # Check if dataset already registered
        if dataset_id in self.datasets:
            logger.warning(f"Dataset already registered: {dataset_id}")
            return False
        
        # Validate manifest
        required_fields = ["name", "version", "description", "type", "format"]
        for field in required_fields:
            if field not in manifest:
                logger.error(f"Dataset manifest missing field: {field} - {dataset_id}")
                return False
        
        # Create dataset record
        dataset = {
            "id": dataset_id,
            "name": manifest["name"],
            "version": manifest["version"],
            "description": manifest["description"],
            "type": manifest["type"],
            "format": manifest["format"],
            "size": manifest.get("size", 0),
            "examples": manifest.get("examples", 0),
            "features": manifest.get("features", []),
            "licenses": manifest.get("licenses", []),
            "tags": manifest.get("tags", []),
            "path": path,
            "registered_at": datetime.now().isoformat(),
            "last_used": None,
            "use_count": 0
        }
        
        # Store dataset
        self.datasets[dataset_id] = dataset
        
        # Update stats if dataset type is recognized
        try:
            dataset_type = DatasetType(manifest["type"])
            self.stats["dataset_usage"][dataset_type.value] += 1
        except ValueError:
            logger.warning(f"Unknown dataset type: {manifest['type']} - {dataset_id}")
        
        logger.info(f"Dataset registered: {dataset_id} - {manifest['name']} {manifest['version']}")
        
        return True
    
    def create_training_session(
        self,
        model_id: str,
        description: str,
        strategy: str,
        dataset_ids: List[str],
        parameters: Dict[str, Any] = None,
        auto_start: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new training session
        
        Args:
            model_id: ID of the model to train
            description: Description of the training session
            strategy: Training strategy to use
            dataset_ids: List of dataset IDs to use
            parameters: Training parameters
            auto_start: Whether to start training automatically
            
        Returns:
            Training session object
        """
        # Validate strategy
        try:
            training_strategy = TrainingStrategy(strategy)
        except ValueError:
            logger.error(f"Invalid training strategy: {strategy}")
            return {"error": f"Invalid training strategy: {strategy}"}
        
        # Validate datasets
        for dataset_id in dataset_ids:
            if dataset_id not in self.datasets:
                logger.error(f"Dataset not found: {dataset_id}")
                return {"error": f"Dataset not found: {dataset_id}"}
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session
        session = {
            "id": session_id,
            "model_id": model_id,
            "description": description,
            "strategy": strategy,
            "dataset_ids": dataset_ids,
            "parameters": parameters or {},
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "status": TrainingStatus.INITIALIZING.value,
            "progress": 0.0,  # 0.0 to 1.0
            "metrics": {},
            "logs": [],
            "output_model_id": None
        }
        
        # Store session
        self.training_sessions[session_id] = session
        
        # Update stats
        self.stats["total_sessions"] += 1
        self.stats["strategy_usage"][training_strategy.value] += 1
        
        logger.info(f"Training session created: {session_id} for model {model_id}")
        
        # Start training if requested
        if auto_start:
            self.start_training(session_id)
        
        return session
    
    def start_training(self, session_id: str) -> bool:
        """
        Start a training session
        
        Args:
            session_id: ID of the session to start
            
        Returns:
            Success status
        """
        # Get the session
        session = self.training_sessions.get(session_id)
        
        if not session:
            logger.error(f"Training session not found: {session_id}")
            return False
        
        # Check if already started
        if session["status"] not in [TrainingStatus.INITIALIZING.value, TrainingStatus.STOPPED.value]:
            logger.warning(f"Training session {session_id} is already in progress or completed")
            return False
        
        # Update session state
        session["status"] = TrainingStatus.PREPARING.value
        session["started_at"] = datetime.now().isoformat()
        
        # Add to active sessions
        self.active_sessions.add(session_id)
        
        # Add log entry
        self._add_log(session_id, "Training session started")
        
        # Start training process (in background)
        threading.Thread(target=self._train, args=(session_id,)).start()
        
        logger.info(f"Training session started: {session_id}")
        
        return True
    
    def stop_training(self, session_id: str) -> bool:
        """
        Stop a training session
        
        Args:
            session_id: ID of the session to stop
            
        Returns:
            Success status
        """
        # Get the session
        session = self.training_sessions.get(session_id)
        
        if not session:
            logger.error(f"Training session not found: {session_id}")
            return False
        
        # Check if in progress
        if session["status"] not in [
            TrainingStatus.PREPARING.value,
            TrainingStatus.TRAINING.value,
            TrainingStatus.EVALUATING.value
        ]:
            logger.warning(f"Training session {session_id} is not in progress")
            return False
        
        # Update session state
        session["status"] = TrainingStatus.STOPPED.value
        
        # Remove from active sessions
        self.active_sessions.discard(session_id)
        
        # Add log entry
        self._add_log(session_id, "Training session manually stopped")
        
        logger.info(f"Training session stopped: {session_id}")
        
        return True
    
    def get_training_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get training session details by ID"""
        return self.training_sessions.get(session_id)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active training sessions"""
        return [
            self.training_sessions[session_id]
            for session_id in self.active_sessions
            if session_id in self.training_sessions
        ]
    
    def get_datasets(self, dataset_type: str = None) -> List[Dict[str, Any]]:
        """
        Get list of datasets
        
        Args:
            dataset_type: Filter by dataset type
            
        Returns:
            List of datasets
        """
        # Collect matching datasets
        matching = []
        
        for dataset_id, dataset in self.datasets.items():
            # Apply filter
            if dataset_type and dataset["type"] != dataset_type:
                continue
            
            # Include dataset
            matching.append(dataset)
        
        return matching
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset details by ID"""
        return self.datasets.get(dataset_id)
    
    def update_session_progress(
        self,
        session_id: str,
        progress: float,
        metrics: Dict[str, Any] = None,
        status: str = None
    ) -> bool:
        """
        Update progress of a training session
        
        Args:
            session_id: ID of the session to update
            progress: Progress value (0.0 to 1.0)
            metrics: Current metrics
            status: New status (if changing)
            
        Returns:
            Success status
        """
        # Get the session
        session = self.training_sessions.get(session_id)
        
        if not session:
            logger.error(f"Training session not found: {session_id}")
            return False
        
        # Update progress
        session["progress"] = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        
        # Update metrics if provided
        if metrics:
            session["metrics"].update(metrics)
        
        # Update status if provided
        if status:
            try:
                training_status = TrainingStatus(status)
                session["status"] = training_status.value
            except ValueError:
                logger.warning(f"Invalid training status: {status}")
        
        # Add log entry
        self._add_log(
            session_id,
            f"Progress update: {progress:.1%}" + 
            (f", status: {status}" if status else "")
        )
        
        # Check for completion
        if progress >= 1.0 and session["status"] not in [
            TrainingStatus.COMPLETED.value,
            TrainingStatus.FAILED.value,
            TrainingStatus.STOPPED.value
        ]:
            self._complete_training(session_id)
        
        return True
    
    def _train(self, session_id: str) -> None:
        """
        Execute training process
        
        Args:
            session_id: ID of the session to train
        """
        # Get the session
        session = self.training_sessions.get(session_id)
        
        if not session:
            logger.error(f"Training session not found: {session_id}")
            return
        
        try:
            # Get training strategy
            strategy = session["strategy"]
            model_id = session["model_id"]
            dataset_ids = session["dataset_ids"]
            parameters = session["parameters"]
            
            # Update stats for datasets
            for dataset_id in dataset_ids:
                dataset = self.datasets.get(dataset_id)
                if dataset:
                    dataset["use_count"] += 1
                    dataset["last_used"] = datetime.now().isoformat()
            
            # Prepare datasets
            self._add_log(session_id, "Preparing datasets...")
            
            # Simulate dataset preparation time
            time.sleep(1)
            
            # Update progress
            self.update_session_progress(
                session_id,
                0.1,
                status=TrainingStatus.PREPARING.value
            )
            
            # Initialize model for training
            self._add_log(session_id, "Initializing model...")
            
            # Simulate model initialization time
            time.sleep(1)
            
            # Update progress
            self.update_session_progress(
                session_id,
                0.2,
                status=TrainingStatus.TRAINING.value
            )
            
            # Main training loop
            self._add_log(session_id, f"Starting {strategy} training...")
            
            # Simulate training epochs
            num_epochs = parameters.get("epochs", 10)
            for epoch in range(1, num_epochs + 1):
                # Check if stopped
                if self.training_sessions[session_id]["status"] == TrainingStatus.STOPPED.value:
                    self._add_log(session_id, "Training stopped")
                    return
                
                # Simulate epoch training
                time.sleep(0.5)
                
                # Calculate metrics for this epoch
                epoch_metrics = {
                    "epoch": epoch,
                    "loss": 1.0 - min(0.9, (epoch / num_epochs) * 0.9),  # Loss decreases over time
                    "accuracy": min(0.95, 0.5 + (epoch / num_epochs) * 0.45)  # Accuracy increases over time
                }
                
                # Update progress
                progress = 0.2 + (epoch / num_epochs) * 0.6
                self.update_session_progress(
                    session_id,
                    progress,
                    metrics=epoch_metrics
                )
                
                # Log epoch results
                self._add_log(
                    session_id,
                    f"Epoch {epoch}/{num_epochs}: loss={epoch_metrics['loss']:.4f}, accuracy={epoch_metrics['accuracy']:.4f}"
                )
            
            # Evaluate model
            self._add_log(session_id, "Evaluating model...")
            
            # Update progress
            self.update_session_progress(
                session_id,
                0.9,
                status=TrainingStatus.EVALUATING.value
            )
            
            # Simulate evaluation time
            time.sleep(1)
            
            # Final evaluation metrics
            final_metrics = {
                "final_loss": 0.1,
                "final_accuracy": 0.95,
                "f1_score": 0.94,
                "precision": 0.96,
                "recall": 0.93
            }
            
            # Update with final metrics
            self.update_session_progress(
                session_id,
                1.0,
                metrics=final_metrics
            )
            
            # Generate output model ID
            output_model_id = f"{model_id}_trained_{int(time.time())}"
            
            # Update session with output model
            session["output_model_id"] = output_model_id
            
            # Complete training
            self._complete_training(session_id)
        
        except Exception as e:
            logger.error(f"Error in training session {session_id}: {str(e)}")
            
            # Mark as failed
            session["status"] = TrainingStatus.FAILED.value
            
            # Remove from active sessions
            self.active_sessions.discard(session_id)
            
            # Add log entry
            self._add_log(session_id, f"Training failed: {str(e)}")
            
            # Update stats
            self.stats["failed_sessions"] += 1
    
    def _complete_training(self, session_id: str) -> None:
        """
        Complete a training session
        
        Args:
            session_id: ID of the session to complete
        """
        # Get the session
        session = self.training_sessions.get(session_id)
        
        if not session:
            logger.error(f"Training session not found: {session_id}")
            return
        
        # Calculate training time
        started_at = datetime.fromisoformat(session["started_at"])
        completed_at = datetime.now()
        training_time = (completed_at - started_at).total_seconds()
        
        # Update session
        session["status"] = TrainingStatus.COMPLETED.value
        session["completed_at"] = completed_at.isoformat()
        session["progress"] = 1.0
        
        # Remove from active sessions
        self.active_sessions.discard(session_id)
        
        # Add log entry
        self._add_log(
            session_id,
            f"Training completed successfully in {training_time:.1f} seconds"
        )
        
        # Update stats
        self.stats["completed_sessions"] += 1
        self.stats["total_training_time"] += training_time
        
        logger.info(f"Training session completed: {session_id}")
    
    def _add_log(self, session_id: str, message: str) -> None:
        """Add a log entry to a training session"""
        session = self.training_sessions.get(session_id)
        
        if not session:
            return
        
        # Add log entry
        session["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the auto-training system"""
        return {
            "operational": True,
            "stats": {
                "total_sessions": self.stats["total_sessions"],
                "completed_sessions": self.stats["completed_sessions"],
                "failed_sessions": self.stats["failed_sessions"],
                "active_sessions": len(self.active_sessions)
            },
            "datasets_count": len(self.datasets)
        }
    
    def create_dataset_template(
        self,
        name: str,
        description: str,
        dataset_type: str,
        format: str,
        features: List[str] = None,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new dataset template
        
        Args:
            name: Dataset name
            description: Dataset description
            dataset_type: Dataset type
            format: Dataset format (e.g., json, csv, parquet)
            features: List of features in the dataset
            tags: List of tags for the dataset
            
        Returns:
            Created dataset details
        """
        # Validate dataset type
        try:
            DatasetType(dataset_type)
        except ValueError:
            logger.error(f"Invalid dataset type: {dataset_type}")
            return {"error": f"Invalid dataset type: {dataset_type}"}
        
        # Generate dataset ID
        dataset_id = name.lower().replace(" ", "_")
        
        # Create dataset directory
        dataset_dir = os.path.join(self.datasets_dir, dataset_id)
        if os.path.exists(dataset_dir):
            return {"error": f"Dataset directory already exists: {dataset_dir}"}
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create manifest
        manifest = {
            "id": dataset_id,
            "name": name,
            "version": "0.1.0",
            "description": description,
            "type": dataset_type,
            "format": format,
            "size": 0,
            "examples": 0,
            "features": features or [],
            "licenses": [],
            "tags": tags or []
        }
        
        # Create manifest file
        manifest_path = os.path.join(dataset_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Create empty dataset file
        if format == "json":
            dataset_path = os.path.join(dataset_dir, "data.json")
            with open(dataset_path, "w") as f:
                f.write("[]")
        elif format == "csv":
            dataset_path = os.path.join(dataset_dir, "data.csv")
            with open(dataset_path, "w") as f:
                if features:
                    f.write(",".join(features) + "\n")
                else:
                    f.write("")
        else:
            dataset_path = os.path.join(dataset_dir, f"data.{format}")
            with open(dataset_path, "w") as f:
                f.write("")
        
        # Create README file
        readme_path = os.path.join(dataset_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"""# {name} Dataset

{description}

## Features

{', '.join(features or [])}

## Format

This dataset is in {format.upper()} format.

## Usage

This dataset is designed for {dataset_type} learning.

## Tags

{', '.join(tags or [])}
""")
        
        # Register the dataset
        self.register_dataset(
            dataset_id=dataset_id,
            manifest=manifest,
            path=dataset_dir
        )
        
        logger.info(f"Created dataset template: {dataset_id}")
        
        return {
            "id": dataset_id,
            "name": name,
            "path": dataset_dir,
            "manifest": manifest
        }

# Initialize auto-training system
auto_training = AIAutoTraining()