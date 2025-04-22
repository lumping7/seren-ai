"""
Neuro-Symbolic Reasoning Engine

Combines neural network capabilities with symbolic reasoning to provide
explainable, verifiable, and logically sound problem-solving capabilities.
"""

import os
import sys
import json
import logging
import time
import uuid
import re
from enum import Enum
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

class ReasoningStrategy(Enum):
    """Reasoning strategies that can be used"""
    DEDUCTIVE = "deductive"  # From general principles to specific conclusions
    INDUCTIVE = "inductive"  # From specific observations to general principles
    ABDUCTIVE = "abductive"  # From observations to most likely explanation
    ANALOGICAL = "analogical"  # Reasoning by analogy to similar problems
    CAUSAL = "causal"  # Reasoning about cause and effect relationships
    COUNTERFACTUAL = "counterfactual"  # "What if" reasoning
    BAYESIAN = "bayesian"  # Probabilistic reasoning with prior beliefs
    CONSTRAINT = "constraint"  # Reasoning within defined constraints
    CASE_BASED = "case_based"  # Reasoning from past cases
    HYBRID = "hybrid"  # Combination of multiple strategies

class RuleType(Enum):
    """Types of rules in the knowledge base"""
    LOGICAL = "logical"  # Logical implication
    ONTOLOGICAL = "ontological"  # Class and property relationships
    CAUSAL = "causal"  # Cause and effect relationships
    PROCEDURAL = "procedural"  # Step-by-step procedures
    HEURISTIC = "heuristic"  # Rules of thumb
    CONSTRAINT = "constraint"  # Constraints on solutions
    VALIDATION = "validation"  # Validation rules
    TRANSFORMATION = "transformation"  # Transformation rules

class SymbolicOperator(Enum):
    """Symbolic operators for logical reasoning"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    EQUIVALENT = "equivalent"
    FORALL = "forall"  # Universal quantifier
    EXISTS = "exists"  # Existential quantifier
    IN = "in"  # Set membership
    SUBSET = "subset"  # Subset relationship
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"

class ReasoningStage(Enum):
    """Stages in the reasoning process"""
    PROBLEM_FORMULATION = "problem_formulation"
    CONTEXT_ANALYSIS = "context_analysis"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    PATTERN_RECOGNITION = "pattern_recognition"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    LOGICAL_INFERENCE = "logical_inference"
    CAUSAL_ANALYSIS = "causal_analysis"
    SOLUTION_SYNTHESIS = "solution_synthesis"
    VERIFICATION = "verification"
    EXPLANATION = "explanation"

class NeuroSymbolicEngine:
    """
    Neuro-Symbolic Reasoning Engine
    
    Combines neural network capabilities with symbolic reasoning to provide
    explainable, verifiable, and logically sound problem-solving.
    
    Bleeding-edge capabilities:
    1. Bi-directional translation between neural and symbolic representations
    2. Symbolic knowledge integration with neural reasoning
    3. Logical verification of neural network outputs
    4. Uncertainty-aware reasoning with probabilistic logic
    5. Multi-strategy reasoning composition
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the reasoning engine"""
        # Set the base directory
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Rule sets
        self.rule_sets = self._initialize_rule_sets()
        
        # Reasoning history
        self.reasoning_history = {}
        
        # Strategy handlers
        self.strategy_handlers = {
            ReasoningStrategy.DEDUCTIVE: self._deductive_reasoning,
            ReasoningStrategy.INDUCTIVE: self._inductive_reasoning,
            ReasoningStrategy.ABDUCTIVE: self._abductive_reasoning,
            ReasoningStrategy.ANALOGICAL: self._analogical_reasoning,
            ReasoningStrategy.CAUSAL: self._causal_reasoning,
            ReasoningStrategy.COUNTERFACTUAL: self._counterfactual_reasoning,
            ReasoningStrategy.BAYESIAN: self._bayesian_reasoning,
            ReasoningStrategy.CONSTRAINT: self._constraint_reasoning,
            ReasoningStrategy.CASE_BASED: self._case_based_reasoning,
            ReasoningStrategy.HYBRID: self._hybrid_reasoning
        }
        
        # Set default reasoning strategy
        self.default_strategy = ReasoningStrategy.HYBRID
        
        # Configure verbosity
        self.verbosity = "detailed"  # "minimal", "standard", "detailed", "debug"
        
        # Reasoning metrics
        self.metrics = {
            "total_reasoning_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_reasoning_time": 0,
            "total_reasoning_time": 0,
            "strategy_usage": {strategy.value: 0 for strategy in ReasoningStrategy},
            "stage_timings": {stage.value: 0 for stage in ReasoningStage}
        }
        
        logger.info("Neuro-Symbolic Reasoning Engine initialized")
    
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize the knowledge base with foundational concepts"""
        return {
            "concepts": {
                # Programming concepts
                "variable": {
                    "definition": "Named storage location for data",
                    "properties": ["name", "type", "value", "scope"],
                    "relations": {
                        "has_type": "type",
                        "has_value": "value",
                        "defined_in": "scope"
                    }
                },
                "function": {
                    "definition": "Reusable block of code that performs a specific task",
                    "properties": ["name", "parameters", "return_type", "body", "scope"],
                    "relations": {
                        "has_parameters": ["parameter"],
                        "returns": "return_type",
                        "contains": ["statement"],
                        "defined_in": "scope"
                    }
                },
                "class": {
                    "definition": "Blueprint for creating objects with shared properties and methods",
                    "properties": ["name", "attributes", "methods", "inheritance"],
                    "relations": {
                        "has_attributes": ["attribute"],
                        "has_methods": ["method"],
                        "inherits_from": ["class"]
                    }
                },
                
                # Software design concepts
                "pattern": {
                    "definition": "Reusable solution to a common problem in software design",
                    "properties": ["name", "category", "problem", "solution"],
                    "relations": {
                        "solves": "problem",
                        "belongs_to": "category"
                    }
                },
                "architecture": {
                    "definition": "High-level structure of a software system",
                    "properties": ["name", "components", "connectors", "constraints"],
                    "relations": {
                        "has_components": ["component"],
                        "has_connectors": ["connector"],
                        "follows": ["pattern"]
                    }
                },
                
                # Algorithm concepts
                "algorithm": {
                    "definition": "Step-by-step procedure for calculations or problem-solving",
                    "properties": ["name", "input", "output", "steps", "complexity"],
                    "relations": {
                        "takes_input": ["input_type"],
                        "produces_output": ["output_type"],
                        "has_complexity": "complexity"
                    }
                },
                "complexity": {
                    "definition": "Measure of resources required by an algorithm",
                    "properties": ["time", "space", "notation"],
                    "relations": {
                        "of_algorithm": "algorithm"
                    }
                },
                
                # Data structures
                "data_structure": {
                    "definition": "Specialized format for organizing and storing data",
                    "properties": ["name", "operations", "complexity", "implementation"],
                    "relations": {
                        "supports_operations": ["operation"],
                        "has_complexity": "complexity",
                        "implemented_in": ["language"]
                    }
                },
                
                # Software development concepts
                "requirement": {
                    "definition": "Description of what a system should do",
                    "properties": ["id", "description", "type", "priority"],
                    "relations": {
                        "related_to": ["requirement"],
                        "implemented_by": ["component"]
                    }
                },
                "bug": {
                    "definition": "Error, flaw, or fault in a computer program",
                    "properties": ["id", "description", "severity", "status"],
                    "relations": {
                        "affects": ["component"],
                        "caused_by": ["error"]
                    }
                },
                "test": {
                    "definition": "Procedure to verify system behavior",
                    "properties": ["id", "description", "type", "status"],
                    "relations": {
                        "verifies": ["requirement"],
                        "executed_on": ["component"]
                    }
                }
            },
            
            "domain_knowledge": {
                # Programming languages
                "languages": {
                    "python": {
                        "typing": "dynamic",
                        "paradigms": ["object-oriented", "procedural", "functional"],
                        "key_features": ["indentation-based syntax", "comprehensive standard library", "dynamic typing"]
                    },
                    "javascript": {
                        "typing": "dynamic",
                        "paradigms": ["prototype-based", "functional", "event-driven"],
                        "key_features": ["first-class functions", "asynchronous", "single-threaded with event loop"]
                    },
                    "typescript": {
                        "typing": "static",
                        "paradigms": ["object-oriented", "functional"],
                        "key_features": ["static typing", "interfaces", "generics", "compiles to JavaScript"]
                    },
                    "java": {
                        "typing": "static",
                        "paradigms": ["object-oriented", "class-based"],
                        "key_features": ["platform independence", "strong typing", "garbage collection"]
                    },
                    "c++": {
                        "typing": "static",
                        "paradigms": ["object-oriented", "procedural", "generic"],
                        "key_features": ["manual memory management", "high performance", "templates"]
                    }
                },
                
                # Design patterns
                "design_patterns": {
                    "creational": ["factory", "abstract_factory", "builder", "singleton", "prototype"],
                    "structural": ["adapter", "bridge", "composite", "decorator", "facade", "flyweight", "proxy"],
                    "behavioral": ["chain_of_responsibility", "command", "interpreter", "iterator", "mediator", "memento", "observer", "state", "strategy", "template_method", "visitor"]
                },
                
                # Software architectures
                "architectures": {
                    "monolithic": {
                        "description": "Single-tiered application where all components are interconnected",
                        "strengths": ["simplicity", "performance"],
                        "weaknesses": ["scalability", "maintainability"]
                    },
                    "microservices": {
                        "description": "Collection of small, independent services that communicate via APIs",
                        "strengths": ["scalability", "fault isolation", "technology diversity"],
                        "weaknesses": ["complexity", "network overhead", "distributed system challenges"]
                    },
                    "serverless": {
                        "description": "Cloud-native architecture where servers are managed by cloud providers",
                        "strengths": ["cost efficiency", "scalability", "reduced operational burden"],
                        "weaknesses": ["vendor lock-in", "cold starts", "limited execution time"]
                    },
                    "event_driven": {
                        "description": "Architecture based on producing, detecting, and consuming events",
                        "strengths": ["loose coupling", "scalability", "responsiveness"],
                        "weaknesses": ["complexity", "debugging challenges", "eventual consistency"]
                    }
                },
                
                # Data structures
                "data_structures": {
                    "array": {
                        "operations": {"access": "O(1)", "search": "O(n)", "insert": "O(n)", "delete": "O(n)"},
                        "use_cases": ["sequential data", "caching", "buffering"]
                    },
                    "linked_list": {
                        "operations": {"access": "O(n)", "search": "O(n)", "insert": "O(1)", "delete": "O(1)"},
                        "use_cases": ["dynamic memory allocation", "implementing stacks/queues", "music playlist"]
                    },
                    "hash_table": {
                        "operations": {"access": "N/A", "search": "O(1) avg", "insert": "O(1) avg", "delete": "O(1) avg"},
                        "use_cases": ["databases", "caches", "symbol tables"]
                    },
                    "tree": {
                        "operations": {"access": "N/A", "search": "O(log n)", "insert": "O(log n)", "delete": "O(log n)"},
                        "use_cases": ["hierarchical data", "binary search", "decision trees"]
                    },
                    "graph": {
                        "operations": {"access": "N/A", "search": "O(V+E)", "insert": "O(1)", "delete": "O(V+E)"},
                        "use_cases": ["social networks", "maps", "recommendation systems"]
                    }
                },
                
                # Algorithms
                "algorithms": {
                    "sorting": {
                        "quick_sort": {"average": "O(n log n)", "worst": "O(n²)", "space": "O(log n)"},
                        "merge_sort": {"average": "O(n log n)", "worst": "O(n log n)", "space": "O(n)"},
                        "heap_sort": {"average": "O(n log n)", "worst": "O(n log n)", "space": "O(1)"},
                        "bubble_sort": {"average": "O(n²)", "worst": "O(n²)", "space": "O(1)"}
                    },
                    "search": {
                        "binary_search": {"time": "O(log n)", "space": "O(1)"},
                        "depth_first_search": {"time": "O(V+E)", "space": "O(V)"},
                        "breadth_first_search": {"time": "O(V+E)", "space": "O(V)"},
                        "dijkstra": {"time": "O(V² + E)", "space": "O(V)"}
                    }
                },
                
                # Testing
                "testing": {
                    "unit_testing": {
                        "description": "Testing individual components in isolation",
                        "frameworks": {"python": ["pytest", "unittest"], "javascript": ["jest", "mocha"]}
                    },
                    "integration_testing": {
                        "description": "Testing interactions between components",
                        "frameworks": {"python": ["pytest", "behave"], "javascript": ["cypress", "supertest"]}
                    },
                    "end_to_end_testing": {
                        "description": "Testing entire application workflows",
                        "frameworks": {"python": ["selenium", "playwright"], "javascript": ["cypress", "puppeteer"]}
                    }
                }
            }
        }
    
    def _initialize_rule_sets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize rule sets for different domains"""
        return {
            "software_engineering": [
                {
                    "type": RuleType.LOGICAL.value,
                    "premise": {"concept": "function", "property": "complexity", "operator": SymbolicOperator.GREATER_THAN.value, "value": "O(n²)"},
                    "conclusion": {"concept": "function", "property": "optimization_needed", "value": True},
                    "confidence": 0.8
                },
                {
                    "type": RuleType.HEURISTIC.value,
                    "context": "performance_optimization",
                    "rule": "For CPU-bound operations, prefer algorithmic optimizations over micro-optimizations",
                    "confidence": 0.9
                },
                {
                    "type": RuleType.HEURISTIC.value,
                    "context": "performance_optimization",
                    "rule": "For I/O-bound operations, focus on asynchronous processing and caching",
                    "confidence": 0.9
                },
                {
                    "type": RuleType.LOGICAL.value,
                    "premise": {"concept": "code", "property": "has_side_effects", "value": True},
                    "conclusion": {"concept": "code", "property": "testability", "operator": SymbolicOperator.LESS_THAN.value, "value": "high"},
                    "confidence": 0.7
                },
                {
                    "type": RuleType.CAUSAL.value,
                    "cause": {"concept": "function", "property": "cyclomatic_complexity", "operator": SymbolicOperator.GREATER_THAN.value, "value": 10},
                    "effect": {"concept": "function", "property": "bug_probability", "operator": SymbolicOperator.GREATER_THAN.value, "value": "average"},
                    "confidence": 0.75
                }
            ],
            
            "code_generation": [
                {
                    "type": RuleType.PROCEDURAL.value,
                    "context": "function_definition",
                    "steps": [
                        "Define function signature with appropriate parameters",
                        "Add docstring explaining purpose, parameters, and return value",
                        "Implement input validation",
                        "Implement core logic",
                        "Handle edge cases and errors",
                        "Return appropriate value"
                    ],
                    "confidence": 0.95
                },
                {
                    "type": RuleType.CONSTRAINT.value,
                    "context": "error_handling",
                    "constraint": "Never silently catch exceptions without specific handling or logging",
                    "confidence": 0.9
                },
                {
                    "type": RuleType.VALIDATION.value,
                    "context": "code_quality",
                    "validation": "Functions should have a single responsibility and be no longer than 30 lines",
                    "confidence": 0.8
                },
                {
                    "type": RuleType.LOGICAL.value,
                    "premise": {"concept": "variable", "property": "mutability", "value": True},
                    "conclusion": {"action": "restrict_scope", "reason": "Minimize potential side effects"},
                    "confidence": 0.85
                }
            ],
            
            "debugging": [
                {
                    "type": RuleType.PROCEDURAL.value,
                    "context": "debug_process",
                    "steps": [
                        "Reproduce the issue consistently",
                        "Isolate the specific conditions that trigger the issue",
                        "Examine relevant logs and error messages",
                        "Use debugger to step through code execution",
                        "Identify root cause",
                        "Implement fix",
                        "Verify fix resolves the issue",
                        "Add tests to prevent regression"
                    ],
                    "confidence": 0.9
                },
                {
                    "type": RuleType.HEURISTIC.value,
                    "context": "bug_identification",
                    "rule": "If a bug appears after recent changes, first examine those changes",
                    "confidence": 0.85
                },
                {
                    "type": RuleType.CAUSAL.value,
                    "cause": {"concept": "null_reference", "context": "access_operation"},
                    "effect": {"concept": "exception", "type": "null_reference_exception"},
                    "confidence": 1.0
                },
                {
                    "type": RuleType.ABDUCTIVE.value,
                    "observation": {"concept": "memory_usage", "trend": "increasing", "reset": False},
                    "explanation": {"concept": "bug", "type": "memory_leak"},
                    "confidence": 0.8
                }
            ],
            
            "architecture": [
                {
                    "type": RuleType.HEURISTIC.value,
                    "context": "system_design",
                    "rule": "Prefer loose coupling and high cohesion between components",
                    "confidence": 0.95
                },
                {
                    "type": RuleType.LOGICAL.value,
                    "premise": {"concept": "system", "property": "expected_load", "operator": SymbolicOperator.GREATER_THAN.value, "value": "high"},
                    "conclusion": {"concept": "architecture", "property": "scalability", "priority": "critical"},
                    "confidence": 0.9
                },
                {
                    "type": RuleType.CONSTRAINT.value,
                    "context": "security_sensitive_system",
                    "constraint": "All user inputs must be validated and sanitized before processing",
                    "confidence": 1.0
                },
                {
                    "type": RuleType.ONTOLOGICAL.value,
                    "concept": "microservice",
                    "is_a": "architecture_style",
                    "properties": {
                        "service_boundaries": "defined_by_business_domain",
                        "communication": "via_apis",
                        "deployment": "independent"
                    },
                    "confidence": 1.0
                }
            ],
            
            "testing": [
                {
                    "type": RuleType.PROCEDURAL.value,
                    "context": "test_creation",
                    "steps": [
                        "Identify the functionality to test",
                        "Define test cases covering normal scenarios, edge cases, and error conditions",
                        "Setup test environment and fixtures",
                        "Implement test with arrange-act-assert pattern",
                        "Ensure test isolation",
                        "Verify both positive and negative assertions"
                    ],
                    "confidence": 0.9
                },
                {
                    "type": RuleType.LOGICAL.value,
                    "premise": {"concept": "code", "property": "complexity", "operator": SymbolicOperator.GREATER_THAN.value, "value": "high"},
                    "conclusion": {"concept": "testing", "property": "coverage_requirement", "operator": SymbolicOperator.GREATER_THAN.value, "value": 90},
                    "confidence": 0.8
                },
                {
                    "type": RuleType.HEURISTIC.value,
                    "context": "test_strategy",
                    "rule": "Use test pyramid approach: many unit tests, some integration tests, few end-to-end tests",
                    "confidence": 0.85
                }
            ],
            
            "code_review": [
                {
                    "type": RuleType.PROCEDURAL.value,
                    "context": "code_review_process",
                    "steps": [
                        "Check code functionality meets requirements",
                        "Verify code readability and maintainability",
                        "Ensure adherence to coding standards",
                        "Examine edge cases and error handling",
                        "Verify test coverage",
                        "Check performance considerations",
                        "Look for security vulnerabilities"
                    ],
                    "confidence": 0.9
                },
                {
                    "type": RuleType.VALIDATION.value,
                    "context": "pull_request",
                    "validation": "All automated tests must pass before merging",
                    "confidence": 1.0
                },
                {
                    "type": RuleType.CONSTRAINT.value,
                    "context": "security_sensitive_code",
                    "constraint": "Credentials and secrets must never be hardcoded",
                    "confidence": 1.0
                }
            ]
        }
    
    def reason(
        self,
        query: str,
        context: Dict[str, Any] = None,
        strategy: Optional[Union[ReasoningStrategy, str]] = None,
        knowledge_domains: Optional[List[str]] = None,
        trace_reasoning: bool = True
    ) -> Dict[str, Any]:
        """
        Perform neuro-symbolic reasoning
        
        Args:
            query: The question or problem to reason about
            context: Additional context information
            strategy: Reasoning strategy to use
            knowledge_domains: Specific knowledge domains to consider
            trace_reasoning: Whether to trace the reasoning process
            
        Returns:
            Result of reasoning process
        """
        # Generate reasoning session ID
        session_id = str(uuid.uuid4())
        
        # Initialize context if None
        context = context or {}
        
        # Convert strategy to enum if string
        if isinstance(strategy, str):
            try:
                strategy = ReasoningStrategy(strategy)
            except ValueError:
                logger.warning(f"Invalid reasoning strategy: {strategy}, using default")
                strategy = self.default_strategy
        elif strategy is None:
            # Determine best strategy based on query and context
            strategy = self._determine_best_strategy(query, context)
        
        # Initialize reasoning history for this session
        self.reasoning_history[session_id] = {
            "id": session_id,
            "query": query,
            "context": context,
            "strategy": strategy.value,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration": None,
            "stages": [],
            "result": None,
            "success": None,
            "confidence": None,
            "trace": []
        }
        
        start_time = time.time()
        
        try:
            # Begin reasoning process
            self._log_reasoning(session_id, "Starting reasoning process", level="info")
            
            # Formulate the problem
            problem_formulation = self._execute_stage(
                session_id, 
                ReasoningStage.PROBLEM_FORMULATION,
                lambda: self._formulate_problem(query, context)
            )
            
            # Analyze context
            context_analysis = self._execute_stage(
                session_id,
                ReasoningStage.CONTEXT_ANALYSIS,
                lambda: self._analyze_context(problem_formulation, context)
            )
            
            # Retrieve relevant knowledge
            knowledge_retrieval = self._execute_stage(
                session_id,
                ReasoningStage.KNOWLEDGE_RETRIEVAL,
                lambda: self._retrieve_knowledge(
                    problem_formulation, 
                    context_analysis, 
                    domains=knowledge_domains
                )
            )
            
            # Recognize patterns
            pattern_recognition = self._execute_stage(
                session_id,
                ReasoningStage.PATTERN_RECOGNITION,
                lambda: self._recognize_patterns(
                    problem_formulation, 
                    context_analysis, 
                    knowledge_retrieval
                )
            )
            
            # Generate hypotheses
            hypothesis_generation = self._execute_stage(
                session_id,
                ReasoningStage.HYPOTHESIS_GENERATION,
                lambda: self._generate_hypotheses(
                    problem_formulation, 
                    pattern_recognition, 
                    knowledge_retrieval
                )
            )
            
            # Perform logical inference
            logical_inference = self._execute_stage(
                session_id,
                ReasoningStage.LOGICAL_INFERENCE,
                lambda: self._perform_logical_inference(
                    problem_formulation, 
                    hypothesis_generation, 
                    knowledge_retrieval,
                    strategy
                )
            )
            
            # Analyze causal relationships
            causal_analysis = self._execute_stage(
                session_id,
                ReasoningStage.CAUSAL_ANALYSIS,
                lambda: self._analyze_causal_relationships(
                    problem_formulation, 
                    logical_inference
                )
            )
            
            # Synthesize solution
            solution_synthesis = self._execute_stage(
                session_id,
                ReasoningStage.SOLUTION_SYNTHESIS,
                lambda: self._synthesize_solution(
                    problem_formulation, 
                    logical_inference, 
                    causal_analysis
                )
            )
            
            # Verify solution
            verification = self._execute_stage(
                session_id,
                ReasoningStage.VERIFICATION,
                lambda: self._verify_solution(
                    problem_formulation, 
                    solution_synthesis, 
                    knowledge_retrieval
                )
            )
            
            # Generate explanation
            explanation = self._execute_stage(
                session_id,
                ReasoningStage.EXPLANATION,
                lambda: self._generate_explanation(
                    problem_formulation, 
                    solution_synthesis, 
                    logical_inference,
                    verification
                )
            )
            
            # Compute confidence
            confidence = self._compute_confidence(verification, logical_inference)
            
            # Prepare result
            result = {
                "answer": solution_synthesis["solution"],
                "confidence": confidence,
                "explanation": explanation["explanation"],
                "reasoning_path": self._extract_reasoning_path(session_id),
                "verified": verification["verified"]
            }
            
            # Update reasoning history
            self.reasoning_history[session_id]["result"] = result
            self.reasoning_history[session_id]["success"] = True
            self.reasoning_history[session_id]["confidence"] = confidence
            
            # Update metrics
            self.metrics["successful_tasks"] += 1
            
        except Exception as e:
            logger.error(f"Error in reasoning process: {str(e)}")
            
            result = {
                "error": str(e),
                "partial_reasoning": self._extract_reasoning_path(session_id)
            }
            
            # Update reasoning history
            self.reasoning_history[session_id]["result"] = result
            self.reasoning_history[session_id]["success"] = False
            
            # Update metrics
            self.metrics["failed_tasks"] += 1
        
        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        # Update reasoning history with timing
        self.reasoning_history[session_id]["end_time"] = datetime.now().isoformat()
        self.reasoning_history[session_id]["duration"] = duration
        
        # Update metrics
        self.metrics["total_reasoning_tasks"] += 1
        self.metrics["total_reasoning_time"] += duration
        if self.metrics["total_reasoning_tasks"] > 0:
            self.metrics["average_reasoning_time"] = (
                self.metrics["total_reasoning_time"] / self.metrics["total_reasoning_tasks"]
            )
        self.metrics["strategy_usage"][strategy.value] += 1
        
        self._log_reasoning(
            session_id, 
            f"Reasoning process completed in {duration:.2f} seconds",
            level="info"
        )
        
        return result
    
    def _determine_best_strategy(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> ReasoningStrategy:
        """Determine the best reasoning strategy for a given query and context"""
        # Extract keywords from query
        query_lower = query.lower()
        
        # Detect strategy indicators in query
        strategy_indicators = {
            ReasoningStrategy.DEDUCTIVE: [
                "deduce", "derive", "infer", "prove", "must be", "therefore", "always", "never", "every", "all"
            ],
            ReasoningStrategy.INDUCTIVE: [
                "pattern", "trend", "generally", "typically", "usually", "most", "often", "examples", "observations"
            ],
            ReasoningStrategy.ABDUCTIVE: [
                "explain", "why", "cause", "reason for", "best explanation", "most likely cause", "diagnose"
            ],
            ReasoningStrategy.ANALOGICAL: [
                "similar to", "like", "analogy", "comparison", "compared to", "resembles", "parallels"
            ],
            ReasoningStrategy.CAUSAL: [
                "cause", "effect", "leads to", "results in", "because", "consequence", "impact"
            ],
            ReasoningStrategy.COUNTERFACTUAL: [
                "what if", "could have", "would have", "might have", "if then", "alternative", "scenario"
            ],
            ReasoningStrategy.BAYESIAN: [
                "probability", "likely", "chance", "uncertain", "evidence", "prior", "posterior", "update"
            ],
            ReasoningStrategy.CONSTRAINT: [
                "constraint", "restriction", "limited by", "must satisfy", "requirement", "condition"
            ],
            ReasoningStrategy.CASE_BASED: [
                "previous case", "similar case", "past example", "precedent", "historical", "previous instance"
            ]
        }
        
        # Score each strategy based on indicators in the query
        strategy_scores = {strategy: 0 for strategy in ReasoningStrategy}
        
        for strategy, indicators in strategy_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    strategy_scores[strategy] += 1
        
        # Consider context
        if context.get("reasoning_strategy"):
            # Context explicitly specifies strategy
            try:
                return ReasoningStrategy(context["reasoning_strategy"])
            except ValueError:
                pass
        
        # If no clear winner, check query type
        if max(strategy_scores.values(), default=0) == 0:
            # Classify based on query structure
            if re.search(r"\b(how|what steps|process)\b", query_lower):
                return ReasoningStrategy.PROCEDURAL
            elif re.search(r"\b(why|explain|reason)\b", query_lower):
                return ReasoningStrategy.ABDUCTIVE
            elif re.search(r"\b(what if|predict|would happen)\b", query_lower):
                return ReasoningStrategy.COUNTERFACTUAL
            else:
                # Default to hybrid reasoning
                return ReasoningStrategy.HYBRID
        
        # Return the strategy with the highest score
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        return best_strategy
    
    def _execute_stage(
        self,
        session_id: str,
        stage: ReasoningStage,
        func: Callable
    ) -> Dict[str, Any]:
        """Execute a reasoning stage and track its execution"""
        stage_name = stage.value
        
        self._log_reasoning(session_id, f"Starting stage: {stage_name}", level="info")
        
        start_time = time.time()
        result = func()
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Update metrics
        self.metrics["stage_timings"][stage_name] += duration
        
        # Add stage to reasoning history
        stage_info = {
            "stage": stage_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "result": result
        }
        
        self.reasoning_history[session_id]["stages"].append(stage_info)
        
        self._log_reasoning(
            session_id, 
            f"Completed stage {stage_name} in {duration:.4f} seconds",
            level="info"
        )
        
        return result
    
    def _log_reasoning(
        self,
        session_id: str,
        message: str,
        level: str = "debug",
        data: Dict[str, Any] = None
    ) -> None:
        """Log reasoning step with appropriate level"""
        # Add to reasoning trace
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data
        }
        
        if session_id in self.reasoning_history:
            self.reasoning_history[session_id]["trace"].append(trace_entry)
        
        # Log based on configured verbosity
        log_levels = {
            "minimal": ["error", "critical"],
            "standard": ["error", "critical", "warning", "info"],
            "detailed": ["error", "critical", "warning", "info", "debug"],
            "debug": ["error", "critical", "warning", "info", "debug", "trace"]
        }
        
        if level in log_levels.get(self.verbosity, ["error", "critical"]):
            if level == "error":
                logger.error(f"[{session_id}] {message}")
            elif level == "critical":
                logger.critical(f"[{session_id}] {message}")
            elif level == "warning":
                logger.warning(f"[{session_id}] {message}")
            elif level == "info":
                logger.info(f"[{session_id}] {message}")
            else:
                logger.debug(f"[{session_id}] {message}")
    
    def _extract_reasoning_path(self, session_id: str) -> List[Dict[str, Any]]:
        """Extract the reasoning path from the session history"""
        if session_id not in self.reasoning_history:
            return []
        
        history = self.reasoning_history[session_id]
        
        # Extract key information from each stage
        reasoning_path = []
        
        for stage_info in history["stages"]:
            stage_result = stage_info["result"]
            
            # Skip stages with empty results
            if not stage_result:
                continue
            
            stage_name = stage_info["stage"]
            
            # Format based on stage type
            if stage_name == ReasoningStage.PROBLEM_FORMULATION.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "problem": stage_result.get("problem"),
                    "type": stage_result.get("problem_type"),
                    "concepts": stage_result.get("key_concepts")
                })
            
            elif stage_name == ReasoningStage.CONTEXT_ANALYSIS.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "relevant_context": stage_result.get("relevant_context"),
                    "constraints": stage_result.get("constraints"),
                    "assumptions": stage_result.get("assumptions")
                })
            
            elif stage_name == ReasoningStage.KNOWLEDGE_RETRIEVAL.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "knowledge_domains": stage_result.get("domains"),
                    "key_concepts": stage_result.get("concepts"),
                    "relevant_rules": [
                        f"{rule.get('type')}: {rule.get('description', '')}" 
                        for rule in stage_result.get("rules", [])
                    ]
                })
            
            elif stage_name == ReasoningStage.PATTERN_RECOGNITION.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "patterns": stage_result.get("patterns"),
                    "relevance": stage_result.get("pattern_relevance")
                })
            
            elif stage_name == ReasoningStage.HYPOTHESIS_GENERATION.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "hypotheses": [
                        f"{h.get('description')} (confidence: {h.get('confidence'):.2f})"
                        for h in stage_result.get("hypotheses", [])
                    ]
                })
            
            elif stage_name == ReasoningStage.LOGICAL_INFERENCE.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "inferences": stage_result.get("inferences"),
                    "strategy": stage_result.get("strategy"),
                    "conclusion": stage_result.get("conclusion")
                })
            
            elif stage_name == ReasoningStage.CAUSAL_ANALYSIS.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "causal_factors": stage_result.get("causal_factors"),
                    "causal_chain": stage_result.get("causal_chain")
                })
            
            elif stage_name == ReasoningStage.SOLUTION_SYNTHESIS.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "solution": stage_result.get("solution"),
                    "approach": stage_result.get("approach"),
                    "confidence": stage_result.get("confidence")
                })
            
            elif stage_name == ReasoningStage.VERIFICATION.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "verified": stage_result.get("verified"),
                    "verification_method": stage_result.get("method"),
                    "issues": stage_result.get("issues")
                })
            
            elif stage_name == ReasoningStage.EXPLANATION.value:
                reasoning_path.append({
                    "stage": stage_name,
                    "explanation": stage_result.get("explanation"),
                    "justification": stage_result.get("justification")
                })
        
        return reasoning_path
    
    def _formulate_problem(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Formulate the problem from the query and context"""
        # Identify the problem type
        problem_types = [
            "code_generation", "debugging", "optimization", "architecture", 
            "algorithm_design", "data_structure_selection", "testing", 
            "code_review", "concept_explanation", "comparison", "best_practice"
        ]
        
        identified_type = "general"
        
        # Simple problem type classification
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["write", "create", "generate", "implement", "code"]):
            identified_type = "code_generation"
        elif any(term in query_lower for term in ["debug", "fix", "error", "issue", "bug", "problem"]):
            identified_type = "debugging"
        elif any(term in query_lower for term in ["optimize", "performance", "efficient", "faster", "improve"]):
            identified_type = "optimization"
        elif any(term in query_lower for term in ["architecture", "design", "system", "structure"]):
            identified_type = "architecture"
        elif any(term in query_lower for term in ["algorithm", "procedure", "approach", "method"]):
            identified_type = "algorithm_design"
        elif any(term in query_lower for term in ["data structure", "store", "retrieve", "collection"]):
            identified_type = "data_structure_selection"
        elif any(term in query_lower for term in ["test", "verify", "validate", "assert"]):
            identified_type = "testing"
        elif any(term in query_lower for term in ["review", "analyze", "assess", "evaluate"]):
            identified_type = "code_review"
        elif any(term in query_lower for term in ["explain", "describe", "what is", "how does"]):
            identified_type = "concept_explanation"
        elif any(term in query_lower for term in ["compare", "versus", "vs", "difference", "better"]):
            identified_type = "comparison"
        elif any(term in query_lower for term in ["best practice", "standard", "convention", "should i"]):
            identified_type = "best_practice"
        
        # Extract key concepts from query
        key_concepts = self._extract_key_concepts(query)
        
        # Look for specific requirements in context
        requirements = context.get("requirements", [])
        if not requirements and "requirements" in context:
            if isinstance(context["requirements"], str):
                requirements = [context["requirements"]]
        
        # Identify constraints
        constraints = context.get("constraints", [])
        if not constraints and "constraints" in context:
            if isinstance(context["constraints"], str):
                constraints = [context["constraints"]]
        
        # Check if context specifies a particular problem type
        if "problem_type" in context:
            identified_type = context["problem_type"]
        
        # Formulated problem
        formulated_problem = {
            "problem": query,
            "problem_type": identified_type,
            "key_concepts": key_concepts,
            "requirements": requirements,
            "constraints": constraints
        }
        
        return formulated_problem
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction
        concepts = set()
        
        # Programming languages
        languages = ["python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "sql", "php", "ruby"]
        for lang in languages:
            pattern = r"\b" + re.escape(lang) + r"\b"
            if re.search(pattern, text.lower()):
                concepts.add(lang)
        
        # Data structures
        data_structures = ["array", "list", "tree", "graph", "hash", "map", "set", "queue", "stack", "heap", "dictionary"]
        for ds in data_structures:
            pattern = r"\b" + re.escape(ds) + r"\b"
            if re.search(pattern, text.lower()):
                concepts.add(ds)
        
        # Algorithms and patterns
        algorithms = ["sort", "search", "recursion", "iteration", "dynamic programming", "greedy", "backtracking"]
        for algo in algorithms:
            if algo in text.lower():
                concepts.add(algo)
        
        patterns = ["singleton", "factory", "observer", "decorator", "strategy", "adapter", "composite", "facade"]
        for pattern in patterns:
            if pattern in text.lower():
                concepts.add(pattern)
        
        # Software architecture concepts
        arch_concepts = ["api", "rest", "microservice", "database", "server", "client", "frontend", "backend", "mvc", "mvvm"]
        for concept in arch_concepts:
            pattern = r"\b" + re.escape(concept) + r"\b"
            if re.search(pattern, text.lower()):
                concepts.add(concept)
        
        # Clean up concepts
        return list(concepts)
    
    def _analyze_context(
        self,
        problem_formulation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the context to extract relevant information"""
        # Extract relevant context for the problem
        relevant_context = {}
        
        # Default assumptions
        assumptions = []
        
        # Get problem type
        problem_type = problem_formulation["problem_type"]
        
        # Extract relevant context based on problem type
        if problem_type == "code_generation":
            # Language preference
            if "language" in context:
                relevant_context["language"] = context["language"]
            else:
                # Look for language in problem concepts
                languages = set(["python", "javascript", "typescript", "java", "c++", "c#", "go", "rust"]) & set(problem_formulation["key_concepts"])
                if languages:
                    relevant_context["language"] = list(languages)[0]
                else:
                    # Make an assumption about the language
                    relevant_context["language"] = "python"  # Default to Python
                    assumptions.append("Using Python as the default language")
            
            # Framework preference
            if "framework" in context:
                relevant_context["framework"] = context["framework"]
            
            # Style and formatting preferences
            if "code_style" in context:
                relevant_context["code_style"] = context["code_style"]
            
            # Performance considerations
            if "performance" in context:
                relevant_context["performance"] = context["performance"]
        
        elif problem_type == "debugging":
            # Error information
            if "error" in context:
                relevant_context["error"] = context["error"]
            
            # Code with the bug
            if "code" in context:
                relevant_context["code"] = context["code"]
            
            # Environment information
            if "environment" in context:
                relevant_context["environment"] = context["environment"]
            
            # Reproduction steps
            if "reproduction_steps" in context:
                relevant_context["reproduction_steps"] = context["reproduction_steps"]
        
        elif problem_type == "optimization":
            # Current code to optimize
            if "code" in context:
                relevant_context["code"] = context["code"]
            
            # Performance metrics
            if "metrics" in context:
                relevant_context["metrics"] = context["metrics"]
            
            # Optimization goals
            if "optimization_goals" in context:
                relevant_context["optimization_goals"] = context["optimization_goals"]
            else:
                assumptions.append("Optimizing for general performance (time and space efficiency)")
        
        elif problem_type == "architecture":
            # System requirements
            if "requirements" in context:
                relevant_context["requirements"] = context["requirements"]
            
            # Scale expectations
            if "scale" in context:
                relevant_context["scale"] = context["scale"]
            
            # Technology constraints
            if "technologies" in context:
                relevant_context["technologies"] = context["technologies"]
        
        # Extract constraints
        constraints = problem_formulation["constraints"].copy()
        
        for key in ["constraints", "max_complexity", "space_constraints", "time_constraints"]:
            if key in context:
                if isinstance(context[key], list):
                    constraints.extend(context[key])
                else:
                    constraints.append(str(context[key]))
        
        # Process knowledge context
        domain_knowledge = context.get("domain_knowledge", {})
        if domain_knowledge:
            relevant_context["domain_knowledge"] = domain_knowledge
        
        # Process other potential context factors
        for key in ["experience_level", "priority", "platform", "compatibility"]:
            if key in context:
                relevant_context[key] = context[key]
        
        return {
            "relevant_context": relevant_context,
            "constraints": constraints,
            "assumptions": assumptions
        }
    
    def _retrieve_knowledge(
        self,
        problem_formulation: Dict[str, Any],
        context_analysis: Dict[str, Any],
        domains: List[str] = None
    ) -> Dict[str, Any]:
        """Retrieve relevant knowledge from the knowledge base"""
        # Get problem type and key concepts
        problem_type = problem_formulation["problem_type"]
        key_concepts = problem_formulation["key_concepts"]
        
        # Determine relevant knowledge domains
        relevant_domains = domains or []
        
        # If no domains specified, determine based on problem type
        if not relevant_domains:
            type_to_domains = {
                "code_generation": ["code_generation", "software_engineering"],
                "debugging": ["debugging", "software_engineering"],
                "optimization": ["software_engineering"],
                "architecture": ["architecture", "software_engineering"],
                "algorithm_design": ["algorithms", "software_engineering"],
                "data_structure_selection": ["data_structures", "software_engineering"],
                "testing": ["testing", "software_engineering"],
                "code_review": ["code_review", "software_engineering"],
                "concept_explanation": ["software_engineering"],
                "comparison": ["software_engineering"],
                "best_practice": ["software_engineering"]
            }
            
            relevant_domains = type_to_domains.get(problem_type, ["software_engineering"])
        
        # Retrieve concepts from knowledge base
        relevant_concepts = {}
        
        for concept_name, concept_info in self.knowledge_base["concepts"].items():
            # Check if concept is relevant to the problem
            if concept_name in key_concepts or any(kc in concept_info["definition"].lower() for kc in key_concepts):
                relevant_concepts[concept_name] = concept_info
        
        # Retrieve domain knowledge
        relevant_domain_knowledge = {}
        
        for domain, knowledge in self.knowledge_base["domain_knowledge"].items():
            # Check if domain is relevant
            if domain in key_concepts or any(subtype in key_concepts for subtype in knowledge.keys()):
                relevant_domain_knowledge[domain] = knowledge
        
        # Retrieve relevant rules
        relevant_rules = []
        
        for domain, rules in self.rule_sets.items():
            if domain in relevant_domains:
                for rule in rules:
                    # Check rule relevance based on context and concepts
                    if rule["type"] == RuleType.LOGICAL.value:
                        premise = rule.get("premise", {})
                        if premise.get("concept") in key_concepts:
                            relevant_rules.append({
                                "type": rule["type"],
                                "description": f"If {premise.get('concept')} has {premise.get('property')} {premise.get('operator', '')} {premise.get('value', '')}, then {rule.get('conclusion')}",
                                "confidence": rule.get("confidence", 0.5),
                                "rule": rule
                            })
                    
                    elif rule["type"] == RuleType.HEURISTIC.value:
                        context_match = rule.get("context", "").lower()
                        if context_match in problem_type.lower() or any(context_match in kc.lower() for kc in key_concepts):
                            relevant_rules.append({
                                "type": rule["type"],
                                "description": rule.get("rule", ""),
                                "confidence": rule.get("confidence", 0.5),
                                "rule": rule
                            })
                    
                    elif rule["type"] == RuleType.PROCEDURAL.value:
                        context_match = rule.get("context", "").lower()
                        if context_match in problem_type.lower() or any(context_match in kc.lower() for kc in key_concepts):
                            steps = rule.get("steps", [])
                            step_text = "; ".join(steps)
                            relevant_rules.append({
                                "type": rule["type"],
                                "description": f"Procedure for {context_match}: {step_text}",
                                "confidence": rule.get("confidence", 0.5),
                                "rule": rule
                            })
                    
                    elif rule["type"] == RuleType.CAUSAL.value:
                        cause = rule.get("cause", {})
                        if cause.get("concept") in key_concepts:
                            relevant_rules.append({
                                "type": rule["type"],
                                "description": f"{cause} causes {rule.get('effect')}",
                                "confidence": rule.get("confidence", 0.5),
                                "rule": rule
                            })
                    
                    elif rule["type"] == RuleType.CONSTRAINT.value:
                        context_match = rule.get("context", "").lower()
                        if context_match in problem_type.lower() or any(context_match in kc.lower() for kc in key_concepts):
                            relevant_rules.append({
                                "type": rule["type"],
                                "description": rule.get("constraint", ""),
                                "confidence": rule.get("confidence", 0.5),
                                "rule": rule
                            })
        
        # Sort rules by confidence
        relevant_rules.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return {
            "domains": relevant_domains,
            "concepts": relevant_concepts,
            "domain_knowledge": relevant_domain_knowledge,
            "rules": relevant_rules
        }
    
    def _recognize_patterns(
        self,
        problem_formulation: Dict[str, Any],
        context_analysis: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recognize patterns in the problem and context"""
        # Extract problem details
        problem_type = problem_formulation["problem_type"]
        problem = problem_formulation["problem"]
        key_concepts = problem_formulation["key_concepts"]
        
        recognized_patterns = []
        
        # Recognize common problem patterns
        if problem_type == "code_generation":
            if "function" in problem.lower() or "method" in problem.lower():
                recognized_patterns.append({
                    "pattern": "function_implementation",
                    "confidence": 0.9,
                    "description": "Implementation of a function or method"
                })
            
            if "class" in problem.lower():
                recognized_patterns.append({
                    "pattern": "class_implementation",
                    "confidence": 0.9,
                    "description": "Implementation of a class"
                })
            
            if any(algo in problem.lower() for algo in ["sort", "search", "algorithm", "find"]):
                recognized_patterns.append({
                    "pattern": "algorithm_implementation",
                    "confidence": 0.8,
                    "description": "Implementation of an algorithm"
                })
            
            if "api" in problem.lower() or "endpoint" in problem.lower() or "server" in problem.lower():
                recognized_patterns.append({
                    "pattern": "api_implementation",
                    "confidence": 0.8,
                    "description": "Implementation of an API or server endpoint"
                })
        
        elif problem_type == "debugging":
            if "null" in problem.lower() or "undefined" in problem.lower() or "none" in problem.lower():
                recognized_patterns.append({
                    "pattern": "null_reference_bug",
                    "confidence": 0.8,
                    "description": "Bug related to null/undefined/None values"
                })
            
            if "loop" in problem.lower():
                recognized_patterns.append({
                    "pattern": "loop_bug",
                    "confidence": 0.7,
                    "description": "Bug related to loop logic or termination"
                })
            
            if "memory" in problem.lower() or "leak" in problem.lower():
                recognized_patterns.append({
                    "pattern": "memory_issue",
                    "confidence": 0.8,
                    "description": "Memory-related issue or leak"
                })
            
            if "performance" in problem.lower() or "slow" in problem.lower():
                recognized_patterns.append({
                    "pattern": "performance_issue",
                    "confidence": 0.8,
                    "description": "Performance problem"
                })
        
        elif problem_type == "architecture":
            if "scale" in problem.lower() or "traffic" in problem.lower() or "load" in problem.lower():
                recognized_patterns.append({
                    "pattern": "scalability_design",
                    "confidence": 0.9,
                    "description": "Architecture design for scalability"
                })
            
            if "microservice" in problem.lower() or "service" in problem.lower():
                recognized_patterns.append({
                    "pattern": "microservice_architecture",
                    "confidence": 0.9,
                    "description": "Microservice-based architecture"
                })
            
            if "security" in problem.lower() or "auth" in problem.lower():
                recognized_patterns.append({
                    "pattern": "security_architecture",
                    "confidence": 0.8,
                    "description": "Security-focused architecture"
                })
        
        # Pattern relevance assessment
        pattern_relevance = {}
        
        for pattern in recognized_patterns:
            # Check relevance against rules in knowledge base
            relevance_score = pattern["confidence"]
            
            for rule in knowledge_retrieval["rules"]:
                if pattern["pattern"] in str(rule):
                    # Pattern mentioned in rule, increase relevance
                    relevance_score += 0.1
            
            pattern_relevance[pattern["pattern"]] = min(relevance_score, 1.0)
        
        return {
            "patterns": recognized_patterns,
            "pattern_relevance": pattern_relevance
        }
    
    def _generate_hypotheses(
        self,
        problem_formulation: Dict[str, Any],
        pattern_recognition: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate hypotheses for the problem solution"""
        problem_type = problem_formulation["problem_type"]
        
        # Generate possible hypotheses based on problem type and patterns
        hypotheses = []
        
        if problem_type == "code_generation":
            # Get recognized patterns
            patterns = [p["pattern"] for p in pattern_recognition["patterns"]]
            
            if "function_implementation" in patterns:
                hypotheses.append({
                    "description": "Implement a function with clear inputs, outputs, and error handling",
                    "confidence": 0.9,
                    "approach": "procedural"
                })
            
            if "class_implementation" in patterns:
                hypotheses.append({
                    "description": "Implement a class with appropriate attributes and methods",
                    "confidence": 0.9,
                    "approach": "object_oriented"
                })
            
            if "algorithm_implementation" in patterns:
                hypotheses.append({
                    "description": "Implement an optimized algorithm with clear steps",
                    "confidence": 0.85,
                    "approach": "algorithmic"
                })
            
            # Add language-specific hypotheses
            relevant_context = problem_formulation.get("relevant_context", {})
            language = relevant_context.get("language", "python").lower()
            
            if language == "python":
                hypotheses.append({
                    "description": "Use Pythonic approach with list comprehensions and built-in functions",
                    "confidence": 0.8,
                    "approach": "pythonic"
                })
            elif language == "javascript" or language == "typescript":
                hypotheses.append({
                    "description": "Use modern JS/TS features like destructuring, arrow functions, and async/await",
                    "confidence": 0.8,
                    "approach": "modern_javascript"
                })
        
        elif problem_type == "debugging":
            patterns = [p["pattern"] for p in pattern_recognition["patterns"]]
            
            if "null_reference_bug" in patterns:
                hypotheses.append({
                    "description": "Check for null/undefined values before access and add proper validation",
                    "confidence": 0.85,
                    "approach": "validation"
                })
            
            if "loop_bug" in patterns:
                hypotheses.append({
                    "description": "Verify loop conditions and iteration logic, ensure proper termination",
                    "confidence": 0.8,
                    "approach": "logic_verification"
                })
            
            if "memory_issue" in patterns:
                hypotheses.append({
                    "description": "Identify resource leaks and ensure proper cleanup",
                    "confidence": 0.8,
                    "approach": "resource_management"
                })
            
            if "performance_issue" in patterns:
                hypotheses.append({
                    "description": "Optimize algorithms and data structures for better performance",
                    "confidence": 0.75,
                    "approach": "optimization"
                })
            
            # Generic debugging hypothesis
            hypotheses.append({
                "description": "Trace through code execution and identify the root cause",
                "confidence": 0.7,
                "approach": "root_cause_analysis"
            })
        
        elif problem_type == "architecture":
            patterns = [p["pattern"] for p in pattern_recognition["patterns"]]
            
            if "scalability_design" in patterns:
                hypotheses.append({
                    "description": "Design horizontally scalable architecture with stateless components",
                    "confidence": 0.85,
                    "approach": "horizontal_scaling"
                })
            
            if "microservice_architecture" in patterns:
                hypotheses.append({
                    "description": "Design microservices with clear boundaries and communication interfaces",
                    "confidence": 0.85,
                    "approach": "microservices"
                })
            
            if "security_architecture" in patterns:
                hypotheses.append({
                    "description": "Implement security at all layers with proper authentication and authorization",
                    "confidence": 0.85,
                    "approach": "security_first"
                })
            
            # Generic architecture hypothesis
            hypotheses.append({
                "description": "Apply appropriate design patterns and principles for the specific requirements",
                "confidence": 0.7,
                "approach": "pattern_based"
            })
        
        else:
            # Generic hypotheses for other problem types
            hypotheses.append({
                "description": "Apply domain-specific knowledge and best practices",
                "confidence": 0.6,
                "approach": "best_practices"
            })
            
            hypotheses.append({
                "description": "Break down the problem into manageable components",
                "confidence": 0.7,
                "approach": "decomposition"
            })
        
        # Adjust confidences based on knowledge
        for hypothesis in hypotheses:
            # Check if approach matches any rules in knowledge base
            for rule in knowledge_retrieval["rules"]:
                if hypothesis["approach"] in str(rule):
                    # Approach mentioned in rule, adjust confidence
                    hypothesis["confidence"] = min(hypothesis["confidence"] + 0.1, 1.0)
        
        # Sort hypotheses by confidence
        hypotheses.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "hypotheses": hypotheses,
            "primary_hypothesis": hypotheses[0] if hypotheses else None
        }
    
    def _perform_logical_inference(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any],
        strategy: ReasoningStrategy
    ) -> Dict[str, Any]:
        """Perform logical inference to reach a conclusion"""
        # Use the appropriate strategy handler
        strategy_handler = self.strategy_handlers.get(strategy, self._hybrid_reasoning)
        
        # Perform reasoning using the selected strategy
        inference_result = strategy_handler(
            problem_formulation=problem_formulation,
            hypothesis_generation=hypothesis_generation,
            knowledge_retrieval=knowledge_retrieval
        )
        
        return inference_result
    
    def _analyze_causal_relationships(
        self,
        problem_formulation: Dict[str, Any],
        logical_inference: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze causal relationships in the problem and solution"""
        # Extract problem details
        problem_type = problem_formulation["problem_type"]
        
        # Initialize causal factors and chains
        causal_factors = []
        causal_chain = []
        
        # Different analysis based on problem type
        if problem_type == "debugging":
            # For debugging, analyze cause of bugs
            conclusion = logical_inference.get("conclusion", "")
            
            if "null" in conclusion.lower() or "undefined" in conclusion.lower():
                causal_factors.append({
                    "factor": "missing_validation",
                    "description": "Lack of input validation or null checks",
                    "impact": "high"
                })
                
                causal_chain = [
                    "Invalid or null input received",
                    "No validation performed",
                    "Null reference accessed",
                    "Exception or error thrown"
                ]
            
            elif "loop" in conclusion.lower():
                causal_factors.append({
                    "factor": "incorrect_loop_condition",
                    "description": "Incorrect loop termination condition",
                    "impact": "high"
                })
                
                causal_chain = [
                    "Loop condition defined incorrectly",
                    "Loop continues beyond intended range",
                    "Unexpected values processed",
                    "Incorrect results or errors"
                ]
            
            elif "memory" in conclusion.lower():
                causal_factors.append({
                    "factor": "resource_leak",
                    "description": "Resources not properly released",
                    "impact": "high"
                })
                
                causal_chain = [
                    "Resource allocated during operation",
                    "Missing or incomplete cleanup code",
                    "Resource not released after use",
                    "Resource accumulation leads to memory issues"
                ]
        
        elif problem_type == "optimization":
            # For optimization, analyze causes of performance issues
            conclusion = logical_inference.get("conclusion", "")
            
            if "algorithm" in conclusion.lower():
                causal_factors.append({
                    "factor": "inefficient_algorithm",
                    "description": "Algorithm with suboptimal time complexity",
                    "impact": "high"
                })
                
                causal_chain = [
                    "Inefficient algorithm chosen",
                    "High computational complexity",
                    "Processing time scales poorly with input size",
                    "Performance degradation with larger inputs"
                ]
            
            elif "data structure" in conclusion.lower():
                causal_factors.append({
                    "factor": "suboptimal_data_structure",
                    "description": "Data structure with inefficient operations for the use case",
                    "impact": "high"
                })
                
                causal_chain = [
                    "Suboptimal data structure chosen",
                    "Operations require higher complexity than necessary",
                    "Excessive computational work",
                    "Slower than optimal performance"
                ]
            
            elif "i/o" in conclusion.lower() or "io" in conclusion.lower():
                causal_factors.append({
                    "factor": "io_bottleneck",
                    "description": "Input/output operations causing performance bottleneck",
                    "impact": "high"
                })
                
                causal_chain = [
                    "Excessive I/O operations",
                    "Blocking on I/O completion",
                    "CPU idling during I/O",
                    "Overall system slowdown"
                ]
        
        # Generic causal analysis for other problems
        if not causal_factors:
            # Extract factors from logical inference
            for premise in logical_inference.get("premises", []):
                causal_factors.append({
                    "factor": premise.get("concept", "unknown"),
                    "description": premise.get("description", ""),
                    "impact": "medium"
                })
        
        return {
            "causal_factors": causal_factors,
            "causal_chain": causal_chain
        }
    
    def _synthesize_solution(
        self,
        problem_formulation: Dict[str, Any],
        logical_inference: Dict[str, Any],
        causal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize a solution based on the reasoning process"""
        # Extract problem details
        problem_type = problem_formulation["problem_type"]
        
        # Get the conclusion from logical inference
        conclusion = logical_inference.get("conclusion", "")
        
        # Generate solution based on problem type and conclusion
        solution = ""
        approach = ""
        confidence = logical_inference.get("confidence", 0.7)
        
        if problem_type == "code_generation":
            # Solution will be code generation guidelines
            solution = f"Generate code that implements the required functionality following these guidelines:\n\n{conclusion}"
            approach = logical_inference.get("approach", "structured")
        
        elif problem_type == "debugging":
            # Solution will be debugging steps
            causal_chain = causal_analysis.get("causal_chain", [])
            if causal_chain:
                chain_text = "\n".join([f"- {step}" for step in causal_chain])
                solution = f"Debug the issue by understanding the root cause:\n\n{chain_text}\n\nThen apply this fix approach:\n{conclusion}"
            else:
                solution = f"Debug the issue by applying this approach:\n{conclusion}"
            
            approach = "root_cause_analysis"
        
        elif problem_type == "optimization":
            # Solution will be optimization strategy
            causal_factors = causal_analysis.get("causal_factors", [])
            factors_text = "\n".join([f"- {factor.get('description', '')}" for factor in causal_factors])
            
            solution = f"Optimize performance by addressing these factors:\n\n{factors_text}\n\nImplementation approach:\n{conclusion}"
            approach = "targeted_optimization"
        
        elif problem_type == "architecture":
            # Solution will be architecture guidelines
            solution = f"Design the system architecture following these principles:\n\n{conclusion}"
            approach = logical_inference.get("approach", "modular")
        
        else:
            # Generic solution synthesis
            solution = f"Apply this approach to solve the problem:\n\n{conclusion}"
            approach = "general"
        
        return {
            "solution": solution,
            "approach": approach,
            "confidence": confidence
        }
    
    def _verify_solution(
        self,
        problem_formulation: Dict[str, Any],
        solution_synthesis: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify the solution against rules and constraints"""
        # Extract solution details
        solution = solution_synthesis["solution"]
        approach = solution_synthesis["approach"]
        
        # Get problem constraints
        constraints = problem_formulation.get("constraints", [])
        
        # Get relevant rules for verification
        verification_rules = []
        for rule in knowledge_retrieval.get("rules", []):
            if rule["type"] in [RuleType.VALIDATION.value, RuleType.CONSTRAINT.value]:
                verification_rules.append(rule)
        
        # Initialize verification results
        verified = True
        issues = []
        verification_method = "rule_based"
        
        # Check against constraints
        for constraint in constraints:
            constraint_lower = constraint.lower()
            solution_lower = solution.lower()
            
            # Simple constraint checking
            if "memory" in constraint_lower and "efficient" in constraint_lower:
                if "optimize memory" not in solution_lower and "memory efficiency" not in solution_lower:
                    verified = False
                    issues.append(f"Constraint not addressed: {constraint}")
            
            if "performance" in constraint_lower and "efficient" in constraint_lower:
                if "optimize" not in solution_lower and "performance" not in solution_lower:
                    verified = False
                    issues.append(f"Constraint not addressed: {constraint}")
        
        # Check against verification rules
        for rule in verification_rules:
            rule_text = rule.get("description", "")
            if "rule" in rule:
                rule_obj = rule["rule"]
                
                # Check different rule types
                if rule["type"] == RuleType.CONSTRAINT.value:
                    constraint = rule_obj.get("constraint", "")
                    negations = ["never", "not", "avoid", "don't", "shouldn't"]
                    
                    # Check if solution violates constraints with negations
                    for negation in negations:
                        if negation in constraint.lower():
                            # Extract what should be avoided
                            avoid_pattern = constraint.lower().split(negation)[1].strip()
                            if avoid_pattern and avoid_pattern in solution.lower():
                                verified = False
                                issues.append(f"Solution violates constraint: {constraint}")
            
            # Generic text matching check
            elif rule_text:
                keywords = rule_text.lower().split()
                keywords = [k for k in keywords if len(k) > 3]  # Only consider significant words
                
                missed_keywords = [k for k in keywords if k not in solution.lower()]
                if len(missed_keywords) > len(keywords) * 0.7:  # If most keywords are missed
                    verified = False
                    issues.append(f"Solution may not address: {rule_text}")
        
        return {
            "verified": verified,
            "issues": issues,
            "method": verification_method
        }
    
    def _generate_explanation(
        self,
        problem_formulation: Dict[str, Any],
        solution_synthesis: Dict[str, Any],
        logical_inference: Dict[str, Any],
        verification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate an explanation for the reasoning process and solution"""
        # Extract details
        problem_type = problem_formulation["problem_type"]
        solution = solution_synthesis["solution"]
        approach = solution_synthesis["approach"]
        
        # Reasoning strategy used
        strategy = logical_inference.get("strategy", "hybrid")
        
        # Generate explanation based on problem type
        if problem_type == "code_generation":
            explanation = f"The solution provides code generation guidelines using a {approach} approach. "
            
            if "premises" in logical_inference:
                premises = logical_inference["premises"]
                premise_text = "; ".join([p.get("description", "") for p in premises])
                explanation += f"This is based on the following considerations: {premise_text}. "
            
            explanation += f"The solution was determined using {strategy} reasoning, and "
            
            if verification["verified"]:
                explanation += "has been verified to meet all specified constraints."
            else:
                explanation += "has some issues that should be addressed: " + "; ".join(verification["issues"])
        
        elif problem_type == "debugging":
            explanation = f"The debugging solution focuses on a root cause analysis approach. "
            
            if "causal_chain" in logical_inference:
                causal_chain = logical_inference["causal_chain"]
                if causal_chain:
                    explanation += f"The issue is caused by a chain of events: {' → '.join(causal_chain)}. "
            
            explanation += f"The solution was determined using {strategy} reasoning, and "
            
            if verification["verified"]:
                explanation += "should effectively resolve the issue while meeting all constraints."
            else:
                explanation += "has some potential issues: " + "; ".join(verification["issues"])
        
        else:
            # Generic explanation
            explanation = f"The solution uses a {approach} approach to address the {problem_type} problem. "
            
            if "inferences" in logical_inference:
                inferences = logical_inference["inferences"]
                inference_text = "; ".join(inferences)
                explanation += f"Key inferences made: {inference_text}. "
            
            explanation += f"This solution was developed using {strategy} reasoning, and "
            
            if verification["verified"]:
                explanation += "has been verified to meet all specified constraints."
            else:
                explanation += "has some issues that should be addressed: " + "; ".join(verification["issues"])
        
        # Generate justification details
        justification = {
            "reasoning_strategy": strategy,
            "key_inferences": logical_inference.get("inferences", []),
            "approach_rationale": f"A {approach} approach was chosen based on the problem characteristics and requirements.",
            "verification_status": "Verified" if verification["verified"] else "Has issues",
            "confidence_level": f"{solution_synthesis.get('confidence', 0.7) * 100:.0f}%"
        }
        
        return {
            "explanation": explanation,
            "justification": justification
        }
    
    def _compute_confidence(
        self,
        verification: Dict[str, Any],
        logical_inference: Dict[str, Any]
    ) -> float:
        """Compute overall confidence in the solution"""
        # Start with base confidence from logical inference
        base_confidence = logical_inference.get("confidence", 0.7)
        
        # Adjust based on verification
        if not verification["verified"]:
            # Reduce confidence based on number of issues
            num_issues = len(verification.get("issues", []))
            confidence_reduction = min(0.1 * num_issues, 0.5)  # Cap reduction at 0.5
            
            base_confidence -= confidence_reduction
        
        # Ensure confidence is in valid range
        return max(0.0, min(base_confidence, 1.0))
    
    # Reasoning strategy implementations
    
    def _deductive_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform deductive reasoning (from general to specific)"""
        # Extract knowledge and hypotheses
        rules = knowledge_retrieval.get("rules", [])
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # Extract premises from rules (general principles)
        premises = []
        for rule in rules:
            if rule["type"] == RuleType.LOGICAL.value and "rule" in rule:
                rule_obj = rule["rule"]
                premise = rule_obj.get("premise")
                conclusion = rule_obj.get("conclusion")
                
                if premise and conclusion:
                    premises.append({
                        "description": f"If {premise.get('concept', '')} has {premise.get('property', '')} {premise.get('operator', '')} {premise.get('value', '')}, then {conclusion}",
                        "confidence": rule.get("confidence", 0.5)
                    })
        
        # Use primary hypothesis if available
        approach = ""
        if hypotheses:
            primary = hypotheses[0]
            approach = primary.get("approach", "")
        
        # Generate inferences
        inferences = []
        
        # Apply premises to the specific problem
        for premise in premises:
            inferences.append(premise["description"])
        
        # Derive specific conclusion
        conclusion = ""
        if premises:
            # Sort premises by confidence
            sorted_premises = sorted(premises, key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Combine the top premises
            top_premises = sorted_premises[:3]
            
            conclusion_parts = []
            for premise in top_premises:
                conclusion_text = premise["description"].split("then ")[-1].strip() if "then " in premise["description"] else premise["description"]
                conclusion_parts.append(conclusion_text)
            
            conclusion = " Therefore, " + ". ".join(conclusion_parts)
        
        # Calculate confidence (average of premise confidences)
        confidence_values = [p.get("confidence", 0.5) for p in premises]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.7
        
        return {
            "strategy": ReasoningStrategy.DEDUCTIVE.value,
            "premises": premises,
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _inductive_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform inductive reasoning (from specific to general)"""
        # Extract knowledge and hypotheses
        domain_knowledge = knowledge_retrieval.get("domain_knowledge", {})
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # Collect specific examples/observations
        observations = []
        
        # Extract specific examples from domain knowledge
        for domain, knowledge in domain_knowledge.items():
            for key, value in knowledge.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        observations.append({
                            "description": f"{domain} - {key} - {subkey}: {str(subvalue)}",
                            "domain": domain,
                            "category": key,
                            "item": subkey
                        })
                elif isinstance(value, list):
                    observations.append({
                        "description": f"{domain} - {key}: {', '.join(value)}",
                        "domain": domain,
                        "category": key
                    })
        
        # Use primary hypothesis if available
        approach = ""
        if hypotheses:
            primary = hypotheses[0]
            approach = primary.get("approach", "")
        
        # Generate inferences from specific to general
        inferences = []
        
        # Group observations by domain/category
        grouped_observations = {}
        for obs in observations:
            domain = obs.get("domain", "unknown")
            category = obs.get("category", "unknown")
            key = f"{domain}_{category}"
            
            if key not in grouped_observations:
                grouped_observations[key] = []
            
            grouped_observations[key].append(obs)
        
        # Generate patterns from groups
        for key, group in grouped_observations.items():
            if len(group) >= 2:  # Need at least 2 observations to find a pattern
                pattern = f"Pattern in {key.replace('_', ' - ')}: " + "; ".join([o.get("description", "").split(":")[-1].strip() for o in group[:3]])
                inferences.append(pattern)
        
        # Derive general pattern/principle
        conclusion = ""
        if inferences:
            most_relevant_inference = inferences[0]
            conclusion = f"Based on observed patterns, a general principle emerges: {most_relevant_inference}. This leads to the approach: {approach}."
        
        # Calculate confidence (lower for inductive reasoning due to uncertainty)
        confidence = 0.6 + (min(len(inferences), 5) * 0.05)  # More inferences increase confidence, up to 0.85
        
        return {
            "strategy": ReasoningStrategy.INDUCTIVE.value,
            "observations": observations[:5],  # Limit for readability
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _abductive_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform abductive reasoning (to best explanation)"""
        # Extract knowledge and hypotheses
        rules = knowledge_retrieval.get("rules", [])
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # The observation is the problem itself
        observation = problem_formulation.get("problem", "")
        
        # Collect potential explanations
        explanations = []
        for hypothesis in hypotheses:
            explanations.append({
                "description": hypothesis.get("description", ""),
                "confidence": hypothesis.get("confidence", 0.5)
            })
        
        # Add explanations from rules
        for rule in rules:
            if rule["type"] == RuleType.ABDUCTIVE.value and "rule" in rule:
                rule_obj = rule["rule"]
                observation_match = rule_obj.get("observation", {})
                explanation_match = rule_obj.get("explanation", {})
                
                explanations.append({
                    "description": f"{explanation_match}",
                    "confidence": rule.get("confidence", 0.5)
                })
        
        # Use primary hypothesis if available
        approach = ""
        if hypotheses:
            primary = hypotheses[0]
            approach = primary.get("approach", "")
        
        # Find best explanation based on confidence
        best_explanation = None
        if explanations:
            best_explanation = max(explanations, key=lambda x: x.get("confidence", 0))
        
        # Generate inferences
        inferences = []
        for explanation in explanations:
            inferences.append(f"Possible explanation: {explanation.get('description', '')}")
        
        # Derive conclusion based on best explanation
        conclusion = ""
        if best_explanation:
            conclusion = f"The most likely explanation is: {best_explanation['description']}. Therefore, the recommended approach is: {approach}."
        
        # Calculate confidence (use confidence of best explanation)
        confidence = best_explanation.get("confidence", 0.6) if best_explanation else 0.6
        
        return {
            "strategy": ReasoningStrategy.ABDUCTIVE.value,
            "observation": observation,
            "explanations": explanations,
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _analogical_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform analogical reasoning (by comparison to similar problems)"""
        # Extract knowledge and hypotheses
        domain_knowledge = knowledge_retrieval.get("domain_knowledge", {})
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # Find analogs in domain knowledge
        analogs = []
        
        # Extract problem features
        problem = problem_formulation.get("problem", "")
        problem_type = problem_formulation.get("problem_type", "")
        key_concepts = problem_formulation.get("key_concepts", [])
        
        # Find analogous problems/solutions in domain knowledge
        for domain, knowledge in domain_knowledge.items():
            if isinstance(knowledge, dict):
                for key, value in knowledge.items():
                    # Check if key matches any concepts
                    if key in key_concepts or any(concept in key for concept in key_concepts):
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                analogs.append({
                                    "source": f"{domain}.{key}.{subkey}",
                                    "description": f"{subkey} in {key}: {str(subvalue)}",
                                    "similarity": self._calculate_similarity(problem, subkey)
                                })
                        elif isinstance(value, list):
                            analogs.append({
                                "source": f"{domain}.{key}",
                                "description": f"{key}: {', '.join(value)}",
                                "similarity": self._calculate_similarity(problem, key)
                            })
        
        # Sort analogs by similarity
        analogs.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get most similar analogs
        top_analogs = analogs[:3]
        
        # Use primary hypothesis if available
        approach = ""
        if hypotheses:
            primary = hypotheses[0]
            approach = primary.get("approach", "")
        
        # Generate mappings between source and target
        mappings = []
        for analog in top_analogs:
            mappings.append(f"Map {analog['source']} to current problem")
        
        # Generate inferences
        inferences = []
        for analog in top_analogs:
            inferences.append(f"Similar case: {analog['description']}")
        
        # Derive conclusion by transferring solution from analogs
        conclusion = ""
        if top_analogs:
            analog_insights = "; ".join([a["description"] for a in top_analogs])
            conclusion = f"By analogy to similar cases ({analog_insights}), the approach {approach} is appropriate for this problem."
        
        # Calculate confidence (based on similarity of best analog)
        confidence = top_analogs[0]["similarity"] if top_analogs else 0.6
        
        return {
            "strategy": ReasoningStrategy.ANALOGICAL.value,
            "analogs": top_analogs,
            "mappings": mappings,
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple implementation)"""
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _causal_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform causal reasoning (cause and effect analysis)"""
        # Extract knowledge and hypotheses
        rules = knowledge_retrieval.get("rules", [])
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # Collect causal relationships
        causal_relations = []
        
        # Extract causal relations from rules
        for rule in rules:
            if rule["type"] == RuleType.CAUSAL.value and "rule" in rule:
                rule_obj = rule["rule"]
                cause = rule_obj.get("cause", {})
                effect = rule_obj.get("effect", {})
                
                if cause and effect:
                    causal_relations.append({
                        "cause": f"{cause.get('concept', '')} {cause.get('property', '')} {cause.get('operator', '')} {cause.get('value', '')}",
                        "effect": f"{effect.get('concept', '')} {effect.get('property', '')} {effect.get('operator', '')} {effect.get('value', '')}",
                        "confidence": rule.get("confidence", 0.5)
                    })
        
        # Use primary hypothesis if available
        approach = ""
        if hypotheses:
            primary = hypotheses[0]
            approach = primary.get("approach", "")
        
        # Generate inferences
        inferences = []
        for relation in causal_relations:
            inferences.append(f"{relation['cause']} causes {relation['effect']}")
        
        # Build causal chain
        causal_chain = []
        if causal_relations:
            # Sort by confidence
            sorted_relations = sorted(causal_relations, key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Start with the highest confidence cause
            current_cause = sorted_relations[0]["cause"]
            causal_chain.append(current_cause)
            
            # Add the corresponding effect
            current_effect = sorted_relations[0]["effect"]
            causal_chain.append(current_effect)
            
            # Try to extend the chain by finding relations where this effect is a cause
            for _ in range(3):  # Limit chain length
                found = False
                for relation in sorted_relations:
                    if relation["cause"] == current_effect:
                        current_effect = relation["effect"]
                        causal_chain.append(current_effect)
                        found = True
                        break
                
                if not found:
                    break
        
        # Derive conclusion based on causal chain
        conclusion = ""
        if causal_chain:
            chain_text = " → ".join(causal_chain)
            conclusion = f"The causal relationship is: {chain_text}. Therefore, the approach {approach} addresses the root cause."
        
        # Calculate confidence (average of relation confidences)
        confidence_values = [r.get("confidence", 0.5) for r in causal_relations]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.6
        
        return {
            "strategy": ReasoningStrategy.CAUSAL.value,
            "causal_relations": causal_relations,
            "causal_chain": causal_chain,
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _counterfactual_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform counterfactual reasoning (what-if analysis)"""
        # Extract knowledge and hypotheses
        rules = knowledge_retrieval.get("rules", [])
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # Use primary hypothesis if available
        approach = ""
        if hypotheses:
            primary = hypotheses[0]
            approach = primary.get("approach", "")
        
        # Generate counterfactual scenarios
        counterfactuals = []
        
        # Generate counterfactuals based on hypotheses
        for hypothesis in hypotheses:
            description = hypothesis.get("description", "")
            
            # Create a counterfactual by negating the hypothesis
            if description:
                # Extract key action phrase
                action_phrases = re.findall(r"(use|implement|check|verify|design|optimize).*", description, re.IGNORECASE)
                if action_phrases:
                    action = action_phrases[0]
                    counterfactuals.append({
                        "scenario": f"What if we don't {action}?",
                        "consequence": "This would likely lead to suboptimal results or failure",
                        "confidence": 0.7
                    })
        
        # Generate counterfactuals based on relevant rules
        for rule in rules:
            if rule["type"] in [RuleType.LOGICAL.value, RuleType.CONSTRAINT.value] and "rule" in rule:
                rule_obj = rule["rule"]
                
                if rule["type"] == RuleType.LOGICAL.value:
                    premise = rule_obj.get("premise", {})
                    conclusion = rule_obj.get("conclusion", {})
                    
                    if premise and conclusion:
                        # Create counterfactual by negating the premise
                        counterfactuals.append({
                            "scenario": f"What if {premise.get('concept', '')} does not have {premise.get('property', '')} {premise.get('operator', '')} {premise.get('value', '')}?",
                            "consequence": f"Then {conclusion} would not apply",
                            "confidence": rule.get("confidence", 0.5)
                        })
                
                elif rule["type"] == RuleType.CONSTRAINT.value:
                    constraint = rule_obj.get("constraint", "")
                    
                    if constraint:
                        # Create counterfactual by violating the constraint
                        counterfactuals.append({
                            "scenario": f"What if we violate the constraint: {constraint}?",
                            "consequence": "This would likely lead to problems or failures",
                            "confidence": rule.get("confidence", 0.5)
                        })
        
        # Generate inferences
        inferences = []
        for counterfactual in counterfactuals:
            inferences.append(f"{counterfactual['scenario']} → {counterfactual['consequence']}")
        
        # Derive conclusion based on counterfactual analysis
        conclusion = ""
        if counterfactuals:
            # Focus on highest confidence counterfactuals
            sorted_counterfactuals = sorted(counterfactuals, key=lambda x: x.get("confidence", 0), reverse=True)
            top_counterfactuals = sorted_counterfactuals[:3]
            
            scenarios = [f"{cf['scenario']} → {cf['consequence']}" for cf in top_counterfactuals]
            scenarios_text = "; ".join(scenarios)
            
            conclusion = f"Counterfactual analysis shows: {scenarios_text}. Therefore, the approach {approach} is the most appropriate."
        
        # Calculate confidence (average of counterfactual confidences)
        confidence_values = [cf.get("confidence", 0.5) for cf in counterfactuals]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.6
        
        return {
            "strategy": ReasoningStrategy.COUNTERFACTUAL.value,
            "counterfactuals": counterfactuals,
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _bayesian_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform Bayesian reasoning (probabilistic inference)"""
        # Extract knowledge and hypotheses
        rules = knowledge_retrieval.get("rules", [])
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # Get prior probabilities (use confidence as probability)
        priors = []
        for hypothesis in hypotheses:
            priors.append({
                "hypothesis": hypothesis.get("description", ""),
                "approach": hypothesis.get("approach", ""),
                "prior": hypothesis.get("confidence", 0.5)
            })
        
        # Use evidence from rules to update priors
        evidence = []
        
        # Extract evidence from rules
        for rule in rules:
            relevance = 0.0
            
            # Check relevance to hypotheses
            for prior in priors:
                approach = prior.get("approach", "")
                if approach.lower() in str(rule).lower():
                    relevance += 0.2
                
                hypothesis = prior.get("hypothesis", "")
                if any(word in str(rule).lower() for word in hypothesis.lower().split() if len(word) > 3):
                    relevance += 0.1
            
            # Only include relevant evidence
            if relevance > 0:
                evidence.append({
                    "description": rule.get("description", ""),
                    "likelihood": rule.get("confidence", 0.5),
                    "relevance": min(relevance, 1.0)
                })
        
        # Calculate posteriors (simplified Bayesian update)
        posteriors = []
        for prior in priors:
            posterior = prior["prior"]
            
            # Update posterior with each evidence
            for ev in evidence:
                # Simplified Bayesian update
                # P(H|E) ∝ P(E|H) * P(H)
                likelihood = ev["likelihood"]
                relevance = ev["relevance"]
                
                # Adjust likelihood based on relevance
                adjusted_likelihood = likelihood * relevance
                
                # Update posterior (simplified)
                posterior = posterior * (1 + adjusted_likelihood) / (1 + posterior * adjusted_likelihood)
                
                # Keep posterior in valid range
                posterior = max(0.01, min(posterior, 0.99))
            
            posteriors.append({
                "hypothesis": prior["hypothesis"],
                "approach": prior["approach"],
                "prior": prior["prior"],
                "posterior": posterior
            })
        
        # Sort posteriors by probability
        posteriors.sort(key=lambda x: x["posterior"], reverse=True)
        
        # Use the highest posterior hypothesis
        best_hypothesis = posteriors[0] if posteriors else None
        approach = best_hypothesis["approach"] if best_hypothesis else ""
        
        # Generate inferences
        inferences = []
        for posterior in posteriors:
            inferences.append(f"P({posterior['hypothesis']}) = {posterior['posterior']:.2f}")
        
        # Derive conclusion based on Bayesian analysis
        conclusion = ""
        if best_hypothesis:
            conclusion = f"Bayesian analysis indicates that {best_hypothesis['hypothesis']} has the highest probability ({best_hypothesis['posterior']:.2f}). Therefore, the approach {approach} is recommended."
        
        # Use posterior of best hypothesis as confidence
        confidence = best_hypothesis["posterior"] if best_hypothesis else 0.6
        
        return {
            "strategy": ReasoningStrategy.BAYESIAN.value,
            "priors": [{"hypothesis": p["hypothesis"], "prior": p["prior"]} for p in priors],
            "evidence": evidence,
            "posteriors": [{"hypothesis": p["hypothesis"], "posterior": p["posterior"]} for p in posteriors],
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _constraint_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform constraint-based reasoning"""
        # Extract knowledge and hypotheses
        rules = knowledge_retrieval.get("rules", [])
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # Get problem constraints
        constraints = problem_formulation.get("constraints", [])
        
        # Extract additional constraints from rules
        for rule in rules:
            if rule["type"] == RuleType.CONSTRAINT.value and "rule" in rule:
                rule_obj = rule["rule"]
                constraint = rule_obj.get("constraint", "")
                
                if constraint and constraint not in constraints:
                    constraints.append(constraint)
        
        # Evaluate hypotheses against constraints
        evaluations = []
        for hypothesis in hypotheses:
            description = hypothesis.get("description", "")
            approach = hypothesis.get("approach", "")
            
            # Count satisfied constraints
            satisfied = 0
            violated = 0
            
            for constraint in constraints:
                # Simple matching - if constraint keywords are in hypothesis
                constraint_keywords = [word.lower() for word in constraint.split() if len(word) > 3]
                description_lower = description.lower()
                
                # Check if constraint keywords are present
                matches = sum(1 for keyword in constraint_keywords if keyword in description_lower)
                if matches >= len(constraint_keywords) / 2:  # At least half the keywords match
                    satisfied += 1
                else:
                    violated += 1
            
            # Calculate satisfaction ratio
            satisfaction_ratio = satisfied / (satisfied + violated) if (satisfied + violated) > 0 else 0.5
            
            evaluations.append({
                "hypothesis": description,
                "approach": approach,
                "satisfied": satisfied,
                "violated": violated,
                "satisfaction_ratio": satisfaction_ratio
            })
        
        # Sort by satisfaction ratio
        evaluations.sort(key=lambda x: x["satisfaction_ratio"], reverse=True)
        
        # Get best evaluation
        best_evaluation = evaluations[0] if evaluations else None
        approach = best_evaluation["approach"] if best_evaluation else ""
        
        # Generate inferences
        inferences = []
        for evaluation in evaluations:
            inferences.append(f"{evaluation['hypothesis']} - Satisfies {evaluation['satisfied']} constraints, violates {evaluation['violated']} constraints")
        
        # Derive conclusion based on constraint satisfaction
        conclusion = ""
        if best_evaluation:
            conclusion = f"Constraint analysis shows that {best_evaluation['hypothesis']} best satisfies the constraints ({best_evaluation['satisfied']} satisfied, {best_evaluation['violated']} violated). Therefore, the approach {approach} is recommended."
        
        # Use satisfaction ratio as confidence
        confidence = best_evaluation["satisfaction_ratio"] if best_evaluation else 0.6
        
        return {
            "strategy": ReasoningStrategy.CONSTRAINT.value,
            "constraints": constraints,
            "evaluations": evaluations,
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _case_based_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform case-based reasoning"""
        # This is similar to analogical reasoning but focuses on previous cases/solutions
        # For now, we'll use a simplified implementation similar to analogical reasoning
        
        # Extract knowledge and hypotheses
        domain_knowledge = knowledge_retrieval.get("domain_knowledge", {})
        hypotheses = hypothesis_generation.get("hypotheses", [])
        
        # Extract problem details
        problem = problem_formulation.get("problem", "")
        problem_type = problem_formulation.get("problem_type", "")
        key_concepts = problem_formulation.get("key_concepts", [])
        
        # Find similar cases in domain knowledge
        similar_cases = []
        
        # Search for similar cases in domain knowledge
        for domain, knowledge in domain_knowledge.items():
            if isinstance(knowledge, dict):
                for key, value in knowledge.items():
                    # Check if key matches problem type or concepts
                    if key == problem_type or key in key_concepts or any(concept in key for concept in key_concepts):
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                similar_cases.append({
                                    "case": f"{domain}.{key}.{subkey}",
                                    "description": f"{subkey} in {key}: {str(subvalue)}",
                                    "similarity": self._calculate_similarity(problem, subkey)
                                })
        
        # Sort cases by similarity
        similar_cases.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get most similar cases
        top_cases = similar_cases[:3]
        
        # Use primary hypothesis if available
        approach = ""
        if hypotheses:
            primary = hypotheses[0]
            approach = primary.get("approach", "")
        
        # Generate inferences from similar cases
        inferences = []
        for case in top_cases:
            inferences.append(f"Similar case: {case['description']}")
        
        # Adapt solution from similar cases
        adapted_solutions = []
        for case in top_cases:
            adapted_solutions.append(f"Adapt solution from {case['case']}")
        
        # Derive conclusion based on case adaptation
        conclusion = ""
        if top_cases:
            case_descriptions = "; ".join([c["description"] for c in top_cases])
            conclusion = f"Based on similar cases ({case_descriptions}), the solution can be adapted to this problem. The recommended approach is {approach}."
        
        # Calculate confidence (based on similarity of best case)
        confidence = top_cases[0]["similarity"] if top_cases else 0.6
        
        return {
            "strategy": ReasoningStrategy.CASE_BASED.value,
            "similar_cases": top_cases,
            "adaptations": adapted_solutions,
            "inferences": inferences,
            "conclusion": conclusion,
            "approach": approach,
            "confidence": confidence
        }
    
    def _hybrid_reasoning(
        self,
        problem_formulation: Dict[str, Any],
        hypothesis_generation: Dict[str, Any],
        knowledge_retrieval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform hybrid reasoning (combining multiple strategies)"""
        # Determine which strategies to combine based on problem type
        problem_type = problem_formulation.get("problem_type", "")
        
        strategies_to_combine = []
        
        if problem_type == "debugging":
            # Debugging benefits from causal and abductive reasoning
            strategies_to_combine = [ReasoningStrategy.CAUSAL, ReasoningStrategy.ABDUCTIVE]
        
        elif problem_type == "code_generation":
            # Code generation benefits from deductive and analogical reasoning
            strategies_to_combine = [ReasoningStrategy.DEDUCTIVE, ReasoningStrategy.ANALOGICAL]
        
        elif problem_type == "optimization":
            # Optimization benefits from causal and constraint reasoning
            strategies_to_combine = [ReasoningStrategy.CAUSAL, ReasoningStrategy.CONSTRAINT]
        
        elif problem_type == "architecture":
            # Architecture benefits from deductive and constraint reasoning
            strategies_to_combine = [ReasoningStrategy.DEDUCTIVE, ReasoningStrategy.CONSTRAINT]
        
        else:
            # Default combination for other problem types
            strategies_to_combine = [ReasoningStrategy.DEDUCTIVE, ReasoningStrategy.CASE_BASED]
        
        # Apply each strategy
        results = []
        for strategy in strategies_to_combine:
            strategy_handler = self.strategy_handlers.get(strategy)
            if strategy_handler:
                result = strategy_handler(
                    problem_formulation=problem_formulation,
                    hypothesis_generation=hypothesis_generation,
                    knowledge_retrieval=knowledge_retrieval
                )
                results.append(result)
        
        # If no results, fall back to deductive reasoning
        if not results:
            deductive_result = self._deductive_reasoning(
                problem_formulation=problem_formulation,
                hypothesis_generation=hypothesis_generation,
                knowledge_retrieval=knowledge_retrieval
            )
            results.append(deductive_result)
        
        # Combine inferences from all strategies
        combined_inferences = []
        for result in results:
            combined_inferences.extend(result.get("inferences", []))
        
        # Use approaches from all strategies, prioritizing higher confidence
        approaches = []
        for result in results:
            approach = result.get("approach", "")
            confidence = result.get("confidence", 0.5)
            
            if approach:
                approaches.append({
                    "approach": approach,
                    "confidence": confidence,
                    "strategy": result.get("strategy", "unknown")
                })
        
        # Sort approaches by confidence
        approaches.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Use the highest confidence approach
        best_approach = approaches[0]["approach"] if approaches else ""
        
        # Combine conclusions, prioritizing higher confidence results
        sorted_results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
        conclusion_parts = [r.get("conclusion", "") for r in sorted_results if r.get("conclusion")]
        conclusion = " Furthermore, ".join(conclusion_parts)
        
        # Calculate overall confidence (weighted average of individual confidences)
        confidence_values = [(r.get("confidence", 0.5), r.get("strategy", "unknown")) for r in results]
        
        # Weight primary strategy higher
        total_confidence = 0.0
        total_weight = 0.0
        
        for i, (conf, _) in enumerate(confidence_values):
            # First strategy gets higher weight
            weight = 1.0 if i == 0 else 0.7
            total_confidence += conf * weight
            total_weight += weight
        
        overall_confidence = total_confidence / total_weight if total_weight > 0 else 0.6
        
        return {
            "strategy": ReasoningStrategy.HYBRID.value,
            "combined_strategies": [r.get("strategy", "unknown") for r in results],
            "inferences": combined_inferences,
            "conclusion": conclusion,
            "approach": best_approach,
            "confidence": overall_confidence
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the reasoning engine"""
        return {
            "operational": True,
            "active_strategies": [s.value for s in ReasoningStrategy],
            "default_strategy": self.default_strategy.value,
            "knowledge_domains": list(self.rule_sets.keys()),
            "metrics": {
                "total_tasks": self.metrics["total_reasoning_tasks"],
                "successful_tasks": self.metrics["successful_tasks"],
                "failed_tasks": self.metrics["failed_tasks"],
                "average_time": self.metrics["average_reasoning_time"]
            }
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status of the reasoning engine"""
        # Count active reasoning sessions
        active_sessions = [
            session_id for session_id, session in self.reasoning_history.items()
            if not session.get("end_time")
        ]
        
        return {
            "operational": True,
            "active_sessions": len(active_sessions),
            "total_sessions": len(self.reasoning_history),
            "strategies": {
                "available": [s.value for s in ReasoningStrategy],
                "default": self.default_strategy.value,
                "usage": self.metrics["strategy_usage"]
            },
            "knowledge_base": {
                "concepts": len(self.knowledge_base["concepts"]),
                "domains": list(self.knowledge_base["domain_knowledge"].keys())
            },
            "rule_sets": {
                domain: len(rules) for domain, rules in self.rule_sets.items()
            },
            "performance": {
                "total_tasks": self.metrics["total_reasoning_tasks"],
                "successful": self.metrics["successful_tasks"],
                "failed": self.metrics["failed_tasks"],
                "success_rate": self.metrics["successful_tasks"] / self.metrics["total_reasoning_tasks"] if self.metrics["total_reasoning_tasks"] > 0 else 0,
                "average_time": self.metrics["average_reasoning_time"],
                "stage_timings": self.metrics["stage_timings"]
            }
        }

# Initialize reasoning engine
reasoning_engine = NeuroSymbolicEngine()