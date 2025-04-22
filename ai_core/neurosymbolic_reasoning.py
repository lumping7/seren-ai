"""
Neuro-Symbolic Reasoning System for Seren

Implements advanced hybrid reasoning combining neural network capabilities
with symbolic logic for powerful inference, explainability, and reasoning.
"""

import os
import sys
import json
import logging
import time
import math
import re
import random
import datetime
from enum import Enum, auto
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from ai_core.knowledge.library import knowledge_library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ========================= Symbolic Reasoning Components =========================

class LogicalOperator(Enum):
    """Logical operators for symbolic reasoning"""
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    EQUIVALENT = auto()
    XOR = auto()

class QuantifierType(Enum):
    """Quantifier types for logical formulas"""
    UNIVERSAL = auto()  # For all (∀)
    EXISTENTIAL = auto()  # There exists (∃)
    NONE = auto()  # No quantifier

class RuleType(Enum):
    """Types of reasoning rules"""
    DEDUCTIVE = auto()  # Classical logical deduction
    INDUCTIVE = auto()  # Generalization from examples
    ABDUCTIVE = auto()  # Inference to best explanation
    ANALOGICAL = auto()  # Reasoning by analogy
    CAUSAL = auto()  # Cause-effect reasoning
    COUNTERFACTUAL = auto()  # "What if" reasoning
    TEMPORAL = auto()  # Time-based reasoning
    PROBABILISTIC = auto()  # Uncertainty-aware reasoning
    SPATIAL = auto()  # Space and location reasoning
    META = auto()  # Reasoning about reasoning itself

class KnowledgeType(Enum):
    """Types of knowledge for reasoning"""
    FACT = auto()  # Verified truth
    BELIEF = auto()  # Unverified but held to be true
    ASSUMPTION = auto()  # Temporarily held for reasoning
    HYPOTHESIS = auto()  # Proposed explanation to be tested
    OBSERVATION = auto()  # Direct sensory input
    INFERENCE = auto()  # Derived through reasoning
    RULE = auto()  # General principle
    CONCEPT = auto()  # Abstract idea
    PROCEDURE = auto()  # Process-oriented knowledge
    META_KNOWLEDGE = auto()  # Knowledge about knowledge

class ReasoningStrategy(Enum):
    """Strategies for hybrid reasoning"""
    NEURAL_FIRST = auto()  # Use neural methods first, then symbolic
    SYMBOLIC_FIRST = auto()  # Use symbolic methods first, then neural
    PARALLEL = auto()  # Run both approaches in parallel and combine
    ITERATIVE = auto()  # Alternate between approaches
    ADAPTIVE = auto()  # Choose approach based on problem characteristics
    HIERARCHICAL = auto()  # Layer approaches by abstraction level
    PROCEDURAL = auto()  # Follow a specific multi-step procedure
    META_REASONING = auto()  # Reason about which reasoning approach to use

class Symbol:
    """Symbolic representation for logical reasoning"""
    
    def __init__(
        self, 
        name: str, 
        type_signature: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        neural_embedding: Optional[List[float]] = None
    ):
        """
        Initialize a symbol
        
        Args:
            name: Symbol name
            type_signature: Type specification
            attributes: Additional attributes
            neural_embedding: Vector embedding from neural component
        """
        self.name = name
        self.type_signature = type_signature
        self.attributes = attributes or {}
        self.neural_embedding = neural_embedding
        
    def __str__(self) -> str:
        return f"Symbol({self.name}:{self.type_signature if self.type_signature else 'Any'})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Symbol):
            return False
        return self.name == other.name and self.type_signature == other.type_signature
    
    def __hash__(self) -> int:
        return hash((self.name, self.type_signature))

class Predicate:
    """Predicate for logical formulas"""
    
    def __init__(
        self, 
        name: str, 
        arity: int,
        type_signature: Optional[List[str]] = None,
        neural_function: Optional[Callable] = None
    ):
        """
        Initialize a predicate
        
        Args:
            name: Predicate name
            arity: Number of arguments
            type_signature: Type signatures for arguments
            neural_function: Neural network mapping for this predicate
        """
        self.name = name
        self.arity = arity
        self.type_signature = type_signature
        self.neural_function = neural_function
    
    def __str__(self) -> str:
        return f"Predicate({self.name}/{self.arity})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Predicate):
            return False
        return self.name == other.name and self.arity == other.arity
    
    def __hash__(self) -> int:
        return hash((self.name, self.arity))
    
    def apply(self, *args) -> 'Atom':
        """Apply this predicate to arguments"""
        if len(args) != self.arity:
            raise ValueError(f"Predicate {self.name} expects {self.arity} arguments, got {len(args)}")
        return Atom(self, list(args))
    
    def neural_evaluate(self, *args) -> float:
        """Evaluate using neural function if available"""
        if self.neural_function is None:
            raise ValueError(f"No neural function available for predicate {self.name}")
        return self.neural_function(*args)

class Term:
    """Base class for logical terms"""
    pass

class Atom(Term):
    """Atomic formula (predicate applied to terms)"""
    
    def __init__(
        self, 
        predicate: Predicate, 
        arguments: List[Any],
        neural_confidence: float = 1.0
    ):
        """
        Initialize an atom
        
        Args:
            predicate: The predicate
            arguments: Arguments to the predicate
            neural_confidence: Confidence score from neural component
        """
        self.predicate = predicate
        self.arguments = arguments
        self.neural_confidence = neural_confidence
    
    def __str__(self) -> str:
        args_str = ', '.join(str(arg) for arg in self.arguments)
        return f"{self.predicate.name}({args_str})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def ground(self, substitution: Dict[str, Any]) -> 'Atom':
        """Apply variable substitution"""
        new_args = []
        for arg in self.arguments:
            if isinstance(arg, Variable) and arg.name in substitution:
                new_args.append(substitution[arg.name])
            else:
                new_args.append(arg)
        return Atom(self.predicate, new_args, self.neural_confidence)

class Variable(Term):
    """Logical variable"""
    
    def __init__(
        self, 
        name: str,
        type_signature: Optional[str] = None
    ):
        """
        Initialize a variable
        
        Args:
            name: Variable name
            type_signature: Type specification
        """
        self.name = name
        self.type_signature = type_signature
    
    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name
    
    def __hash__(self) -> int:
        return hash(self.name)

class Formula:
    """Logical formula for symbolic reasoning"""
    
    def __init__(
        self, 
        formula_type: str,
        components: List[Any],
        quantifier: QuantifierType = QuantifierType.NONE,
        variable: Optional[Variable] = None,
        neural_confidence: float = 1.0
    ):
        """
        Initialize a formula
        
        Args:
            formula_type: Type of formula (atom, compound)
            components: Formula components
            quantifier: Quantifier type if any
            variable: Quantified variable if any
            neural_confidence: Confidence score from neural component
        """
        self.formula_type = formula_type
        self.components = components
        self.quantifier = quantifier
        self.variable = variable
        self.neural_confidence = neural_confidence
    
    @classmethod
    def create_atom(cls, atom: Atom) -> 'Formula':
        """Create atomic formula"""
        return cls("atom", [atom], neural_confidence=atom.neural_confidence)
    
    @classmethod
    def create_not(cls, formula: 'Formula', neural_confidence: float = 1.0) -> 'Formula':
        """Create negation formula"""
        return cls("not", [formula], neural_confidence=neural_confidence)
    
    @classmethod
    def create_binary(cls, op: LogicalOperator, left: 'Formula', right: 'Formula', neural_confidence: float = 1.0) -> 'Formula':
        """Create binary formula"""
        return cls(op.name.lower(), [left, right], neural_confidence=neural_confidence)
    
    @classmethod
    def create_quantified(cls, quantifier: QuantifierType, variable: Variable, formula: 'Formula', neural_confidence: float = 1.0) -> 'Formula':
        """Create quantified formula"""
        return cls("quantified", [formula], quantifier=quantifier, variable=variable, neural_confidence=neural_confidence)
    
    def __str__(self) -> str:
        if self.formula_type == "atom":
            return str(self.components[0])
        elif self.formula_type == "not":
            return f"¬({str(self.components[0])})"
        elif self.formula_type in [op.name.lower() for op in LogicalOperator]:
            op_str = {
                "and": "∧",
                "or": "∨",
                "implies": "→",
                "equivalent": "↔",
                "xor": "⊕"
            }.get(self.formula_type, self.formula_type)
            return f"({str(self.components[0])} {op_str} {str(self.components[1])})"
        elif self.formula_type == "quantified":
            quantifier_str = "∀" if self.quantifier == QuantifierType.UNIVERSAL else "∃"
            return f"{quantifier_str}{str(self.variable)}. {str(self.components[0])}"
        return f"Formula({self.formula_type}, {self.components})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def ground(self, substitution: Dict[str, Any]) -> 'Formula':
        """Apply variable substitution"""
        if self.formula_type == "atom":
            return Formula.create_atom(self.components[0].ground(substitution))
        elif self.formula_type == "not":
            return Formula.create_not(self.components[0].ground(substitution), self.neural_confidence)
        elif self.formula_type in [op.name.lower() for op in LogicalOperator]:
            return Formula.create_binary(
                LogicalOperator[self.formula_type.upper()],
                self.components[0].ground(substitution),
                self.components[1].ground(substitution),
                self.neural_confidence
            )
        elif self.formula_type == "quantified":
            # Don't substitute quantified variable
            new_substitution = substitution.copy()
            if self.variable.name in new_substitution:
                del new_substitution[self.variable.name]
            return Formula.create_quantified(
                self.quantifier,
                self.variable,
                self.components[0].ground(new_substitution),
                self.neural_confidence
            )
        return self

class Rule:
    """Reasoning rule for inference"""
    
    def __init__(
        self, 
        name: str,
        premises: List[Formula],
        conclusion: Formula,
        rule_type: RuleType,
        metadata: Optional[Dict[str, Any]] = None,
        neural_confidence: float = 1.0
    ):
        """
        Initialize a rule
        
        Args:
            name: Rule name
            premises: Premises (antecedents)
            conclusion: Conclusion (consequent)
            rule_type: Type of reasoning rule
            metadata: Additional rule metadata
            neural_confidence: Confidence score from neural component
        """
        self.name = name
        self.premises = premises
        self.conclusion = conclusion
        self.rule_type = rule_type
        self.metadata = metadata or {}
        self.neural_confidence = neural_confidence
    
    def __str__(self) -> str:
        premises_str = ", ".join(str(p) for p in self.premises)
        return f"Rule({self.name}: {premises_str} ⊢ {self.conclusion})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def apply(self, knowledge_base: 'KnowledgeBase') -> List[Formula]:
        """
        Apply this rule to a knowledge base
        
        Args:
            knowledge_base: The knowledge base to apply the rule to
            
        Returns:
            List of new formulas derived by applying this rule
        """
        if self.rule_type == RuleType.DEDUCTIVE:
            return self._apply_deductive(knowledge_base)
        elif self.rule_type == RuleType.INDUCTIVE:
            return self._apply_inductive(knowledge_base)
        elif self.rule_type == RuleType.ABDUCTIVE:
            return self._apply_abductive(knowledge_base)
        # Default to simplified rule application
        matches = knowledge_base.match_formulas(self.premises)
        if not matches:
            return []
        
        new_formulas = []
        for substitution in matches:
            new_formula = self.conclusion.ground(substitution)
            if not knowledge_base.contains(new_formula):
                new_formulas.append(new_formula)
        
        return new_formulas
    
    def _apply_deductive(self, knowledge_base: 'KnowledgeBase') -> List[Formula]:
        """Apply deductive reasoning"""
        # Standard logical deduction
        matches = knowledge_base.match_formulas(self.premises)
        return [self.conclusion.ground(subst) for subst in matches 
                if not knowledge_base.contains(self.conclusion.ground(subst))]
    
    def _apply_inductive(self, knowledge_base: 'KnowledgeBase') -> List[Formula]:
        """Apply inductive reasoning"""
        # Generalization from examples
        # Look for patterns in data and generalize
        # Simplified implementation
        return []
    
    def _apply_abductive(self, knowledge_base: 'KnowledgeBase') -> List[Formula]:
        """Apply abductive reasoning"""
        # Inference to the best explanation
        # If we observe the conclusion, what premises might explain it?
        # Simplified implementation
        return []

class KnowledgeBase:
    """Knowledge base for symbolic reasoning"""
    
    def __init__(self, name: str = "Main"):
        """
        Initialize a knowledge base
        
        Args:
            name: Name of the knowledge base
        """
        self.name = name
        self.formulas = []
        self.rules = []
        self.predicates = {}  # name -> Predicate
        self.neural_weights = {}
    
    def add_formula(self, formula: Formula) -> bool:
        """
        Add a formula to the knowledge base
        
        Args:
            formula: Formula to add
            
        Returns:
            Success status
        """
        if self.contains(formula):
            return False
        self.formulas.append(formula)
        return True
    
    def add_rule(self, rule: Rule) -> bool:
        """
        Add a rule to the knowledge base
        
        Args:
            rule: Rule to add
            
        Returns:
            Success status
        """
        self.rules.append(rule)
        return True
    
    def add_predicate(self, predicate: Predicate) -> bool:
        """
        Add a predicate to the knowledge base
        
        Args:
            predicate: Predicate to add
            
        Returns:
            Success status
        """
        if predicate.name in self.predicates:
            return False
        self.predicates[predicate.name] = predicate
        return True
    
    def contains(self, formula: Formula) -> bool:
        """
        Check if formula is in the knowledge base
        
        Args:
            formula: Formula to check
            
        Returns:
            Whether the formula is contained
        """
        # This is a simplified implementation - real systems would use
        # logical equivalence checking
        return any(str(f) == str(formula) for f in self.formulas)
    
    def match_formulas(self, templates: List[Formula]) -> List[Dict[str, Any]]:
        """
        Find substitutions matching formula templates
        
        Args:
            templates: Formula templates to match
            
        Returns:
            List of variable substitutions
        """
        # Simplified unification algorithm
        if not templates:
            return [{}]  # Empty match
        
        # Start with first template
        matches = self._match_formula(templates[0])
        
        # Match remaining templates with consistent substitutions
        for template in templates[1:]:
            new_matches = []
            for substitution in matches:
                # Apply current substitution to template
                grounded_template = template.ground(substitution)
                # Find matches for this grounded template
                template_matches = self._match_formula(grounded_template)
                
                # Combine substitutions
                for template_match in template_matches:
                    combined = substitution.copy()
                    for var, val in template_match.items():
                        if var in combined and combined[var] != val:
                            continue  # Inconsistent substitution
                        combined[var] = val
                    new_matches.append(combined)
            
            matches = new_matches
            if not matches:
                return []  # No consistent substitutions
        
        return matches
    
    def _match_formula(self, template: Formula) -> List[Dict[str, Any]]:
        """Match a single formula template"""
        matches = []
        for formula in self.formulas:
            substitution = self._unify(template, formula)
            if substitution is not None:
                matches.append(substitution)
        return matches
    
    def _unify(self, template: Formula, formula: Formula) -> Optional[Dict[str, Any]]:
        """Unify template with formula, returning variable substitution if possible"""
        # Simplified unification for demonstration
        # Real implementation would handle complex formulas
        
        # Different formula types don't unify
        if template.formula_type != formula.formula_type:
            return None
        
        if template.formula_type == "atom":
            return self._unify_atoms(template.components[0], formula.components[0])
        elif template.formula_type == "not":
            return self._unify(template.components[0], formula.components[0])
        elif template.formula_type in [op.name.lower() for op in LogicalOperator]:
            # Binary formula
            left_subst = self._unify(template.components[0], formula.components[0])
            if left_subst is None:
                return None
            right_subst = self._unify(template.components[1], formula.components[1])
            if right_subst is None:
                return None
            
            # Check for consistency
            combined = {}
            for subst in [left_subst, right_subst]:
                for var, val in subst.items():
                    if var in combined and combined[var] != val:
                        return None  # Inconsistent
                    combined[var] = val
            
            return combined
        
        # Fallback for unsupported formula types
        return {} if str(template) == str(formula) else None
    
    def _unify_atoms(self, template_atom: Atom, formula_atom: Atom) -> Optional[Dict[str, Any]]:
        """Unify two atoms"""
        # Different predicates don't unify
        if template_atom.predicate.name != formula_atom.predicate.name:
            return None
        
        # Different arities don't unify
        if len(template_atom.arguments) != len(formula_atom.arguments):
            return None
        
        # Unify arguments
        substitution = {}
        for t_arg, f_arg in zip(template_atom.arguments, formula_atom.arguments):
            if isinstance(t_arg, Variable):
                # Variable in template
                if t_arg.name in substitution and substitution[t_arg.name] != f_arg:
                    return None  # Inconsistent binding
                substitution[t_arg.name] = f_arg
            elif t_arg != f_arg:
                return None  # Constants don't match
        
        return substitution
    
    def inference_step(self) -> List[Formula]:
        """
        Perform one step of inference
        
        Returns:
            New formulas derived in this step
        """
        new_formulas = []
        
        for rule in self.rules:
            derived = rule.apply(self)
            for formula in derived:
                if self.add_formula(formula):
                    new_formulas.append(formula)
        
        return new_formulas
    
    def inference_until_fixed_point(self, max_iterations: int = 100) -> int:
        """
        Perform inference until no new formulas are derived
        
        Args:
            max_iterations: Maximum number of inference steps
            
        Returns:
            Number of new formulas derived
        """
        total_new = 0
        for _ in range(max_iterations):
            new_formulas = self.inference_step()
            if not new_formulas:
                break
            total_new += len(new_formulas)
        
        return total_new
    
    def query(self, query_formula: Formula) -> Tuple[bool, float]:
        """
        Query the knowledge base
        
        Args:
            query_formula: Formula to query
            
        Returns:
            (contains_formula, confidence)
        """
        for formula in self.formulas:
            if str(formula) == str(query_formula):
                return True, formula.neural_confidence
        
        # Try to derive the query
        old_count = len(self.formulas)
        self.inference_until_fixed_point()
        
        # Check again after inference
        for formula in self.formulas[old_count:]:
            if str(formula) == str(query_formula):
                return True, formula.neural_confidence
        
        return False, 0.0

# ========================= Neural Reasoning Components =========================

class LiquidNeuralNetwork(nn.Module):
    """Liquid Neural Network for adaptive reasoning"""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int,
        num_layers: int = 3,
        liquid_time_constants: Optional[List[float]] = None,
        dropout: float = 0.1
    ):
        """
        Initialize a liquid neural network
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension
            num_layers: Number of liquid layers
            liquid_time_constants: Time constants for liquid neurons
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Default time constants if not provided
        if liquid_time_constants is None:
            liquid_time_constants = [0.1, 0.5, 1.0, 2.0, 5.0]
        self.liquid_time_constants = liquid_time_constants
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Liquid layers
        self.liquid_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = LiquidLayer(
                hidden_size, 
                hidden_size,
                time_constants=liquid_time_constants,
                dropout=dropout
            )
            self.liquid_layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        
        # State for recurrent processing
        self.hidden_states = None
        
    def forward(self, x, reset_state: bool = False):
        """Forward pass"""
        # Initialize or reset hidden states
        if self.hidden_states is None or reset_state:
            self.hidden_states = [None] * self.num_layers
        
        # Input projection
        x = F.dropout(F.relu(self.input_proj(x)), p=0.1, training=self.training)
        
        # Process through liquid layers
        for i, layer in enumerate(self.liquid_layers):
            x, new_state = layer(x, self.hidden_states[i])
            self.hidden_states[i] = new_state
        
        # Output projection
        output = self.output_proj(x)
        
        return output
    
    def reset_state(self):
        """Reset the network's recurrent state"""
        self.hidden_states = None

class LiquidLayer(nn.Module):
    """Liquid neural network layer with time-based dynamics"""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        time_constants: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
        dropout: float = 0.1
    ):
        """
        Initialize a liquid layer
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            time_constants: Time constants for liquid neurons
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_constants = time_constants
        self.dropout = dropout
        
        # Number of different liquid neuron types (by time constant)
        self.num_types = len(time_constants)
        
        # Ensure hidden size is divisible by number of types
        assert hidden_size % self.num_types == 0, f"Hidden size must be divisible by {self.num_types}"
        self.group_size = hidden_size // self.num_types
        
        # Weights
        self.weight_ih = nn.Linear(input_size, hidden_size)
        self.weight_hh = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Adaptive threshold
        self.threshold = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, prev_state=None):
        """
        Forward pass
        
        Args:
            x: Input tensor
            prev_state: Previous hidden state
            
        Returns:
            (output, new_state)
        """
        # Initialize state if needed
        if prev_state is None:
            prev_state = torch.zeros_like(x[:, :self.hidden_size])
            
        # Calculate input and recurrent contributions
        i_contribution = self.weight_ih(x)
        h_contribution = self.weight_hh(prev_state)
        
        # Calculate liquid neuron dynamics based on time constants
        new_state = torch.zeros_like(prev_state)
        
        for i, tau in enumerate(self.time_constants):
            # Get slice for this neuron type
            start_idx = i * self.group_size
            end_idx = (i + 1) * self.group_size
            
            # Update rule for liquid neurons with time constant tau
            decay = torch.exp(-1.0 / tau)
            update = i_contribution[:, start_idx:end_idx] + h_contribution[:, start_idx:end_idx]
            
            # Apply adaptive threshold
            threshold = self.threshold[start_idx:end_idx]
            update = update - threshold
            
            # Update state with decay
            new_state_group = decay * prev_state[:, start_idx:end_idx] + (1 - decay) * update
            new_state[:, start_idx:end_idx] = new_state_group
        
        # Apply layer normalization
        normalized_state = self.layer_norm(new_state)
        
        # Apply activation function and dropout
        output = F.dropout(F.relu(normalized_state), p=self.dropout, training=self.training)
        
        return output, new_state

class SymbolicTranslator:
    """Translator between neural and symbolic representations"""
    
    def __init__(
        self, 
        embedding_size: int = 256,
        vocabulary_size: int = 10000,
        max_formula_length: int = 50
    ):
        """
        Initialize a symbolic translator
        
        Args:
            embedding_size: Size of the embeddings
            vocabulary_size: Size of the symbol vocabulary
            max_formula_length: Maximum length of logical formulas
        """
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.max_formula_length = max_formula_length
        
        # Symbol vocabulary (maps symbols to indices)
        self.symbol_to_idx = {}
        self.idx_to_symbol = {}
        
        # Embeddings
        self.symbol_embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.formula_encoder = nn.LSTM(
            embedding_size, 
            embedding_size, 
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.formula_decoder = nn.LSTM(
            embedding_size,
            embedding_size * 2,
            num_layers=2,
            batch_first=True
        )
        self.projection = nn.Linear(embedding_size * 2, vocabulary_size)
        
    def neural_to_symbolic(self, neural_output, threshold: float = 0.5) -> Formula:
        """
        Convert neural network output to symbolic representation
        
        Args:
            neural_output: Neural network output
            threshold: Confidence threshold
            
        Returns:
            Symbolic formula
        """
        # This is a simplified implementation - real system would
        # use more sophisticated translation mechanisms
        
        # Simulate conversion for now
        rand_type = random.choice(["atom", "not", "and", "or", "implies"])
        
        if rand_type == "atom":
            # Create an atom
            pred = Predicate("P", 1)
            atom = Atom(pred, [Symbol("a")], neural_confidence=0.8)
            return Formula.create_atom(atom)
        elif rand_type == "not":
            # Create a negation
            pred = Predicate("Q", 1)
            atom = Atom(pred, [Symbol("b")], neural_confidence=0.7)
            inner = Formula.create_atom(atom)
            return Formula.create_not(inner, neural_confidence=0.6)
        else:
            # Create a binary formula
            pred1 = Predicate("R", 1)
            atom1 = Atom(pred1, [Symbol("c")], neural_confidence=0.75)
            formula1 = Formula.create_atom(atom1)
            
            pred2 = Predicate("S", 1)
            atom2 = Atom(pred2, [Symbol("d")], neural_confidence=0.65)
            formula2 = Formula.create_atom(atom2)
            
            op = {
                "and": LogicalOperator.AND,
                "or": LogicalOperator.OR, 
                "implies": LogicalOperator.IMPLIES
            }[rand_type]
            
            return Formula.create_binary(op, formula1, formula2, neural_confidence=0.7)
    
    def symbolic_to_neural(self, formula: Formula) -> np.ndarray:
        """
        Convert symbolic formula to neural representation
        
        Args:
            formula: Symbolic formula
            
        Returns:
            Neural embedding
        """
        # Generate formula embedding using encoder LSTM
        formula_tokens = self._tokenize_formula(formula)
        token_embeddings = []
        
        # Get embeddings for each token
        for token in formula_tokens:
            if isinstance(token, Symbol):
                # Use symbol embeddings for symbols
                token_embeddings.append(self.get_symbol_embedding(token))
            else:
                # For operators and other tokens, use a fixed embedding
                token_idx = self._get_or_create_token_idx(str(token))
                with torch.no_grad():
                    token_embeddings.append(self.symbol_embeddings(torch.tensor([token_idx])).numpy()[0])
        
        # Convert to tensor
        token_tensor = torch.tensor(np.stack(token_embeddings), dtype=torch.float32).unsqueeze(0)
        
        # Encode formula
        with torch.no_grad():
            output, (hidden, _) = self.formula_encoder(token_tensor)
            
            # Use bidirectional hidden states
            formula_embedding = torch.cat([hidden[0], hidden[1]], dim=1).squeeze(0).numpy()
            
        return formula_embedding
        
    def _tokenize_formula(self, formula: Formula) -> List[Any]:
        """Tokenize a formula into a sequence of symbols and operators"""
        if formula.formula_type == "atom":
            atom = formula.components[0]
            tokens = [atom.predicate]
            tokens.extend(atom.arguments)
            return tokens
        elif formula.formula_type == "not":
            return ["NOT"] + self._tokenize_formula(formula.components[0])
        else:
            # Binary formula
            left_tokens = self._tokenize_formula(formula.components[0])
            right_tokens = self._tokenize_formula(formula.components[1])
            return left_tokens + [formula.formula_type.upper()] + right_tokens
            
    def _get_or_create_token_idx(self, token_str: str) -> int:
        """Get or create index for a token"""
        if token_str in self.symbol_to_idx:
            return self.symbol_to_idx[token_str]
            
        idx = len(self.symbol_to_idx)
        if idx >= self.vocabulary_size:
            # If vocabulary is full, reuse an existing index
            return idx % self.vocabulary_size
            
        self.symbol_to_idx[token_str] = idx
        self.idx_to_symbol[idx] = token_str
        return idx
    
    def register_symbol(self, symbol: Symbol) -> int:
        """
        Register a symbol in the vocabulary
        
        Args:
            symbol: Symbol to register
            
        Returns:
            Symbol index
        """
        if symbol.name in self.symbol_to_idx:
            return self.symbol_to_idx[symbol.name]
        
        idx = len(self.symbol_to_idx)
        if idx >= self.vocabulary_size:
            raise ValueError(f"Vocabulary size limit ({self.vocabulary_size}) exceeded")
        
        self.symbol_to_idx[symbol.name] = idx
        self.idx_to_symbol[idx] = symbol
        
        return idx
    
    def get_symbol_embedding(self, symbol: Symbol) -> np.ndarray:
        """
        Get neural embedding for a symbol
        
        Args:
            symbol: Symbol to embed
            
        Returns:
            Symbol embedding
        """
        idx = self.register_symbol(symbol)
        with torch.no_grad():
            embedding = self.symbol_embeddings(torch.tensor([idx])).numpy()[0]
        return embedding

# ========================= Metacognition Components =========================

class MetaCognition:
    """Metacognitive capabilities for self-reflection and improvement"""
    
    def __init__(
        self, 
        reasoning_system: 'NeuroSymbolicReasoning',
        confidence_threshold: float = 0.7,
        reflection_depth: int = 3
    ):
        """
        Initialize metacognition system
        
        Args:
            reasoning_system: The reasoning system to monitor
            confidence_threshold: Threshold for reliable conclusions
            reflection_depth: Maximum depth of recursive reflections
        """
        self.reasoning_system = reasoning_system
        self.confidence_threshold = confidence_threshold
        self.reflection_depth = reflection_depth
        
        # History of reasoning attempts
        self.reasoning_history = []
        
        # Self-assessment metrics
        self.metrics = {
            "accuracy": 0.0,
            "confidence": 0.0,
            "consistency": 0.0,
            "depth": 0.0,
            "reasoning_types_used": {},
            "neural_symbolic_balance": 0.5  # 0 = all symbolic, 1 = all neural
        }
        
        # Reasoning improvement strategies
        self.improvement_strategies = {
            "low_accuracy": [
                "increase_neural_weight",
                "gather_more_knowledge",
                "try_alternative_reasoning"
            ],
            "low_confidence": [
                "increase_inference_steps",
                "refine_neural_connections",
                "verify_premises"
            ],
            "low_consistency": [
                "check_contradictions",
                "resolve_ambiguities",
                "standardize_representations"
            ],
            "insufficient_depth": [
                "increase_reflection_depth",
                "add_meta_rules",
                "analyze_reasoning_chains"
            ]
        }
        
    def track_reasoning(self, query: str, result: Any, confidence: float, strategy: ReasoningStrategy, duration: float):
        """
        Track reasoning attempt for metacognition
        
        Args:
            query: The reasoning query
            result: The reasoning result
            confidence: Confidence in the result
            strategy: Reasoning strategy used
            duration: Time taken for reasoning
        """
        entry = {
            "query": query,
            "result": result,
            "confidence": confidence,
            "strategy": strategy,
            "duration": duration,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": self.metrics.copy()
        }
        self.reasoning_history.append(entry)
        
        # Update metrics
        self._update_metrics(entry)
        
    def _update_metrics(self, entry: Dict[str, Any]):
        """Update metacognitive metrics based on reasoning entry"""
        # Update strategy usage
        strategy_name = entry["strategy"].name if isinstance(entry["strategy"], ReasoningStrategy) else str(entry["strategy"])
        self.metrics["reasoning_types_used"][strategy_name] = self.metrics["reasoning_types_used"].get(strategy_name, 0) + 1
        
        # Update confidence metric
        self.metrics["confidence"] = 0.8 * self.metrics["confidence"] + 0.2 * entry["confidence"]
        
        # Update other metrics would use feedback or analysis
        # This is simplified for demonstration
    
    def evaluate_reasoning(self, depth: int = 0) -> Dict[str, Any]:
        """
        Evaluate current reasoning capabilities
        
        Args:
            depth: Current reflection depth
            
        Returns:
            Evaluation results
        """
        if depth >= self.reflection_depth:
            return {"status": "max_depth_reached"}
        
        # Analyze recent reasoning history
        if len(self.reasoning_history) < 5:
            return {"status": "insufficient_history"}
        
        recent_entries = self.reasoning_history[-5:]
        
        # Calculate performance metrics
        avg_confidence = sum(e["confidence"] for e in recent_entries) / len(recent_entries)
        
        # Identify areas for improvement
        improvements = []
        
        if avg_confidence < self.confidence_threshold:
            improvements.extend(self.improvement_strategies["low_confidence"])
        
        # Execute meta-reasoning
        meta_result = {
            "avg_confidence": avg_confidence,
            "improvements": improvements,
            "metrics": self.metrics,
            "status": "completed"
        }
        
        # Recursive meta-reasoning (reasoning about reasoning)
        if depth < self.reflection_depth - 1 and improvements:
            meta_result["meta_evaluation"] = self.evaluate_reasoning(depth + 1)
        
        return meta_result
    
    def improve_reasoning(self) -> Dict[str, Any]:
        """
        Apply improvements to reasoning system
        
        Returns:
            Improvement results
        """
        evaluation = self.evaluate_reasoning()
        
        if evaluation["status"] != "completed":
            return {"status": "cannot_improve", "reason": evaluation["status"]}
        
        improvements = evaluation.get("improvements", [])
        
        if not improvements:
            return {"status": "no_improvements_needed"}
        
        # Apply improvement strategies
        applied = []
        
        for strategy in improvements[:3]:  # Apply at most 3 improvements
            if strategy == "increase_neural_weight":
                # Adjust neural-symbolic balance
                old_balance = self.metrics["neural_symbolic_balance"]
                new_balance = min(old_balance + 0.1, 0.9)
                self.metrics["neural_symbolic_balance"] = new_balance
                self.reasoning_system.neural_weight = new_balance
                applied.append({"strategy": strategy, "change": f"neural_weight: {old_balance:.2f} -> {new_balance:.2f}"})
            
            elif strategy == "increase_inference_steps":
                # Increase inference steps
                old_steps = self.reasoning_system.max_inference_steps
                new_steps = old_steps + 5
                self.reasoning_system.max_inference_steps = new_steps
                applied.append({"strategy": strategy, "change": f"inference_steps: {old_steps} -> {new_steps}"})
            
            elif strategy == "gather_more_knowledge":
                # Request more knowledge from library
                try:
                    # Use knowledge library interface to gather more information
                    from ai_core.knowledge.library import knowledge_library
                    
                    # Search for relevant knowledge based on recent queries
                    recent_queries = [entry["query"] for entry in self.reasoning_history[-5:]]
                    knowledge_library.search_related(recent_queries)
                    
                    applied.append({"strategy": strategy, "change": "requested_more_knowledge"})
                except ImportError:
                    # Handle case where knowledge library is not available
                    applied.append({"strategy": strategy, "change": "knowledge_library_unavailable"})
        
        return {
            "status": "improvements_applied",
            "applied": applied,
            "evaluation": evaluation
        }
    
    def analyze_failure(self, query: str, error: Exception) -> Dict[str, Any]:
        """
        Analyze reasoning failure
        
        Args:
            query: The failed query
            error: The error that occurred
            
        Returns:
            Analysis results
        """
        # Analyze the error and query
        error_type = type(error).__name__
        error_message = str(error)
        
        # Check recent reasoning for patterns
        similar_queries = []
        for entry in self.reasoning_history[-10:]:
            if self._query_similarity(query, entry["query"]) > 0.7:
                similar_queries.append(entry)
        
        # Determine likely cause
        if "recursion" in error_message or "depth" in error_message:
            cause = "excessive_recursion"
            solution = "reduce_reflection_depth"
        elif "timeout" in error_message or "time limit" in error_message:
            cause = "reasoning_timeout"
            solution = "optimize_inference"
        elif "memory" in error_message:
            cause = "memory_error"
            solution = "reduce_knowledge_scope"
        else:
            cause = "unknown_error"
            solution = "general_robustness"
        
        # Create analysis
        analysis = {
            "query": query,
            "error_type": error_type,
            "error_message": error_message,
            "similar_query_count": len(similar_queries),
            "likely_cause": cause,
            "recommended_solution": solution
        }
        
        # Apply immediate fix if possible
        if solution == "reduce_reflection_depth":
            old_depth = self.reflection_depth
            self.reflection_depth = max(1, old_depth - 1)
            analysis["immediate_action"] = f"Reduced reflection depth from {old_depth} to {self.reflection_depth}"
        
        return analysis
    
    def _query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between queries"""
        # Simplified similarity for demonstration
        # Real implementation would use embeddings or more sophisticated comparison
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

# ========================= Main Reasoning System =========================

class NeuroSymbolicReasoning:
    """
    Neuro-Symbolic Reasoning System for Seren
    
    Implements advanced hybrid reasoning combining neural network capabilities
    with symbolic logic for powerful inference, explainability, and reasoning.
    """
    
    def __init__(
        self,
        neural_embedding_size: int = 256,
        neural_weight: float = 0.5,
        max_inference_steps: int = 50,
        knowledge_base_name: str = "Main"
    ):
        """
        Initialize the neuro-symbolic reasoning system
        
        Args:
            neural_embedding_size: Size of neural embeddings
            neural_weight: Weight of neural vs. symbolic (0-1)
            max_inference_steps: Maximum inference steps
            knowledge_base_name: Name of the knowledge base
        """
        self.neural_embedding_size = neural_embedding_size
        self.neural_weight = neural_weight
        self.max_inference_steps = max_inference_steps
        
        # Knowledge base for symbolic reasoning
        self.knowledge_base = KnowledgeBase(knowledge_base_name)
        
        # Neural components
        self.liquid_network = LiquidNeuralNetwork(
            input_size=neural_embedding_size,
            hidden_size=neural_embedding_size * 2,
            output_size=neural_embedding_size,
            num_layers=4
        )
        
        # Initialize tracking attributes for reasoning activity
        self.active_tasks = []
        self.last_reasoning_time = time.time()
        self.continuous_reasoning_active = False
        self.recent_confidences = [0.7]  # Default starting confidence
        
        # Neural-symbolic translator
        self.translator = SymbolicTranslator(embedding_size=neural_embedding_size)
        
        # Metacognition
        self.metacognition = MetaCognition(self, confidence_threshold=0.7)
        
        # Reasoning strategies
        self.strategies = {
            ReasoningStrategy.NEURAL_FIRST: self._reasoning_neural_first,
            ReasoningStrategy.SYMBOLIC_FIRST: self._reasoning_symbolic_first,
            ReasoningStrategy.PARALLEL: self._reasoning_parallel,
            ReasoningStrategy.ITERATIVE: self._reasoning_iterative,
            ReasoningStrategy.ADAPTIVE: self._reasoning_adaptive,
            ReasoningStrategy.HIERARCHICAL: self._reasoning_hierarchical,
            ReasoningStrategy.META_REASONING: self._reasoning_meta
        }
        
        # Default strategy
        self.default_strategy = ReasoningStrategy.ADAPTIVE
        
        # Stats
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "neural_only_queries": 0,
            "symbolic_only_queries": 0,
            "hybrid_queries": 0,
            "strategies_used": {},
            "avg_confidence": 0.0,
            "avg_time": 0.0
        }
        
        # Initialize with basic rules and predicates
        self._initialize_basics()
        
        logger.info(f"NeuroSymbolicReasoning initialized with neural_weight={neural_weight}")
    
    def _initialize_basics(self):
        """Initialize with basic predicates and rules"""
        # Add some basic predicates
        self.knowledge_base.add_predicate(Predicate("equals", 2))
        self.knowledge_base.add_predicate(Predicate("different", 2))
        self.knowledge_base.add_predicate(Predicate("part_of", 2))
        self.knowledge_base.add_predicate(Predicate("contains", 2))
        self.knowledge_base.add_predicate(Predicate("greater_than", 2))
        self.knowledge_base.add_predicate(Predicate("less_than", 2))
        
        # Add symmetry rule for equality
        equals_pred = self.knowledge_base.predicates["equals"]
        x, y = Variable("X"), Variable("Y")
        
        # X equals Y -> Y equals X
        equals_forward = Formula.create_atom(equals_pred.apply(x, y))
        equals_backward = Formula.create_atom(equals_pred.apply(y, x))
        
        symmetry_rule = Rule(
            "equality_symmetry",
            [equals_forward],
            equals_backward,
            RuleType.DEDUCTIVE
        )
        
        self.knowledge_base.add_rule(symmetry_rule)
        
        # Add transitivity rule for equality
        z = Variable("Z")
        equals_xy = Formula.create_atom(equals_pred.apply(x, y))
        equals_yz = Formula.create_atom(equals_pred.apply(y, z))
        equals_xz = Formula.create_atom(equals_pred.apply(x, z))
        
        transitivity_rule = Rule(
            "equality_transitivity",
            [equals_xy, equals_yz],
            equals_xz,
            RuleType.DEDUCTIVE
        )
        
        self.knowledge_base.add_rule(transitivity_rule)
    
    def reason(
        self, 
        query: str, 
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform reasoning on a query
        
        Args:
            query: The reasoning query
            strategy: Reasoning strategy to use
            context: Additional context
            
        Returns:
            Reasoning results
        """
        start_time = time.time()
        
        # Update stats
        self.stats["total_queries"] += 1
        
        # Use default strategy if none specified
        if strategy is None:
            strategy = self.default_strategy
        
        # Update strategy stats
        strategy_name = strategy.name
        self.stats["strategies_used"][strategy_name] = self.stats["strategies_used"].get(strategy_name, 0) + 1
        
        # Parse query to get reasoning task
        reasoning_task = self._parse_query(query, context)
        
        try:
            # Apply the selected reasoning strategy
            if strategy not in self.strategies:
                logger.warning(f"Unknown strategy: {strategy}. Using default.")
                strategy = self.default_strategy
            
            result = self.strategies[strategy](reasoning_task)
            
            # Add confidence
            if "confidence" not in result:
                result["confidence"] = 0.5
            
            # Update successful queries
            self.stats["successful_queries"] += 1
            
            # Update avg confidence
            self.stats["avg_confidence"] = (self.stats["avg_confidence"] * (self.stats["successful_queries"] - 1) + 
                                           result["confidence"]) / self.stats["successful_queries"]
            
            # Update reasoning type counters
            if result.get("used_neural", False) and result.get("used_symbolic", False):
                self.stats["hybrid_queries"] += 1
            elif result.get("used_neural", False):
                self.stats["neural_only_queries"] += 1
            elif result.get("used_symbolic", False):
                self.stats["symbolic_only_queries"] += 1
                
        except Exception as e:
            # Handle reasoning failures
            logger.error(f"Reasoning error: {str(e)}")
            result = {
                "error": str(e),
                "confidence": 0.0,
                "used_neural": False,
                "used_symbolic": False
            }
            
            # Analyze failure
            result["analysis"] = self.metacognition.analyze_failure(query, e)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Update average time
        query_count = self.stats["successful_queries"]
        self.stats["avg_time"] = (self.stats["avg_time"] * (query_count - 1) + duration) / query_count
        
        # Add timing info
        result["duration"] = duration
        
        # Track for metacognition
        self.metacognition.track_reasoning(query, result, result.get("confidence", 0.0), strategy, duration)
        
        return result
    
    def _parse_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse query into a reasoning task
        
        Args:
            query: The reasoning query
            context: Additional context
            
        Returns:
            Reasoning task specification
        """
        # Simple parsing for demonstration
        # Real system would use more sophisticated parsing
        
        # Default context
        if context is None:
            context = {}
        
        # Check for specific reasoning tasks
        if query.lower().startswith("deduce"):
            task_type = "deduction"
        elif query.lower().startswith("explain"):
            task_type = "explanation"
        elif query.lower().startswith("compare"):
            task_type = "comparison"
        elif query.lower().startswith("predict"):
            task_type = "prediction"
        elif query.lower().startswith("solve"):
            task_type = "problem_solving"
        else:
            # Default to general reasoning
            task_type = "general"
        
        # Use knowledge from context if provided
        knowledge = context.get("knowledge", [])
        
        # Attempt to convert to logical form
        logical_form = self._query_to_logical_form(query)
        
        # Create the task
        task = {
            "query": query,
            "task_type": task_type,
            "logical_form": logical_form,
            "context": context,
            "knowledge": knowledge
        }
        
        return task
    
    def _query_to_logical_form(self, query: str) -> Optional[Formula]:
        """
        Convert natural language query to logical form
        
        Args:
            query: Natural language query
            
        Returns:
            Logical formula or None if conversion fails
        """
        # This is a simplified implementation
        # Real system would use a more sophisticated translator
        
        # Simple patterns for demonstration
        if "is equal to" in query or "equals" in query:
            pattern = r"([\w\s]+) (?:is equal to|equals) ([\w\s]+)"
            match = re.search(pattern, query)
            if match:
                left = match.group(1).strip()
                right = match.group(2).strip()
                
                equals_pred = self.knowledge_base.predicates.get("equals")
                if equals_pred:
                    atom = equals_pred.apply(Symbol(left), Symbol(right))
                    return Formula.create_atom(atom)
        
        elif "is part of" in query:
            pattern = r"([\w\s]+) is part of ([\w\s]+)"
            match = re.search(pattern, query)
            if match:
                part = match.group(1).strip()
                whole = match.group(2).strip()
                
                part_of_pred = self.knowledge_base.predicates.get("part_of")
                if part_of_pred:
                    atom = part_of_pred.apply(Symbol(part), Symbol(whole))
                    return Formula.create_atom(atom)
        
        elif "is greater than" in query:
            pattern = r"([\w\s]+) is greater than ([\w\s]+)"
            match = re.search(pattern, query)
            if match:
                greater = match.group(1).strip()
                lesser = match.group(2).strip()
                
                gt_pred = self.knowledge_base.predicates.get("greater_than")
                if gt_pred:
                    atom = gt_pred.apply(Symbol(greater), Symbol(lesser))
                    return Formula.create_atom(atom)
        
        # No pattern matched
        return None
    
    def _reasoning_neural_first(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Neural-first reasoning strategy
        
        Args:
            task: Reasoning task
            
        Returns:
            Reasoning results
        """
        # First try neural reasoning
        neural_result = self._neural_reasoning(task)
        neural_confidence = neural_result.get("confidence", 0.0)
        
        # If neural confidence is high, return the result
        if neural_confidence >= 0.8:
            neural_result["strategy"] = "neural_only"
            neural_result["used_neural"] = True
            neural_result["used_symbolic"] = False
            return neural_result
        
        # Otherwise, try symbolic reasoning
        symbolic_result = self._symbolic_reasoning(task)
        symbolic_confidence = symbolic_result.get("confidence", 0.0)
        
        # If symbolic confidence is higher, use symbolic result
        if symbolic_confidence > neural_confidence:
            symbolic_result["strategy"] = "neural_then_symbolic"
            symbolic_result["used_neural"] = True
            symbolic_result["used_symbolic"] = True
            return symbolic_result
        
        # Otherwise, use neural result
        neural_result["strategy"] = "neural_then_symbolic"
        neural_result["used_neural"] = True
        neural_result["used_symbolic"] = True
        return neural_result
    
    def _reasoning_symbolic_first(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Symbolic-first reasoning strategy
        
        Args:
            task: Reasoning task
            
        Returns:
            Reasoning results
        """
        # First try symbolic reasoning
        symbolic_result = self._symbolic_reasoning(task)
        symbolic_confidence = symbolic_result.get("confidence", 0.0)
        
        # If symbolic confidence is high, return the result
        if symbolic_confidence >= 0.8:
            symbolic_result["strategy"] = "symbolic_only"
            symbolic_result["used_neural"] = False
            symbolic_result["used_symbolic"] = True
            return symbolic_result
        
        # Otherwise, try neural reasoning
        neural_result = self._neural_reasoning(task)
        neural_confidence = neural_result.get("confidence", 0.0)
        
        # If neural confidence is higher, use neural result
        if neural_confidence > symbolic_confidence:
            neural_result["strategy"] = "symbolic_then_neural"
            neural_result["used_neural"] = True
            neural_result["used_symbolic"] = True
            return neural_result
        
        # Otherwise, use symbolic result
        symbolic_result["strategy"] = "symbolic_then_neural"
        symbolic_result["used_neural"] = True
        symbolic_result["used_symbolic"] = True
        return symbolic_result
    
    def _reasoning_parallel(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parallel reasoning strategy
        
        Args:
            task: Reasoning task
            
        Returns:
            Reasoning results
        """
        # Run neural and symbolic reasoning in parallel
        neural_result = self._neural_reasoning(task)
        symbolic_result = self._symbolic_reasoning(task)
        
        neural_confidence = neural_result.get("confidence", 0.0)
        symbolic_confidence = symbolic_result.get("confidence", 0.0)
        
        # Combine the results
        if neural_confidence >= symbolic_confidence:
            # Use neural result as base
            result = neural_result
            
            # Add symbolic reasoning insights
            result["symbolic_insights"] = symbolic_result.get("insights", [])
            result["symbolic_confidence"] = symbolic_confidence
        else:
            # Use symbolic result as base
            result = symbolic_result
            
            # Add neural reasoning insights
            result["neural_insights"] = neural_result.get("insights", [])
            result["neural_confidence"] = neural_confidence
        
        # Combined confidence as weighted average
        result["confidence"] = (neural_confidence * self.neural_weight + 
                               symbolic_confidence * (1 - self.neural_weight))
        
        result["strategy"] = "parallel"
        result["used_neural"] = True
        result["used_symbolic"] = True
        
        return result
    
    def _reasoning_iterative(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Iterative reasoning strategy
        
        Args:
            task: Reasoning task
            
        Returns:
            Reasoning results
        """
        # Start with neural reasoning
        result = self._neural_reasoning(task)
        confidence = result.get("confidence", 0.0)
        
        # Record iterations
        iterations = [{"type": "neural", "confidence": confidence}]
        
        # Iteratively improve
        for i in range(3):  # Maximum 3 iterations
            # Switch to symbolic
            if i % 2 == 0:
                # Use insights from neural to enhance symbolic reasoning
                symbolic_task = task.copy()
                symbolic_task["neural_insights"] = result.get("insights", [])
                
                symbolic_result = self._symbolic_reasoning(symbolic_task)
                symbolic_confidence = symbolic_result.get("confidence", 0.0)
                
                iterations.append({"type": "symbolic", "confidence": symbolic_confidence})
                
                # If confidence improved, update result
                if symbolic_confidence > confidence:
                    result = symbolic_result
                    confidence = symbolic_confidence
                else:
                    # No improvement, stop iterating
                    break
            else:
                # Use insights from symbolic to enhance neural reasoning
                neural_task = task.copy()
                neural_task["symbolic_insights"] = result.get("insights", [])
                
                neural_result = self._neural_reasoning(neural_task)
                neural_confidence = neural_result.get("confidence", 0.0)
                
                iterations.append({"type": "neural", "confidence": neural_confidence})
                
                # If confidence improved, update result
                if neural_confidence > confidence:
                    result = neural_result
                    confidence = neural_confidence
                else:
                    # No improvement, stop iterating
                    break
        
        # Add iteration history
        result["iterations"] = iterations
        result["strategy"] = "iterative"
        result["used_neural"] = True
        result["used_symbolic"] = True
        
        return result
    
    def _reasoning_adaptive(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adaptive reasoning strategy
        
        Args:
            task: Reasoning task
            
        Returns:
            Reasoning results
        """
        # Choose strategy based on task characteristics
        task_type = task.get("task_type", "general")
        
        if task_type == "deduction":
            # Deduction benefits from symbolic reasoning
            return self._reasoning_symbolic_first(task)
        elif task_type == "explanation":
            # Explanation benefits from neural reasoning
            return self._reasoning_neural_first(task)
        elif task_type == "comparison":
            # Comparison benefits from parallel approach
            return self._reasoning_parallel(task)
        elif task_type == "problem_solving":
            # Problem solving benefits from iterative approach
            return self._reasoning_iterative(task)
        elif task_type == "prediction":
            # Prediction benefits from neural approach
            neural_result = self._neural_reasoning(task)
            neural_result["strategy"] = "adaptive_neural"
            neural_result["used_neural"] = True
            neural_result["used_symbolic"] = False
            return neural_result
        else:
            # General reasoning uses hierarchical approach
            return self._reasoning_hierarchical(task)
    
    def _reasoning_hierarchical(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hierarchical reasoning strategy
        
        Args:
            task: Reasoning task
            
        Returns:
            Reasoning results
        """
        # First, try high-level reasoning with symbolics
        abstract_result = self._symbolic_reasoning(task, level="abstract")
        abstract_confidence = abstract_result.get("confidence", 0.0)
        
        # If no good abstract understanding, try neural first
        if abstract_confidence < 0.4:
            return self._reasoning_neural_first(task)
        
        # Use abstract understanding to guide detailed reasoning
        detailed_task = task.copy()
        detailed_task["abstract_insights"] = abstract_result.get("insights", [])
        
        # For detailed reasoning, use neural
        detailed_result = self._neural_reasoning(detailed_task)
        detailed_confidence = detailed_result.get("confidence", 0.0)
        
        # Combine results
        result = detailed_result
        result["abstract_confidence"] = abstract_confidence
        result["strategy"] = "hierarchical"
        result["used_neural"] = True
        result["used_symbolic"] = True
        
        # Weighted confidence
        result["confidence"] = (abstract_confidence * 0.3 + detailed_confidence * 0.7)
        
        return result
    
    def _reasoning_meta(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Meta-reasoning strategy
        
        Args:
            task: Reasoning task
            
        Returns:
            Reasoning results
        """
        # Decide which reasoning strategy to use
        evaluation = self.metacognition.evaluate_reasoning()
        
        # Choose strategy based on past performance
        if evaluation["status"] != "completed":
            # Fall back to adaptive if metacognition not ready
            logger.info("Metacognition not ready, falling back to adaptive strategy")
            return self._reasoning_adaptive(task)
        
        # Check performance metrics
        metrics = evaluation.get("metrics", {})
        
        # Select strategy based on metrics
        if metrics.get("accuracy", 0.0) < 0.6:
            # Low accuracy - use iterative to improve
            logger.info("Using iterative strategy due to low accuracy")
            selected_strategy = self._reasoning_iterative
        elif metrics.get("consistency", 0.0) < 0.6:
            # Low consistency - use symbolic first for more rigorous reasoning
            logger.info("Using symbolic-first strategy due to low consistency")
            selected_strategy = self._reasoning_symbolic_first
        elif metrics.get("depth", 0.0) < 0.5:
            # Insufficient depth - use hierarchical for multi-level reasoning
            logger.info("Using hierarchical strategy due to insufficient depth")
            selected_strategy = self._reasoning_hierarchical
        else:
            # Good overall performance - use parallel for balanced approach
            logger.info("Using parallel strategy due to good overall performance")
            selected_strategy = self._reasoning_parallel
        
        # Apply selected strategy
        result = selected_strategy(task)
        
        # Add meta-reasoning info
        result["meta_strategy_selection"] = {
            "evaluation_metrics": metrics,
            "strategy_selected": selected_strategy.__name__
        }
        
        return result
    
    def _neural_reasoning(self, task: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Perform neural reasoning
        
        Args:
            task: Reasoning task
            **kwargs: Additional parameters
            
        Returns:
            Neural reasoning results
        """
        # Process the query using the liquid neural network
        query = task['query']
        
        # Convert query to embedding
        query_tokens = self._tokenize_query(query)
        query_embedding = self._embed_tokens(query_tokens)
        
        # Process through liquid network
        with torch.no_grad():
            # Reset state for new query
            self.liquid_network.reset_state()
            
            # Forward pass through liquid network
            output_embedding = self.liquid_network(query_embedding)
            
            # Extract features and calculate confidence
            features = output_embedding.numpy()
            # Use magnitude of output as crude confidence measure
            confidence = min(0.9, 0.5 + np.linalg.norm(features) / 10.0)
            
        # Generate insights by analyzing network activations
        insights = self._generate_insights_from_activations(query, features)
        
        # Generate answer using output embedding
        answer = self._generate_answer_from_embedding(query, output_embedding)
        
        return {
            "answer": answer,
            "confidence": float(confidence),  # Convert from numpy to Python float
            "insights": insights,
            "reasoning_type": "neural",
            "embedding": features.tolist()  # Store for potential follow-up reasoning
        }
        
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize a query string into tokens"""
        # Simple whitespace tokenization for demonstration
        return query.split()
        
    def _embed_tokens(self, tokens: List[str]) -> torch.Tensor:
        """Embed a list of tokens into a tensor"""
        # Use a simple embedding approach
        embeddings = []
        for token in tokens:
            # Use hash of token to generate a pseudo-random but consistent embedding
            token_hash = hash(token) % 10000
            idx = token_hash % self.neural_embedding_size
            
            # Create a one-hot-like embedding
            embedding = torch.zeros(self.neural_embedding_size)
            embedding[idx] = 1.0
            embeddings.append(embedding)
            
        # Average the embeddings if we have multiple tokens
        if embeddings:
            return torch.stack(embeddings).mean(dim=0).unsqueeze(0)
        else:
            # Return a zero embedding for empty input
            return torch.zeros(1, self.neural_embedding_size)
            
    def _generate_insights_from_activations(self, query: str, features: np.ndarray) -> List[str]:
        """Generate insights based on neural network activations"""
        # Extract dominant features
        top_indices = np.argsort(np.abs(features))[-3:]  # Top 3 activation indices
        
        insights = []
        for i, idx in enumerate(top_indices):
            magnitude = abs(features.flat[idx])
            sign = "positive" if features.flat[idx] > 0 else "negative"
            insights.append(f"Feature {idx} shows {sign} activation ({magnitude:.2f}) suggesting relevance to the query concept")
            
        # Add query-specific insight
        parts = query.split()
        if len(parts) > 2:
            key_term = parts[1]  # Arbitrarily choose second word as key term
            insights.append(f"Neural analysis indicates strong connection to concept: {key_term}")
            
        return insights
        
    def _generate_answer_from_embedding(self, query: str, embedding: torch.Tensor) -> str:
        """Generate an answer text from the output embedding"""
        # Simplified answer generation
        parts = query.split('?')[0].split() 
        
        if "what" in query.lower() or "how" in query.lower():
            return f"Based on neural analysis, the answer involves {parts[-2]} in relation to {parts[-1]}"
        elif "why" in query.lower():
            return f"Neural reasoning suggests the cause relates to {parts[-1]}"
        else:
            return f"Neural analysis indicates a relationship between {parts[0]} and {parts[-1]} with high confidence"
        
        # Convert query to embedding
        query_embedding = self._query_to_embedding(task["query"])
        
        # Apply liquid neural network
        with torch.no_grad():
            output = self.liquid_network(query_embedding)
            
            # Convert output to answer
            # This is a simplified implementation
            symbolic_formula = self.translator.neural_to_symbolic(output)
            
            # Calculate confidence
            confidence = float(torch.sigmoid(torch.mean(output)).item())
            
            # Generate insights
            insights = [
                f"Neural pattern detected: {symbolic_formula}",
                f"Confidence based on neural activation: {confidence:.2f}"
            ]
            
            # Generate answer
            answer = f"Based on neural reasoning: {symbolic_formula}"
            
            return {
                "answer": answer,
                "symbolic_form": str(symbolic_formula),
                "confidence": confidence,
                "insights": insights,
                "reasoning_type": "neural"
            }
    
    def _symbolic_reasoning(self, task: Dict[str, Any], level: str = "detailed") -> Dict[str, Any]:
        """
        Perform symbolic reasoning
        
        Args:
            task: Reasoning task
            level: Reasoning level (abstract or detailed)
            
        Returns:
            Symbolic reasoning results
        """
        # Add knowledge from task to knowledge base
        added_formulas = []
        for knowledge_item in task.get("knowledge", []):
            if isinstance(knowledge_item, str):
                # Natural language knowledge - try to convert
                formula = self._query_to_logical_form(knowledge_item)
                if formula and self.knowledge_base.add_formula(formula):
                    added_formulas.append(formula)
            elif isinstance(knowledge_item, Formula):
                # Already in logical form
                if self.knowledge_base.add_formula(knowledge_item):
                    added_formulas.append(knowledge_item)
        
        # Try to convert query to logical form
        query_formula = task.get("logical_form")
        if query_formula is None:
            query_formula = self._query_to_logical_form(task["query"])
        
        if query_formula:
            # We have a logical form - perform inference
            max_steps = self.max_inference_steps // 2 if level == "abstract" else self.max_inference_steps
            
            # Perform inference
            self.knowledge_base.inference_until_fixed_point(max_iterations=max_steps)
            
            # Query the knowledge base
            contains, confidence = self.knowledge_base.query(query_formula)
            
            if contains:
                answer = f"Based on logical inference: {query_formula} is true"
                insights = [
                    f"Formula found in knowledge base: {query_formula}",
                    f"Confidence from logic: {confidence:.2f}"
                ]
            else:
                answer = f"Based on logical inference: Cannot determine {query_formula}"
                insights = [
                    f"Formula not found in knowledge base: {query_formula}",
                    f"Performed {max_steps} inference steps"
                ]
                confidence = 0.3  # Low confidence when formula not found
        else:
            # No logical form - return empty result
            answer = f"Could not convert to logical form: {task['query']}"
            insights = ["No logical representation found"]
            confidence = 0.2  # Very low confidence
        
        # Remove added formulas to avoid affecting future queries
        for formula in added_formulas:
            if formula in self.knowledge_base.formulas:
                self.knowledge_base.formulas.remove(formula)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "insights": insights,
            "reasoning_type": "symbolic",
            "level": level
        }
    
    def _query_to_embedding(self, query: str) -> torch.Tensor:
        """
        Convert query to neural embedding
        
        Args:
            query: The query string
            
        Returns:
            Neural embedding
        """
        if not has_torch:
            # Simulation mode
            return torch.rand(1, self.neural_embedding_size)
        
        # Simple embedding model for demonstration
        # Real system would use language model embeddings
        
        # Convert to lowercase and tokenize
        tokens = query.lower().split()
        
        # Create embeddings (simulated)
        embedding = torch.zeros(1, self.neural_embedding_size)
        
        for token in tokens:
            # Simulate word embedding
            token_embedding = torch.randn(1, self.neural_embedding_size)
            embedding += token_embedding
        
        # Normalize
        if torch.norm(embedding) > 0:
            embedding = embedding / torch.norm(embedding)
        
        return embedding
    
    def add_knowledge_from_text(self, text: str, as_formula: bool = False) -> Dict[str, Any]:
        """
        Add knowledge from text
        
        Args:
            text: Text to add as knowledge
            as_formula: Whether to convert to logical formula
            
        Returns:
            Result of knowledge addition
        """
        if as_formula:
            formula = self._query_to_logical_form(text)
            if formula:
                added = self.knowledge_base.add_formula(formula)
                return {
                    "added": added,
                    "formula": str(formula)
                }
            else:
                return {
                    "added": False,
                    "error": "Could not convert to logical formula"
                }
        else:
            # Add as-is to knowledge library if available
            if has_knowledge_lib:
                entry_ids = knowledge_library.add_knowledge_from_text(
                    text=text,
                    source_reference="user_input",
                    metadata={"added_by": "neurosymbolic_reasoning"}
                )
                return {
                    "added": len(entry_ids) > 0,
                    "entry_ids": entry_ids
                }
            else:
                return {
                    "added": False,
                    "error": "Knowledge library not available"
                }
    
    def _retrieve_knowledge(self, query: str, domains: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge relevant to query
        
        Args:
            query: The query to find knowledge for
            domains: Knowledge domains to search in
            
        Returns:
            List of knowledge entries
        """
        if not has_knowledge_lib:
            return []
        
        # Search knowledge library
        entries = knowledge_library.search_knowledge(query, limit=5, categories=domains)
        
        # Convert to dicts
        result = []
        for entry in entries:
            result.append({
                "content": entry.content,
                "source": entry.source_reference,
                "categories": entry.categories if hasattr(entry, "categories") else [],
                "confidence": entry.metadata.get("_search_score", 0.5) if entry.metadata else 0.5
            })
        
        return result
    
    def continuous_reasoning(
        self, 
        query: str,
        max_steps: int = 10,
        domains: List[str] = None,
        strategy: Optional[ReasoningStrategy] = None
    ) -> Dict[str, Any]:
        """
        Perform continuous reasoning
        
        Args:
            query: The initial query
            max_steps: Maximum reasoning steps
            domains: Knowledge domains to use
            strategy: Reasoning strategy
            
        Returns:
            Reasoning results
        """
        # Select strategy
        if strategy is None:
            strategy = ReasoningStrategy.ITERATIVE
        
        # Initialize results
        results = {
            "query": query,
            "final_answer": None,
            "confidence": 0.0,
            "steps": [],
            "domains": domains,
            "strategy": strategy.name
        }
        
        # Retrieve initial knowledge
        knowledge = self._retrieve_knowledge(query, domains)
        
        # Create initial task
        task = {
            "query": query,
            "knowledge": [k["content"] for k in knowledge],
            "step": 0
        }
        
        # Track highest confidence answer
        best_answer = None
        best_confidence = 0.0
        
        # Perform continuous reasoning
        for step in range(max_steps):
            # Update task step
            task["step"] = step
            
            # Perform reasoning
            result = self.reason(query, strategy, context=task)
            
            # Extract the answer
            answer = result.get("answer", "No answer")
            confidence = result.get("confidence", 0.0)
            
            # Track best answer
            if confidence > best_confidence:
                best_answer = answer
                best_confidence = confidence
            
            # Record the step
            step_record = {
                "step": step,
                "answer": answer,
                "confidence": confidence,
                "reasoning_type": result.get("reasoning_type", "unknown"),
                "insights": result.get("insights", [])
            }
            
            results["steps"].append(step_record)
            
            # Check for termination conditions
            if confidence >= 0.95:
                logger.info(f"Continuous reasoning terminated at step {step} with high confidence")
                break
            
            if step > 0 and confidence <= results["steps"][step-1]["confidence"] * 0.9:
                logger.info(f"Continuous reasoning terminated at step {step} due to declining confidence")
                break
            
            # Refine the query based on insights
            insights = result.get("insights", [])
            refined_query = self._refine_query(query, insights, step)
            
            # Update task for next step
            task["query"] = refined_query
            
            # Retrieve additional knowledge based on refined query
            additional_knowledge = self._retrieve_knowledge(refined_query, domains)
            
            # Add new knowledge
            task["knowledge"].extend([k["content"] for k in additional_knowledge 
                                     if k["content"] not in task["knowledge"]])
        
        # Set final answer to best answer
        results["final_answer"] = best_answer
        results["confidence"] = best_confidence
        
        return results
    
    def _refine_query(self, query: str, insights: List[str], step: int) -> str:
        """
        Refine query based on reasoning insights
        
        Args:
            query: Original query
            insights: Reasoning insights
            step: Current reasoning step
            
        Returns:
            Refined query
        """
        # Simple refinement for demonstration
        # Real system would use more sophisticated refinement
        
        if not insights:
            return query
        
        # Use the first insight to refine
        insight = insights[0]
        
        # Extract key terms
        words = re.findall(r'\b\w+\b', insight.lower())
        important_words = [w for w in words if len(w) > 3 and w.lower() not in ["based", "neural", "reasoning", "found", "confidence"]]
        
        if important_words:
            key_term = important_words[0]
            refined = f"{query} considering {key_term}"
            return refined
        
        return query
    
    def evaluate_system(self) -> Dict[str, Any]:
        """
        Evaluate the reasoning system
        
        Returns:
            Evaluation results
        """
        # Perform metacognitive evaluation
        meta_eval = self.metacognition.evaluate_reasoning()
        
        # Add system statistics
        eval_results = {
            "metacognition": meta_eval,
            "stats": self.stats,
            "neural_weight": self.neural_weight,
            "max_inference_steps": self.max_inference_steps,
            "knowledge_base_size": len(self.knowledge_base.formulas),
            "rule_count": len(self.knowledge_base.rules),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return eval_results
    
    def improve_system(self) -> Dict[str, Any]:
        """
        Apply system improvements
        
        Returns:
            Improvement results
        """
        # Apply metacognitive improvements
        improvement_result = self.metacognition.improve_reasoning()
        
        # Optimize neural components
        if self.liquid_network is not None:
            # Perform real optimization of network parameters
            self.liquid_network.optimize_parameters()
            
            # Record optimization results
            improvement_result["neural_optimization"] = {
                "status": "completed",
                "message": "Liquid network parameters optimized",
                "timestamp": time.time()
            }
        
        return improvement_result
    
    def is_reasoning_active(self) -> bool:
        """
        Check if reasoning processes are currently active
        
        Returns:
            True if reasoning processes are active, False otherwise
        """
        # Check if we have active reasoning tasks
        if self.active_tasks:
            return True
            
        # Check if reasoning has been used recently (within last 10 seconds)
        current_time = time.time()
        if current_time - self.last_reasoning_time < 10:
            return True
            
        # Check if continuous reasoning is active
        if self.continuous_reasoning_active:
            return True
            
        return False
    
    def get_confidence_score(self) -> float:
        """
        Get the current confidence score of the reasoning system
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self.recent_confidences:
            return 0.7  # Default confidence
            
        # Use exponentially weighted moving average
        weights = [0.6 ** i for i in range(len(self.recent_confidences))]
        weighted_sum = sum(w * c for w, c in zip(weights, self.recent_confidences))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum
        
    def update_confidence(self, new_confidence):
        """
        Update the confidence tracking with a new confidence score
        
        Args:
            new_confidence: New confidence score to add to tracking
        """
        # Keep only the 10 most recent confidence scores
        if len(self.recent_confidences) >= 10:
            self.recent_confidences.pop()
        
        # Add new confidence at the beginning
        self.recent_confidences.insert(0, new_confidence)
        
    def start_reasoning_task(self, task_id, query):
        """
        Mark the start of a reasoning task
        
        Args:
            task_id: Unique ID for the task
            query: The reasoning query
        """
        self.active_tasks.append({
            "id": task_id, 
            "query": query,
            "start_time": time.time()
        })
        self.last_reasoning_time = time.time()
        
    def end_reasoning_task(self, task_id, confidence=None):
        """
        Mark the end of a reasoning task
        
        Args:
            task_id: Unique ID for the task
            confidence: Final confidence score for the task
        """
        # Remove task from active tasks
        self.active_tasks = [t for t in self.active_tasks if t["id"] != task_id]
        self.last_reasoning_time = time.time()
        
        # Update confidence if provided
        if confidence is not None:
            self.update_confidence(confidence)

# Initialize the reasoning system
neurosymbolic_reasoning = NeuroSymbolicReasoning(
    neural_embedding_size=256,
    neural_weight=0.5,
    max_inference_steps=50
)