"""
Neuro-Symbolic Reasoning Engine

Combines neural network capabilities with symbolic reasoning to enable
more robust, interpretable, and logically sound AI reasoning.

This is a critical component that elevates the system beyond traditional LLMs
by integrating explicit symbolic reasoning with neural network capabilities.
"""

import os
import sys
import json
import logging
import time
import re
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from datetime import datetime
import enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ReasoningStrategy(str, enum.Enum):
    """Strategies for neuro-symbolic reasoning"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    BAYESIAN = "bayesian"
    COUNTERFACTUAL = "counterfactual"

class SymbolicOperator(str, enum.Enum):
    """Symbolic logic operators"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    EQUIVALENT = "equivalent"
    EXISTS = "exists"
    FORALL = "forall"

class NeuroSymbolicEngine:
    """
    Neuro-Symbolic Reasoning Engine
    
    Combines neural network-based reasoning with symbolic logic
    to achieve more robust, interpretable, and logical reasoning.
    
    Key capabilities:
    1. Extract symbolic knowledge from neural outputs
    2. Perform logical reasoning over symbolic knowledge
    3. Verify consistency and detect contradictions
    4. Generate explanations for reasoning steps
    5. Incorporate domain-specific reasoning rules
    """
    
    def __init__(self):
        """Initialize the reasoning engine"""
        # Knowledge base of facts and rules
        self.knowledge_base = {
            "facts": [],
            "rules": [],
            "ontology": {},
            "constraints": []
        }
        
        # Cache for reasoning results
        self.reasoning_cache = {}
        
        # Domain-specific reasoning components
        self.domain_reasoners = {}
        
        # Load built-in reasoning resources
        self._load_reasoning_resources()
        
        logger.info("Neuro-Symbolic Reasoning Engine initialized")
    
    def _load_reasoning_resources(self):
        """Load built-in reasoning resources"""
        # In a real implementation, this would load resources from files
        # For now, we initialize with some basic rules
        
        # Basic logical rules
        self.knowledge_base["rules"].extend([
            {
                "name": "modus_ponens",
                "description": "If P implies Q, and P is true, then Q is true",
                "formal": "P → Q, P ⊢ Q",
                "type": "deductive"
            },
            {
                "name": "modus_tollens",
                "description": "If P implies Q, and Q is false, then P is false",
                "formal": "P → Q, ¬Q ⊢ ¬P",
                "type": "deductive"
            },
            {
                "name": "conjunction_introduction",
                "description": "If P is true and Q is true, then P and Q is true",
                "formal": "P, Q ⊢ P ∧ Q",
                "type": "deductive"
            },
            {
                "name": "disjunction_introduction",
                "description": "If P is true, then P or Q is true",
                "formal": "P ⊢ P ∨ Q",
                "type": "deductive"
            }
        ])
        
        # Basic reasoning patterns
        self.knowledge_base["ontology"] = {
            "entity_types": ["person", "organization", "location", "concept", "event", "object"],
            "relation_types": ["is_a", "part_of", "located_in", "created_by", "depends_on", "causes"],
            "property_types": ["attribute", "state", "measurement", "capability"]
        }
        
        # Load domain-specific reasoners
        self._load_domain_reasoners()
    
    def _load_domain_reasoners(self):
        """Load domain-specific reasoning components"""
        # In a real implementation, these would be more complex
        self.domain_reasoners = {
            "programming": {
                "rules": [
                    "If a function has side effects, it is not pure",
                    "If a value is immutable, it cannot be changed after creation",
                    "If code uses recursion without a base case, it may cause a stack overflow"
                ],
                "concepts": ["function", "variable", "class", "module", "algorithm", "data structure"],
                "relations": ["implements", "extends", "imports", "calls", "depends_on"]
            },
            "mathematics": {
                "rules": [
                    "If A equals B and B equals C, then A equals C",
                    "If A is less than B and B is less than C, then A is less than C",
                    "If a number is divisible by both X and Y, it is divisible by their least common multiple"
                ],
                "concepts": ["number", "set", "function", "equation", "inequality", "proof"],
                "relations": ["equals", "less_than", "greater_than", "element_of", "subset_of"]
            },
            "design": {
                "rules": [
                    "If a design increases cognitive load, it may reduce usability",
                    "If users cannot find a feature, the feature is effectively absent",
                    "If a design lacks consistency, it may confuse users"
                ],
                "concepts": ["user", "interface", "interaction", "feedback", "affordance", "constraint"],
                "relations": ["uses", "interacts_with", "perceives", "understands", "prefers"]
            }
        }
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[ReasoningStrategy] = None
    ) -> Dict[str, Any]:
        """
        Apply neuro-symbolic reasoning to a query
        
        Args:
            query: The question or statement to reason about
            context: Additional context information
            strategy: Specific reasoning strategy to employ
            
        Returns:
            Reasoning results including path, explanations, and conclusion
        """
        logger.info(f"Reasoning about: {query[:100]}...")
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Initialize context if not provided
        context = context or {}
        
        # Check cache for identical queries
        cache_key = f"{query}_{json.dumps(context)}_{strategy}"
        if cache_key in self.reasoning_cache:
            logger.info("Returning cached reasoning result")
            return self.reasoning_cache[cache_key]
        
        # Determine the appropriate reasoning strategies
        strategies = self._determine_reasoning_strategies(query, context, strategy)
        logger.info(f"Using reasoning strategies: {strategies}")
        
        # Extract key concepts and relations from the query
        concepts, relations = self._extract_concepts_and_relations(query, context)
        logger.info(f"Extracted {len(concepts)} concepts and {len(relations)} relations")
        
        # Identify relevant domain knowledge
        domains = self._identify_relevant_domains(concepts, relations)
        logger.info(f"Identified relevant domains: {domains}")
        
        # Extract known facts from context
        facts = self._extract_facts(query, context)
        logger.info(f"Extracted {len(facts)} facts from context")
        
        # Generate reasoning path
        reasoning_path = []
        for strategy_name in strategies:
            strategy_result = self._apply_reasoning_strategy(
                strategy_name, query, concepts, relations, facts, domains, context
            )
            reasoning_path.extend(strategy_result)
        
        # Extract conclusion from reasoning path
        conclusion = self._extract_conclusion(reasoning_path)
        
        # Verify logical consistency
        consistency_result = self._verify_consistency(reasoning_path, conclusion)
        
        # Prepare result
        result = {
            "query": query,
            "reasoning_path": reasoning_path,
            "conclusion": conclusion,
            "concepts": concepts,
            "relations": relations,
            "domains": domains,
            "consistency": consistency_result,
            "elapsed_time": time.time() - start_time
        }
        
        # Cache the result
        self.reasoning_cache[cache_key] = result
        
        # If cache is too large, remove oldest entries
        if len(self.reasoning_cache) > 100:
            # Remove 20% of the oldest entries
            keys_to_remove = sorted(self.reasoning_cache.keys())[:20]
            for key in keys_to_remove:
                del self.reasoning_cache[key]
        
        return result
    
    def _determine_reasoning_strategies(
        self,
        query: str,
        context: Dict[str, Any],
        requested_strategy: Optional[ReasoningStrategy] = None
    ) -> List[ReasoningStrategy]:
        """Determine appropriate reasoning strategies for the query"""
        # If a specific strategy is requested, use it
        if requested_strategy:
            return [requested_strategy]
        
        strategies = []
        
        # Analyze query to determine appropriate strategies
        lower_query = query.lower()
        
        # Check for strategy indicators in the query
        if ("why" in lower_query or "because" in lower_query or "reason" in lower_query):
            strategies.append(ReasoningStrategy.CAUSAL)
        
        if ("if" in lower_query and "then" in lower_query) or ("implies" in lower_query):
            strategies.append(ReasoningStrategy.DEDUCTIVE)
        
        if ("like" in lower_query or "similar" in lower_query or "analogy" in lower_query):
            strategies.append(ReasoningStrategy.ANALOGICAL)
        
        if ("could" in lower_query or "might" in lower_query or "probability" in lower_query):
            strategies.append(ReasoningStrategy.BAYESIAN)
        
        if ("what if" in lower_query or "would" in lower_query):
            strategies.append(ReasoningStrategy.COUNTERFACTUAL)
        
        if ("observed" in lower_query or "examples" in lower_query or "pattern" in lower_query):
            strategies.append(ReasoningStrategy.INDUCTIVE)
        
        if ("best explanation" in lower_query or "hypothesis" in lower_query):
            strategies.append(ReasoningStrategy.ABDUCTIVE)
        
        # Default to deductive reasoning if no clear indicators
        if not strategies:
            strategies.append(ReasoningStrategy.DEDUCTIVE)
        
        return strategies
    
    def _extract_concepts_and_relations(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract key concepts and relations from the query"""
        # In a real implementation, this would use more sophisticated NLP
        # For now, we use a simplified approach
        
        concepts = []
        relations = []
        
        # Simple concept extraction - look for nouns and noun phrases
        words = re.findall(r'\b\w+\b', query)
        for word in words:
            word = word.lower()
            
            # Filter out common stop words
            if word in ["the", "a", "an", "in", "on", "at", "to", "for", "by", "with"]:
                continue
            
            # Check if word is in our ontology
            for entity_type in self.knowledge_base["ontology"]["entity_types"]:
                # This is very simplified - in reality, this would be more sophisticated
                if self._is_concept_of_type(word, entity_type):
                    concept = {
                        "name": word,
                        "type": entity_type,
                        "source": "query"
                    }
                    
                    # Avoid duplicates
                    if not any(c["name"] == word for c in concepts):
                        concepts.append(concept)
                    break
        
        # Simple relation extraction - look for relation patterns
        for relation_type in self.knowledge_base["ontology"]["relation_types"]:
            relation_patterns = self._get_relation_patterns(relation_type)
            for pattern in relation_patterns:
                matches = re.finditer(pattern, query.lower())
                for match in matches:
                    if match and len(match.groups()) >= 2:
                        subject = match.group(1)
                        object = match.group(2)
                        
                        relation = {
                            "type": relation_type,
                            "subject": subject,
                            "object": object,
                            "source": "query"
                        }
                        
                        relations.append(relation)
        
        # Enhance with concepts from context if available
        if context.get("concepts"):
            for concept in context["concepts"]:
                if isinstance(concept, dict) and "name" in concept and "type" in concept:
                    # Avoid duplicates
                    if not any(c["name"] == concept["name"] for c in concepts):
                        concept["source"] = "context"
                        concepts.append(concept)
        
        # Enhance with relations from context if available
        if context.get("relations"):
            for relation in context["relations"]:
                if isinstance(relation, dict) and "subject" in relation and "object" in relation:
                    relation["source"] = "context"
                    relations.append(relation)
        
        return concepts, relations
    
    def _is_concept_of_type(self, word: str, entity_type: str) -> bool:
        """Check if a word represents a concept of the given type"""
        # This is a placeholder - in a real system, this would use more sophisticated methods
        
        # Some simple rules for demonstration
        if entity_type == "person" and word in ["user", "developer", "designer", "person"]:
            return True
        
        if entity_type == "concept" and word in ["code", "design", "architecture", "pattern", "algorithm"]:
            return True
        
        if entity_type == "object" and word in ["system", "application", "database", "interface"]:
            return True
        
        # Default to True with low probability to allow exploration
        return False
    
    def _get_relation_patterns(self, relation_type: str) -> List[str]:
        """Get regex patterns for extracting relations of the given type"""
        # This is a placeholder - in a real system, this would be more sophisticated
        
        patterns = []
        
        if relation_type == "is_a":
            patterns = [
                r"(\w+) is an? (\w+)",
                r"(\w+) are (\w+)",
                r"(\w+) as an? (\w+)"
            ]
        
        elif relation_type == "part_of":
            patterns = [
                r"(\w+) is part of (\w+)",
                r"(\w+) belongs to (\w+)",
                r"(\w+) in (\w+)",
                r"(\w+) of the (\w+)"
            ]
        
        elif relation_type == "depends_on":
            patterns = [
                r"(\w+) depends on (\w+)",
                r"(\w+) requires (\w+)",
                r"(\w+) needs (\w+)"
            ]
        
        elif relation_type == "causes":
            patterns = [
                r"(\w+) causes (\w+)",
                r"(\w+) leads to (\w+)",
                r"(\w+) results in (\w+)",
                r"if (\w+) then (\w+)"
            ]
        
        return patterns
    
    def _identify_relevant_domains(
        self,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify relevant knowledge domains based on concepts and relations"""
        domain_scores = {domain: 0 for domain in self.domain_reasoners.keys()}
        
        # Score each domain based on concept and relation overlap
        for concept in concepts:
            concept_name = concept["name"].lower()
            
            for domain, reasoner in self.domain_reasoners.items():
                # Check if concept matches domain concepts
                if concept_name in reasoner["concepts"]:
                    domain_scores[domain] += 2
                elif any(domain_concept in concept_name for domain_concept in reasoner["concepts"]):
                    domain_scores[domain] += 1
        
        for relation in relations:
            relation_type = relation["type"].lower()
            
            for domain, reasoner in self.domain_reasoners.items():
                # Check if relation matches domain relations
                if relation_type in reasoner["relations"]:
                    domain_scores[domain] += 2
        
        # Select domains above a threshold score
        relevant_domains = [
            domain for domain, score in domain_scores.items()
            if score >= 2  # Arbitrary threshold
        ]
        
        # If no domains score high enough, use the highest scoring domain
        if not relevant_domains and domain_scores:
            max_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            relevant_domains = [max_domain]
        
        return relevant_domains
    
    def _extract_facts(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract known facts from the query and context"""
        facts = []
        
        # Extract facts from explicit statements in the query
        # This is simplified - a real system would use more sophisticated NLP
        fact_patterns = [
            r"(\w+) is (\w+)",
            r"(\w+) are (\w+)",
            r"(\w+) has (\w+)",
            r"(\w+) have (\w+)",
            r"(\w+) contains (\w+)",
            r"all (\w+) are (\w+)",
            r"no (\w+) are (\w+)",
            r"if (\w+) then (\w+)"
        ]
        
        for pattern in fact_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                if match and len(match.groups()) >= 2:
                    subject = match.group(1)
                    predicate = match.group(2)
                    
                    fact = {
                        "subject": subject,
                        "predicate": predicate,
                        "source": "query",
                        "confidence": 0.8  # Arbitrary confidence for query-derived facts
                    }
                    
                    facts.append(fact)
        
        # Extract facts from context if available
        if context.get("facts"):
            for fact in context["facts"]:
                if isinstance(fact, dict) and "subject" in fact and "predicate" in fact:
                    fact["source"] = "context"
                    fact["confidence"] = fact.get("confidence", 0.9)  # Higher confidence for explicit facts
                    facts.append(fact)
        
        # Add facts from relevant domain knowledge
        if context.get("domains"):
            for domain in context["domains"]:
                if domain in self.domain_reasoners:
                    for rule in self.domain_reasoners[domain]["rules"]:
                        # Simplified rule parsing - in reality, this would be more sophisticated
                        if "if" in rule.lower() and "then" in rule.lower():
                            parts = rule.lower().split("then")
                            if len(parts) == 2:
                                condition = parts[0].replace("if", "").strip()
                                result = parts[1].strip()
                                
                                fact = {
                                    "type": "implication",
                                    "condition": condition,
                                    "result": result,
                                    "source": f"domain:{domain}",
                                    "confidence": 0.95  # High confidence for domain knowledge
                                }
                                
                                facts.append(fact)
        
        return facts
    
    def _apply_reasoning_strategy(
        self,
        strategy: ReasoningStrategy,
        query: str,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        domains: List[str],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply a specific reasoning strategy"""
        logger.info(f"Applying {strategy} reasoning")
        
        # Initialize reasoning path
        reasoning_path = []
        
        # Apply strategy-specific reasoning
        if strategy == ReasoningStrategy.DEDUCTIVE:
            reasoning_path = self._apply_deductive_reasoning(query, concepts, relations, facts, domains)
        
        elif strategy == ReasoningStrategy.INDUCTIVE:
            reasoning_path = self._apply_inductive_reasoning(query, concepts, relations, facts, domains, context)
        
        elif strategy == ReasoningStrategy.ABDUCTIVE:
            reasoning_path = self._apply_abductive_reasoning(query, concepts, relations, facts, domains)
        
        elif strategy == ReasoningStrategy.ANALOGICAL:
            reasoning_path = self._apply_analogical_reasoning(query, concepts, relations, facts, domains, context)
        
        elif strategy == ReasoningStrategy.CAUSAL:
            reasoning_path = self._apply_causal_reasoning(query, concepts, relations, facts, domains)
        
        elif strategy == ReasoningStrategy.BAYESIAN:
            reasoning_path = self._apply_bayesian_reasoning(query, concepts, relations, facts, domains, context)
        
        elif strategy == ReasoningStrategy.COUNTERFACTUAL:
            reasoning_path = self._apply_counterfactual_reasoning(query, concepts, relations, facts, domains, context)
        
        # Add strategy information to each step
        for step in reasoning_path:
            step["strategy"] = strategy
        
        return reasoning_path
    
    def _apply_deductive_reasoning(
        self,
        query: str,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply deductive reasoning (from general principles to specific conclusions)"""
        # Simplified implementation - in reality, this would be much more sophisticated
        
        reasoning_path = []
        
        # Extract general principles (rules)
        principles = []
        for fact in facts:
            if fact.get("type") == "implication":
                principles.append(fact)
        
        # Add domain-specific principles
        for domain in domains:
            if domain in self.domain_reasoners:
                for rule_text in self.domain_reasoners[domain]["rules"]:
                    if "if" in rule_text.lower() and "then" in rule_text.lower():
                        principle = {
                            "type": "implication",
                            "text": rule_text,
                            "source": f"domain:{domain}",
                            "confidence": 0.9
                        }
                        principles.append(principle)
        
        # Select relevant principles based on concepts
        concept_names = [c["name"].lower() for c in concepts]
        relevant_principles = []
        
        for principle in principles:
            # Check if the principle mentions any of the concepts
            if principle.get("text"):
                principle_text = principle["text"].lower()
                if any(concept in principle_text for concept in concept_names):
                    relevant_principles.append(principle)
                    
                    # Add as first reasoning step
                    reasoning_path.append({
                        "type": "principle",
                        "content": principle["text"],
                        "explanation": f"Applying general principle from {principle['source']}",
                        "confidence": principle.get("confidence", 0.8)
                    })
        
        # Apply principles to derive conclusions
        if relevant_principles:
            # For each relevant principle, check if conditions are met
            for principle in relevant_principles:
                principle_text = principle.get("text", "")
                
                if "if" in principle_text.lower() and "then" in principle_text.lower():
                    parts = principle_text.lower().split("then")
                    if len(parts) == 2:
                        condition = parts[0].replace("if", "").strip()
                        result = parts[1].strip()
                        
                        # Check if condition is met (simplified)
                        condition_met = False
                        for fact in facts:
                            fact_text = ""
                            if "subject" in fact and "predicate" in fact:
                                fact_text = f"{fact['subject']} {fact['predicate']}"
                            elif "text" in fact:
                                fact_text = fact["text"]
                                
                            if condition.lower() in fact_text.lower():
                                condition_met = True
                                
                                # Add reasoning step showing condition is met
                                reasoning_path.append({
                                    "type": "condition_match",
                                    "content": f"Condition '{condition}' is met based on established fact: '{fact_text}'",
                                    "explanation": "Verifying that the condition in the principle is satisfied",
                                    "confidence": fact.get("confidence", 0.7)
                                })
                                break
                        
                        # If condition is met, derive the conclusion
                        if condition_met:
                            reasoning_path.append({
                                "type": "conclusion",
                                "content": f"Therefore, {result}",
                                "explanation": "Deriving conclusion by applying the principle (modus ponens)",
                                "confidence": 0.85  # Confidence in deductive step
                            })
        
        # If no specific principles apply, use general logical rules
        if not reasoning_path:
            # Try to apply basic logical rules
            reasoning_path.append({
                "type": "observation",
                "content": "No specific principles found for deductive reasoning",
                "explanation": "Moving to general logical analysis",
                "confidence": 0.7
            })
            
            # Apply general logic (simplified)
            for relation in relations:
                reasoning_path.append({
                    "type": "analysis",
                    "content": f"Relationship identified: {relation['subject']} {relation['type']} {relation['object']}",
                    "explanation": "Extracting relationships for logical analysis",
                    "confidence": 0.7
                })
        
        return reasoning_path
    
    def _apply_inductive_reasoning(
        self,
        query: str,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        domains: List[str],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply inductive reasoning (from specific observations to general patterns)"""
        # Simplified implementation
        reasoning_path = []
        
        # Check for examples or instances in facts and context
        examples = []
        for fact in facts:
            # This is a simple heuristic - in reality, would be more sophisticated
            if "example" in str(fact).lower() or "instance" in str(fact).lower():
                examples.append(fact)
        
        # Extract examples from context if available
        if context.get("examples"):
            examples.extend(context["examples"])
        
        # If no examples found, note this limitation
        if not examples:
            reasoning_path.append({
                "type": "limitation",
                "content": "Insufficient examples for strong inductive reasoning",
                "explanation": "Inductive reasoning requires multiple examples or instances to identify patterns",
                "confidence": 0.4
            })
            return reasoning_path
        
        # Analyze examples to identify patterns
        reasoning_path.append({
            "type": "observation",
            "content": f"Analyzing {len(examples)} examples for patterns",
            "explanation": "Gathering specific instances for inductive analysis",
            "confidence": 0.8
        })
        
        # Identify common attributes or patterns (simplified)
        common_attributes = {}
        
        for example in examples:
            # Extract attributes from example
            attributes = {}
            
            if isinstance(example, dict):
                attributes = {k: v for k, v in example.items() if k not in ["source", "confidence"]}
            
            # Update common attributes
            for attr, value in attributes.items():
                if attr not in common_attributes:
                    common_attributes[attr] = []
                common_attributes[attr].append(value)
        
        # Identify attributes that appear consistently
        patterns = []
        for attr, values in common_attributes.items():
            # Check if attribute has consistent values
            unique_values = set(str(v) for v in values if v is not None)
            
            if len(unique_values) == 1 and len(values) >= 2:
                # All examples have the same value for this attribute
                patterns.append({
                    "attribute": attr,
                    "value": next(iter(unique_values)),
                    "consistency": 1.0,
                    "support": len(values)
                })
            elif len(unique_values) / len(values) < 0.5:
                # Most examples have similar values
                most_common = max(unique_values, key=lambda x: values.count(x))
                patterns.append({
                    "attribute": attr,
                    "value": most_common,
                    "consistency": values.count(most_common) / len(values),
                    "support": values.count(most_common)
                })
        
        # Add pattern identification to reasoning path
        for pattern in patterns:
            reasoning_path.append({
                "type": "pattern",
                "content": f"Pattern identified: {pattern['attribute']} is typically {pattern['value']}",
                "explanation": f"Found in {pattern['support']}/{len(examples)} examples ({pattern['consistency']*100:.1f}% consistent)",
                "confidence": pattern['consistency'] * 0.9  # Confidence based on consistency
            })
        
        # Generate inductive conclusion if patterns found
        if patterns:
            conclusion_parts = []
            for pattern in patterns:
                if pattern['consistency'] > 0.7:  # Only use strong patterns
                    conclusion_parts.append(f"{pattern['attribute']} is typically {pattern['value']}")
            
            if conclusion_parts:
                conclusion = "; ".join(conclusion_parts)
                
                reasoning_path.append({
                    "type": "conclusion",
                    "content": f"Inductive conclusion: {conclusion}",
                    "explanation": "Generalizing from specific examples to a broader pattern",
                    "confidence": 0.7  # Inductive conclusions have lower confidence than deductive ones
                })
        else:
            reasoning_path.append({
                "type": "conclusion",
                "content": "No clear patterns identified from the examples",
                "explanation": "Insufficient consistency in the available examples",
                "confidence": 0.4
            })
        
        return reasoning_path
    
    def _apply_abductive_reasoning(
        self,
        query: str,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply abductive reasoning (inference to the best explanation)"""
        # Simplified implementation
        reasoning_path = []
        
        # Identify observations that need explanation
        observations = []
        for fact in facts:
            # Simplistic approach - in reality would be more sophisticated
            if not fact.get("type") == "implication" and fact.get("confidence", 0) > 0.7:
                observations.append(fact)
        
        if not observations:
            reasoning_path.append({
                "type": "limitation",
                "content": "No clear observations to explain with abductive reasoning",
                "explanation": "Abductive reasoning requires observations to generate explanatory hypotheses",
                "confidence": 0.4
            })
            return reasoning_path
        
        # List observations
        observation_descriptions = []
        for obs in observations:
            if "subject" in obs and "predicate" in obs:
                observation_descriptions.append(f"{obs['subject']} {obs['predicate']}")
            elif "text" in obs:
                observation_descriptions.append(obs["text"])
            else:
                observation_descriptions.append(str(obs))
        
        reasoning_path.append({
            "type": "observation",
            "content": f"Observations requiring explanation: {'; '.join(observation_descriptions[:3])}",
            "explanation": "Identifying key observations for abductive reasoning",
            "confidence": 0.8
        })
        
        # Generate potential explanations from domain knowledge
        explanations = []
        
        for domain in domains:
            if domain in self.domain_reasoners:
                # Generate domain-specific explanations
                domain_explanations = self._generate_domain_explanations(
                    domain, observations, concepts, relations
                )
                
                for explanation in domain_explanations:
                    explanations.append({
                        "content": explanation["content"],
                        "domain": domain,
                        "coverage": explanation["coverage"],  # How many observations it explains
                        "complexity": explanation["complexity"],  # Simplicity principle
                        "score": explanation["score"]  # Overall score
                    })
        
        # Sort explanations by score
        explanations.sort(key=lambda x: x["score"], reverse=True)
        
        # Add top explanations to reasoning path
        for i, explanation in enumerate(explanations[:3]):  # Consider top 3 explanations
            reasoning_path.append({
                "type": "hypothesis",
                "content": explanation["content"],
                "explanation": f"Potential explanation from {explanation['domain']} domain (explains {explanation['coverage']*100:.0f}% of observations)",
                "confidence": 0.7 * explanation["score"]  # Confidence based on explanation score
            })
        
        # Select best explanation
        if explanations:
            best_explanation = explanations[0]
            
            reasoning_path.append({
                "type": "conclusion",
                "content": f"Best explanation: {best_explanation['content']}",
                "explanation": "Selected based on explanatory power, simplicity, and domain relevance",
                "confidence": 0.7 * best_explanation["score"]  # Abductive conclusions have moderate confidence
            })
        else:
            reasoning_path.append({
                "type": "conclusion",
                "content": "No clear explanation found for the observations",
                "explanation": "Insufficient domain knowledge to generate explanatory hypotheses",
                "confidence": 0.3
            })
        
        return reasoning_path
    
    def _generate_domain_explanations(
        self,
        domain: str,
        observations: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate domain-specific explanations for observations"""
        # Simplified implementation
        explanations = []
        
        # Get domain rules
        domain_rules = self.domain_reasoners[domain]["rules"]
        
        # For each rule, check if it could explain the observations
        for rule in domain_rules:
            # Extract cause and effect from rule (simplified parsing)
            cause = ""
            effect = ""
            
            if "if" in rule.lower() and "then" in rule.lower():
                parts = rule.lower().split("then")
                if len(parts) == 2:
                    cause = parts[0].replace("if", "").strip()
                    effect = parts[1].strip()
            
            # Check if rule effect matches any observations
            matching_observations = 0
            for obs in observations:
                obs_text = ""
                if "subject" in obs and "predicate" in obs:
                    obs_text = f"{obs['subject']} {obs['predicate']}".lower()
                elif "text" in obs:
                    obs_text = obs["text"].lower()
                else:
                    obs_text = str(obs).lower()
                
                # Simple string matching - in reality, would use more sophisticated semantic matching
                if effect and effect.lower() in obs_text:
                    matching_observations += 1
            
            # If rule explains some observations, add it as potential explanation
            if matching_observations > 0:
                coverage = matching_observations / len(observations)
                complexity = 0.5  # Default complexity
                
                # Calculate complexity (prefer simpler explanations)
                if len(rule.split()) < 10:
                    complexity = 0.3  # Simpler
                elif len(rule.split()) > 20:
                    complexity = 0.7  # More complex
                
                # Calculate overall score (higher is better)
                score = coverage * (1 - complexity)
                
                explanations.append({
                    "content": f"Because {cause}, therefore {effect}",
                    "rule": rule,
                    "coverage": coverage,
                    "complexity": complexity,
                    "score": score
                })
        
        # Sort by score
        explanations.sort(key=lambda x: x["score"], reverse=True)
        
        return explanations
    
    def _apply_analogical_reasoning(
        self,
        query: str,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        domains: List[str],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply analogical reasoning (drawing parallels between situations)"""
        # Simplified implementation
        reasoning_path = []
        
        # Extract source domain (what we're comparing from)
        source_domain = None
        source_concepts = []
        
        # Try to identify if query explicitly mentions analogy
        analogy_pattern = r"how is ([\w\s]+) like ([\w\s]+)"
        analogy_match = re.search(analogy_pattern, query.lower())
        
        if analogy_match and len(analogy_match.groups()) >= 2:
            source_domain = analogy_match.group(2)
            target_domain = analogy_match.group(1)
            
            reasoning_path.append({
                "type": "identification",
                "content": f"Analogy between '{target_domain}' and '{source_domain}'",
                "explanation": "Explicit analogy identified in query",
                "confidence": 0.9
            })
        else:
            # Try to find source domain from context
            if context.get("analogy_source"):
                source_domain = context["analogy_source"]
                reasoning_path.append({
                    "type": "identification",
                    "content": f"Using '{source_domain}' as source domain for analogy",
                    "explanation": "Source domain provided in context",
                    "confidence": 0.85
                })
            else:
                # No explicit source domain - limit reasoning
                reasoning_path.append({
                    "type": "limitation",
                    "content": "No clear source domain for analogical reasoning",
                    "explanation": "Analogical reasoning requires a source domain for comparison",
                    "confidence": 0.4
                })
                return reasoning_path
        
        # Extract source domain concepts and relations
        for concept in concepts:
            concept_name = concept["name"].lower()
            
            # Very simplified check - in reality, would use more sophisticated methods
            if source_domain and source_domain.lower() in concept_name:
                source_concepts.append(concept)
        
        source_relations = []
        for relation in relations:
            subject = relation["subject"].lower()
            object = relation["object"].lower()
            
            # Check if relation involves source domain concepts
            if source_domain and (source_domain.lower() in subject or source_domain.lower() in object):
                source_relations.append(relation)
        
        # If no source domain concepts or relations, abort
        if not source_concepts and not source_relations:
            reasoning_path.append({
                "type": "limitation",
                "content": f"Insufficient knowledge about source domain '{source_domain}'",
                "explanation": "Cannot perform analogical reasoning without understanding the source domain",
                "confidence": 0.3
            })
            return reasoning_path
        
        # Map source domain to target domain
        mapping = self._create_analogical_mapping(source_domain, domains[0] if domains else "general")
        
        reasoning_path.append({
            "type": "mapping",
            "content": f"Mapping from '{source_domain}' to target domain",
            "explanation": "Creating structural mapping between domains",
            "confidence": 0.7
        })
        
        # Add mapping details
        for source, target in list(mapping.items())[:3]:  # Show top 3 mappings
            reasoning_path.append({
                "type": "correspondence",
                "content": f"'{source}' corresponds to '{target}'",
                "explanation": "Identified semantic correspondence between domains",
                "confidence": 0.65
            })
        
        # Transfer insights from source to target
        insights = []
        
        # Get relations from source domain
        for relation in source_relations:
            relation_type = relation["type"]
            source_subject = relation["subject"]
            source_object = relation["object"]
            
            # Map to target domain
            target_subject = mapping.get(source_subject, source_subject)
            target_object = mapping.get(source_object, source_object)
            
            # Create analogical insight
            insights.append({
                "content": f"If '{source_subject}' {relation_type} '{source_object}' in {source_domain}, " +
                          f"then '{target_subject}' may {relation_type} '{target_object}' in {domains[0] if domains else 'this domain'}",
                "confidence": 0.6  # Analogical insights have moderate confidence
            })
        
        # Add insights to reasoning path
        for insight in insights[:3]:  # Show top 3 insights
            reasoning_path.append({
                "type": "insight",
                "content": insight["content"],
                "explanation": "Transferring relationship from source to target domain",
                "confidence": insight["confidence"]
            })
        
        # Generate analogical conclusion
        if insights:
            reasoning_path.append({
                "type": "conclusion",
                "content": f"By analogy with {source_domain}, we can understand {domains[0] if domains else 'this domain'} in similar terms",
                "explanation": "Analogical reasoning provides useful but inexact correspondence between domains",
                "confidence": 0.6  # Analogical conclusions have moderate confidence
            })
        else:
            reasoning_path.append({
                "type": "conclusion",
                "content": f"The analogy with {source_domain} provides limited insight into the target domain",
                "explanation": "Insufficient structural similarity between domains",
                "confidence": 0.4
            })
        
        return reasoning_path
    
    def _create_analogical_mapping(self, source_domain: str, target_domain: str) -> Dict[str, str]:
        """Create a mapping from source domain concepts to target domain concepts"""
        # This is a placeholder - in reality, this would use more sophisticated methods
        
        # Some example mappings for demonstration
        if source_domain == "architecture" and target_domain == "programming":
            return {
                "building": "program",
                "architect": "developer",
                "blueprint": "design document",
                "foundation": "core library",
                "room": "module",
                "door": "interface",
                "window": "API",
                "structure": "architecture",
                "material": "language",
                "construction": "implementation"
            }
        
        elif source_domain == "journey" and target_domain == "project":
            return {
                "traveler": "team member",
                "destination": "goal",
                "map": "plan",
                "vehicle": "methodology",
                "obstacle": "challenge",
                "distance": "timeline",
                "supplies": "resources",
                "guide": "manager",
                "path": "approach",
                "milestone": "milestone"
            }
        
        # Default to empty mapping
        return {}
    
    def _apply_causal_reasoning(
        self,
        query: str,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply causal reasoning (identifying cause-effect relationships)"""
        # Simplified implementation
        reasoning_path = []
        
        # Extract causal relationships from relations
        causal_relations = []
        for relation in relations:
            if relation["type"] == "causes":
                causal_relations.append(relation)
        
        # Extract causal implications from facts
        causal_implications = []
        for fact in facts:
            if fact.get("type") == "implication":
                if "condition" in fact and "result" in fact:
                    causal_implications.append(fact)
        
        # If no causal relationships found, look in domain knowledge
        if not causal_relations and not causal_implications:
            for domain in domains:
                if domain in self.domain_reasoners:
                    for rule in self.domain_reasoners[domain]["rules"]:
                        if "if" in rule.lower() and "then" in rule.lower() and "cause" in rule.lower():
                            causal_implications.append({
                                "type": "implication",
                                "text": rule,
                                "source": f"domain:{domain}",
                                "confidence": 0.85
                            })
        
        # If still no causal relationships, abort
        if not causal_relations and not causal_implications:
            reasoning_path.append({
                "type": "limitation",
                "content": "No causal relationships identified for causal reasoning",
                "explanation": "Need cause-effect relationships to perform causal analysis",
                "confidence": 0.4
            })
            return reasoning_path
        
        # Build causal chain
        causal_network = {}
        
        # Add relationships from relations
        for relation in causal_relations:
            cause = relation["subject"]
            effect = relation["object"]
            
            if cause not in causal_network:
                causal_network[cause] = []
            
            causal_network[cause].append({
                "effect": effect,
                "confidence": relation.get("confidence", 0.7),
                "source": relation.get("source", "relation")
            })
            
            reasoning_path.append({
                "type": "causal_link",
                "content": f"'{cause}' causes '{effect}'",
                "explanation": f"Causal relationship from {relation.get('source', 'analysis')}",
                "confidence": relation.get("confidence", 0.7)
            })
        
        # Add relationships from implications
        for implication in causal_implications:
            cause = ""
            effect = ""
            
            if "condition" in implication and "result" in implication:
                cause = implication["condition"]
                effect = implication["result"]
            elif "text" in implication:
                text = implication["text"].lower()
                if "if" in text and "then" in text:
                    parts = text.split("then")
                    if len(parts) == 2:
                        cause = parts[0].replace("if", "").strip()
                        effect = parts[1].strip()
            
            if cause and effect:
                if cause not in causal_network:
                    causal_network[cause] = []
                
                causal_network[cause].append({
                    "effect": effect,
                    "confidence": implication.get("confidence", 0.75),
                    "source": implication.get("source", "implication")
                })
                
                reasoning_path.append({
                    "type": "causal_link",
                    "content": f"'{cause}' leads to '{effect}'",
                    "explanation": f"Causal implication from {implication.get('source', 'analysis')}",
                    "confidence": implication.get("confidence", 0.75)
                })
        
        # Identify root causes and ultimate effects
        root_causes = set(causal_network.keys())
        all_effects = set()
        
        for cause, effects in causal_network.items():
            for effect_info in effects:
                all_effects.add(effect_info["effect"])
        
        # Root causes are causes that are not effects of anything else
        root_causes = root_causes - all_effects
        
        # Ultimate effects are effects that are not causes of anything else
        ultimate_effects = all_effects - set(causal_network.keys())
        
        # Add analysis to reasoning path
        if root_causes:
            reasoning_path.append({
                "type": "analysis",
                "content": f"Root causes identified: {', '.join(root_causes)}",
                "explanation": "These are initial factors that aren't caused by other factors in the analysis",
                "confidence": 0.8
            })
        
        if ultimate_effects:
            reasoning_path.append({
                "type": "analysis",
                "content": f"Ultimate effects identified: {', '.join(ultimate_effects)}",
                "explanation": "These are end results that don't cause further effects in the analysis",
                "confidence": 0.8
            })
        
        # Trace causal chains
        causal_chains = []
        for root in root_causes:
            chains = self._trace_causal_chains(root, causal_network)
            causal_chains.extend(chains)
        
        # Add longest causal chain to reasoning
        if causal_chains:
            longest_chain = max(causal_chains, key=len)
            
            chain_description = " → ".join(longest_chain)
            reasoning_path.append({
                "type": "causal_chain",
                "content": f"Causal chain: {chain_description}",
                "explanation": "Tracing the sequence of cause and effect relationships",
                "confidence": 0.7  # Confidence decreases with chain length
            })
        
        # Generate causal conclusion
        if causal_chains:
            reasoning_path.append({
                "type": "conclusion",
                "content": f"Causal analysis shows that {root_causes.pop() if root_causes else 'factors'} ultimately lead to {ultimate_effects.pop() if ultimate_effects else 'results'}",
                "explanation": "Conclusion based on identified causal relationships",
                "confidence": 0.75
            })
        else:
            reasoning_path.append({
                "type": "conclusion",
                "content": "Causal relationships exist but don't form a clear causal chain",
                "explanation": "Identified individual causal links but not a complete causal pathway",
                "confidence": 0.6
            })
        
        return reasoning_path
    
    def _trace_causal_chains(
        self,
        start: str,
        causal_network: Dict[str, List[Dict[str, Any]]],
        current_chain: List[str] = None,
        visited: Set[str] = None
    ) -> List[List[str]]:
        """Recursively trace all causal chains starting from a root cause"""
        if current_chain is None:
            current_chain = [start]
        
        if visited is None:
            visited = set([start])
        
        # Get effects of the current cause
        effects = causal_network.get(start, [])
        
        # If no further effects, this chain is complete
        if not effects:
            return [current_chain]
        
        # Otherwise, continue tracing the chain for each effect
        chains = []
        for effect_info in effects:
            effect = effect_info["effect"]
            
            # Avoid cycles
            if effect in visited:
                chains.append(current_chain + [f"{effect} (cycle)"])
                continue
            
            # Continue the chain
            effect_chains = self._trace_causal_chains(
                effect,
                causal_network,
                current_chain + [effect],
                visited.union([effect])
            )
            
            chains.extend(effect_chains)
        
        return chains
    
    def _apply_bayesian_reasoning(
        self,
        query: str,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        domains: List[str],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply Bayesian reasoning (probabilistic inference)"""
        # Simplified implementation
        reasoning_path = []
        
        # Check if we have prior probabilities
        priors = {}
        for fact in facts:
            # Look for probability statements
            if fact.get("type") == "probability" and "subject" in fact and "probability" in fact:
                priors[fact["subject"]] = fact["probability"]
        
        # Extract from context if available
        if context.get("probabilities"):
            for subject, probability in context["probabilities"].items():
                priors[subject] = probability
        
        # If no probability information, note limitation
        if not priors:
            reasoning_path.append({
                "type": "limitation",
                "content": "Insufficient probability information for strong Bayesian reasoning",
                "explanation": "Bayesian reasoning requires prior probabilities and conditional relationships",
                "confidence": 0.4
            })
            
            # Continue with qualitative probabilistic reasoning
            reasoning_path.append({
                "type": "approach",
                "content": "Using qualitative probabilistic analysis instead of exact Bayesian calculations",
                "explanation": "Qualitative analysis can still provide useful insights without exact probabilities",
                "confidence": 0.7
            })
        else:
            # List available priors
            prior_descriptions = []
            for subject, probability in priors.items():
                prior_descriptions.append(f"P({subject}) = {probability:.2f}")
            
            reasoning_path.append({
                "type": "priors",
                "content": f"Prior probabilities: {'; '.join(prior_descriptions)}",
                "explanation": "Starting points for Bayesian analysis",
                "confidence": 0.85
            })
        
        # Extract conditional probabilities from relations
        conditionals = []
        for relation in relations:
            if relation["type"] == "depends_on" or relation["type"] == "causes":
                subject = relation["subject"]
                condition = relation["object"]
                
                # Simplified way to assign conditional probability
                # In reality, would extract from data or knowledge base
                conditionals.append({
                    "event": subject,
                    "condition": condition,
                    "probability": 0.7,  # Default conditional probability
                    "source": relation.get("source", "derived")
                })
        
        # Add conditionals to reasoning path
        if conditionals:
            conditional_descriptions = []
            for cond in conditionals[:3]:  # Top 3 conditionals
                conditional_descriptions.append(
                    f"P({cond['event']} | {cond['condition']}) = {cond['probability']:.2f}"
                )
            
            reasoning_path.append({
                "type": "conditionals",
                "content": f"Conditional probabilities: {'; '.join(conditional_descriptions)}",
                "explanation": "Relationships between events for Bayesian inference",
                "confidence": 0.75
            })
        
        # Identify hypothesis and evidence
        hypothesis = None
        evidence = []
        
        # Try to extract from query
        hypothesis_pattern = r"(how likely|probability|chance|likelihood) (?:of|that) ([\w\s]+)"
        hypothesis_match = re.search(hypothesis_pattern, query.lower())
        
        if hypothesis_match and len(hypothesis_match.groups()) >= 2:
            hypothesis = hypothesis_match.group(2).strip()
            
            reasoning_path.append({
                "type": "identification",
                "content": f"Hypothesis identified: {hypothesis}",
                "explanation": "Extracted from the query",
                "confidence": 0.85
            })
            
            # Look for given evidence
            evidence_pattern = r"given (?:that|the fact that) ([\w\s]+)"
            evidence_match = re.search(evidence_pattern, query.lower())
            
            if evidence_match and len(evidence_match.groups()) >= 1:
                evidence_text = evidence_match.group(1).strip()
                evidence.append(evidence_text)
                
                reasoning_path.append({
                    "type": "identification",
                    "content": f"Evidence identified: {evidence_text}",
                    "explanation": "Extracted from the query",
                    "confidence": 0.85
                })
        
        # If no hypothesis identified, try to infer from concepts
        if not hypothesis and concepts:
            # Use the most prominent concept
            hypothesis = concepts[0]["name"]
            
            reasoning_path.append({
                "type": "identification",
                "content": f"Hypothesis inferred: {hypothesis}",
                "explanation": "Based on key concept in the query",
                "confidence": 0.6
            })
        
        # Apply Bayesian reasoning if we have hypothesis
        if hypothesis:
            # Calculate posterior probability (simplified)
            prior = priors.get(hypothesis, 0.5)  # Default prior if not available
            
            # Find relevant conditionals
            relevant_conditionals = []
            for cond in conditionals:
                if cond["event"].lower() == hypothesis.lower():
                    relevant_conditionals.append(cond)
            
            # Apply evidence (simplified)
            posterior = prior
            likelihood_ratio = 1.0
            
            for e in evidence:
                # Find conditional probability for this evidence
                for cond in relevant_conditionals:
                    if cond["condition"].lower() == e.lower():
                        # Simplified Bayesian update
                        likelihood_ratio *= cond["probability"] / 0.5  # Normalized by baseline
            
            # Apply likelihood ratio
            posterior = (prior * likelihood_ratio) / ((prior * likelihood_ratio) + (1 - prior))
            
            # Add Bayesian update to reasoning path
            if evidence:
                reasoning_path.append({
                    "type": "bayesian_update",
                    "content": f"Prior P({hypothesis}) = {prior:.2f} → Posterior P({hypothesis} | evidence) = {posterior:.2f}",
                    "explanation": "Applied Bayesian updating based on evidence",
                    "confidence": 0.8
                })
            
            # Generate probabilistic conclusion
            confidence_terms = {
                (0.0, 0.1): "highly unlikely",
                (0.1, 0.3): "unlikely",
                (0.3, 0.4): "somewhat unlikely",
                (0.4, 0.6): "uncertain",
                (0.6, 0.7): "somewhat likely",
                (0.7, 0.9): "likely",
                (0.9, 1.0): "highly likely"
            }
            
            # Find appropriate confidence term
            confidence_term = "uncertain"
            for (lower, upper), term in confidence_terms.items():
                if lower <= posterior < upper:
                    confidence_term = term
                    break
            
            reasoning_path.append({
                "type": "conclusion",
                "content": f"Based on Bayesian analysis, {hypothesis} is {confidence_term} (probability: {posterior:.2f})",
                "explanation": "Conclusion based on prior probabilities and available evidence",
                "confidence": 0.7
            })
        else:
            reasoning_path.append({
                "type": "conclusion",
                "content": "Insufficient information to draw a Bayesian conclusion",
                "explanation": "Unable to identify a clear hypothesis for probability analysis",
                "confidence": 0.4
            })
        
        return reasoning_path
    
    def _apply_counterfactual_reasoning(
        self,
        query: str,
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        domains: List[str],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply counterfactual reasoning (what-if scenarios)"""
        # Simplified implementation
        reasoning_path = []
        
        # Try to identify the counterfactual condition
        counterfactual_condition = None
        
        # Check for explicit counterfactual in query
        cf_patterns = [
            r"what if ([\w\s]+)",
            r"if ([\w\s]+) (?:had|were|would)",
            r"suppose that ([\w\s]+)"
        ]
        
        for pattern in cf_patterns:
            cf_match = re.search(pattern, query.lower())
            if cf_match and len(cf_match.groups()) >= 1:
                counterfactual_condition = cf_match.group(1).strip()
                break
        
        # Check context for counterfactual
        if not counterfactual_condition and context.get("counterfactual"):
            counterfactual_condition = context["counterfactual"]
        
        # If no counterfactual found, abort
        if not counterfactual_condition:
            reasoning_path.append({
                "type": "limitation",
                "content": "No clear counterfactual condition identified",
                "explanation": "Counterfactual reasoning requires a specific 'what if' scenario",
                "confidence": 0.4
            })
            return reasoning_path
        
        # Add counterfactual to reasoning path
        reasoning_path.append({
            "type": "counterfactual",
            "content": f"Counterfactual condition: What if {counterfactual_condition}?",
            "explanation": "Analyzing an alternate scenario that differs from reality",
            "confidence": 0.85
        })
        
        # Identify relationships affected by the counterfactual
        affected_relations = []
        
        for relation in relations:
            # Check if relation involves counterfactual
            relation_text = f"{relation['subject']} {relation['type']} {relation['object']}"
            
            if (counterfactual_condition in relation['subject'] or 
                counterfactual_condition in relation['object'] or
                relation['subject'] in counterfactual_condition or
                relation['object'] in counterfactual_condition):
                
                affected_relations.append(relation)
        
        # If no affected relations found directly, use causal relations
        if not affected_relations:
            for relation in relations:
                if relation["type"] == "causes" or relation["type"] == "depends_on":
                    affected_relations.append(relation)
        
        # Identify domain-specific effects
        domain_effects = []
        for domain in domains:
            if domain in self.domain_reasoners:
                for rule in self.domain_reasoners[domain]["rules"]:
                    # Check if rule mentions counterfactual
                    if counterfactual_condition.lower() in rule.lower():
                        # Extract effect from rule
                        if "if" in rule.lower() and "then" in rule.lower():
                            parts = rule.lower().split("then")
                            if len(parts) == 2:
                                effect = parts[1].strip()
                                
                                domain_effects.append({
                                    "domain": domain,
                                    "effect": effect,
                                    "rule": rule,
                                    "confidence": 0.7
                                })
        
        # Generate counterfactual effects
        if affected_relations or domain_effects:
            # Add first-order effects
            reasoning_path.append({
                "type": "analysis",
                "content": "Analyzing first-order effects of the counterfactual",
                "explanation": "Immediate consequences that directly follow from the counterfactual",
                "confidence": 0.8
            })
            
            # Add effects from relations
            for relation in affected_relations[:3]:  # Top 3 affected relations
                reasoning_path.append({
                    "type": "effect",
                    "content": f"If {counterfactual_condition}, then the relationship '{relation['subject']} {relation['type']} {relation['object']}' would change",
                    "explanation": "Direct effect based on identified relationship",
                    "confidence": 0.7
                })
            
            # Add effects from domain knowledge
            for effect in domain_effects[:3]:  # Top 3 domain effects
                reasoning_path.append({
                    "type": "effect",
                    "content": f"In {effect['domain']} domain: If {counterfactual_condition}, then {effect['effect']}",
                    "explanation": f"Effect based on domain knowledge: {effect['rule']}",
                    "confidence": effect["confidence"]
                })
            
            # Consider second-order effects
            reasoning_path.append({
                "type": "analysis",
                "content": "Considering second-order effects (effects of effects)",
                "explanation": "Consequences that arise from the first-order effects",
                "confidence": 0.7
            })
            
            # Add a plausible second-order effect (simplified)
            if affected_relations:
                relation = affected_relations[0]
                reasoning_path.append({
                    "type": "effect",
                    "content": f"Changes to '{relation['subject']} {relation['type']} {relation['object']}' would likely affect dependent systems or processes",
                    "explanation": "Second-order effect based on cascading implications",
                    "confidence": 0.6  # Lower confidence for second-order effects
                })
            
            # Generate counterfactual conclusion
            reasoning_path.append({
                "type": "conclusion",
                "content": f"If {counterfactual_condition}, multiple aspects of the system would be affected, with cascading consequences",
                "explanation": "Conclusion from counterfactual analysis, considering direct and indirect effects",
                "confidence": 0.65
            })
        else:
            reasoning_path.append({
                "type": "conclusion",
                "content": f"The counterfactual '{counterfactual_condition}' would have limited identifiable effects given available knowledge",
                "explanation": "Unable to trace specific effects of this counterfactual scenario",
                "confidence": 0.5
            })
        
        return reasoning_path
    
    def _extract_conclusion(self, reasoning_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract overall conclusion from reasoning path"""
        # Look for explicit conclusion steps
        conclusion_steps = [step for step in reasoning_path if step["type"] == "conclusion"]
        
        if conclusion_steps:
            # Use the last conclusion as the overall conclusion
            return conclusion_steps[-1]
        
        # If no explicit conclusion, synthesize from last few steps
        if reasoning_path:
            last_steps = reasoning_path[-3:] if len(reasoning_path) >= 3 else reasoning_path
            
            conclusion_content = "Based on the reasoning process: " + "; ".join(
                [step["content"] for step in last_steps]
            )
            
            return {
                "type": "synthesized_conclusion",
                "content": conclusion_content,
                "explanation": "Synthesized from reasoning steps",
                "confidence": 0.6
            }
        
        # Default if no reasoning path
        return {
            "type": "no_conclusion",
            "content": "No conclusion could be reached",
            "explanation": "Insufficient reasoning steps",
            "confidence": 0.2
        }
    
    def _verify_consistency(
        self,
        reasoning_path: List[Dict[str, Any]],
        conclusion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify logical consistency of reasoning path and conclusion"""
        # Check for contradictions in reasoning path
        contradictions = []
        for i, step1 in enumerate(reasoning_path):
            for step2 in reasoning_path[i+1:]:
                # Simplified contradiction detection - in reality, this would be more sophisticated
                if self._are_contradictory(step1, step2):
                    contradictions.append({
                        "step1": step1["content"],
                        "step2": step2["content"],
                        "explanation": "These steps appear to contradict each other"
                    })
        
        # Check if conclusion follows from reasoning
        supports_conclusion = True
        explanation = "Conclusion follows logically from reasoning steps"
        
        # If there are contradictions, conclusion is questionable
        if contradictions:
            supports_conclusion = False
            explanation = "Contradictions in reasoning undermine the conclusion"
        
        return {
            "is_consistent": len(contradictions) == 0,
            "contradictions": contradictions,
            "supports_conclusion": supports_conclusion,
            "explanation": explanation
        }
    
    def _are_contradictory(self, step1: Dict[str, Any], step2: Dict[str, Any]) -> bool:
        """Check if two reasoning steps contradict each other"""
        # Simplified implementation - in reality, this would be more sophisticated
        
        # Check for simple negation patterns
        content1 = step1["content"].lower()
        content2 = step2["content"].lower()
        
        # Look for direct contradictions like "X is Y" vs "X is not Y"
        negation_patterns = [
            (r"(\w+) is (\w+)", r"(\1) is not (\2)"),
            (r"(\w+) are (\w+)", r"(\1) are not (\2)"),
            (r"all (\w+) are (\w+)", r"some (\1) are not (\2)"),
            (r"none of the (\w+) are (\w+)", r"some (\1) are (\2)")
        ]
        
        for pattern, negation in negation_patterns:
            if re.search(pattern, content1) and re.search(negation, content2):
                return True
            if re.search(pattern, content2) and re.search(negation, content1):
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get basic status information"""
        return {
            "name": "Neuro-Symbolic Reasoning Engine",
            "status": "operational",
            "domains": list(self.domain_reasoners.keys()),
            "capabilities": [str(s) for s in ReasoningStrategy]
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            "name": "Neuro-Symbolic Reasoning Engine",
            "status": "operational",
            "domains": {
                domain: {
                    "rules": len(info["rules"]),
                    "concepts": len(info["concepts"]),
                    "relations": len(info["relations"])
                }
                for domain, info in self.domain_reasoners.items()
            },
            "capabilities": [str(s) for s in ReasoningStrategy],
            "knowledge_base": {
                "facts": len(self.knowledge_base["facts"]),
                "rules": len(self.knowledge_base["rules"]),
                "entity_types": len(self.knowledge_base["ontology"].get("entity_types", [])),
                "relation_types": len(self.knowledge_base["ontology"].get("relation_types", []))
            },
            "cache_size": len(self.reasoning_cache)
        }