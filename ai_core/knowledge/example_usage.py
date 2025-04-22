"""
Example usage of the Knowledge Library System

This script demonstrates how to use the knowledge library
to add, retrieve, and learn from knowledge.
"""

import os
import sys
import json
import logging
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import knowledge components
from ai_core.knowledge.library import knowledge_library, KnowledgeSource
from ai_core.knowledge.web_retriever import web_retriever
from ai_core.knowledge.self_learning import self_learning_system, LearningPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_text():
    """Create a sample text file for knowledge demonstration"""
    sample_dir = os.path.join(parent_dir, "data", "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_file = os.path.join(sample_dir, "quantum_computing.txt")
    
    content = """# Introduction to Quantum Computing

Quantum computing is a rapidly emerging technology that harnesses the laws of quantum mechanics to solve complex problems much faster than classical computers.

## Key Concepts

### Qubits
Unlike classical bits that exist in a state of either 0 or 1, quantum bits (qubits) can exist in a superposition of both states simultaneously. This unique property enables quantum computers to process vast amounts of information in parallel.

### Superposition
Superposition allows quantum systems to exist in multiple states at once. When applied to computing, this means a quantum computer with n qubits can represent 2^n states simultaneously, compared to a single state for classical computing.

### Entanglement
Quantum entanglement is a phenomenon where pairs or groups of particles interact in ways such that the quantum state of each particle cannot be described independently of the others. This property enables quantum computers to perform complex operations with remarkable efficiency.

### Quantum Gates
Similar to logical gates in classical computing, quantum gates manipulate qubits to perform operations. Common quantum gates include the Hadamard gate, CNOT gate, and Pauli gates.

## Applications

- **Cryptography**: Quantum computers could potentially break widely-used encryption methods, but also enable new quantum-secure cryptographic techniques.
- **Drug Discovery**: Simulating molecular interactions with high precision could revolutionize pharmaceutical research.
- **Optimization Problems**: Complex optimization tasks in logistics, finance, and machine learning could be solved exponentially faster.
- **Material Science**: Quantum computers could help design new materials with specific properties by simulating their quantum behavior.

## Current Challenges

Despite tremendous progress, quantum computing faces several challenges:
- Maintaining quantum coherence (qubits are fragile and susceptible to environmental interference)
- Scaling up quantum systems to handle practical applications
- Developing quantum algorithms that provide clear advantages over classical methods
- Building fault-tolerant systems that can correct quantum errors

As research continues to advance, quantum computing promises to transform fields ranging from materials science to artificial intelligence, offering computational capabilities beyond what classical computers can achieve."""
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Created sample file: {sample_file}")
    return sample_file

def example_add_knowledge_from_file():
    """Example of adding knowledge from a file"""
    logger.info("=== Adding Knowledge from File ===")
    
    # Create a sample file
    sample_file = create_sample_text()
    
    # Add the file to the knowledge library
    entry_ids = knowledge_library.add_knowledge_from_file(
        file_path=sample_file,
        context_name="quantum_computing",
        metadata={
            "topic": "quantum computing",
            "category": "computer science",
            "importance": "high"
        }
    )
    
    logger.info(f"Added {len(entry_ids)} entries to the knowledge library")
    logger.info(f"First entry ID: {entry_ids[0] if entry_ids else 'N/A'}")
    return entry_ids

def example_add_knowledge_from_text():
    """Example of adding knowledge from text"""
    logger.info("=== Adding Knowledge from Text ===")
    
    # Sample text
    text = """# Neuro-Symbolic AI

Neuro-symbolic AI combines neural networks with symbolic reasoning to create more robust AI systems. 
This approach integrates the learning capabilities of neural networks with the logical reasoning of symbolic AI.

Key benefits include:
1. Better explainability - decisions can be traced through symbolic rules
2. More efficient learning - requiring less training data
3. Improved generalization - can reason about new situations
4. Stronger reasoning capabilities - can apply logical rules consistently

This hybrid approach represents a promising direction for more capable and trustworthy AI systems."""
    
    # Add to knowledge library
    entry_ids = knowledge_library.add_knowledge_from_text(
        text=text,
        source_reference="neuro-symbolic lecture notes",
        context_name="ai_techniques",
        metadata={
            "topic": "neuro-symbolic AI",
            "category": "artificial intelligence",
            "importance": "high"
        }
    )
    
    logger.info(f"Added {len(entry_ids)} entries to the knowledge library")
    logger.info(f"First entry ID: {entry_ids[0] if entry_ids else 'N/A'}")
    return entry_ids

def example_search_knowledge():
    """Example of searching knowledge"""
    logger.info("=== Searching Knowledge ===")
    
    # Search for knowledge
    queries = [
        "How do quantum computers work?",
        "What is superposition in quantum computing?",
        "Explain neuro-symbolic AI advantages"
    ]
    
    for query in queries:
        logger.info(f"Query: {query}")
        entries = knowledge_library.search_knowledge(query, limit=3)
        
        logger.info(f"Found {len(entries)} relevant entries")
        for i, entry in enumerate(entries):
            # Display a snippet of each entry
            snippet = entry.content[:150] + "..." if len(entry.content) > 150 else entry.content
            logger.info(f"  [{i+1}] Source: {entry.source_reference}")
            logger.info(f"      Snippet: {snippet}")
        
        # Get context for query
        context = knowledge_library.extract_context_for_query(query)
        logger.info(f"Context length: {len(context)} characters")

def example_self_learning():
    """Example of self-learning system"""
    logger.info("=== Self-Learning System ===")
    
    # Add a learning task
    self_learning_system.add_learning_task(
        content="""Quantum error correction is a technique used to protect quantum information from errors due to decoherence and other quantum noise. 
        It is essential for fault-tolerant quantum computation.
        
        The most common approach uses quantum error-correcting codes, such as the Surface code, which spreads quantum information across many physical qubits.
        This allows the detection and correction of errors without disrupting the quantum state itself.""",
        source="quantum error correction lecture",
        priority=LearningPriority.HIGH,
        metadata={
            "topic": "quantum error correction",
            "related_to": "quantum computing"
        }
    )
    
    # Check learning status
    status = self_learning_system.get_learning_status()
    logger.info(f"Learning status: {json.dumps(status, indent=2)}")
    
    # Wait for processing (in a real system, would happen in background)
    logger.info("Waiting for self-learning system to process task...")
    time.sleep(2)
    
    # Check learning status again
    status = self_learning_system.get_learning_status()
    logger.info(f"Updated learning status: {json.dumps(status, indent=2)}")
    
    # Search for the learned knowledge
    entries = knowledge_library.search_knowledge("quantum error correction", limit=2)
    logger.info(f"Found {len(entries)} entries about quantum error correction")

def example_web_retrieval():
    """Example of web retrieval (simulated)"""
    logger.info("=== Web Knowledge Retrieval ===")
    
    # Note: Web access is disabled by default for safety
    # This demonstrates how it would work if enabled
    
    if web_retriever.enable_web_access:
        # Retrieve from URL
        entries = web_retriever.retrieve_and_add_to_library(
            url="https://example.com/quantum-computing",
            context_name="web_retrieved"
        )
        logger.info(f"Retrieved {len(entries)} entries from web")
    else:
        logger.info("Web access is disabled")
        logger.info("To enable web access (for demo purposes only):")
        logger.info("web_retriever.enable_web_access = True")
        
        # Simulate web retrieval
        logger.info("Simulating web retrieval...")
        
        # Simulated web content
        simulated_content = """# Latest Advances in Quantum Computing

Scientists have recently demonstrated a 127-qubit quantum processor, 
achieving a new milestone in quantum computing scale.

The new processor shows improved error correction and longer coherence times,
bringing quantum computing closer to practical applications in fields such
as material science, cryptography, and drug discovery."""
        
        # Add to knowledge library
        entry_ids = knowledge_library.add_knowledge_from_text(
            text=simulated_content,
            source_reference="simulated_web:quantum_news",
            context_name="web_retrieved_simulated",
            metadata={
                "source_type": "web_simulated",
                "topic": "quantum computing news"
            }
        )
        
        logger.info(f"Added {len(entry_ids)} simulated web entries to the knowledge library")

def main():
    """Main function to run all examples"""
    logger.info("Starting Knowledge Library System Examples")
    
    # Run examples
    example_add_knowledge_from_file()
    logger.info("\n")
    
    example_add_knowledge_from_text()
    logger.info("\n")
    
    example_search_knowledge()
    logger.info("\n")
    
    example_self_learning()
    logger.info("\n")
    
    example_web_retrieval()
    
    logger.info("\nAll examples completed successfully!")

if __name__ == "__main__":
    main()