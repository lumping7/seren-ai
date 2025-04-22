"""
Example of Model Communication and Knowledge Sharing

Demonstrates how Qwen2.5 and OlympicCoder models can communicate
and share knowledge with each other.
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

# Import knowledge and communication components
try:
    from ai_core.model_relay import model_relay
    from ai_core.model_communication import MessageType, CommunicationMode
    from ai_core.knowledge.library import knowledge_library, KnowledgeSource
    from ai_core.model_manager import ModelType
    
    has_imports = True
except ImportError as e:
    print(f"Error importing components: {str(e)}")
    has_imports = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def simulate_qwen_message(content: str) -> str:
    """Simulate a message from Qwen model (in real system, this would call the actual model)"""
    return f"[QWEN RESPONSE]: {content}"

def simulate_olympic_message(content: str) -> str:
    """Simulate a message from OlympicCoder model (in real system, this would call the actual model)"""
    return f"[OLYMPIC RESPONSE]: {content}"

def example_basic_communication():
    """Example of basic communication between models"""
    logger.info("=== Basic Communication Example ===")
    
    # Start a new conversation
    conversation = model_relay.start_conversation(
        mode=CommunicationMode.COLLABORATIVE,
        topic="Discussing quantum computing algorithms"
    )
    
    conversation_id = conversation["id"]
    logger.info(f"Started conversation: {conversation_id}")
    
    # Qwen sends a message
    qwen_query = "What are the advantages of Grover's algorithm over classical search algorithms?"
    model_relay.send_message(
        conversation_id=conversation_id,
        from_model="qwen",
        to_model="olympic",
        message_type=MessageType.QUERY,
        content=qwen_query
    )
    logger.info(f"Qwen sent query: {qwen_query}")
    
    # Olympic gets the message and responds
    messages = model_relay.get_messages(
        conversation_id=conversation_id,
        for_model="olympic",
        message_types=[MessageType.QUERY]
    )
    
    if messages:
        query_message = messages[0]
        query = query_message["content"]
        logger.info(f"Olympic received query: {query}")
        
        # Generate response (simulated)
        response_content = simulate_olympic_message(
            "Grover's algorithm provides a quadratic speedup for unstructured search problems, "
            "reducing the time complexity from O(N) to O(√N) by amplifying the probability of "
            "measuring the correct answer through quantum superposition and quantum amplitude amplification."
        )
        
        # Send response
        model_relay.send_message(
            conversation_id=conversation_id,
            from_model="olympic",
            to_model="qwen",
            message_type=MessageType.RESPONSE,
            content=response_content
        )
        logger.info(f"Olympic sent response")
    
    # Qwen gets the response
    responses = model_relay.get_messages(
        conversation_id=conversation_id,
        for_model="qwen",
        message_types=[MessageType.RESPONSE]
    )
    
    if responses:
        response_message = responses[0]
        response = response_message["content"]
        logger.info(f"Qwen received response: {response}")
    
    return conversation_id

def example_knowledge_sharing(conversation_id: str):
    """Example of knowledge sharing between models"""
    logger.info("\n=== Knowledge Sharing Example ===")
    
    # Qwen shares knowledge about quantum computing
    qwen_knowledge = """
    # Quantum Computing Hardware Platforms
    
    There are several hardware platforms for quantum computing:
    
    1. **Superconducting qubits**: Used by IBM, Google, and Rigetti. These are artificial atoms made from superconducting circuits. They offer fast gate operations but have relatively short coherence times.
    
    2. **Trapped ions**: Used by IonQ and Honeywell. These are actual atoms held in place by electromagnetic fields. They have long coherence times but slower gate operations.
    
    3. **Photonic quantum computers**: Use photons as qubits. They can operate at room temperature but creating interactions between photons is challenging.
    
    4. **Topological qubits**: Microsoft's approach, using Majorana fermions. These are theoretically more resistant to errors but are still in early development.
    
    5. **Neutral atoms**: Atoms arranged in arrays using optical tweezers. This approach offers scalability and good coherence times.
    """
    
    sharing_result = model_relay.share_knowledge(
        conversation_id=conversation_id,
        from_model="qwen",
        to_model="olympic",
        content=qwen_knowledge,
        categories=["quantum_computing", "hardware"]
    )
    
    logger.info(f"Qwen shared knowledge about quantum computing hardware")
    
    # Olympic shares knowledge about quantum algorithms
    olympic_knowledge = """
    # Quantum Algorithms Beyond Grover and Shor
    
    While Grover's search algorithm and Shor's factoring algorithm are the most famous quantum algorithms, several others show promise:
    
    1. **Quantum approximate optimization algorithm (QAOA)**: Used for combinatorial optimization problems like MaxCut. QAOA can potentially outperform classical algorithms for certain NP-hard problems.
    
    2. **Variational quantum eigensolver (VQE)**: Used for simulating quantum systems in chemistry and materials science. It's a hybrid algorithm that uses a classical computer to optimize a quantum circuit.
    
    3. **Quantum machine learning algorithms**: Including quantum neural networks, quantum support vector machines, and quantum principal component analysis. These leverage quantum superposition and entanglement for potential speedups in machine learning tasks.
    
    4. **HHL algorithm**: For solving linear systems of equations, offering exponential speedup for certain well-conditioned sparse matrices.
    
    5. **Quantum random walk algorithms**: Provide polynomial speedups for various graph and network analysis problems.
    """
    
    sharing_result = model_relay.share_knowledge(
        conversation_id=conversation_id,
        from_model="olympic",
        to_model="qwen",
        content=olympic_knowledge,
        categories=["quantum_computing", "algorithms"]
    )
    
    logger.info(f"Olympic shared knowledge about quantum algorithms")
    
    # Now Qwen retrieves the knowledge shared by Olympic
    olympic_shared = model_relay.get_shared_knowledge(
        for_model="qwen",
        categories=["algorithms"],
        limit=1
    )
    
    if olympic_shared:
        logger.info(f"Qwen retrieved knowledge from Olympic:")
        logger.info(f"- Categories: {olympic_shared[0]['categories']}")
        logger.info(f"- Content snippet: {olympic_shared[0]['content'][:100]}...")
    
    # Olympic retrieves the knowledge shared by Qwen
    qwen_shared = model_relay.get_shared_knowledge(
        for_model="olympic",
        categories=["hardware"],
        limit=1
    )
    
    if qwen_shared:
        logger.info(f"Olympic retrieved knowledge from Qwen:")
        logger.info(f"- Categories: {qwen_shared[0]['categories']}")
        logger.info(f"- Content snippet: {qwen_shared[0]['content'][:100]}...")

def example_help_request(conversation_id: str):
    """Example of a help request between models"""
    logger.info("\n=== Help Request Example ===")
    
    # Olympic requests help from Qwen
    help_request = model_relay.request_help(
        conversation_id=conversation_id,
        from_model="olympic",
        to_model="qwen",
        query="Can you explain the relationship between quantum entanglement and quantum teleportation?",
        context={"purpose": "Technical writing clarification"}
    )
    
    request_id = help_request["request_id"]
    logger.info(f"Olympic requested help: {help_request['query']}")
    
    # Qwen receives the help request
    help_requests = model_relay.get_messages(
        conversation_id=conversation_id,
        for_model="qwen",
        message_types=[MessageType.REQUEST_HELP]
    )
    
    if help_requests:
        request_message = help_requests[0]
        request_content = json.loads(request_message["content"])
        logger.info(f"Qwen received help request: {request_content['query']}")
        
        # Generate help response (simulated)
        help_response = simulate_qwen_message(
            "Quantum entanglement is the fundamental resource enabling quantum teleportation. "
            "In quantum teleportation, a sender (Alice) and receiver (Bob) share an entangled pair. "
            "Alice performs a Bell measurement on her part of the entangled pair along with the qubit "
            "she wants to teleport. This measurement destroys the original qubit and the entanglement, "
            "but gives Alice two classical bits of information. Alice sends these bits to Bob, who "
            "then applies specific quantum gates to his part of the formerly entangled pair, reconstructing "
            "the original quantum state. Without entanglement, this process would be impossible, as the "
            "quantum information would be lost in measurement."
        )
        
        # Provide help
        help_provision = model_relay.provide_help(
            conversation_id=conversation_id,
            from_model="qwen",
            to_model="olympic",
            request_id=request_id,
            response=help_response,
            add_to_knowledge=True,
            categories=["quantum_physics", "entanglement"]
        )
        
        logger.info(f"Qwen provided help response")
    
    # Olympic receives the help
    help_responses = model_relay.get_messages(
        conversation_id=conversation_id,
        for_model="olympic",
        message_types=[MessageType.PROVIDE_HELP]
    )
    
    if help_responses:
        response_message = help_responses[0]
        response_content = json.loads(response_message["content"])
        logger.info(f"Olympic received help: {response_content['response'][:100]}...")

def example_knowledge_search():
    """Example of searching shared knowledge"""
    logger.info("\n=== Knowledge Search Example ===")
    
    # Search for knowledge about quantum algorithms
    algorithm_results = model_relay.search_knowledge(
        query="quantum optimization algorithms",
        categories=["algorithms"],
        limit=2
    )
    
    if algorithm_results:
        logger.info(f"Found {len(algorithm_results)} entries about quantum optimization algorithms:")
        for i, result in enumerate(algorithm_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"- Created by: {result['created_by']}")
            logger.info(f"- Categories: {result['categories']}")
            logger.info(f"- Content snippet: {result['content'][:100]}...")
    
    # Search for knowledge about quantum hardware
    hardware_results = model_relay.search_knowledge(
        query="superconducting qubits",
        categories=["hardware"],
        limit=2
    )
    
    if hardware_results:
        logger.info(f"Found {len(hardware_results)} entries about superconducting qubits:")
        for i, result in enumerate(hardware_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"- Created by: {result['created_by']}")
            logger.info(f"- Categories: {result['categories']}")
            logger.info(f"- Content snippet: {result['content'][:100]}...")

def example_export_knowledge(conversation_id: str):
    """Example of exporting knowledge from a conversation"""
    logger.info("\n=== Export Knowledge Example ===")
    
    # Create export directory
    export_dir = os.path.join(parent_dir, "data", "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Export as plain text
    txt_file = os.path.join(export_dir, f"conversation_{conversation_id}_knowledge.txt")
    txt_result = model_relay.export_conversation_knowledge(
        conversation_id=conversation_id,
        output_file=txt_file,
        format="plain_text"
    )
    
    if "error" not in txt_result:
        logger.info(f"Exported {txt_result['count']} entries to plain text file: {txt_file}")
    
    # Export as JSON
    json_file = os.path.join(export_dir, f"conversation_{conversation_id}_knowledge.json")
    json_result = model_relay.export_conversation_knowledge(
        conversation_id=conversation_id,
        output_file=json_file,
        format="json"
    )
    
    if "error" not in json_result:
        logger.info(f"Exported {json_result['count']} entries to JSON file: {json_file}")

def main():
    """Main function to run all examples"""
    if not has_imports:
        logger.error("Required components not available. Cannot run examples.")
        return
    
    logger.info("Starting Model Communication Examples")
    
    # Run communication example
    conversation_id = example_basic_communication()
    
    # Run knowledge sharing example
    example_knowledge_sharing(conversation_id)
    
    # Run help request example
    example_help_request(conversation_id)
    
    # Run knowledge search example
    example_knowledge_search()
    
    # Run export example
    example_export_knowledge(conversation_id)
    
    logger.info("\nAll examples completed successfully!")

if __name__ == "__main__":
    main()