"""
Example of Categorized Knowledge Management

Demonstrates how to organize, store, and retrieve knowledge
using categories in the Knowledge Library system.
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
try:
    from ai_core.knowledge.library import knowledge_library, KnowledgeSource, KnowledgeEntry
    has_knowledge_lib = True
except ImportError as e:
    print(f"Error importing knowledge library: {str(e)}")
    has_knowledge_lib = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def add_hierarchical_knowledge():
    """Add knowledge in a hierarchical category structure"""
    logger.info("=== Adding Hierarchical Categorized Knowledge ===")
    
    # Define category structure
    categories = {
        "programming": [
            "languages", "frameworks", "algorithms", "best_practices"
        ],
        "machine_learning": [
            "deep_learning", "nlp", "computer_vision", "reinforcement_learning"
        ],
        "quantum_computing": [
            "algorithms", "hardware", "theory", "applications"
        ]
    }
    
    # Create main categories with information
    main_category_info = {
        "programming": """
        # Programming Knowledge Category
        
        This category contains knowledge about programming languages, frameworks, algorithms, 
        and best practices in software development. It covers both theory and practical aspects
        of computer programming.
        """,
        
        "machine_learning": """
        # Machine Learning Knowledge Category
        
        This category contains knowledge about machine learning techniques, models, frameworks,
        and applications. It includes deep learning, natural language processing, computer vision,
        and reinforcement learning.
        """,
        
        "quantum_computing": """
        # Quantum Computing Knowledge Category
        
        This category contains knowledge about quantum computing principles, algorithms,
        hardware platforms, theoretical foundations, and practical applications in the real world.
        """
    }
    
    # Add main categories
    main_category_ids = {}
    for category, description in main_category_info.items():
        entry_id = knowledge_library.add_knowledge_entry(
            content=description,
            source_type=KnowledgeSource.TEXT_CONTENT,
            source_reference=f"category_description:{category}",
            categories=[category, "category_description"],
            created_by="system",
            metadata={
                "is_category_description": True,
                "category_level": "main"
            }
        )
        main_category_ids[category] = entry_id
        logger.info(f"Added main category description for: {category}")
    
    # Add subcategory descriptions
    subcategory_descriptions = {
        "programming.languages": """
        # Programming Languages
        
        Knowledge about various programming languages including Python, JavaScript, Java, C++, Rust, Go,
        and others. Includes syntax, features, best practices, and comparisons between languages.
        """,
        
        "programming.frameworks": """
        # Programming Frameworks
        
        Knowledge about software frameworks for different purposes, including web development,
        data processing, scientific computing, and more. Includes React, Django, TensorFlow, etc.
        """,
        
        "machine_learning.deep_learning": """
        # Deep Learning
        
        Knowledge about neural network architectures, training methods, optimization techniques,
        and applications. Includes CNNs, RNNs, Transformers, and other architectures.
        """,
        
        "machine_learning.nlp": """
        # Natural Language Processing
        
        Knowledge about processing, understanding, and generating human language with computers.
        Includes tokenization, embeddings, language models, sentiment analysis, and more.
        """,
        
        "quantum_computing.algorithms": """
        # Quantum Algorithms
        
        Knowledge about algorithms specifically designed for quantum computers, including
        Grover's search, Shor's factoring, quantum Fourier transform, QAOA, VQE, and others.
        """,
        
        "quantum_computing.hardware": """
        # Quantum Hardware
        
        Knowledge about physical implementations of quantum computers, including
        superconducting qubits, trapped ions, photonic quantum computers, and topological qubits.
        """
    }
    
    # Add subcategory descriptions
    subcategory_ids = {}
    for full_category, description in subcategory_descriptions.items():
        main_cat, sub_cat = full_category.split(".")
        entry_id = knowledge_library.add_knowledge_entry(
            content=description,
            source_type=KnowledgeSource.TEXT_CONTENT,
            source_reference=f"category_description:{full_category}",
            categories=[main_cat, sub_cat, "category_description"],
            created_by="system",
            metadata={
                "is_category_description": True,
                "category_level": "sub",
                "parent_category": main_cat
            }
        )
        subcategory_ids[full_category] = entry_id
        logger.info(f"Added subcategory description for: {full_category}")
    
    # Add actual knowledge entries in different categories
    knowledge_entries = [
        # Programming - Python
        {
            "title": "Python Type Hints and Type Checking",
            "content": """
            # Python Type Hints and Type Checking
            
            Python 3.5+ supports type hints that make code more readable and maintainable.
            
            ```python
            def greeting(name: str) -> str:
                return f"Hello, {name}"
            ```
            
            You can use the `mypy` static type checker to verify type correctness:
            
            ```bash
            pip install mypy
            mypy your_script.py
            ```
            
            Type hints also improve IDE autocompletion and documentation.
            """,
            "categories": ["programming", "languages", "python", "best_practices"],
            "created_by": "qwen"
        },
        
        # Machine Learning - Deep Learning
        {
            "title": "Transformer Architecture Overview",
            "content": """
            # Transformer Architecture Overview
            
            The Transformer architecture is a neural network model introduced in the paper 
            "Attention Is All You Need" (Vaswani et al., 2017).
            
            Key components:
            
            1. **Multi-head self-attention**: Allows the model to focus on different parts of the input sequence
            2. **Position-wise feed-forward networks**: Processes each position independently
            3. **Residual connections and layer normalization**: Stabilizes training
            4. **Positional encodings**: Provides information about token position
            
            Transformers have revolutionized NLP and are now being applied to computer vision, 
            time series analysis, and many other domains.
            """,
            "categories": ["machine_learning", "deep_learning", "transformers", "nlp"],
            "created_by": "olympic"
        },
        
        # Quantum Computing - Algorithms
        {
            "title": "Quantum Phase Estimation Algorithm",
            "content": """
            # Quantum Phase Estimation Algorithm
            
            Quantum Phase Estimation (QPE) is a fundamental quantum algorithm that estimates the eigenvalues
            of a unitary operator.
            
            Key applications:
            
            1. It's a subroutine in Shor's factoring algorithm
            2. Used in quantum chemistry to find molecular energy levels
            3. Enables solving systems of linear equations in the HHL algorithm
            
            The algorithm uses quantum Fourier transform and controlled-U operations to extract phase information
            with exponential precision compared to classical methods.
            """,
            "categories": ["quantum_computing", "algorithms", "phase_estimation"],
            "created_by": "qwen"
        }
    ]
    
    # Add knowledge entries
    entry_ids = []
    for entry in knowledge_entries:
        entry_id = knowledge_library.add_knowledge_entry(
            content=entry["content"],
            source_type=KnowledgeSource.TEXT_CONTENT,
            source_reference=f"knowledge:{entry['title']}",
            categories=entry["categories"],
            created_by=entry["created_by"],
            metadata={
                "title": entry["title"]
            }
        )
        entry_ids.append(entry_id)
        logger.info(f"Added knowledge entry: {entry['title']}")
    
    return {
        "main_categories": main_category_ids,
        "subcategories": subcategory_ids,
        "entries": entry_ids
    }

def retrieve_by_category():
    """Retrieve knowledge by category"""
    logger.info("\n=== Retrieving Knowledge by Category ===")
    
    # Get all categories
    all_categories = knowledge_library.get_categories()
    logger.info(f"All categories in the knowledge library ({len(all_categories)}):")
    for category in all_categories:
        logger.info(f"- {category}")
    
    # Get entries in specific categories
    category_examples = ["programming", "deep_learning", "algorithms"]
    for category in category_examples:
        entries = knowledge_library.get_entries_by_category(category)
        logger.info(f"\nEntries in category '{category}' ({len(entries)}):")
        
        for i, entry in enumerate(entries[:3]):  # Show up to 3 entries
            metadata = entry.metadata or {}
            title = metadata.get("title", f"Entry {entry.id}")
            is_description = metadata.get("is_category_description", False)
            
            if is_description:
                logger.info(f"  - [{i+1}] Category Description: {title}")
            else:
                logger.info(f"  - [{i+1}] Entry: {title} (by {entry.created_by})")
                logger.info(f"      Categories: {entry.categories}")
                
                # Show a snippet of content
                content_snippet = entry.content.strip().split("\n")[0][:100]
                logger.info(f"      Content: {content_snippet}...")
        
        if len(entries) > 3:
            logger.info(f"      ... and {len(entries) - 3} more entries")

def search_within_categories():
    """Search for knowledge within specific categories"""
    logger.info("\n=== Searching Within Categories ===")
    
    search_examples = [
        {"query": "python", "categories": ["programming"]},
        {"query": "transformer", "categories": ["machine_learning"]},
        {"query": "algorithm", "categories": ["quantum_computing"]}
    ]
    
    for search in search_examples:
        query = search["query"]
        categories = search["categories"]
        
        logger.info(f"\nSearching for '{query}' within categories {categories}:")
        
        results = knowledge_library.search_knowledge(
            query=query,
            limit=5,
            categories=categories
        )
        
        if not results:
            logger.info(f"  No results found")
            continue
        
        for i, entry in enumerate(results):
            metadata = entry.metadata or {}
            title = metadata.get("title", f"Entry {entry.id}")
            
            logger.info(f"  - [{i+1}] {title} (by {entry.created_by})")
            logger.info(f"      Categories: {entry.categories}")
            
            # Show a snippet of content where the query appears
            content_lower = entry.content.lower()
            query_lower = query.lower()
            
            if query_lower in content_lower:
                # Find a snippet containing the query
                start_idx = content_lower.find(query_lower)
                # Get a window of text around the query
                start = max(0, start_idx - 50)
                end = min(len(content_lower), start_idx + len(query_lower) + 50)
                
                snippet = "..." if start > 0 else ""
                snippet += entry.content[start:end]
                snippet += "..." if end < len(entry.content) else ""
                
                logger.info(f"      Matching snippet: {snippet}")

def export_by_category():
    """Export knowledge by category"""
    logger.info("\n=== Exporting Knowledge by Category ===")
    
    # Create export directory
    export_dir = os.path.join(parent_dir, "data", "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Export a few categories
    categories_to_export = ["programming", "deep_learning", "algorithms"]
    
    for category in categories_to_export:
        # Export as plain text
        txt_file = os.path.join(export_dir, f"category_{category}_knowledge.txt")
        result = knowledge_library.export_category_to_plain_text(category, txt_file)
        
        if result:
            logger.info(f"Exported category '{category}' to plain text file: {txt_file}")
        else:
            logger.warning(f"Failed to export category '{category}' to plain text")

def example_import_export_formats():
    """Example of importing and exporting in different formats"""
    logger.info("\n=== Example of Different Storage Formats ===")
    
    # Create test directories
    export_dir = os.path.join(parent_dir, "data", "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    # Create a sample entry for export
    entry = KnowledgeEntry(
        content="""
        # Storage Format Example
        
        This is a sample knowledge entry created to demonstrate different storage formats.
        
        ## Key Points
        
        1. Knowledge can be stored in plain text format, which is human-readable and easy to edit
        2. Knowledge can also be stored in JSON format, which preserves all metadata
        3. Both formats support import/export operations
        4. Categories are preserved in both formats
        """,
        source_type=KnowledgeSource.TEXT_CONTENT,
        source_reference="storage_format_example",
        categories=["examples", "storage", "formats"],
        created_by="system",
        metadata={
            "title": "Storage Format Example",
            "importance": "high",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    
    # Export to different formats
    # 1. Plain text format
    txt_file = os.path.join(export_dir, "format_example.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(entry.to_plain_text())
    logger.info(f"Exported entry to plain text format: {txt_file}")
    
    # 2. JSON format
    json_file = os.path.join(export_dir, "format_example.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Exported entry to JSON format: {json_file}")
    
    # Import from different formats
    # 1. Plain text format
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    from_txt = KnowledgeEntry.from_plain_text(content)
    if from_txt:
        logger.info(f"Successfully imported from plain text format:")
        logger.info(f"- Categories: {from_txt.categories}")
        logger.info(f"- Created by: {from_txt.created_by}")
        logger.info(f"- Content snippet: {from_txt.content[:50]}...")
    
    # 2. JSON format
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    from_json = KnowledgeEntry.from_dict(data)
    if from_json:
        logger.info(f"Successfully imported from JSON format:")
        logger.info(f"- Categories: {from_json.categories}")
        logger.info(f"- Created by: {from_json.created_by}")
        logger.info(f"- Metadata: {list(from_json.metadata.keys())}")
        logger.info(f"- Content snippet: {from_json.content[:50]}...")

def main():
    """Main function to run all examples"""
    if not has_knowledge_lib:
        logger.error("Knowledge library not available. Cannot run examples.")
        return
    
    logger.info("Starting Categorized Knowledge Examples")
    
    # Add hierarchical categorized knowledge
    add_hierarchical_knowledge()
    
    # Retrieve knowledge by category
    retrieve_by_category()
    
    # Search within categories
    search_within_categories()
    
    # Export by category
    export_by_category()
    
    # Example import/export formats
    example_import_export_formats()
    
    logger.info("\nAll examples completed successfully!")

if __name__ == "__main__":
    main()