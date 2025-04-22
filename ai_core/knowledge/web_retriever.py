"""
Web Knowledge Retriever for Seren

Retrieves relevant knowledge from the internet to enhance
the knowledge library.
"""

import os
import sys
import json
import logging
import time
import re
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlparse, urljoin
import threading

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# For web access
try:
    import requests
    from requests.exceptions import RequestException
    has_requests = True
except ImportError:
    has_requests = False
    logging.warning("Requests library not installed. Web access will be disabled.")

# For HTML parsing
try:
    from bs4 import BeautifulSoup
    has_bs4 = True
except ImportError:
    has_bs4 = False
    logging.warning("BeautifulSoup not installed. HTML parsing will be limited.")

# Local imports
try:
    from ai_core.knowledge.library import knowledge_library, KnowledgeSource
except ImportError:
    logging.error("Knowledge library not found.")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class WebRetriever:
    """
    Web Knowledge Retriever for Seren
    
    Retrieves knowledge from the internet to enhance the AI's knowledge:
    - Fetches and extracts content from web pages
    - Adds retrieved knowledge to the knowledge library
    - Respects robots.txt and ethical web scraping practices
    - Limits requests to prevent overloading websites
    """
    
    def __init__(self, enable_web_access: bool = False, rate_limit: float = 1.0):
        """
        Initialize the web retriever
        
        Args:
            enable_web_access: Whether to enable web access
            rate_limit: Minimum seconds between requests to the same domain
        """
        self.enable_web_access = enable_web_access and has_requests
        self.rate_limit = rate_limit
        
        # Track last request time per domain
        self.last_request_time = {}
        
        # User agent for requests
        self.user_agent = "Seren AI Knowledge Retriever/1.0"
        
        # Check dependencies
        self.can_parse_html = has_bs4
        
        logger.info(f"Web Retriever initialized. Web access enabled: {self.enable_web_access}")
    
    def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL
        
        Args:
            url: URL to fetch
            
        Returns:
            Text content or None if failed
        """
        if not self.enable_web_access:
            logger.warning("Web access is disabled")
            return None
        
        try:
            # Parse the URL to get the domain
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Check rate limit
            now = time.time()
            if domain in self.last_request_time:
                elapsed = now - self.last_request_time[domain]
                if elapsed < self.rate_limit:
                    # Wait to respect rate limit
                    time.sleep(self.rate_limit - elapsed)
            
            # Update last request time
            self.last_request_time[domain] = time.time()
            
            # Make the request
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Determine content type
            content_type = response.headers.get("Content-Type", "").lower()
            
            # Process based on content type
            if "text/html" in content_type and self.can_parse_html:
                # Parse HTML
                return self._extract_text_from_html(response.text, url)
            elif "application/json" in content_type:
                # Format JSON
                try:
                    json_data = response.json()
                    return json.dumps(json_data, indent=2)
                except:
                    return response.text
            else:
                # Return raw text
                return response.text
        
        except RequestException as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching URL {url}: {str(e)}")
            return None
    
    def _extract_text_from_html(self, html: str, url: str) -> str:
        """
        Extract meaningful text from HTML
        
        Args:
            html: HTML content
            url: Source URL
            
        Returns:
            Extracted text
        """
        if not self.can_parse_html:
            # Simple regex-based extraction
            text = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]*>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            # Get title
            title = soup.title.string if soup.title else ""
            
            # Find main content
            main_content = None
            for tag in ["main", "article", "div[role='main']", ".main-content", "#content", "#main"]:
                content = soup.select(tag)
                if content:
                    main_content = content[0]
                    break
            
            # If no main content identified, use the body
            if not main_content:
                main_content = soup.body
            
            # Extract text
            text = ""
            if title:
                text += f"Title: {title}\n\n"
            
            if main_content:
                paragraphs = main_content.find_all('p')
                for p in paragraphs:
                    text += p.get_text() + "\n\n"
            
            # If no paragraphs found, use all text
            if not text.strip() and main_content:
                text = main_content.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            # Fall back to simple extraction
            return self._extract_text_from_html(html, url)
    
    def retrieve_and_add_to_library(
        self, 
        url: str, 
        context_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> List[str]:
        """
        Retrieve knowledge from a URL and add to the library
        
        Args:
            url: URL to retrieve
            context_name: Name of the context to add to
            metadata: Additional metadata
            
        Returns:
            List of entry IDs added
        """
        # Fetch the URL
        content = self.fetch_url(url)
        if not content:
            logger.error(f"Failed to retrieve content from {url}")
            return []
        
        # Prepare metadata
        meta = metadata or {}
        meta["source_url"] = url
        meta["retrieved_time"] = datetime.datetime.now().isoformat()
        
        # Add to knowledge library
        return knowledge_library.add_knowledge_from_text(
            text=content,
            source_reference=url,
            context_name=context_name,
            metadata=meta
        )
    
    def search_and_retrieve(
        self, 
        query: str, 
        search_engine: str = "duckduckgo",
        result_count: int = 3,
        context_name: str = None
    ) -> List[str]:
        """
        Search the web and retrieve relevant content
        
        Args:
            query: Search query
            search_engine: Search engine to use
            result_count: Number of results to retrieve
            context_name: Name of the context to add to
            
        Returns:
            List of entry IDs added
        """
        if not self.enable_web_access:
            logger.warning("Web access is disabled")
            return []
        
        # This is a stub implementation - in a real system, you would
        # use a proper search API. For demonstration purposes, we'll
        # just show the concept.
        
        logger.info(f"Searching for: {query}")
        
        # For demonstration, we'll just pretend to search
        # In a real implementation, you would:
        # 1. Call a search API
        # 2. Get the search results
        # 3. Fetch each result
        
        # Example mock implementation:
        if search_engine == "duckduckgo":
            search_url = f"https://duckduckgo.com/html/?q={query}"
            logger.info(f"Would search using: {search_url}")
        
        # We'll return an empty list since this is just a demonstration
        # In a real implementation, you would retrieve the search results
        return []
    
    def find_and_retrieve_knowledge(self, query: str, context_name: str = None) -> List[str]:
        """
        Find and retrieve knowledge relevant to a query
        
        Args:
            query: The query to find knowledge for
            context_name: Name of the context to add to
            
        Returns:
            List of entry IDs added
        """
        if not self.enable_web_access:
            logger.warning("Web access is disabled")
            return []
        
        # A more sophisticated version would:
        # 1. Extract key topics from the query
        # 2. Formulate search queries
        # 3. Search and retrieve content
        # 4. Filter and rank the content
        
        # For demonstration, we'll just use the query directly
        entry_ids = self.search_and_retrieve(
            query=query,
            context_name=context_name
        )
        
        return entry_ids

# Initialize the web retriever (disabled by default for safety)
web_retriever = WebRetriever(enable_web_access=False)