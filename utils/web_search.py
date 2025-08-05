"""Web search utilities using DuckDuckGo"""

from typing import List, Dict, Optional
import asyncio
from duckduckgo_search import DDGS
import re
from bs4 import BeautifulSoup
import requests
from config.config import CONFIG


class WebSearcher:
    """Web search functionality using DuckDuckGo"""
    
    def __init__(self):
        self.ddgs = DDGS()
        
    def search(
        self,
        query: str,
        max_results: int = None,
        region: str = "wt-wt"  # worldwide
    ) -> List[Dict[str, str]]:
        """Search the web and return results"""
        
        max_results = max_results or CONFIG.max_search_results
        
        try:
            results = []
            for result in self.ddgs.text(
                query,
                region=region,
                max_results=max_results
            ):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("body", "")
                })
            
            return results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def search_code(
        self,
        query: str,
        language: str = "python",
        max_results: int = None
    ) -> List[Dict[str, str]]:
        """Search specifically for code examples"""
        
        # Add code-specific keywords
        code_query = f"{query} {language} code example implementation github kaggle"
        
        results = self.search(code_query, max_results)
        
        # Filter for likely code sources
        code_sources = ["github.com", "kaggle.com", "stackoverflow.com", 
                       "medium.com", "towardsdatascience.com"]
        
        filtered_results = []
        for result in results:
            url = result.get("url", "")
            if any(source in url for source in code_sources):
                filtered_results.append(result)
        
        return filtered_results
    
    def extract_code_from_url(self, url: str) -> Optional[str]:
        """Extract code snippets from a URL"""
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for code blocks
            code_blocks = []
            
            # GitHub code blocks
            for code in soup.find_all('pre'):
                code_text = code.get_text().strip()
                if code_text:
                    code_blocks.append(code_text)
            
            # Markdown code blocks
            for code in soup.find_all('code'):
                code_text = code.get_text().strip()
                if len(code_text) > 50:  # Filter out inline code
                    code_blocks.append(code_text)
            
            # Join all code blocks
            if code_blocks:
                return "\n\n".join(code_blocks[:3])  # Limit to first 3 blocks
            
            return None
            
        except Exception as e:
            print(f"Error extracting code from {url}: {str(e)}")
            return None
    
    def search_ml_models(
        self,
        task_description: str,
        task_type: str,
        num_results: int = 5
    ) -> List[Dict[str, str]]:
        """Search for ML models suitable for a specific task"""
        
        # Construct search query based on task
        queries = [
            f"best {task_type} models {task_description}",
            f"{task_type} state-of-the-art models kaggle competition",
            f"{task_type} winning solution code implementation"
        ]
        
        all_results = []
        seen_urls = set()
        
        for query in queries:
            results = self.search_code(query, max_results=num_results)
            
            for result in results:
                url = result.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    
                    # Try to extract code
                    code = self.extract_code_from_url(url)
                    if code:
                        result["code_snippet"] = code
                    
                    all_results.append(result)
        
        return all_results[:num_results]


# Global search instance
web_searcher = WebSearcher()