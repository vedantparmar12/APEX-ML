"""OpenRouter API Client for LLM interactions"""

import os
import json
import requests
from typing import List, Dict, Optional, Any
import time
from config.config import CONFIG


class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or CONFIG.openrouter_api_key
        self.model = model or CONFIG.model_name
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/one-above-all",
            "X-Title": "One-Above-All ML System"
        }
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Send chat completion request to OpenRouter"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if response_format:
            payload["response_format"] = response_format
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    wait_time = min(2 ** attempt * 5, 60)
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"OpenRouter API error: {str(e)}")
                time.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded for OpenRouter API")
    
    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        """Get a simple text completion"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response_format = {"type": "json_object"} if json_mode else None
        
        response = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        return response["choices"][0]["message"]["content"]
    
    def get_structured_output(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Get structured JSON output from the model"""
        
        json_prompt = prompt + "\n\nRespond with valid JSON only, no additional text."
        
        response = self.get_completion(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=True
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response}")


# Global client instance
llm_client = OpenRouterClient()