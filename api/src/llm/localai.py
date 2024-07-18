import aiohttp
import json
from typing import Callable, List, Any
from llm.basellm import BaseLLM
from abc import ABC, abstractmethod
import requests

class LocalAIChat(BaseLLM):
    """Wrapper around a local AI model running on localhost."""

    def __init__(self, api_url: str, model_name: str, temperature: float, max_tokens: int = 1000, timeout: int = 60):
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, messages: List[str]) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "prompt": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            print("------------------------------------------------------------------")
            print(data)
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"

    async def generateStreaming(self, messages: List[str], onTokenCallback: Callable[[str], None]) -> List[str]:
        payload = {
            "model": self.model_name,
            "prompt": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        headers = {
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=self.timeout) as response:
                result = []
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                result.append(delta["content"])
                                await onTokenCallback(delta["content"])
                    return result
                else:
                    print(f"Failed to get a response. Status code: {response.status}, Response: {await response.text()}")
                    return []
        return []

    async def num_tokens_from_string(self, string: str) -> int:
        # Example tokenization method: simple whitespace split
        return len(string.split())

    async def max_allowed_token_length(self) -> int:
        # Example token limit
        return 2048
