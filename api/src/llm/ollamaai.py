import requests
import json
from typing import Callable, List, Dict
from llm.basellm import BaseLLM
from retry import retry

class OllamaMistral(BaseLLM):
    """Wrapper around Ollama Mistral model."""

    def __init__(self, api_endpoint: str) -> None:
        self.api_endpoint = api_endpoint

    @retry(tries=3, delay=1)
    def generate(self, messages: List[Dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        prompt = self._construct_prompt(messages)
        payload = {
            "model": "mistral",
            "prompt": prompt
        }

        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()

            responses = []
            for line in response.text.splitlines():
                if line.strip():  # Ignore empty lines
                    entry = json.loads(line)
                    responses.append(entry['response'])

            final_response = ''.join(responses)
            return final_response

        except requests.exceptions.HTTPError as e:
            return str(f"Error: {e.response.text}")
        except Exception as e:
            print(f"Retrying LLM call {e}")
            raise Exception()

    async def generateStreaming(self, messages: List[Dict[str, str]], onTokenCallback=Callable[[str], None]) -> str:
        headers = {"Content-Type": "application/json"}
        prompt = self._construct_prompt(messages)
        payload = {
            "model": "mistral",
            "prompt": prompt
        }

        response = requests.post(self.api_endpoint, headers=headers, json=payload, stream=True)
        result = []
        for line in response.iter_lines():
            if line:
                token = line.decode("utf-8")
                result.append(token)
                await onTokenCallback(token)
        return result

    def _construct_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            prompt += f"{role}: {content}\n"
        return prompt

    def num_tokens_from_string(self, string: str) -> int:
        # Assuming the encoding method for Mistral model is similar to OpenAI's GPT models.
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use appropriate encoding method if available
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def max_allowed_token_length(self) -> int:
        # Assuming similar token limits for simplicity; adjust if different for Mistral model.
        return 2049