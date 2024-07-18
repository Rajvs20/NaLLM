import aiohttp
from embedding.base_embedding import BaseEmbedding

class LocalAIEmbedding(BaseEmbedding):
    """Wrapper around a local AI embedding endpoint."""

    def __init__(self, endpoint_url: str =
"http://74.225.197.5:8000/v1/embeddings", model_name: str =
"all-minilm-l6-v2") -> None:
        self.endpoint_url = endpoint_url
        self.model = model_name

    async def generate(self, input: str) -> str:
        payload = {
            "input": input,
            "model": self.model
        }
        headers = {
            "Content-Type": "application/json"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint_url, json=payload,
headers=headers) as response:
                response.raise_for_status()
                embedding = await response.json()
                return embedding["data"][0]["embedding"]