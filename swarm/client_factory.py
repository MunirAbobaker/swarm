from abc import ABC, abstractmethod 
from typing import Dict, Type
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class BaseClient(ABC):
    @abstractmethod
    def create(self):
        """Method to create and return the client instance."""
        pass

class OpenAIClient(BaseClient):
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either as a parameter or set in the environment variable 'OPENAI_API_KEY'.")
        self.base_url = base_url or 'https://api.openai.com/v1'

    def create(self):
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

class OllamaClient(BaseClient):
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or "ollama"
        self.base_url = base_url or 'http://localhost:11434/v1'

    def create(self):
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

class HuggingFaceClient(BaseClient):
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or os.getenv('HF_API_TOKEN')
        if not self.api_key:
            raise ValueError("API key must be provided either as a parameter or set in the environment variable 'HF_API_TOKEN'.")
        self.base_url = base_url or os.getenv('ENDPOINT_URL')
        if not self.base_url:
            raise ValueError("Endpoint URL must be provided either as a parameter or set in the environment variable 'ENDPOINT_URL'.")

    def create(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

class ClientFactory:
    _clients: Dict[str, Type[BaseClient]] = {
        'openai': OpenAIClient,
        'ollama': OllamaClient,
        'huggingface': HuggingFaceClient
    }

    @classmethod
    def create_client(cls, client_name: str, api_key: str = None, base_url: str = None) -> OpenAI:
        """
        Factory method to create a client instance based on the client name.

        Parameters:
        - client_name (str): The name of the client to create ('openai', 'ollama', 'huggingface').
        - api_key (str): The API key for authenticating with the service. If not provided, it will be retrieved from the environment.
        - base_url (str): The base URL for the API. If not provided, defaults will be used.

        Returns:
        - An instance of the specified client.

        Raises:
        - ValueError: If an invalid client name is provided.
        """
        client_class = cls._clients.get(client_name.lower())
        if not client_class:
            raise ValueError(f"Unknown client name: {client_name}. Valid options are {', '.join(cls._clients.keys())}.")
        return client_class(api_key, base_url).create()

    @classmethod
    def register_client(cls, name: str, client_class: Type[BaseClient]):
        cls._clients[name.lower()] = client_class

# Example usage
if __name__ == "__main__":
    print("Start factory")
    try:
        openai_client = ClientFactory.create_client('openai')
        print("OpenAI client created successfully.")
    except ValueError as e:
        print(f"Error creating OpenAI client: {e}")

    try:
        huggingface_client = ClientFactory.create_client('huggingface')
        print("Hugging Face client created successfully.")
    except ValueError as e:
        print(f"Error creating Hugging Face client: {e}")

    try:
        ollama_client = ClientFactory.create_client('ollama')
        print("Ollama client created successfully.")
    except ValueError as e:
        print(f"Error creating Ollama client: {e}")