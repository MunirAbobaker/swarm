import unittest
from unittest.mock import patch, MagicMock
import os
from openai import OpenAI

from swarm.client_factory import ClientFactory, BaseClient, HuggingFaceClient, OpenAIClient, OllamaClient

class TestClientFactory(unittest.TestCase):

    def setUp(self):
        # Reset environment variables before each test
        if 'HF_API_TOKEN' in os.environ:
            del os.environ['HF_API_TOKEN']
        if 'ENDPOINT_URL' in os.environ:
            del os.environ['ENDPOINT_URL']

    def test_create_client_huggingface(self):
        with patch.object(HuggingFaceClient, 'create') as mock_create:
            mock_create.return_value = MagicMock()
            client = ClientFactory.create_client('huggingface', api_key='test_key', base_url='http://test.url')
            self.assertIsInstance(client, MagicMock)
            mock_create.assert_called_once()

    def test_create_client_openai(self):
        with patch.object(OpenAIClient, 'create') as mock_create:
            mock_create.return_value = MagicMock()
            client = ClientFactory.create_client('openai', api_key='test_key')
            self.assertIsInstance(client, MagicMock)
            mock_create.assert_called_once()

    def test_create_client_ollama(self):
        with patch.object(OllamaClient, 'create') as mock_create:
            mock_create.return_value = MagicMock()
            client = ClientFactory.create_client('ollama', base_url='http://test.url')
            self.assertIsInstance(client, MagicMock)
            mock_create.assert_called_once()

    def test_create_client_unknown(self):
        with self.assertRaises(ValueError):
            ClientFactory.create_client('unknown_client')

    def test_register_new_client(self):
        class TestClient(BaseClient):
            def __init__(self, api_key=None, base_url=None):
                self.api_key = api_key or os.getenv('TEST_API_KEY')
                if not self.api_key:
                    raise ValueError("API key must be provided either as a parameter or set in the environment variable 'TEST_API_KEY'.")
                self.base_url = base_url or os.getenv('TEST_ENDPOINT_URL')
                if not self.base_url:
                    raise ValueError("Endpoint URL must be provided either as a parameter or set in the environment variable 'TEST_ENDPOINT_URL'.")

            def create(self):
                return MagicMock()

        ClientFactory.register_client('test', TestClient)
        
        with patch.dict(os.environ, {'TEST_API_KEY': 'test_key', 'TEST_ENDPOINT_URL': 'http://test.url'}):
            client = ClientFactory.create_client('test')
            self.assertIsInstance(client, MagicMock)

class TestHuggingFaceClient(unittest.TestCase):

    def setUp(self):
        # Reset environment variables before each test
        if 'HF_API_TOKEN' in os.environ:
            del os.environ['HF_API_TOKEN']
        if 'ENDPOINT_URL' in os.environ:
            del os.environ['ENDPOINT_URL']

    def test_init_with_params(self):
        client = HuggingFaceClient(api_key='test_key', base_url='http://test.url')
        self.assertEqual(client.api_key, 'test_key')
        self.assertEqual(client.base_url, 'http://test.url')

    def test_init_with_env_vars(self):
        os.environ['HF_API_TOKEN'] = 'env_test_key'
        os.environ['ENDPOINT_URL'] = 'http://env.test.url'
        client = HuggingFaceClient()
        self.assertEqual(client.api_key, 'env_test_key')
        self.assertEqual(client.base_url, 'http://env.test.url')

    def test_init_missing_api_key(self):
        with self.assertRaises(ValueError):
            HuggingFaceClient(base_url='http://test.url')

    def test_init_missing_base_url(self):
        with self.assertRaises(ValueError):
            HuggingFaceClient(api_key='test_key')

    def test_create_method(self):
        with patch('swarm.client_factory.OpenAI') as mock_openai:
            client = HuggingFaceClient(api_key='test_key', base_url='http://test.url')
            result = client.create()
            mock_openai.assert_called_once_with(base_url='http://test.url', api_key='test_key')
            self.assertEqual(result, mock_openai.return_value)
