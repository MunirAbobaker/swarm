from swarm import Swarm, Agent, ClientFactory
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

client = Swarm(ClientFactory.create_client('ollama', base_url=os.getenv('ENDPOINT_URL'), api_key=os.getenv('HF_API_TOKEN')))

english_agent = Agent(
    name="English Agent",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish.",
)


def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent


english_agent.functions.append(transfer_to_spanish_agent)

messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
response = client.run(agent=english_agent, messages=messages)

print(response.messages[-1]["content"])
