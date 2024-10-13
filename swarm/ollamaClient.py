from openai import OpenAI

def getOpenAIClient() -> OpenAI:
    client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama'  # required, but unused
            ) 
    return client