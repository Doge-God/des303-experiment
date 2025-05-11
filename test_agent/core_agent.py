import logfire
from pydantic_ai.models import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import Agent

class CoreAgent:
    def __init__(self):
        logfire.configure()
        self.model = OpenAIModel(
            model_name='qwen3:latest', provider=OpenAIProvider(base_url='http://127.0.0.1:11434/v1')
        )
        self.agent = Agent(self.model,
            system_prompt="You are a helpful assistent with some tools, try to connect with the user. /nothink", #The user input is an audio transcript that might not be directed at you. Based on the conversation history to determine if the input is valid. If so, give response accordingly, else leave model response empty. Do not use plain text, only call functions.
            retries=3,
            instrument=True
        )
        Agent.instrument_all()

    