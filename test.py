import random
from pydantic import BaseModel

import pprint

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models import ModelMessage
from pydantic_ai.messages import TextPart, ModelResponse, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import InstrumentationSettings
import logfire

logfire.configure()

import asyncio

Agent.instrument_all()

class CityLocation(BaseModel):
    city: str
    country: str

class UserInputResult(BaseModel):
    is_complete: bool
    is_relevant: bool

class AgentResponseResult(BaseModel):
    is_input_valid: bool
    model_response: str

ollama_model = OpenAIModel(
    model_name='qwen2.5:7b-instruct', provider=OpenAIProvider(base_url='http://127.0.0.1:11434/v1')
)


main_agent = Agent(ollama_model, output_type=AgentResponseResult, 
                   system_prompt="You are a helpful assistent with some tools, try to connect with the user.", 
                   retries=3) 

filter_agent = Agent(ollama_model, output_type=bool,
                      system_prompt="The user input is an audio transcript that might not be directed at you. Determine if the input is directed at you. Do not use plain text, only call functions")

@main_agent.tool_plain
def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

async def main():
    model_messages:list[ModelMessage] = None

    while True: 
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        with capture_run_messages() as capture:
            try:
                result = await main_agent.run(user_input, message_history=model_messages)
            except Exception as e:
                pprint.pprint(capture)
        
        if result.output.is_input_valid:
            if not model_messages:
                model_messages = [result.new_messages()[0]]
            else:
                model_messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))

            model_messages.append(ModelResponse(model_name=ollama_model.model_name, parts=[TextPart(content=result.output.model_response)]))
            
            print("\nAssistant:", result.output.model_response)
        else:
            print("\nAssistant: (Input not valid, try again.)")


if __name__ == "__main__":
    asyncio.run(main())