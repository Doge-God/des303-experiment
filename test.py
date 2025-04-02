from pydantic import BaseModel

import pprint

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models import ModelMessage
from pydantic_ai.messages import TextPart, ModelResponse, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import InstrumentationSettings

import asyncio

instrumentation_settings = InstrumentationSettings(event_mode='logs')


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


main_agent = Agent(ollama_model, result_type=AgentResponseResult, 
                   system_prompt="You are a helpful assistent, try to connect with the user. The user input is an audio transcript that might not be directed at you. Based on the conversation history to determine if the input is valid. If so, give response accordingly, else leave model response empty. Do not use plain text, only call functions.",
                   retries=3) 



async def main():
    model_messages:list[ModelMessage] = None

    while True: 
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        result = await main_agent.run(user_input, message_history=model_messages)
        
        if result.data.is_input_valid:
            if not model_messages:
                model_messages = [result.new_messages()[0]]
            # Append the new message as a single text response
            else:
                model_messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))

            model_messages.append(ModelResponse(model_name=ollama_model.model_name, parts=[TextPart(content=result.data.model_response)]))
            
            print("\nAssistant:", result.data.model_response)
        else:
            print("\nAssistant: (Input not valid, try again.)")
        
        pprint.pprint(model_messages)


if __name__ == "__main__":
    asyncio.run(main())