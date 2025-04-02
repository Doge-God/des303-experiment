from pydantic import BaseModel

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models import ModelMessage
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
    user_input: str
    model_response: str

ollama_model = OpenAIModel(
    model_name='qwen2.5:14b', provider=OpenAIProvider(base_url='http://127.0.0.1:11434/v1')
)


main_agent = Agent(ollama_model, result_type=AgentResponseResult, system_prompt="You are a helpful assistent. The user input is an audio transcript that might not be directed at you. Respond accordingly based on the conversation history to determine if the input is valid. If so, give response accordingly and format the user input, else leave model response and user input empty. Do not use plain text, only call functions.") 

model_messages:list[ModelMessage] = []


with capture_run_messages() as messages:
    try:
        result = main_agent.run_sync('What is the')
        print(result.data)
        print(result.all_messages_json())
    except Exception:
        print(messages)
        raise

# async def main():
#     # result = await agent.run('What is the capital of France?')
#     # print(result.data)
#     # #> Paris
     
#     async with agent.run_stream('What is the capital of the UK?') as response:
#         print(response.all_messages_json())
#         print(await response.get_data())

# if __name__ == "__main__":
#     asyncio.run(main())