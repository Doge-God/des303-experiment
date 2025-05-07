import random
from pydantic_ai.models import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import Agent
import asyncio
import logfire

logfire.configure()

ollama_model = OpenAIModel(
    model_name='qwen2.5:7b-instruct', provider=OpenAIProvider(base_url='http://127.0.0.1:11434/v1')
)

main_agent = Agent(ollama_model,
                   system_prompt="You are a helpful assistent with some tools, try to connect with the user.", #The user input is an audio transcript that might not be directed at you. Based on the conversation history to determine if the input is valid. If so, give response accordingly, else leave model response empty. Do not use plain text, only call functions.
                   retries=3,
                   instrument=True)
Agent.instrument_all()

@main_agent.tool_plain
def roll_die() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))

async def main():
    model_messages:list[ModelMessage] = None

    while True: 
        user_input = input("You: ")

        result = await main_agent.run(user_input, message_history=model_messages)

        model_messages = result.all_messages()
        print("\nAssistant:", result.output)


if __name__ == "__main__":
    asyncio.run(main())