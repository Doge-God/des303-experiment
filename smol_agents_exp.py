from smolagents import MultiStepAgent,CodeAgent, LiteLLMModel, PromptTemplates, tool, GradioUI, ToolCallingAgent

model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:7b-instruct", 
    api_base="http://localhost:11434",
    num_ctx=8192, # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

@tool
def roll_dice() -> int:
    '''
    This is a random number generator that simulate a die. It generate a number between 1 and 6.
    '''
    return 5


agent = CodeAgent(tools=[roll_dice], model=model, add_base_tools=False, max_steps=5, verbosity_level=3)

# agent.run(
#     "Roll a dice for me."
# )




# print(agent.logs)

def main():
    GradioUI(agent).launch()

if __name__ == "__main__":
    main()