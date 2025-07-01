from smolagents import CodeAgent, OpenAIServerModel, FinalAnswerTool, DuckDuckGoSearchTool, tool
import os

# Set up model
model = OpenAIServerModel(
    temperature=0.5,
    model_id=os.environ["OPENROUTER_MODEL_NAME"],
    api_base=os.environ["OPENROUTER_API_BASE"],
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# # Tool example
# @tool
# def add(a: int, b: int) -> int:
#     """Add two numbers.
    
#     Args:
#         a (int): The first number.
#         b (int): The second number.
        
#     Returns:
#         int: The sum of the two numbers.
#     """
#     return a + b

# Load system prompt
with open('./system-prompt.md', 'r', encoding='utf-8') as file:
    system_prompt = file.read()

# Agent setup
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), FinalAnswerTool()],
    model=model,
    stream_outputs=False,
)

# Start chat memory as plain string
conversation_memory = f"system: {system_prompt}\n"

def chat_loop():
    print("🤖 Hello! I'm your AI assistant. Type 'exit' to quit.\n")
    global conversation_memory

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("🤖 Goodbye!")
            break

        # Add user input to memory
        conversation_memory += f"user: {user_input}\n"

        # Run agent
        response = agent.run(task=conversation_memory, max_steps=3)

        # Extract clean reply
        reply = response if isinstance(response, str) else getattr(response, "final_answer", str(response))
        
        # Print and store response
        print(f"🤖: {reply}")
        conversation_memory += f"assistant: {reply}\n"

if __name__ == "__main__":
    chat_loop()
