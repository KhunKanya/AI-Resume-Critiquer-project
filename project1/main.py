from langchain_core.messages import HumanMessage   # for user input messages
from langchain_openai import ChatOpenAI            # for OpenAI chat models
from langchain.tools import tool                   # for defining tools
from langgraph.prebuilt import create_react_agent  # for creating an agent with tools
from dotenv import load_dotenv    # for loading .env files

load_dotenv()   # Load environment variables from .env file

@tool # Define a tool for basic arithmetic calculations
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmeric calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"
    
@tool # Define a tool for greeting users
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    print("Tool has been called.")
    return f"Hello {name}, I hope you are well today"
# Main function to run the interactive agent
def main():   
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    tools = [calculator, say_hello]
    agent_executor = create_react_agent(model, tools)
    
    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")
    # Interactive loop for user input and agent response
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input == "quit":
            break
        
        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()
        

if __name__ == "__main__":
    main()
