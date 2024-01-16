

from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain

# Load environment variables
load_dotenv(find_dotenv())

# --------------------------------------------------------------
# LLMs: Get predictions from a language model
# --------------------------------------------------------------

llm = OpenAI(model_name="text-davinci-003")
prompt = "Write a haiku about natural language processing"
print(llm(prompt))


# --------------------------------------------------------------
# Prompt Templates: Manage prompts for LLMs
# --------------------------------------------------------------

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

prompt.format(product="Smart Apps using Large Language Models (LLMs)")

# --------------------------------------------------------------
# Memory: Add state to chains and agents
# --------------------------------------------------------------

llm = OpenAI()
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(
    input="I'm doing well! The weather is great and I'm having a conversation with an AI."
)
print(output)

# --------------------------------------------------------------
# Chains: Combine LLMs and prompts
# --------------------------------------------------------------

llm = OpenAI()
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("AI assistant to study the content of web pages related to academic articles"))

# --------------------------------------------------------------
# Agents: Call chains based on user input
# --------------------------------------------------------------

llm = OpenAI()

get_all_tool_names()
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Initialize an agent with the tools, the language model, and the type of agent we want to use
# Create an agent, give it access to wikipedia, it should be able to do some math
# Initialize the agent, provide it with tools
# Zero-shot react description: based on the prompt, it will pick the best tool to solve the problem 

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Testing
result = agent.run(
    "In what year was the movie The Shining released and who was the director? Multiply the number corresponding to the year that The Shining was released in by 5"
)
print(result)


