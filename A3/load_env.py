from dotenv import load_dotenv, find_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv(find_dotenv())

api_key = os.environ['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings(api_key=api_key)
