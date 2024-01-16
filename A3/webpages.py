from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

url = "https://docs.google.com/document/d/17uFz_AGQ1C6ZQ7vRR2qPszMtN2Yg29AgUGcUrRxanYc/edit"

# gpt-3.5- turbo can handle up to 4097 tokens, setting the chunksize to 1000 and k to 4 maximizes the number of tokens to analyze

def create_db(url: str) -> FAISS:
    loader = WebBaseLoader(url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def query_response(db, query, k = 4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

     # System message prompt
    template = """
        You are a helpful assistant that that can answer questions about webpages 
        based on the page content: {docs}

        The webpages used as input for this task are academic papers with a focus on subfields of artificial intelligence, 
        specifically natural language processing.
        
        Your task is to be an agent that can help users study, understand and learn the content of the articles from the webpages. 
        Your goal is to help users comprehend complex concepts and ideas presented in academic articles. 
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose, detailed, informative and clearly explain the subject matter in a clear and concise manner.
        Your answers should be easy to understand by someone without much domain knowledge within the field of the article. 

        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


# Example usage:
url = "https://docs.google.com/document/d/17uFz_AGQ1C6ZQ7vRR2qPszMtN2Yg29AgUGcUrRxanYc/edit"
db = create_db(url)

query = "Can you make a quiz consisting of 10 questions for me to study related to sequence transduction model architecture?"
response, docs = query_response(db, query)
print(textwrap.fill(response, width=50))








