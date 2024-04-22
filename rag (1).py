# -*- coding: utf-8 -*-
"""rag.ipynb


"""### RAG on using this dataset -> # GPT 3.5"""

!pip install openai

#!pip install --upgrade pip

!pip install transformers torch

!pip install llama-index llama-index-experimental

import logging
import sys
from IPython.display import Markdown, display

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", api_key="sk-proj-fPxf5ypW6BhdQP5XMgetT3BlbkFJI5hOUj2DsSwlUI5ZwN5b")

import openai
openai.api_key = 'sk-proj-fPxf5ypW6BhdQP5XMgetT3BlbkFJI5hOUj2DsSwlUI5ZwN5b'

query_engine = PandasQueryEngine(df=haa_trainChronologies,llm=llm, verbose=True)



file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

# Read the Pandas DataFrame

df = haa_develChronologies

!pip install langchain

!pip install langchain_openai

from langchain_openai.chat_models import ChatOpenAI

# Set your API key and model choice
OPENAI_API_KEY = "sk-proj-0oig5Ll0qDjkUPCGlzq8T3BlbkFJHuYBzj0UJkkUy2Wgfh3O"
MODEL_NAME = "gpt-3.5-turbo"  # Specify the model you want to use

# Initialize the ChatOpenAI model with a specific temperature
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL_NAME, temperature=0.5)

# Example usage
response = model.invoke("What MLB team won the World Series during the COVID-19 pandemic?")
print(response)

!pip show langchain  # Check the installed version
!pip install --upgrade langchain  # Update langchain

!pip install langchain_experimental



#from langchain.agents.agent_types import AgentType

df= haa_develChronologies
from langchain.agents.agent_types import AgentType  # Use the experimental version of AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent  # Assuming you also need this experimental feature

from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key="sk-proj-0oig5Ll0qDjkUPCGlzq8T3BlbkFJHuYBzj0UJkkUy2Wgfh3O", streaming=True)
pandas_df_agent = create_pandas_dataframe_agent(llm,df,verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS,handle_parsing_errors=True,)

response = pandas_df_agent.run("provide me time stemp for observations C0392747")



dataFrameResponse = pandas_df_agent.run("provide me dataframe as string")

type(dataFrameResponse)

!pip install langchain_core

!pip install ChatOpenAI

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
model = llm
chain = model | parser

try:
    chain.invoke({
        "context":dataFrameResponse,
        "question":"which observations are more prone have Hospital acquired pressure injury? "
    })
except Exception as e:
    print(e)

#String Dataframe respone to Text Splitter by langchain

# Note : Chunk size dependent on task you want to achieve with it

from langchain.text_splitter import RecursiveCharacterTextSplitter

df_responseSplitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=15)
df_responseSplitter.split_documents(dataFrameResponse)[:3]



# openai embedding :

from langchain_openai.embeddings import OpenAIEmbeddings

ai_embeddings = OpenAIEmbeddings()
ai_embed_query = embeddings.embed_query("provide me time stemp for observations C0392747")

print(f"length: {len(ai_embed_query)}")



medical_sentence1 = embeddings.embed_query("The timestamps for observations containing C0392747 are as follows - 2104-08-05 - 2104-08-07 - 2104-08-08")
medical_sentence2 = embeddings.embed_query("Subject id : 75")

# now let's examine performance of this medical query

from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(embedded_query, embedded_sentence):

    similarities = [cosine_similarity([embedded_query], [sentence])[0][0] for sentence in embedded_sentences]
    return similarities



similarities_query1 = calculate_cosine_similarity(embedded_medical_query1, medical_sentence1)
similarities_query2 = calculate_cosine_similarity(embedded_medical_query2, medical_sentence2)


print('Similarities for Query 1:', similarities_query1)
print('Similarities for Query 2:', similarities_query2)

# now adding vector store

# Concatenate the columns with a separator for clarity
haa_develAdmittimes['combined'] = haa_develAdmittimes['hadm_id'].astype(str) + " at " + haa_develAdmittimes['admittime'].astype(str)

# Now extract the combined text data
text_data = haa_develAdmittimes['combined'].tolist()

# here we can change multiple vector store as per application

from langchain_community.vectorstores import DocArrayInMemorySearch

# Create vector store from the combined text data
admittime_vectorstore = DocArrayInMemorySearch.from_texts(text_data, embedding=embeddings)

admittime_vectorstore.similarity_search_with_score(query="give me hadm_id of paintents accociated with 3rd january", k=2)

#  retriever

med_retriever = admittime_vectorstore.as_retriever()

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

setup = RunnableParallel(context=med_retriever, question=RunnablePassthrough())
setup.invoke("give me hadm_id of paintents accociated with 3rd january?")

chain = setup | prompt | model | parser
chain.invoke("give me hadm_id of paintents accociated with 2nd january")

