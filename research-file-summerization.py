import os
import openai
import sys
import streamlit as st

from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI

sys.path.append('../..')

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

# llm_name = "gpt-3.5-turbo-0301"
llm_name = "gpt-4"

llm = ChatOpenAI(model_name=llm_name, temperature=0)


def summerise_large_pdf_document(fileUrl):
    url = fileUrl
    loader = PyPDFLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    return chain.run(texts)


def is_url_empty(url):
    if url:
        return False
    else:
        return True


st.title("Research Document Summarizer")
# url = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf'
document_url = st.text_input('Pls enter the URL of the Research Paper')

with get_openai_callback() as cb:
    if document_url:
        response = summerise_large_pdf_document(document_url)
        st.write('The Summery of the research paper is ', response)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
