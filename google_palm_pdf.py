!pip install pinecone-client==2.2.4
!pip install pypdf

!pip install -q google-generativeai

from langchain.llms import GooglePalm
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import pinecone
import os
import sys

!mkdir pdfs

loader = PyPDFDirectoryLoader("pdfs")
data = loader.load()

print(data)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

text_chunks = text_splitter.split_documents(data)

len(text_chunks)

print(text_chunks[2])

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

embedding = GooglePalmEmbeddings()

query_result = embedding.embed_query("Hello World")
print(len(query_result))

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY',"YOUR_PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV","gcp-starter")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

index_name="langchainbot"

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks],embedding=embedding,index_name=index_name)

#docsearch = Pinecone.from_existing_index(index_name, embeddings)

query = "What are the applications of Machine Learning?"

docs = docsearch.similarity_search(query,k=4)

docs

from langchain.llms import GooglePalm
api_key = "YOUR_GOOGLEPALM_API_KEY"

llm = GooglePalm(google_api_key=api_key,temperature=0.7)

QA = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=docsearch.as_retriever())

QA.run(query)

query1 = "what is bank fraud?"

QA.run(query1)

while True:
  user_input = input("Input Prompt:")
  if user_input=='exit':
    print("Exit...")
    sys.exit()
  if user_input=='':
    continue

  result = QA({'query':user_input})
  print("Answer : ",{result['result']})
