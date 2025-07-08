from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

dataset = load_dataset("gfissore/arxiv-abstracts-2021")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

raw_text = [
    f"{entry['title']}\n\n{entry['abstract']}"
    for entry in dataset['train']
    if entry.get('abstract') and entry.get('title')
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

documents = text_splitter.create_documents(raw_text)

vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory="chroma_db")
vectorstore.persist()

print("✅ Vector store created and saved locally.")