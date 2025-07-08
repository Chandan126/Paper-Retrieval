from sqlalchemy.orm import query
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOllama(model="Gemma3")

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

query = input("Ask a question about arXiv abstracts: ")

result = qa_chain(query)


print("\n💬 Answer:")
print(result["result"])

print("\n📚 Source Documents:")
for doc in result["source_documents"]:
    print("-" * 40)
    print(doc.page_content[:500])