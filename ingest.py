import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PDF_PATH = "data/Charaka-Samhita-Acharya-Charaka.pdf"  

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("Creating embeddings and saving to database...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

print("Done! Database saved in db/ folder")