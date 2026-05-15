import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_vaidya(question):
    docs = vectorstore.similarity_search(question, k=4)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are AI Vaidya, an expert Ayurvedic assistant.
Answer the question below using ONLY the context provided from Ayurveda texts.
If the answer is not in the context, say "I could not find this in the provided Ayurveda text."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    sources = [doc.page_content[:200] for doc in docs]
    
    return answer, sources

if __name__ == "__main__":
    print("AI Vaidya is ready! Type your question below.")
    print("Type 'quit' to exit\n")
    while True:
        question = input("Your question: ")
        if question.lower() == "quit":
            break
        answer, sources = ask_vaidya(question)
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources found: {len(sources)} passages")
        print("-" * 50)