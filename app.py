import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

load_dotenv()

st.set_page_config(
    page_title="AI Vaidya",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #f9f5f0; }
    .answer-box {
        background-color: #ffffff;
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 16px;
        line-height: 1.8;
        color: #000000 !important;
    }
    .allergy-box {
        background-color: #fff8e1;
        border-left: 5px solid #FFA726;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 15px;
        line-height: 1.8;
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 AI Vaidya")
st.subheader("Your Intelligent Ayurvedic Assistant")
st.markdown("Upload any Ayurveda PDF and ask questions from it.")

@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

client = load_groq_client()

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.unlink(tmp_path)
    return vectorstore

with st.sidebar:
    st.markdown("### 📂 Upload Ayurveda PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF... please wait..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
        st.success(f"✅ {uploaded_file.name} loaded successfully!")

    st.markdown("---")
    st.markdown("### 🤧 Your Allergies & Restrictions")
    st.markdown("Enter anything you are allergic to or want to avoid:")

    allergy_input = st.text_area(
        "Allergies / Restrictions",
        placeholder="e.g. dairy, nuts, sesame, gluten, honey...",
        height=120
    )

    if allergy_input.strip() != "":
        st.warning(f"⚠️ Avoiding: {allergy_input}")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    AI Vaidya is a retrieval-based Q&A system
    built on classical Ayurveda texts.
    Upload any Ayurveda PDF and ask questions.
    """)
    st.markdown("### 🔍 Sample Questions")
    st.markdown("""
    - What is Vata dosha?
    - What are the benefits of Ashwagandha?
    - How does Ayurveda describe digestion?
    - What is Panchakarma?
    - What foods balance Pitta dosha?
    """)

if "vectorstore" not in st.session_state:
    st.info("👈 Please upload an Ayurveda PDF from the sidebar to get started.")
else:
    question = st.text_input(
        "Ask your Ayurveda question:",
        placeholder="e.g. What is Vata dosha?"
    )

    if st.button("🔍 Ask AI Vaidya", use_container_width=True):
        if question.strip() == "":
            st.warning("Please type a question first!")
        else:
            with st.spinner("Searching Ayurveda texts..."):
                docs = st.session_state.vectorstore.similarity_search(question, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])

                if allergy_input.strip() != "":
                    allergy_section = f"""
The person asking this question has the following allergies or restrictions: {allergy_input}
Make sure your answer does NOT recommend anything the person is allergic to.
If the standard Ayurvedic remedy contains something they are allergic to, suggest a safe alternative instead.
Clearly mention if you are suggesting an alternative due to their allergy."""
                else:
                    allergy_section = ""

                prompt = f"""You are AI Vaidya, an expert Ayurvedic assistant.
Your job is to give a clear, detailed, and helpful answer to the question.
Use the context below from Ayurveda texts to form your answer.
Write the answer in simple English in paragraph form.
Do NOT repeat the context. Do NOT list references. Just answer the question directly.
If the answer is not in the context, say "I could not find this in the provided Ayurveda text."
{allergy_section}

Context:
{context}

Question: {question}

Answer in clear paragraphs:"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content

            st.markdown("### 💬 Answer")

            if allergy_input.strip() != "":
                st.markdown(
                    f'<div class="allergy-box">⚠️ This answer has been personalised considering your allergies/restrictions: <b>{allergy_input}</b></div>',
                    unsafe_allow_html=True
                )

            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)