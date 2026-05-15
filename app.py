import os
import tempfile
import streamlit as st
from datetime import datetime
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
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #f9f5f0 0%, #e8f5e9 100%);
    }

    .hero {
        background: linear-gradient(135deg, #1b5e20, #2e7d32, #388e3c);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        color: white;
    }

    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 52px;
        margin: 0;
        color: white;
    }

    .hero p {
        font-size: 18px;
        color: #c8e6c9;
        margin-top: 10px;
    }

    .user-bubble {
        background: linear-gradient(135deg, #2e7d32, #43a047);
        border-radius: 20px 20px 0px 20px;
        padding: 14px 20px;
        margin: 8px 0px;
        font-size: 15px;
        color: #ffffff;
        text-align: right;
        margin-left: 25%;
        box-shadow: 0 2px 8px rgba(46,125,50,0.3);
    }

    .ai-bubble {
        background: #ffffff;
        border-left: 5px solid #4CAF50;
        border-radius: 0px 20px 20px 20px;
        padding: 18px 22px;
        margin: 8px 0px;
        font-size: 15px;
        line-height: 1.9;
        color: #1a1a1a;
        margin-right: 25%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .allergy-badge {
        background: linear-gradient(135deg, #fff8e1, #ffecb3);
        border-left: 5px solid #FFA726;
        padding: 12px 18px;
        border-radius: 12px;
        font-size: 13px;
        color: #5d4037;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(255,167,38,0.2);
    }

    .chat-header {
        font-size: 12px;
        color: #888888;
        margin-bottom: 3px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .chat-header-right {
        font-size: 12px;
        color: #888888;
        margin-bottom: 3px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-align: right;
    }

    .journal-card {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        margin: 12px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #4CAF50;
        color: #1a1a1a;
    }

    .journal-date {
        font-size: 12px;
        color: #888;
        margin-bottom: 6px;
        font-weight: 500;
    }

    .journal-mood {
        font-size: 22px;
        margin-right: 8px;
    }

    .journal-title {
        font-family: 'Playfair Display', serif;
        font-size: 18px;
        color: #2e7d32;
        margin-bottom: 8px;
    }

    .journal-body {
        font-size: 14px;
        color: #444;
        line-height: 1.7;
    }

    .journal-tag {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        margin: 4px 3px 0 0;
        font-weight: 500;
    }

    .ai-insight-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border-left: 4px solid #66bb6a;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 14px;
        color: #1b5e20;
        line-height: 1.7;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #43a047) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 15px rgba(46,125,50,0.4) !important;
    }

    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #c8e6c9 !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        background: white !important;
        color: #1a1a1a !important;
    }

    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 2px solid #c8e6c9 !important;
        background: white !important;
        color: #1a1a1a !important;
    }

    .stats-bar {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }

    .stat-chip {
        background: white;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 13px;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        font-weight: 500;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 100%) !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #4CAF50, transparent);
        margin: 20px 0;
    }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #555;
    }

    .empty-state h2 {
        font-family: 'Playfair Display', serif;
        color: #2e7d32;
        font-size: 28px;
    }

    .page-title {
        font-family: 'Playfair Display', serif;
        font-size: 32px;
        color: #1b5e20;
        margin-bottom: 5px;
    }

    .page-subtitle {
        font-size: 15px;
        color: #666;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    os.unlink(tmp_path)
    return vectorstore, len(chunks), len(documents)

def get_ai_insight(entry_text, mood, symptoms):
    prompt = f"""You are AI Vaidya, an Ayurvedic wellness assistant.
A person has written this journal entry:
"{entry_text}"
Their mood today: {mood}
Their symptoms today: {symptoms if symptoms else "None mentioned"}

Give a short, warm, personalised Ayurvedic insight (3-4 sentences) based on their journal entry.
Suggest one simple Ayurvedic tip or remedy that could help them today.
Be encouraging and supportive."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = 0
if "journal_entries" not in st.session_state:
    st.session_state.journal_entries = []

with st.sidebar:
    st.markdown("## 🌿 AI Vaidya")
    st.markdown("*Your Ayurvedic Knowledge Assistant*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["💬 Ask AI Vaidya", "📔 My Wellness Journal"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose an Ayurveda PDF", type="pdf")

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                vectorstore, chunks, pages = process_pdf(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_pages = pages
                st.session_state.pdf_chunks = chunks
            st.success("✅ Loaded!")
            st.markdown(f"📄 **{uploaded_file.name}**")
            st.markdown(f"📃 {pages} pages · 🧩 {chunks} chunks")

    st.markdown("---")
    st.markdown("### 🤧 Allergies & Restrictions")
    allergy_input = st.text_area(
        "What should AI Vaidya avoid?",
        placeholder="e.g. dairy, nuts, gluten...",
        height=80
    )
    if allergy_input.strip():
        st.warning(f"⚠️ Avoiding: {allergy_input}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("""
<div class="hero">
    <h1>🌿 AI Vaidya</h1>
    <p>Ancient Ayurvedic Wisdom · Powered by Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)

if page == "💬 Ask AI Vaidya":
    if st.session_state.vectorstore is None:
        st.markdown("""
        <div class="empty-state">
            <h2>Welcome to AI Vaidya 🌿</h2>
            <p>Upload an Ayurveda PDF from the sidebar to begin.</p>
            <br>
            <p>📚 Supports any Ayurveda book or research paper</p>
            <p>🤖 Powered by advanced AI for accurate answers</p>
            <p>🌱 Personalised responses based on your allergies</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.chat_history:
            st.markdown(f"""
            <div class="stats-bar">
                <div class="stat-chip">📄 {st.session_state.pdf_name}</div>
                <div class="stat-chip">💬 {len(st.session_state.chat_history)} questions asked</div>
                <div class="stat-chip">📃 {st.session_state.pdf_pages} pages loaded</div>
            </div>
            """, unsafe_allow_html=True)

        for chat in st.session_state.chat_history:
            st.markdown('<div class="chat-header-right">🧑 You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-bubble">{chat["question"]}</div>', unsafe_allow_html=True)
            if chat["allergies"]:
                st.markdown(f'<div class="allergy-badge">⚠️ Personalised for: <b>{chat["allergies"]}</b></div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-header">🌿 AI Vaidya</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-bubble">{chat["answer"]}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        question = st.text_input(
            "Ask your Ayurveda question:",
            placeholder="e.g. What is Vata dosha?",
            key="question_input"
        )

        if st.button("🔍 Ask AI Vaidya", use_container_width=True):
            if question.strip() == "":
                st.warning("Please type a question first!")
            else:
                with st.spinner("🌿 Searching Ayurveda wisdom..."):
                    docs = st.session_state.vectorstore.similarity_search(question, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    allergy_section = ""
                    if allergy_input.strip():
                        allergy_section = f"""
The person has these allergies: {allergy_input}
Do NOT recommend anything they are allergic to.
If the remedy contains their allergen suggest a safe alternative."""

                    prompt = f"""You are AI Vaidya, an expert Ayurvedic assistant.
Give a clear, detailed, helpful answer using the Ayurveda context below.
Write in simple English in paragraph form.
Do NOT repeat the context or list references. Answer directly.
If not found say: "I could not find this in the provided Ayurveda text."
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
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "allergies": allergy_input.strip()
                    })
                st.rerun()

elif page == "📔 My Wellness Journal":
    st.markdown('<div class="page-title">📔 My Wellness Journal</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Track your daily health, mood, and get personalised Ayurvedic insights</div>', unsafe_allow_html=True)

    with st.expander("✏️ Write a New Journal Entry", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            entry_title = st.text_input("Title", placeholder="e.g. Feeling better today")
            mood = st.selectbox("Mood", ["😊 Happy", "😔 Sad", "😴 Tired", "😤 Stressed", "😌 Calm", "🤒 Unwell", "💪 Energetic"])
            symptoms = st.text_input("Any symptoms?", placeholder="e.g. headache, bloating, fatigue")

        with col2:
            entry_body = st.text_area(
                "How are you feeling today?",
                placeholder="Write about your health, diet, sleep, energy levels...",
                height=150
            )
            tags = st.text_input("Tags", placeholder="e.g. sleep, digestion, stress")

        get_insight = st.checkbox("🌿 Get AI Ayurvedic Insight for this entry")

        if st.button("💾 Save Journal Entry", use_container_width=True):
            if entry_body.strip() == "":
                st.warning("Please write something in your journal first!")
            else:
                insight = ""
                if get_insight:
                    with st.spinner("🌿 Getting your Ayurvedic insight..."):
                        insight = get_ai_insight(entry_body, mood, symptoms)

                entry = {
                    "date": datetime.now().strftime("%d %B %Y · %I:%M %p"),
                    "title": entry_title if entry_title.strip() else "My Journal Entry",
                    "mood": mood,
                    "symptoms": symptoms,
                    "body": entry_body,
                    "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    "insight": insight
                }
                st.session_state.journal_entries.insert(0, entry)
                st.success("✅ Journal entry saved!")
                st.rerun()

    st.markdown("---")

    if not st.session_state.journal_entries:
        st.markdown("""
        <div class="empty-state">
            <h2>No Journal Entries Yet 📔</h2>
            <p>Write your first entry above to start tracking your wellness journey.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"### 📋 Your Entries ({len(st.session_state.journal_entries)} total)")

        for i, entry in enumerate(st.session_state.journal_entries):
            tags_html = "".join([f'<span class="journal-tag">#{t}</span>' for t in entry["tags"]])

            st.markdown(f"""
            <div class="journal-card">
                <div class="journal-date">🗓️ {entry["date"]}</div>
                <div class="journal-title"><span class="journal-mood">{entry["mood"].split()[0]}</span>{entry["title"]}</div>
                <div class="journal-body">{entry["body"]}</div>
                <br>
                {f'<div><b>Symptoms:</b> {entry["symptoms"]}</div>' if entry["symptoms"] else ""}
                <div style="margin-top:8px">{tags_html}</div>
                {f'<div class="ai-insight-box">🌿 <b>AI Vaidya Insight:</b><br>{entry["insight"]}</div>' if entry["insight"] else ""}
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                st.session_state.journal_entries.pop(i)
                st.rerun()