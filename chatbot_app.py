import streamlit as st
import PyPDF2
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("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") or "sk-..."  # Replace or use env

st.title("📄 Chat with your File")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)

    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(raw_text)

    # Embed text and create FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = FAISS.from_texts(texts, embeddings)

    # Input box
    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        # Search similar chunks
        docs = docsearch.similarity_search(user_input)

        # Load QA chain
        llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain.run(input_documents=docs, question=user_input)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for sender, msg in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {msg}")

