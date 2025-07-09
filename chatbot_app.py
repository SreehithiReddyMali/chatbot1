import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.title("📊 Chat with your Excel file")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def read_excel_to_text(file) -> str:
    # Read Excel and join all rows and columns into one big string
    df = pd.read_excel(file)
    # Convert all cells to string and concatenate by rows, then by entire dataframe
    return df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ')

if uploaded_file:
    raw_text = read_excel_to_text(uploaded_file)

    # Split large text into smaller chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(raw_text)

    # Create embeddings for the chunks (OpenAI API key taken from env)
    embeddings = OpenAIEmbeddings()

    # Build vector store with embeddings for similarity search
    db = FAISS.from_texts(texts, embeddings)

    user_input = st.text_input("Ask a question about the Excel data:")

    if user_input:
        # Retrieve relevant chunks by similarity search
        docs = db.similarity_search(user_input)

        # Load LLM and QA chain
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")

        # Run the chain with retrieved docs and user question
        response = chain.run(input_documents=docs, question=user_input)

        # Append messages to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")
else:
    st.info("Please upload an Excel file to start chatting!")
