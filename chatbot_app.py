import streamlit as st
import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# You must set your OPENAI_API_KEY in environment or Streamlit secrets,
# no need to pass it explicitly to LangChain classes

st.title("📊 Chat with your Excel file")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def read_excel_to_text(file):
    df = pd.read_excel(file)
    return df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ')

if uploaded_file:
    raw_text = read_excel_to_text(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()  # no key param
    db = FAISS.from_texts(texts, embeddings)

    user_input = st.text_input("Ask a question about the Excel data:")

    if user_input:
        docs = db.similarity_search(user_input)
        llm = OpenAI()  # no key param
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_input)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")
else:
    st.info("Please upload an Excel file to start chatting!")
