
import os
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
server_ip = os.getenv("SERVER_IP_ADDRESS", "localhost")
server_port = os.getenv("SERVER_PORT_NUMBER", "8000")
server_url = f"http://{server_ip}:{server_port}"

st.set_page_config(page_title="ChatBot")
st.title("RAG & LLM Chatbot")
uploaded_file = st.file_uploader("Upload a text document for indexing", type=["pdf"])

if uploaded_file:
    if st.button("Upload"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{server_url}/upload", files=files)
        st.success(response.json().get("detail", "Uploaded successfully."))

query = st.text_input("Enter your question:")

if st.button("Ask"):
    response = requests.get(f"{server_url}/query", params={"q": query})
    data = response.json()

    if "error" in data:
        st.error(f"Error: {data['error']}")
    else:
        st.success(f"Answer: {data['answer']}")
        st.write("Sources:", data["sources"])
