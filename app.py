import os
import streamlit as st
import pickle
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores.faiss import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize the HuggingFaceHub LLM (Mistral model)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
    model_kwargs={
        "temperature": 0.6,
        "max_new_tokens": 512,
        "huggingfacehub_api_token": os.getenv("HUGGINGFACEHUB_API_TOKEN")
    }
)

# Set up Streamlit page config
st.set_page_config(page_title="Article Research Tool", layout="wide")
st.title("📰 News Research Tool")
st.sidebar.title("🔗 Enter Article URLs")

# Add input fields for URLs in the sidebar
urls = []
for i in range(3): 
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)
url_btn = st.sidebar.button("Process URLs")

# Function to process URLs and create a FAISS index
def process_urls(urls):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n', '.', ','],
            chunk_size=256
        )
        docs = text_splitter.split_documents(data)

        # Create embeddings and FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_index = FAISS.from_documents(docs, embedding=embeddings)

        # Save the index to a pickle file
        with open('vector_index.pkl', 'wb') as f: 
            pickle.dump({
                'index': vector_index.index, 
                'docstore': vector_index.docstore, 
                'index_to_docstore_id': vector_index.index_to_docstore_id
            }, f)
        return True
    except Exception as e:
        st.error(f"Error processing URLs: {str(e)}")
        return False

# Handle URL button click
if url_btn:
    if urls:
        success = process_urls(urls)
        if success:
            st.success("✅ URLs processed and index created successfully!")
    else:
        st.warning("⚠️ Please enter at least one URL.")

# Text input for user query
query = st.text_input("🔍 Ask a Question:", placeholder="Enter your question here...")

# Function to extract answer and source using regex
def extract_answer_and_source(final_answer):
    answer = re.search(r"Helpful Answer:\s*(.+?)(?:Retrieved from|$)", final_answer, re.DOTALL)
    source = re.search(r"Retrieved from:\s*(.+)", final_answer)
    return (answer.group(1).strip() if answer else "Answer not found.", 
            source.group(1).strip() if source else "Source not found.")

# Function to get answer based on the query
def get_answer(query):
    try:
        # Load embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

        # Load saved FAISS index
        with open('vector_index.pkl', 'rb') as f:
            saved_data = pickle.load(f)
            index = saved_data['index']
            docstore = saved_data['docstore']
            index_to_docstore_id = saved_data['index_to_docstore_id']
        
        # Initialize FAISS store and retrieval chain
        faiss_store = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embeddings.embed_query)
        chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff",
            retriever=faiss_store.as_retriever()
        )
        
        # Get the result from the retrieval chain
        result = chain.run(query)
        return result

    except Exception as e:
        st.error(f"Error retrieving answer: {str(e)}")
        return None

# Handle query submission
if query:
    with st.spinner("🔄 Generating answer..."):
        final_answer = get_answer(query)
        if final_answer:
            # Extract the answer and source
            answer_start, source_start = extract_answer_and_source(final_answer)

            # Display the answer
            st.subheader("✅ Answer:")
            st.write(answer_start)
            
            # Display the source only if it's available
            if source_start != "Source not found.":
                st.subheader("✅ Source:")
                st.write(source_start)
        else:
            st.warning("⚠️ No answer could be generated. Please try again.")
