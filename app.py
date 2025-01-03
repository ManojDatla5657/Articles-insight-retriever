import os
import streamlit as st
import pickle
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import re

# Set up Streamlit page config
st.set_page_config(page_title="Article Research Tool", layout="wide")
st.title("📰 Article Research Tool")
st.sidebar.title("🔧 **Configuration Panel**")

# Sidebar for user inputs
with st.sidebar:
    st.markdown("### 🔑 **API Keys**")

    # Hugging Face API token input
    huggingfacehub_api_token = st.text_input(
        "Hugging Face API Token",
        placeholder="Enter your Hugging Face API token",
        type="password"
    )

    # Google API key input
    google_api_key = st.text_input(
        "Google API Key",
        placeholder="Enter your Google API key",
        type="password"
    )

    # Validation check for API keys
    if not huggingfacehub_api_token or not google_api_key:
        st.warning("⚠️ Please enter both API keys to proceed.")
        st.stop()
    else:
        st.success("✅ API keys loaded successfully!")

    # Load the Hugging Face API token into environment variables
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_api_token
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Horizontal divider
    st.markdown("---")

    # Section for URL input
    st.markdown("### 🌐 **Article URLs**")
    st.markdown("Enter up to **3 article URLs** for processing.")
    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i + 1}", placeholder=f"Enter article URL {i + 1}")
        if url:
            urls.append(url)

    # Submit button for processing URLs
    st.markdown("---")
    url_btn = st.button("🚀 **Process URLs**")

    # Footer Instructions
    st.markdown("""
    📝 **Instructions**:
    - Enter your API tokens for authentication.
    - Input up to 3 article URLs.
    - Submit the URLs and ask any question based on the articles.
    """)

# Initialize the HuggingFaceHub model
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 0.0, "max_new_tokens": 512},  # Low temperature to reduce hallucination
    huggingfacehub_api_token=huggingfacehub_api_token
)

# Function to process URLs and create a FAISS index
def process_urls(urls):
    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split the data into chunks with overlap
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n', '.', ','],
            chunk_size=256,
            chunk_overlap=50
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
query = st.text_input("🔍 **Ask a Question:**", placeholder="Enter your question here...")

# Prompt template for strict retrieval-based QA
prompt_template = (
    "Based on the following context, answer the question strictly using the information provided. "
    "Do not add information that is not in the context. If unsure, reply 'I don't know.'\n\n"
    "Context: {context}\n\nQuestion: {question}"
)

# Function to extract answer and source using regex
def extract_answer_and_source(final_answer):
    answer = re.search(r"Helpful Answer:\s*(.*?)(?:\s*Retrieved from|$)", final_answer, re.DOTALL)
    source = re.search(r"Retrieved from:\s*(.*)", final_answer)
    return (
        answer.group(1).strip() if answer else "Answer not found.",
        source.group(1).strip() if source else "Source not found."
    )

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
        retriever = faiss_store.as_retriever()

        # Retrieve relevant chunks
        retrieved_docs = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        # Use the strict prompt template
        input_prompt = prompt_template.format(context=context, question=query)

        # Get the result from the LLM
        result = llm(input_prompt)
        return result

    except Exception as e:
        st.error(f"Error retrieving answer: {str(e)}")
        return None

# Handle query submission
if query:
    with st.spinner("🔄 Generating answer..."):
        final_answer = get_answer(query)
        if final_answer:
            st.subheader("✅ **Answer:**")
            st.write(final_answer)
        else:
            st.warning("⚠️ No answer could be generated. Please try again.")
