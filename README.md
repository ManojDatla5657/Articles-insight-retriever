
# 📰 Article Research Tool

A Streamlit application for researching news articles by querying multiple sources. The tool processes URLs, creates embeddings, and answers questions based on the content of those articles using a combination of LangChain, HuggingFace, and Google Generative AI embeddings.

## Features

- **Process URLs**: Input up to 3 article URLs to process and create embeddings.
- **FAISS Indexing**: Articles are split and indexed using FAISS for fast retrieval.
- **Query Answering**: Ask questions based on article content, and retrieve relevant answers.
- **HuggingFace Integration**: Uses the Mistral-7B-Instruct model for language understanding.
- **Google Generative AI Embeddings**: Used for creating embeddings and text search.

## Requirements

This project requires the following Python packages:

- **Python 3.7+**
- **Streamlit**
- **LangChain**
- **HuggingFaceHub**
- **GoogleGenerativeAIEmbeddings**
- **FAISS**
- **python-dotenv**
- **Pickle**
- **regex**

## Setup Instructions

### Step 1: Install Dependencies

Ensure that you have Python 3.7+ installed. Install the necessary Python packages listed in the `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 2: Environment Variables

You will need to store your HuggingFace API token in a `.env` file in the root directory of the project.

Create a `.env` file with the following content:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

Replace `your_huggingface_api_token` with your actual API token from [HuggingFace](https://huggingface.co/).

### Step 3: Run the Streamlit Application

After setting up the environment variables and installing the dependencies, run the Streamlit app locally using the following command:

```bash
streamlit run app.py
```

This will start the application and open it in your default web browser at `http://localhost:8501`.

## Usage

1. **Enter URLs**: In the sidebar, input up to 3 article URLs that you want to process.
2. **Process the URLs**: Click on the "Process URLs" button to fetch, process, and index the articles.
3. **Ask a Question**: Once the articles are processed, enter a query in the input field and get an answer based on the indexed articles.

## Troubleshooting

- **Bad Request Error**: If you get a "Bad Request" error when loading the HuggingFace model, ensure that your API token is correctly set in the `.env` file and the token is valid.
  
- **No Answer Generated**: If no answer is generated, try rephrasing your query or ensure that the URLs you provided contain sufficient content for the model to process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
