import logging
import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import requests
import openai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from fpdf import FPDF
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load environment variables for OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the path for uploaded files
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory storage for conversation context
conversation_history = {}

# LangChain setup
VECTOR_STORE_PATH = "vector_db"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_content():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded!"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file!"}), 400

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(pdf_path)

        # Process PDF with LangChain
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separator="\n"
        )
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)

        # Update conversation history with the new PDF path
        conversation_history['pdf_path'] = pdf_path

        return jsonify({"message": "File uploaded and processed successfully."})

    except Exception as e:
        logging.error(f"Error in upload_content: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

@app.route('/ask', methods=['POST'])
def ask_chatbot():
    try:
        question = request.form.get('question')
        pdf_path = conversation_history.get('pdf_path')

        if not question:
            return jsonify({"error": "Question not provided!"}), 400

        if 'conversation' not in conversation_history:
            conversation_history['conversation'] = []

        conversation_history['conversation'].append({"role": "user", "content": question})

        # Get the chatbot's response using LangChain
        answer = get_chatbot_response(question)

        conversation_history['conversation'].append({"role": "assistant", "content": answer})

        return jsonify({
            "answer": answer,
            "conversation": conversation_history['conversation']
        })

    except Exception as e:
        logging.error(f"Error in ask_chatbot: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

def get_chatbot_response(question):
    try:
        embeddings = OpenAIEmbeddings()
        retriever = FAISS.load_local(VECTOR_STORE_PATH, embeddings).as_retriever()
        
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        llm = ChatOpenAI()

        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        response = retrieval_chain.invoke({"input": question})
        answer = response.get('answer', 'No answer found.')

        return answer

    except Exception as e:
        logging.error(f"Error in get_chatbot_response: {e}")
        return "Error getting response."

def create_pdf(content):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Times", size=12)
        pdf.multi_cell(0, 10, content)
        pdf_filename = "generated_answer.pdf"
        pdf.output(pdf_filename)
        return pdf_filename

    except Exception as e:
        logging.error(f"Error in create_pdf: {e}")
        return "Error creating PDF."

if __name__ == "__main__":
    app.run(port=5000, debug=True)
