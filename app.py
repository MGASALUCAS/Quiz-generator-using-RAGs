import logging
from flask import Flask, request, render_template, send_file, jsonify
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from fpdf import FPDF
import os

app = Flask(__name__)
load_dotenv()

# Load environment variables for OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# In-memory storage for conversation context
conversation_history = {}

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

        pdf_path = os.path.join('uploads', file.filename)
        file.save(pdf_path)

        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "No prompt provided!"}), 400

        answer = generate_questions(pdf_path, prompt)
        save_as_pdf = request.form.get('save_as_pdf')

        if save_as_pdf == 'yes':
            pdf_filename = create_pdf(answer)
            return send_file(pdf_filename, as_attachment=True)
        else:
            return render_template('index.html', pdf_path=pdf_path, answer=answer, prompt=prompt)

    except Exception as e:
        logging.error(f"Error in upload_content: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded!"}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file!"}), 400

            pdf_path = os.path.join('uploads', file.filename)
            file.save(pdf_path)

            # Initialize conversation history for the new PDF
            conversation_history[pdf_path] = []

            return jsonify({"pdf_path": pdf_path})

        return render_template('chatbot.html')

    except Exception as e:
        logging.error(f"Error in chatbot: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

@app.route('/ask', methods=['POST'])
def ask_chatbot():
    try:
        question = request.form.get('question')
        pdf_path = request.form.get('pdf_path')

        if not question or not pdf_path:
            return jsonify({"error": "Question or PDF path not provided!"}), 400

        # Initialize conversation history for the current PDF if not already done
        if pdf_path not in conversation_history:
            conversation_history[pdf_path] = []

        # Add the user's question to the conversation history
        conversation_history[pdf_path].append({"role": "user", "content": question})

        # Get the chatbot's response
        answer = get_chatbot_response(pdf_path, question)

        # Add the chatbot's response to the conversation history
        conversation_history[pdf_path].append({"role": "assistant", "content": answer})

        return jsonify({
            "answer": answer,
            "conversation": conversation_history[pdf_path]
        })

    except Exception as e:
        logging.error(f"Error in ask_chatbot: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500

def generate_questions(pdf_path, prompt):
    try:
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separator="\n"
        )
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("vector_db")

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        llm = ChatOpenAI()
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        retriever = FAISS.load_local("vector_db", embeddings).as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        response = retrieval_chain.invoke({"input": prompt})

        return response["answer"]

    except Exception as e:
        logging.error(f"Error in generate_questions: {e}")
        return "Error generating questions."

def get_chatbot_response(pdf_path, question):
    try:
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separator="\n"
        )
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("vector_db")

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        llm = ChatOpenAI()
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        retriever = FAISS.load_local("vector_db", embeddings).as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        response = retrieval_chain.invoke({"input": question})

        return response["answer"]

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
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
