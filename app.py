from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Flask app
app = Flask(__name__)

# Load Pinecone index
index_name = "medicalbot"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load Pinecone index with embeddings
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Load LLM model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token": HF_TOKEN, "max_length": 1024}
    )

llm = load_llm(HUGGINGFACE_REPO_ID)

# Define custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_query = request.json.get('query')
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        # Invoke the RAG pipeline
        response = qa_chain.invoke({'query': user_query})

        # Extract the answer
        answer = response.get("result", "No answer found.")

        # Extract and serialize source documents
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content[:300]  # Limit content to avoid huge responses
            })

        return jsonify({"result": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
