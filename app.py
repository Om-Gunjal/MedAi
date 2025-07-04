import os
from flask import Flask, render_template, request, jsonify

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

from src.prompt import system_prompt

app = Flask(__name__)

# --------------------------
# Load PDF and create FAISS
# --------------------------
loader = PyPDFLoader("Data/Medical_book.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_folder = "faiss_index"

if os.path.exists(index_folder):
    vectorstore = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    vectorstore.save_local(index_folder)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --------------------------
# Load LLM
# --------------------------
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to("cpu")  # Use "cuda" if GPU

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=500)
llm = HuggingFacePipeline(pipeline=pipe)

# --------------------------
# Routes
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Please enter a question."})

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = (
        system_prompt.replace("{context}", context)
        + f"\n\nQuestion: {question}\nAnswer:"
    )

    response = llm.invoke(final_prompt)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
