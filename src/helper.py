from langchain_community.embeddings import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
