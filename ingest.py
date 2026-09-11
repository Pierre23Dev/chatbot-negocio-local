import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "diseno_grafico_knowledge"

def cargar_documentos():
    archivos = ["data/servicios.txt", "data/politicas.txt"]
    docs = []
    for ruta in archivos:
        if not os.path.exists(ruta):
            print(f"[ERROR] No se encuentra el archivo: {ruta}")
            sys.exit(1)
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()
            docs.append(Document(page_content=contenido, metadata={"source": ruta}))
    return docs

def indexar_documentos():
    documentos = cargar_documentos()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Total de fragmentos generados: {len(chunks)}")

    print("Cargando modelo local de embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient(url=QDRANT_URL)

    # Recrear la colección en Qdrant (384 dimensiones)
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Guardar en Qdrant
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print("✅ ¡Base vectorial indexada correctamente en Qdrant local!")

if __name__ == "__main__":
    indexar_documentos()