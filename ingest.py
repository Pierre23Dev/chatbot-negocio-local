import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "diseno_grafico_knowledge"

if not GOOGLE_API_KEY or "tu_gemini_api_key" in GOOGLE_API_KEY:
    print("\n[ERROR CRÍTICO] La variable GOOGLE_API_KEY no es válida en el archivo .env.")
    print("Revisa tu .env y pega tu API key de Google AI Studio sin comillas.\n")
    sys.exit(1)

def indexar_documentos():
    archivos = ["data/servicios.txt", "data/politicas.txt"]
    documentos = []
    
    for archivo in archivos:
        if not os.path.exists(archivo):
            print(f"[ERROR] No se encuentra el archivo: {archivo}")
            return
        loader = TextLoader(archivo, encoding="utf-8")
        documentos.extend(loader.load())

    # Dividir texto en chunks contextuales
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Total de fragmentos generados: {len(chunks)}")

    # Inicializar embeddings de Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

    # Conectar a Qdrant y recrear colección usando la API actual
    client = QdrantClient(url=QDRANT_URL)
    
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # Guardar en base vectorial
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print("✅ ¡Base vectorial indexada correctamente en Qdrant local!")

if __name__ == "__main__":
    indexar_documentos()