import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. Cambiamos la importación al ecosistema de Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

# Validar que la API key de Google esté presente
if not os.getenv("GOOGLE_API_KEY"):
    print("[ERROR] No se encontró la variable GOOGLE_API_KEY en las variables de entorno.")
    sys.exit(1)

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

    # 1. Fragmentación del texto base
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Total de fragmentos generados: {len(chunks)}")

    # 2. Configuración del modelo de Embedding de Gemini
    print("Inicializando Gemini Embeddings (models/gemini-embedding-001)...")
    # Al ser una llamada de API externa, no necesitas configurar torch, CUDA o CPU locales.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # El modelo oficial activo de Google
        output_dimensionality=384            # Recorta nativamente el vector a 384 dimensiones
    )

    client = QdrantClient(url=QDRANT_URL)

    # 3. Recrear colección limpia 
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Eliminando colección anterior: {COLLECTION_NAME}")
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # 4. Indexación en Qdrant (Instanciación directa + add_documents)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    print("Generando embeddings a través de la API e indexando en Qdrant...")
    vector_store.add_documents(chunks)

    print("✅ ¡Base vectorial indexada correctamente con Gemini Embeddings!")

if __name__ == "__main__":
    indexar_documentos()
