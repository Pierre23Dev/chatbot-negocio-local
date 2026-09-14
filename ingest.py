import os
import sys
import torch
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

    # 1. Fragmentación del texto base
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Total de fragmentos generados: {len(chunks)}")

    # 2. Inyección del prefijo 'passage: ' tras el split
    # Es crucial aplicarlo después de dividir para evitar que el splitter corte el prefijo
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content.strip()}"

    # 3. Configuración del modelo multilingual-e5-small en float16
    print("Cargando modelo local multilingual-e5-small (float16)...")
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    encode_kwargs = {
        "normalize_embeddings": True  # Normalización coseno requerida para modelos E5
    }

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    client = QdrantClient(url=QDRANT_URL)

    # 4. Recrear colección limpia (multilingual-e5-small usa 384 dimensiones)
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Eliminando colección anterior: {COLLECTION_NAME}")
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # 5. Indexación en Qdrant
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )
    print("✅ ¡Base vectorial indexada correctamente con prefijos 'passage:' y float16!")

if __name__ == "__main__":
    indexar_documentos()