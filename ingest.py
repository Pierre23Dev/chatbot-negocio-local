import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Validar API key de Google
if not os.getenv("GOOGLE_API_KEY"):
    print("[ERROR] No se encontró la variable GOOGLE_API_KEY en las variables de entorno.")
    sys.exit(1)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("[ERROR] No se encontró la variable DATABASE_URL en las variables de entorno.")
    print("Asegúrate de definir DATABASE_URL=postgresql+psycopg://postgres:[PASSWORD]@[HOST]:5432/postgres")
    sys.exit(1)

# Normalizar URL para dialecto psycopg3 en SQLAlchemy / PGVector
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

COLLECTION_NAME = "diseno_grafico_knowledge"

def cargar_documentos(tenant_phone: str, archivos: list) -> list:
    docs = []
    for ruta in archivos:
        if not os.path.exists(ruta):
            print(f"[ADVERTENCIA] No se encuentra el archivo: {ruta}")
            continue
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()
            docs.append(Document(
                page_content=contenido,
                metadata={
                    "tenant_phone": tenant_phone,
                    "source": ruta
                }
            ))
    return docs

def indexar_documentos_tenant(tenant_phone: str, archivos: list, limpiar_anterior: bool = False):
    documentos = cargar_documentos(tenant_phone, archivos)
    if not documentos:
        print("[ERROR] No se pudo cargar ningún documento para ingestar.")
        sys.exit(1)

    # 1. Fragmentación del texto base
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Total de fragmentos generados para tenant '{tenant_phone}': {len(chunks)}")

    # 2. Configuración del modelo de Embedding de Gemini
    print("Inicializando Gemini Embeddings (models/gemini-embedding-001)...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=384
    )

    # 3. Instanciación e indexación en Supabase PGVector
    print(f"Conectando e indexando en Supabase PGVector (colección '{COLLECTION_NAME}')...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True
    )

    if limpiar_anterior:
        print(f"🗑️ Limpiando conocimientos RAG anteriores del tenant '{tenant_phone}'...")
        try:
            vector_store.delete(filter={"tenant_phone": tenant_phone})
            print("✅ Registros anteriores eliminados exitosamente de Supabase.")
        except Exception as e:
            print(f"⚠️ Nota al limpiar registros previos: {e}")

    # Generar IDs únicos y deterministas por fragmento para evitar duplicación mediante Upsert automático
    ids = [f"{tenant_phone}_{chunk.metadata.get('source', 'doc')}_{idx}" for idx, chunk in enumerate(chunks)]

    vector_store.add_documents(chunks, ids=ids)
    print(f"✅ ¡Base vectorial RAG indexada correctamente en Supabase para tenant {tenant_phone}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de conocimientos RAG multi-tenant para Supabase PGVector")
    parser.add_argument("--tenant", required=True, help="Número telefónico del tenant/negocio (ej: 5215512345678)")
    parser.add_argument("--archivos", nargs="+", default=["data/servicios.txt", "data/politicas.txt"], help="Ruta de archivos a ingestar")
    parser.add_argument("--limpiar", action="store_true", help="Elimina los documentos anteriores del tenant antes de ingestar los nuevos")

    args = parser.parse_args()
    indexar_documentos_tenant(args.tenant, args.archivos, args.limpiar)
