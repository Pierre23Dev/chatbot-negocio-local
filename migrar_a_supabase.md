# Plan de Migración de Qdrant + SQLite a Supabase (pgvector + PostgreSQL)

## 📌 Descripción del Objetivo
El objetivo principal es consolidar la infraestructura de almacenamiento del chatbot de WhatsApp en **Supabase**, reemplazando:
1. **Qdrant** (Base de datos vectorial) por **Supabase pgvector** (vía `langchain-postgres`).
2. **SQLite (`ventas.db`)** (Base de datos relacional) por **Tabla PostgreSQL en Supabase**.

### Beneficios de la Migración:
- **Centralización**: Tanto la información vectorial (RAG + memoria de clientes) como los datos relacionales (registro de ventas) residirán en un solo servicio en la nube (Supabase).
- **Reducción de Complejidad**: Unifica variables de entorno y drivers de conexión en Python.
- **Filtrado Avanzado**: Supabase pgvector permite consultas híbridas SQL + vectoriales eficientes con filtros metadata nativos sobre JSONB (`metadata->>'telefono'`).

---

## 🔍 Análisis del Estado Actual del Proyecto

Actualmente el proyecto consta de los siguientes componentes clave:

1. **RAG Base de Conocimientos (`ingest.py` y `consultar_servicios_y_politicas`)**:
   - Carga `data/servicios.txt` y `data/politicas.txt`.
   - Genera embeddings usando `GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", output_dimensionality=384)`.
   - Indexa en la colección `diseno_grafico_knowledge` de Qdrant.

2. **Memoria de Clientes a Largo Plazo (`clientes_memoria`)**:
   - Extrae hechos clave del cliente en `extraer_y_guardar_hechos_cliente`.
   - Indexa en la colección `clientes_memoria` de Qdrant con metadata de teléfono.
   - Recupera antecedentes mediante filtrado de payload en `recuperar_memoria_cliente`.

3. **Base de Datos Relacional de Ventas (`ventas.db`)**:
   - Tabla SQLite `ventas` (`id`, `fecha`, `nombre_cliente`, `telefono`, `servicio`, `monto`, `estado`).
   - Operaciones asíncronas con `aiosqlite` en `src/app/main.py` (`init_db`, `registrar_venta_cerrada`, `enviar_reporte_diario_discord`, `obtener_resumen_ventas_hoy`).
   - Script independiente `exportar_ventas.py` con `sqlite3`.

4. **Alertas a Discord y Reportes**:
   - Notificación inmediata al registrar venta (`notificar_venta_discord`).
   - Reporte diario a las 20:00 con CSV adjunto (`enviar_reporte_diario_discord`).

---

## ⚠️ Revisión Requerida por el Usuario

> [!IMPORTANT]
> **Requisitos previos en Supabase**:
> 1. Crear un proyecto en [Supabase](https://supabase.com).
> 2. Habilitar la extensión `vector` en la base de datos de Supabase.
> 3. Obtener las credenciales de conexión:
>    - `SUPABASE_URL` y `SUPABASE_KEY` (o `SUPABASE_SERVICE_ROLE_KEY`).
>    - `DATABASE_URL` en formato conexión PostgreSQL: `postgresql+psycopg://postgres:[PASSWORD]@[HOST]:5432/postgres` (o Transaction Pooler port 6543 / Direct 5432).

> [!NOTE]
> **Checkpointer de Conversaciones de LangGraph**:
> Actualmente las conversaciones se guardan localmente en SQLite (`conversations.db`). Se propone mantener SQLite local para los checkpoints o migrar a `langgraph-checkpoint-postgres` si deseas centralizar también las conversaciones en Supabase.

---

## ❓ Preguntas Abiertas

> [!QUESTION]
> ¿Tienes un proyecto activo de Supabase con las credenciales listas en tu archivo `.env` o prefieres que preparemos las variables para que las completes?

---

## 🛠️ Cambios Propuestos

### Componente 1: Dependencias (`requirements.txt`)

#### [MODIFY] `requirements.txt`
- Eliminar: `langchain-qdrant`, `qdrant-client`, `aiosqlite`.
- Agregar: `langchain-postgres`, `psycopg[binary]`, `psycopg-pool`, `supabase`, `asyncpg`.

```diff
  fastapi
  uvicorn
  langchain-core
  langgraph
  langchain-google-genai
- langchain-qdrant
- qdrant-client
+ langchain-postgres
+ psycopg[binary]
+ psycopg-pool
+ supabase
+ asyncpg
  python-dotenv
  apscheduler
  httpx
- aiosqlite
  langchain-text-splitters
  langgraph-checkpoint-sqlite
```

---

### Componente 2: Script SQL de Inicialización en Supabase

#### [NEW] `schema.sql` (Instrucciones SQL para ejecutar en el Editor SQL de Supabase)

```sql
-- 1. Habilitar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Tabla relacional de ventas
CREATE TABLE IF NOT EXISTS ventas (
    id SERIAL PRIMARY KEY,
    fecha TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    nombre_cliente TEXT NOT NULL,
    telefono TEXT NOT NULL,
    servicio TEXT NOT NULL,
    monto NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    estado TEXT DEFAULT 'Pendiente Confirmación Anticipo'
);

-- Índice para búsquedas por fecha y teléfono en ventas
CREATE INDEX IF NOT EXISTS idx_ventas_fecha ON ventas (fecha);
CREATE INDEX IF NOT EXISTS idx_ventas_telefono ON ventas (telefono);

-- 3. Las tablas vectoriales para LangChain PGVector serán gestionadas automáticamente 
-- por langchain-postgres (o mediante creación explícita de índices HNSW sobre el vector de 384 dimensiones).
```

---

### Componente 3: Script de Ingesta (`ingest.py`)

#### [MODIFY] `ingest.py`
- Reemplazar `QdrantVectorStore` y `QdrantClient` por `PGVector` de `langchain_postgres`.
- Conectar mediante `DATABASE_URL`.
- Mantener los embeddings de Gemini (`models/gemini-embedding-001` con dimensión 384).

```python
import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("[ERROR] Variable DATABASE_URL no encontrada en .env")
    sys.exit(1)

COLLECTION_KNOWLEDGE = "diseno_grafico_knowledge"

def indexar_documentos():
    documentos = cargar_documentos()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks = text_splitter.split_documents(documentos)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=384
    )

    # Inicializar PGVector en Supabase
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_KNOWLEDGE,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    print("Indexando documentos en Supabase PGVector...")
    vector_store.add_documents(chunks)
    print("✅ Indexación completada en Supabase.")
```

---

### Componente 4: Aplicación Principal (`src/app/main.py`)

#### [MODIFY] `src/app/main.py`
1. **Configuración de Variables**:
   - Reemplazar `QDRANT_URL` por `DATABASE_URL` y/o `SUPABASE_URL` + `SUPABASE_KEY`.
2. **Inicialización de Vectorstores**:
   - Usar `PGVector` para `vectorstore_knowledge` y `vectorstore_clientes`.
3. **Memoria de Clientes (`recuperar_memoria_cliente` & `extraer_y_guardar_hechos_cliente`)**:
   - Ajustar el filtro metadata en PostgreSQL: `{"telefono": telefono}`.
4. **Base de Datos de Ventas (`ventas`)**:
   - Sustituir `aiosqlite` por un pool asíncrono con `psycopg` / `asyncpg`.
   - Consultas SQL adaptadas a PostgreSQL (`SERIAL`, sintaxis de fecha `NOW()`, etc.).

---

### Componente 5: Exportación de Ventas (`exportar_ventas.py`)

#### [MODIFY] `exportar_ventas.py`
- Reemplazar `sqlite3` por `psycopg` / `psycopg2` para consultar Supabase PostgreSQL directamente.

---

## 🧪 Plan de Verificación

### Pruebas Automatizadas y Scripts de Verificación
1. **Verificación de Ingesta Vectorial**:
   - Ejecutar `python ingest.py` y verificar que los fragmentos se insertan en Supabase sin errores de dimensión.
2. **Verificación de Exportación**:
   - Ejecutar `python exportar_ventas.py` para validar la consulta y generación de CSV desde Supabase PostgreSQL.

### Verificación Manual
1. **Simulación de Consulta RAG por WhatsApp**:
   - Enviar pregunta sobre precios/servicios al bot y confirmar respuesta basada en Supabase PGVector.
2. **Simulación de Cierre de Venta**:
   - Confirmar un pedido vía WhatsApp, verificar que se guarda en la tabla `ventas` de Supabase y la alerta llega a Discord.
3. **Verificación de Memoria de Cliente**:
   - Interactuar con el bot y validar en la tabla de vectores que los hechos extraídos del cliente se almacenan y consultan filtrados por teléfono.
