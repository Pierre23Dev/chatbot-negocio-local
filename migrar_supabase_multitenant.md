# Plan de Migración a Supabase Multi-Tenant y Despliegue en Vercel

## 📌 Visión General y Objetivos
Este documento establece la arquitectura y el plan de implementación para convertir el chatbot de ventas en una **solución Multi-Tenant**, desplegada en **Vercel** y respaldada completamente por **Supabase** (PostgreSQL + pgvector).

### Principales Pilares:
1. **Multi-Tenancy por Número Telefónico (`tenant_phone`)**:
   - Cada negocio/negocio local que usa el chatbot es un **Tenant**, identificado por su número de teléfono oficial de WhatsApp (extraído de `metadata.display_phone_number` o `phone_number_id` en el Webhook de Meta).
   - **RAG Multi-Tenant**: Los documentos de conocimiento de un negocio se indexan con su `tenant_phone`. Las búsquedas RAG filtran estrictamente por este ID.
   - **Memoria de Clientes Multi-Tenant**: Los hechos de clientes se aíslan por `(tenant_phone, client_phone)`.
   - **Ventas Multi-Tenant**: La tabla `ventas` registra la columna `tenant_phone` para separar ingresos y reportes de cada negocio.
2. **Alojamiento en Vercel (Serverless)**:
   - La aplicación FastAPI corre en Vercel como Serverless Function (`@vercel/python`).
   - **Reemplazo de `APScheduler`**: Dado que Vercel congela procesos en reposo, los reportes diarios automáticos se ejecutarán mediante **Vercel Cron Jobs**.
   - **Reemplazo de Checkpointer SQLite**: Al ser Vercel un entorno efímero (sistema de archivos de solo lectura y volátil), los checkpoints de conversación de LangGraph se persisten en **Supabase PostgreSQL** (`langgraph-checkpoint-postgres` o tabla de checkpoints).

---

## 🏗️ Arquitectura Multi-Tenant

```
+-----------------------------------------------------------------------------------+
|                                  META CLOUD API                                   |
+-----------------------------------------------------------------------------------+
                                          |
                                 Webhook (JSON Payload)
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                VERCEL SERVERLESS                                  |
|  - FastAPI (src/app/main.py)                                                      |
|  - Extracción de tenant_phone desde payload                                        |
|  - LangGraph Workflow (Estado aislado por tenant_phone + client_phone)             |
|  - Vercel Cron Endpoint (/api/cron/daily-report)                                  |
+-----------------------------------------------------------------------------------+
                                          |
                       Conexión PostgreSQL (psycopg3 / asyncpg)
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                SUPABASE POSTGRESQL                                |
|  1. PGVector (RAG Conocimientos) -> Filter: metadata @> '{"tenant_phone": "..."}'  |
|  2. PGVector (Memoria Clientes)  -> Filter: metadata @> '{"tenant_phone": "..."}'  |
|  3. Tabla `ventas`              -> WHERE tenant_phone = '...'                     |
|  4. Tabla `checkpoints`         -> LangGraph Thread Checkpoints                   |
+-----------------------------------------------------------------------------------+
```

---

## ⚠️ Revisiones y Decisiones de Diseño

> [!IMPORTANT]
> **Identificación del Tenant en Meta Webhook**:
> En el webhook de Meta Cloud API (`POST /webhook`), el objeto `value` contiene:
> - `metadata.display_phone_number`: Número receptor de la cuenta de WhatsApp Business.
> - `metadata.phone_number_id`: ID único del número en Meta.
> Se utilizará `display_phone_number` (o `phone_number_id`) para derivar el `tenant_phone` en cada petición entrante.

> [!WARNING]
> **Entorno Efímero en Vercel**:
> 1. **No usar SQLite local**: `conversations.db` y `ventas.db` deben eliminarse por completo. Toda persistencia debe ir a Supabase.
> 2. **Deshabilitar APScheduler**: Los hilos de fondo en memoria no sobreviven en Vercel. Se usará `crons` en `vercel.json`.

---

## ❓ Preguntas Abiertas

> [!QUESTION]
> ¿Deseas identificar a los negocios por su número telefónico de WhatsApp en formato internacional (ej: `5215512345678`) o por el `PHONE_NUMBER_ID` asignado por Meta? (Recomendado: Número telefónico formateado).

---

## 🛠️ Cambios Propuestos por Componente

### 1. Configuración de Despliegue (`vercel.json`)

#### [MODIFY] `vercel.json`
- Añadir configuración de Cron Job para el reporte diario de ventas (ej: 20:00 UTC / hora local).

```json
{
  "version": 2,
  "builds": [
    {
      "src": "src/app/main.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "src/app/main.py"
    }
  ],
  "crons": [
    {
      "path": "/api/cron/daily-report",
      "schedule": "0 20 * * *"
    }
  ]
}
```

---

### 2. Dependencias (`requirements.txt`)

#### [MODIFY] `requirements.txt`
- Eliminar `apscheduler`, `aiosqlite`, `langgraph-checkpoint-sqlite`, `langchain-qdrant`, `qdrant-client`.
- Agregar `langchain-postgres`, `langgraph-checkpoint-postgres`, `psycopg[binary]`, `psycopg-pool`, `supabase`, `asyncpg`.

```diff
  fastapi
  uvicorn
  langchain-core
  langgraph
  langchain-google-genai
- langchain-qdrant
- qdrant-client
+ langchain-postgres
+ langgraph-checkpoint-postgres
+ psycopg[binary]
+ psycopg-pool
+ supabase
+ asyncpg
  python-dotenv
- apscheduler
  httpx
- aiosqlite
  langchain-text-splitters
- langgraph-checkpoint-sqlite
```

---

### 3. Esquema SQL en Supabase (`schema.sql`)

#### [NEW] `schema.sql`
Script SQL para ejecutar en el SQL Editor de Supabase:

```sql
-- 1. Habilitar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Tabla Multi-Tenant de Ventas
CREATE TABLE IF NOT EXISTS ventas (
    id SERIAL PRIMARY KEY,
    tenant_phone TEXT NOT NULL,
    fecha TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    nombre_cliente TEXT NOT NULL,
    telefono TEXT NOT NULL,
    servicio TEXT NOT NULL,
    monto NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    estado TEXT DEFAULT 'Pendiente Confirmación Anticipo'
);

-- Índices optimizados para consultas por tenant y fecha
CREATE INDEX IF NOT EXISTS idx_ventas_tenant_fecha ON ventas (tenant_phone, fecha DESC);

-- 3. Tabla opcional de configuración por Tenant (para Discord Webhooks personalizados por negocio)
CREATE TABLE IF NOT EXISTS tenant_config (
    tenant_phone TEXT PRIMARY KEY,
    nombre_negocio TEXT NOT NULL,
    discord_webhook_url TEXT,
    activo BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

### 4. Script de Ingesta Multi-Tenant (`ingest.py`)

#### [MODIFY] `ingest.py`
- Aceptar parámetro `--tenant` o variable de entorno para ingestar información específica de un negocio sin borrar la de los demás.
- Inserción de documentos etiquetados con `metadata={"tenant_phone": tenant_phone, "source": ruta}`.

```python
import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_KNOWLEDGE = "diseno_grafico_knowledge"

def indexar_documentos_tenant(tenant_phone: str, archivos: list):
    docs = []
    for ruta in archivos:
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = f.read()
                docs.append(Document(
                    page_content=contenido,
                    metadata={"tenant_phone": tenant_phone, "source": ruta}
                ))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        output_dimensionality=384
    )

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_KNOWLEDGE,
        connection=DATABASE_URL,
        use_jsonb=True
    )

    print(f"Indexando {len(chunks)} fragmentos para el Tenant {tenant_phone}...")
    vector_store.add_documents(chunks)
    print(f"✅ Ingesta multi-tenant completada para {tenant_phone}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos RAG por Tenant")
    parser.add_argument("--tenant", required=True, help="Número telefónico o ID del Tenant")
    args = parser.parse_args()
    
    indexar_documentos_tenant(args.tenant, ["data/servicios.txt", "data/politicas.txt"])
```

---

### 5. Código Principal (`src/app/main.py`)

#### [MODIFY] `src/app/main.py`
1. **Extracción del Tenant**:
   - De la petición entrante del Webhook, leer el receptor:
     `tenant_phone = entry.get("value", {}).get("metadata", {}).get("display_phone_number", "default_tenant")`.
2. **Herramienta RAG Filtrada por Tenant**:
   - `consultar_servicios_y_politicas(consulta: str, tenant_phone: str)` ejecuta búsqueda de similitud con filtro `filter={"tenant_phone": tenant_phone}`.
3. **Memoria de Clientes Multi-Tenant**:
   - `recuperar_memoria_cliente` busca con `filter={"tenant_phone": tenant_phone, "telefono": client_phone}`.
4. **Registro de Ventas Multi-Tenant**:
   - `registrar_venta_cerrada` incluye `tenant_phone` en la sentencia `INSERT INTO ventas`.
5. **Persistencia LangGraph con Checkpointer de Postgres**:
   - Se utiliza `AsyncPostgresSaver` de `langgraph-checkpoint-postgres` pasando la cadena de conexión de Supabase.
6. **Endpoint para Vercel Cron (`/api/cron/daily-report`)**:
   - Reemplaza `APScheduler`. Vercel invocará automáticamente este endpoint a las 20:00.

---

### 6. Script de Exportación (`exportar_ventas.py`)

#### [MODIFY] `exportar_ventas.py`
- Aceptar `--tenant` como argumento para filtrar la exportación del reporte CSV por negocio específico.

```python
import os
import csv
import argparse
from datetime import datetime
import psycopg

DATABASE_URL = os.getenv("DATABASE_URL")

def exportar_ventas_tenant(tenant_phone: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_salida = f"reporte_ventas_{tenant_phone}_{timestamp}.csv"

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, fecha, nombre_cliente, telefono, servicio, monto, estado FROM ventas WHERE tenant_phone = %s ORDER BY id ASC",
                (tenant_phone,)
            )
            filas = cursor.fetchall()
            columnas = [desc[0] for desc in cursor.description]

            with open(archivo_salida, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(columnas)
                writer.writerows(filas)

    print(f"✅ Reporte generado para tenant {tenant_phone}: {archivo_salida}")
```

---

## 🧪 Plan de Verificación

### 1. Pruebas de Ingesta Multi-Tenant
- Ingestar documentos para Tenant A (`+5215511111111`) y Tenant B (`+5215522222222`).
- Consultar el RAG especificando Tenant A y verificar que no devuelva información de Tenant B.

### 2. Pruebas de Ventas Ailadas
- Registrar ventas para Tenant A y Tenant B en Supabase.
- Verificar que `exportar_ventas.py --tenant +5215511111111` genere un CSV que contiene únicamente las ventas del Tenant A.

### 3. Pruebas en Vercel
- Desplegar la aplicación con `vercel deploy`.
- Enviar mensaje de prueba al webhook en Vercel y comprobar respuesta adecuada.
- Disparar manualmente la ruta `/api/cron/daily-report` para verificar la generación de reportes desde Serverless.
