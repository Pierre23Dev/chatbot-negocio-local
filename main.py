import os
import httpx
import csv
import re
import json
import asyncio
import base64
from datetime import datetime
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from fastapi import FastAPI, Request, BackgroundTasks, Response
from fastapi.responses import PlainTextResponse
import uvicorn

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Distance, VectorParams

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing_extensions import TypedDict

from contextlib import asynccontextmanager

import aiosqlite

from apscheduler.schedulers.asyncio import AsyncIOScheduler

load_dotenv()

# --- Configuración de Entorno ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
QDRANT_URL = os.getenv("QDRANT_URL","http://localhost:6333/")
ADMIN_PHONE = os.getenv("ADMIN_PHONE")

# --- Configuración Meta Cloud API ---
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN") 
GRAPH_API_URL = "https://graph.facebook.com/v25.0"


COLLECTION_KNOWLEDGE = "diseno_grafico_knowledge"
COLLECTION_CLIENTS = "clientes_memoria"

DB_VENTAS = "ventas.db"

# --- Control de Concurrencia para Gemini ---
gemini_semaphore = asyncio.Semaphore(5)
scheduler = AsyncIOScheduler()

# --- Inicialización de SQLite con Modo WAL ---
async def init_db():
    async with get_ventas_db() as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=5000;")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS ventas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT,
                nombre_cliente TEXT,
                telefono TEXT,
                servicio TEXT,
                monto TEXT,
                estado TEXT
            )
        """)
        await db.commit()


# Cliente HTTP asíncrono compartido
http_client: httpx.AsyncClient = None


# --- Conexión RAG con Gemini Embeddings ---
print("Inicializando Gemini Embeddings para consultas...")
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # El modelo oficial activo de Google
        output_dimensionality=384            # Recorta nativamente el vector a 384 dimensiones
)

client_qdrant = QdrantClient(url=QDRANT_URL)

# 1. Vectorstore de Conocimientos del Negocio
vectorstore_knowledge = QdrantVectorStore(
    client=client_qdrant,
    collection_name=COLLECTION_KNOWLEDGE,
    embedding=embeddings
)


# 2. Vectorstore de Memoria a Largo Plazo de Clientes
if not client_qdrant.collection_exists(COLLECTION_CLIENTS):
    print(f"Creando colección de clientes limpia: {COLLECTION_CLIENTS}")
    client_qdrant.create_collection(
        collection_name=COLLECTION_CLIENTS,
        # IMPORTANTE: Cambiado a 768 para que coincida con el tamaño de vector de Gemini
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    # Índice payload sobre el teléfono para búsquedas instantáneas y aisladas
    client_qdrant.create_payload_index(
        collection_name=COLLECTION_CLIENTS,
        field_name="metadata.telefono",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )

vectorstore_clientes = QdrantVectorStore(
    client=client_qdrant,
    collection_name=COLLECTION_CLIENTS,
    embedding=embeddings,
)

retriever = vectorstore_knowledge.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,                    # Límite máximo de fragmentos
        "score_threshold": 0.75    # Solo devuelve fragmentos con similitud >= 75%
    }
)

@asynccontextmanager
async def get_ventas_db():
    """Generador de conexiones asíncronas seguras hacia ventas.db."""
    async with aiosqlite.connect(DB_VENTAS, timeout=15.0) as db:
        db.row_factory = aiosqlite.Row
        yield db


RE_MONTO = re.compile(r"[-+]?\d*\.\d+|\d+")

def limpiar_monto(valor) -> float:
    if valor is None:
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    coincidencias = RE_MONTO.findall(str(valor).replace(",", "."))
    return float(coincidencias[0]) if coincidencias else 0.0



def sanitizar_mensajes_para_gemini(messages: list) -> list:
    """
    Sanea el historial recortando un número máximo de mensajes, asegurando
    que no queden ToolMessages huérfanos y respetando las reglas de la API de Gemini.
    """
    if not messages:
        return []

    # 1. Si el último mensaje es un AIMessage que solicita herramientas,
    # significa que el flujo quedó a la mitad. Retornamos todo el bloque actual.
    if isinstance(messages[-1], AIMessage) and getattr(messages[-1], "tool_calls", None):
        return messages[-2:] if len(messages) >= 2 else messages

    recientes = []
    limite_mensajes = 8
    
    # 2. Recorremos al revés para no romper secuencias ToolMessage -> AIMessage
    for msg in reversed(messages):
        if len(recientes) >= limite_mensajes and isinstance(msg, HumanMessage):
            # Solo dejamos de añadir si ya cumplimos la cuota y garantizamos 
            # que el bloque comience con un mensaje del usuario.
            recientes.insert(0, msg)
            break
        recientes.insert(0, msg)

    # 3. Limpieza de seguridad al inicio (Garantizar que empiece con HumanMessage)
    while recientes and not isinstance(recientes[0], HumanMessage):
        recientes.pop(0)

    # 4. Si la limpieza vació la lista, devolvemos al menos el último mensaje válido
    return recientes if recientes else [messages[-1]]


async def notificar_venta_discord(nombre: str, telefono: str, servicio: str, monto: float):
    """Envía un Embed enriquecido al canal de Discord vía Webhook."""
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ Advertencia: DISCORD_WEBHOOK_URL no configurado.")
        return

    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "username": "Bot de Ventas",
        "embeds": [
            {
                "title": "🎉 ¡Nueva Venta Registrada!",
                "description": "Se ha cerrado un pedido desde el bot de WhatsApp.",
                "color": 3066993,  # Verde #2ECC71
                "fields": [
                    {"name": "👤 Cliente", "value": nombre, "inline": True},
                    {"name": "📱 Teléfono / JID", "value": telefono, "inline": True},
                    {"name": "💼 Servicio Contratado", "value": servicio, "inline": False},
                    {"name": "💵 Monto Acordado", "value": f"${monto:.2f} USD", "inline": True},
                    {"name": "⏳ Estado", "value": "Pendiente Confirmación Anticipo (50%)", "inline": True},
                ],
                "footer": {
                    "text": f"Registrado el {ahora} | WhatsApp AI Gateway"
                }
            }
        ]
    }

    try:
        res = await http_client.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5.0)
        res.raise_for_status()
        print(f"✅ Alerta de venta enviada a Discord para {nombre}.")
    except Exception as e:
        print(f"❌ Error al enviar notificación a Discord: {e}")


# --- Funciones de Memoria RAG de Clientes ---

async def recuperar_memoria_cliente(telefono: str, consulta_actual: str) -> str:
    """Busca antecedentes del cliente en Qdrant filtrando strictly por su número."""
    try:
        filtro_cliente = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.telefono",
                    match=qmodels.MatchValue(value=telefono),
                )
            ]
        )
        # E5 requiere 'query: ' para buscar
        resultados = await vectorstore_clientes.asimilarity_search(
            query=f"query: {consulta_actual}",
            k=2,
            filter=filtro_cliente
        )
        if not resultados:
            return "Sin registros previos de este cliente."
        
        recuerdos = [doc.page_content.replace("passage: ", "") for doc in resultados]
        return "\n".join(f"- {r}" for r in recuerdos)
    except Exception as e:
        print(f"Error al recuperar memoria de cliente: {e}")
        return "Sin registros previos."

# Instancia ligera de Gemini dedicada solo a extraer hechos/resúmenes
llm_resumen = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    request_timeout=12,
    max_retries=1
)


def extraer_y_guardar_hechos_cliente(telefono: str, nombre: str, texto_cliente: str, respuesta_bot: str):
        """Ejecuta la extracción de datos clave en un hilo desacoplado del servidor."""
        prompt = f"""Eres un analista de datos comerciales. Analiza esta interacción reciente de WhatsApp y extrae información CLAVE sobre el cliente en
  1 frase concisa (rubro del negocio,nombre del cliente, preferencias visuales, requerimientos especiales, presupuesto mencionado).
    Si el mensaje solo contiene saludos, despedidas o preguntas genéricas sin datos sobre el perfil del cliente, responde únicamente la palabra:
  DESCARTAR.

    Cliente ({nombre}): {texto_cliente}
    Bot: {respuesta_bot}

    Hecho clave extraído:"""

        try:
            res_message = llm_resumen.invoke([HumanMessage(content=prompt)])
            raw_content = res_message.content

            # Validación de tipo para str o list
            if isinstance(raw_content, list):
                resultado = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_content]).strip()
            else:
                resultado = str(raw_content).strip()

            if "DESCARTAR" not in resultado and len(resultado) > 10:
                doc = Document(
                    page_content=f"passage: {resultado}",
                    metadata={
                        "telefono": telefono,
                        "nombre": nombre,
                        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                )
                vectorstore_clientes.add_documents([doc])
                print(f"🧠 Memoria guardada en Qdrant para {telefono}: {resultado}")
        except Exception as e:
            print(f"Error al sintetizar o guardar memoria de cliente: {e}")



# --- Herramientas del Agente ---

@tool
async def consultar_servicios_y_politicas(consulta: str) -> str:
    """Consulta la base de conocimientos sobre precios, servicios, tiempos de entrega y políticas del negocio de diseño gráfico."""
    
    # 1. Gemini no requiere prefijos como 'query: '. Pasamos el texto limpio directamente.
    docs = await retriever.ainvoke(consulta.strip())
    
    if not docs:
        return "No se encontró información específica en los documentos del negocio."
        
    # 2. Unimos los fragmentos de forma limpia. 
    # Ya no hace falta remover "passage: " porque Gemini indexó el texto puro.
    contenido = "\n\n---\n\n".join([d.page_content for d in docs])
    
    # 3. Límite estricto de seguridad para el contexto
    return contenido[:2000]

@tool
async def registrar_venta_cerrada(nombre_cliente: str, telefono: str, servicio: str, monto: str) -> str:
    """Registra una venta cerrada en la base de datos local y envía una alerta a Discord. Usar solo cuando el cliente confirme el servicio."""
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        async with get_ventas_db() as db:
            await db.execute(
                "INSERT INTO ventas (fecha, nombre_cliente, telefono, servicio, monto, estado) VALUES (?, ?, ?, ?, ?, ?)",
                (fecha_actual, nombre_cliente, telefono, servicio, monto, "Pendiente Confirmación Anticipo")
            )
            await db.commit()
    except Exception as e:
        print(f"Error al escribir en SQLite con aiosqlite: {e}")

    # Notificación a Discord
    monto_numerico = limpiar_monto(monto)
    await notificar_venta_discord(
        nombre=nombre_cliente,
        telefono=telefono,
        servicio=servicio,
        monto=monto_numerico
    )

    return "Venta registrada con éxito. Se ha guardado en la base de datos y alertado al equipo."


tools = [consultar_servicios_y_politicas, registrar_venta_cerrada]
tools_by_name = {t.name: t for t in tools}

# --- Inicialización del Modelo Gemini ---
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    request_timeout=10,
    max_retries=1
)

# 2. Inicializamos el cliente
llm_gemini_fallback = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    request_timeout=10,
    max_retries=1
)

# 3. Equipar herramientas a cada LLM y construir la cascada
gemini_con_tools = llm_gemini.bind_tools(tools)
fallback_con_tools = llm_gemini_fallback.bind_tools(tools)

# Este es el objeto final que pasarás directamente a tus nodos de LangGraph
router_llm = gemini_con_tools.with_fallbacks(
    [fallback_con_tools],
    exceptions_to_handle=(Exception,)
)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SYSTEM_PROMPT = SystemMessage(content="""Eres el asesor comercial de la agencia. Tu objetivo es asesorar a clientes potenciales sobre nuestros servicios digitales y cerrar ventas.

REGLAS DE OPERACIÓN:
1. Solo puedes ofrecer información y tarifas obtenidas a través de `consultar_servicios_y_politicas`. Nunca inventes precios ni descuentos.
2. Solo invoca `registrar_venta_cerrada` si el cliente confirma explícitamente la compra y tienes su nombre. Recuerda siempre el anticipo del 50%.
3. Si en el contexto del mensaje se incluyen "Antecedentes del cliente", úsalos para personalizar tu trato amablemente sin ser invasivo.
4. Respuestas concisas, comerciales y directas, óptimas para WhatsApp.
""")


# 4. Nodos de LangGraph optimizados para Gemini
async def call_model(state: AgentState):
    # Asegúrate de que esta función devuelva una lista de mensajes (HumanMessage, AIMessage, ToolMessage)
    mensajes_recientes = sanitizar_mensajes_para_gemini(state["messages"])

    # OPCIÓN RECOMENDADA PARA GEMINI:
    # Si SYSTEM_PROMPT es un string, lo ideal es pasarlo como SystemMessage al inicio.
    # Evita concatenarlo si 'mensajes_recientes' ya incluye un SystemMessage previo.
    system_msg = SystemMessage(content=SYSTEM_PROMPT) if isinstance(SYSTEM_PROMPT, str) else SYSTEM_PROMPT
    
    messages = [system_msg] + mensajes_recientes
    
    # Invocación con tolerancia a fallos
    response = await router_llm.ainvoke(messages)
    return {"messages": [response]}


async def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return {"messages": []}
        
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        
        # Ejecución asíncrona de la tool (aquí llamará a tu buscador Qdrant optimizado sin prefijos)
        output = await tool_fn.ainvoke(tool_call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        
    return {"messages": results}

def route_after_model(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    # Validación extra de seguridad sobre el atributo de llamada a herramientas
    if getattr(last_message, "tool_calls", None) and len(last_message.tool_calls) > 0:
        return "tools"
    return "__end__"

# 5. Compilación del Workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_model, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "agent")

# El checkpointer y graph se inicializan en el lifespan asíncrono
graph = None

# --- Reporte Diario Discord ---


async def enviar_reporte_diario_discord():
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ DISCORD_WEBHOOK_URL no configurado para el reporte diario.")
        return

    hoy_str = datetime.now().strftime("%Y-%m-%d")
    archivo_csv = f"reporte_ventas_{hoy_str}.csv"

    try:
        async with get_ventas_db() as db:
            async with db.execute(
                "SELECT * FROM ventas WHERE fecha LIKE ? ORDER BY id ASC", 
                (f"{hoy_str}%",)
            ) as cursor:
                filas = await cursor.fetchall()
                columnas = [desc[0] for desc in cursor.description]

        total_ventas = len(filas)
        if total_ventas == 0:
            payload = {
                "username": "Cierre Diario de Ventas",
                "embeds": [{
                    "title": f"📊 Cierre Diario — {hoy_str}",
                    "description": "Hoy no se registraron nuevas ventas confirmadas.",
                    "color": 9807270
                }]
            }
            await http_client.post(DISCORD_WEBHOOK_URL, json=payload)
            return

        # Escritura de CSV y envío multipart mediante httpx
        with open(archivo_csv, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(columnas)
            for fila in filas:
                writer.writerow([fila[col] for col in columnas])

        col_cliente = next((c for c in columnas if c in ["nombre", "cliente", "nombre_cliente"]), columnas[1])
        col_servicio = next((c for c in columnas if c in ["servicio", "producto", "item"]), columnas[2])
        col_precio = next((c for c in columnas if c in ["precio", "monto", "total"]), columnas[3])

        total_recaudado = sum(limpiar_monto(fila[col_precio]) for fila in filas)

        resumen_items = "\n".join([
            f"• **{fila[col_cliente]}**: {fila[col_servicio]} (${limpiar_monto(fila[col_precio]):.2f})" 
            for fila in filas[:10]
        ])
        if total_ventas > 10:
            resumen_items += f"\n*... y {total_ventas - 10} más en el archivo adjunto.*"

        embed = {
            "title": f"📊 Cierre de Ventas Diario — {hoy_str}",
            "description": f"Resumen consolidado a las 20:00:\n\n{resumen_items}",
            "color": 3066993,
            "fields": [
                {"name": "💼 Total Contratos", "value": str(total_ventas), "inline": True},
                {"name": "💵 Monto Proyectado", "value": f"${total_recaudado:.2f} USD", "inline": True},
                {"name": "🏦 Anticipos (50%)", "value": f"${(total_recaudado * 0.5):.2f} USD", "inline": True}
            ],
            "footer": {"text": "WhatsApp AI Gateway | Adjunto: CSV oficial del día"}
        }

        with open(archivo_csv, "rb") as f:
            files = {"file": (archivo_csv, f.read(), "text/csv")}
            data = {"payload_json": json.dumps({"username": "Cierre Diario de Ventas", "embeds": [embed]})}
            res = await http_client.post(DISCORD_WEBHOOK_URL, data=data, files=files)
            if res.status_code in [200, 204]:
                print(f"✅ Reporte diario enviado a Discord ({total_ventas} ventas).")
            else:
                print(f"❌ Error al enviar reporte a Discord: {res.text}")

        if os.path.exists(archivo_csv):
            os.remove(archivo_csv)

    except Exception as e:
        print(f"❌ Error generando reporte diario: {e}")


# --- Servidor FastAPI ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, graph
    # Inicializa el pool de conexiones asíncronas optimizado
    http_client = httpx.AsyncClient(
        timeout=10.0,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200)
    )

    await init_db()
    
    # Inicializar checkpointer asíncrono y compilar grafo
    async with AsyncSqliteSaver.from_conn_string("conversations.db") as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Tareas programadas
        scheduler.add_job(enviar_reporte_diario_discord, "cron", hour=20, minute=0)
        scheduler.start()
        
        yield
    
    # Cierre ordenado de conexiones
    await http_client.aclose()
    scheduler.shutdown()

app = FastAPI(title="WhatsApp AI Sales Agent", lifespan=lifespan)

# --- Límites y Funciones Auxiliares para Meta Cloud API ---
LIMITES_META = {
    "texto": 3000,                      # caracteres
    "imagen": 5 * 1024 * 1024,          # 5 MB
    "audio": 3 * 1024 * 1024            # 3 MB (~90 seg)
}

async def descargar_media_meta_con_limite(media_id: str):
    """Consulta la URL del media_id en Meta Graph API y descarga el contenido si cumple los límites de tamaño."""
    if not WHATSAPP_TOKEN or not media_id:
        return None, None, "error_config"

    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    
    try:
        # 1. Obtener URL y metadata del archivo en Meta
        res_info = await http_client.get(
            f"{GRAPH_API_URL}/{media_id}", 
            headers=headers, 
            timeout=10.0
        )
        if res_info.status_code != 200:
            print(f"❌ Error al consultar metadata de media_id {media_id} en Meta ({res_info.status_code}): {res_info.text}")
            return None, None, "error_metadata"
        
        info = res_info.json()
        file_size = info.get("file_size", 0)
        mime = info.get("mime_type", "")
        media_url = info.get("url")

        # 2. Aplicar validación de límites de tamaño
        if "image" in mime and file_size > LIMITES_META["imagen"]:
            print(f"⚠️ Imagen descartada: {file_size} bytes excede límite de 5 MB.")
            return None, mime, "imagen_muy_grande"
        if "audio" in mime and file_size > LIMITES_META["audio"]:
            print(f"⚠️ Audio descartado: {file_size} bytes excede límite de 3 MB.")
            return None, mime, "audio_muy_largo"

        # 3. Descargar el archivo binario desde la URL temporal de Meta
        if not media_url:
            return None, mime, "url_invalida"

        res_bin = await http_client.get(media_url, headers=headers, timeout=20.0)
        if res_bin.status_code != 200:
            print(f"❌ Error al descargar binario de media desde Meta ({res_bin.status_code})")
            return None, mime, "error_descarga"
        
        b64_data = base64.b64encode(res_bin.content).decode("utf-8")
        return b64_data, mime, "ok"

    except Exception as e:
        print(f"❌ Excepción al descargar media desde Meta ({media_id}): {e}")
        return None, None, "error_excepcion"

# --- HELPERS CLOUD API ---
def limpiar_numero(numero: str) -> str:
    return numero.replace("@s.whatsapp.net", "").replace("+", "").replace(" ", "")


async def enviar_mensaje_whatsapp(to: str, mensaje: str):
    """Envía un mensaje de texto al usuario utilizando la Cloud API oficial de Meta."""
    
    phone = limpiar_numero(to)
    
    if not PHONE_NUMBER_ID or not WHATSAPP_TOKEN:
        print("⚠️ PHONE_NUMBER_ID o WHATSAPP_TOKEN no configurado en entorno.")
        return

    url = f"{GRAPH_API_URL}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": mensaje}
    }
    
    try:
        response = await http_client.post(url, headers=headers, json=payload, timeout=10.0)
        response.raise_for_status()
        print(f"📤 Mensaje oficial enviado vía Meta Graph API a {phone}.")
    except httpx.HTTPStatusError as e:
        print(f"❌ Error HTTP al enviar WhatsApp vía Meta ({e.response.status_code}): {e.response.text}")
    except Exception as e:
        print(f"❌ Error de conexión con Meta Graph API: {e}")


async def procesar_mensaje_ia(
    remote_jid: str, 
    nombre_remitente: str, 
    texto: str,
    message_type: str = "text",
    media_id_or_b64: Optional[str] = None,
    mime_type: Optional[str] = None
):
    telefono = remote_jid.replace("@s.whatsapp.net", "").replace("@lid", "")
    config = {"configurable": {"thread_id": telefono}}
    
    # 1. Validación de longitud de texto
    if message_type == "text" and len(texto) > LIMITES_META["texto"]:
        await enviar_mensaje_whatsapp(
            remote_jid,
            "⚠️ Tu mensaje es demasiado largo. Por favor envíame tu consulta resumida en un par de líneas."
        )
        return

    base64_final = None
    mime_final = mime_type

    # 2. Si viene un media_id de Meta (o Base64), procesarlo/descargarlo
    if message_type in ["image", "audio", "video", "document"] and media_id_or_b64:
        if len(media_id_or_b64) < 300 and "/" not in media_id_or_b64[:20]:
            # Es un MEDIA_ID de Meta Cloud API
            b64_data, mime_detected, estado = await descargar_media_meta_con_limite(media_id_or_b64)
            if estado == "imagen_muy_grande":
                await enviar_mensaje_whatsapp(
                    remote_jid, 
                    "⚠️ La imagen enviada pesa más de 5 MB. Por favor envíala comprimida."
                )
                return
            elif estado == "audio_muy_largo":
                await enviar_mensaje_whatsapp(
                    remote_jid, 
                    "⚠️ El audio enviado es muy largo (máx 3 MB / ~90 seg). Por favor envía una nota de voz más corta."
                )
                return
            elif estado == "ok":
                base64_final = b64_data
                mime_final = mime_detected
        else:
            base64_final = media_id_or_b64

    # 3. Recuperar memoria RAG del cliente
    consulta_memoria = texto or ("imagen adjunta" if message_type == "image" else "audio de voz")
    antecedentes = await recuperar_memoria_cliente(telefono, consulta_memoria)
    
    prompt_contexto = (
        f"[Antecedentes del cliente en sistema: {antecedentes}]\n"
        f"[Cliente: {nombre_remitente}, Tel: {telefono}]"
    )
    
    content_blocks = []
    
    if message_type == "image" and base64_final:
        clean_mime = mime_final.split(";")[0] if mime_final else "image/jpeg"
        prompt_texto = f"{prompt_contexto}\n[El cliente envió una imagen/foto con el comentario: '{texto}']\nAnaliza la imagen enviada para comprender su solicitud de diseño gráfico."
        content_blocks.append({"type": "text", "text": prompt_texto})
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{clean_mime};base64,{base64_final}"}
        })
    elif message_type == "audio" and base64_final:
        clean_mime = mime_final.split(";")[0] if mime_final else "audio/ogg"
        prompt_texto = f"{prompt_contexto}\n[El cliente envió una nota de voz/audio]. Escucha el audio atentamente y responde su solicitud comercial."
        content_blocks.append({"type": "text", "text": prompt_texto})
        content_blocks.append({
            "type": "media",
            "mime_type": clean_mime,
            "data": base64_final
        })
    else:
        prompt_texto = f"{prompt_contexto}: {texto}"
        content_blocks.append({"type": "text", "text": prompt_texto})
    
    # Restringe a 5 llamadas simultáneas hacia la API de Google
    async with gemini_semaphore:
        output = await graph.ainvoke(
            {"messages": [HumanMessage(content=content_blocks)]},
            config=config
        )
    
    raw_content = output["messages"][-1].content
    if isinstance(raw_content, list):
        respuesta_final = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_content])
    else:
        respuesta_final = str(raw_content)
        
    await enviar_mensaje_whatsapp(remote_jid, respuesta_final)
    
    # Tarea en background para guardar memoria del cliente sin bloquear
    texto_resumen = texto or (f"[{message_type.upper()} enviado por cliente]" if message_type != "text" else "")
    asyncio.create_task(
        asyncio.to_thread(
            extraer_y_guardar_hechos_cliente,
            telefono, nombre_remitente, texto_resumen, respuesta_final
        )
    )


async def obtener_resumen_ventas_hoy() -> str:
    """Consulta SQLite asíncronamente y devuelve un resumen formateado para WhatsApp."""
    hoy_str = datetime.now().strftime("%Y-%m-%d")
    try:
        async with get_ventas_db() as db:
            async with db.execute(
                "SELECT * FROM ventas WHERE fecha LIKE ? ORDER BY id ASC", 
                (f"{hoy_str}%",)
            ) as cursor:
                filas = await cursor.fetchall()
                columnas = [col[0] for col in cursor.description]

        if not filas:
            return f"📊 *Reporte del Día ({hoy_str})*\n\nNo se han registrado ventas el día de hoy."

        # Detección dinámica de columnas según el esquema actual
        col_cliente = next((c for c in columnas if c in ["nombre", "cliente", "nombre_cliente"]), columnas[1])
        col_servicio = next((c for c in columnas if c in ["servicio", "producto", "item"]), columnas[2])
        col_precio = next((c for c in columnas if c in ["precio", "monto", "total"]), columnas[3])

        total_ventas = len(filas)
        total_recaudado = sum(limpiar_monto(f[col_precio]) for f in filas)
        anticipos = total_recaudado * 0.5

        lineas = [f"📊 *REPORTE DIARIO DE VENTAS ({hoy_str})*\n"]
        for idx, f in enumerate(filas, 1):
            monto = limpiar_monto(f[col_precio])
            lineas.append(f"{idx}. *{f[col_cliente]}* — {f[col_servicio]} (${monto:.2f})")

        lineas.append("\n" + "—" * 20)
        lineas.append(f"💼 *Total contratos:* {total_ventas}")
        lineas.append(f"💵 *Monto proyectado:* ${total_recaudado:.2f} USD")
        lineas.append(f"🏦 *Anticipos requeridos (50%):* ${anticipos:.2f} USD")

        return "\n".join(lineas)
    except Exception as e:
        return f"❌ Error generando resumen: {e}"


@app.get("/webhook")
async def verificar_webhook(request: Request):
    """Verifica la suscripción del webhook ante el reto inicial de Meta Cloud API."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    print(f"Meta está verificando: mode={mode} token={token}") # para ver en consola

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Webhook verificado exitosamente por Meta Cloud API.")
        return PlainTextResponse(content=challenge) # devolver como texto plano, no JSON ni int
        #return Response(content=challenge, media_type="text/plain", status_code=200)
    
    print("❌ Fallo en la verificación del webhook de Meta (token inválido).")
    return Response(content="Error de verificación", status_code=403)


# --- TU ENDPOINT ADAPTADO (POST) ---
@app.post("/webhook")
async def recibir_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    print("🔥 WEBHOOK CRUDO:", body) # <--- ESTO ES CLAVE
    try:
        entry = body.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {})
        if "messages" not in entry:
            return {"status": "received"}

        msg = entry["messages"][0]
        contact = entry.get("contacts", [{}])[0]

        sender_phone = limpiar_numero(msg.get("from", ""))
        remote_jid = sender_phone # YA NO usamos @s.whatsapp.net para Cloud API
        nombre = contact.get("profile", {}).get("name", "Cliente")
        msg_type = msg.get("type", "text")

        texto = ""
        media_id = mime = None

        if msg_type == "text":
            texto = msg.get("text", {}).get("body", "")
        elif msg_type in ["image", "audio", "video", "document"]:
            media_id = msg.get(msg_type, {}).get("id")
            mime = msg.get(msg_type, {}).get("mime_type")
            texto = msg.get(msg_type, {}).get("caption", "")

        es_admin = bool(ADMIN_PHONE and limpiar_numero(ADMIN_PHONE) in sender_phone)

        if es_admin and texto.lower() in ["/reporte", "/ventas", "!reporte"]:
            print(f"👑 Comando admin desde {sender_phone}: {texto}")
            reporte_texto = await obtener_resumen_ventas_hoy()
            await enviar_mensaje_whatsapp(remote_jid, reporte_texto)
            return {"status": "received"}

        if remote_jid and (texto or media_id):
            background_tasks.add_task(
                procesar_mensaje_ia, remote_jid, nombre, texto, msg_type, media_id, mime
            )

    except Exception as e:
        print(f"❌ Error procesando webhook: {e}")

    return {"status": "received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
