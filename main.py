import os
import sqlite3
import requests
import csv
import re
import json
from datetime import datetime
from typing import Annotated, Literal
from dotenv import load_dotenv

from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
import torch

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Distance, VectorParams

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

# --- Configuración de Entorno ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "http://localhost:8080")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "clave_secreta_para_tu_api_local")
ADMIN_PHONE = os.getenv("ADMIN_PHONE")

COLLECTION_KNOWLEDGE = "diseno_grafico_knowledge"
COLLECTION_CLIENTS = "clientes_memoria"

# --- Inicialización de SQLite ---
def init_db():
    conn = sqlite3.connect("ventas.db")
    cursor = conn.cursor()
    cursor.execute("""
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
    conn.commit()
    conn.close()

init_db()

# --- Configuración de Embeddings con multilingual-e5-small (float16) ---
model_kwargs = {
    "torch_dtype": torch.float16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

client_qdrant = QdrantClient(url=QDRANT_URL)

# 1. Vectorstore de Conocimientos del Negocio
vectorstore_knowledge = QdrantVectorStore(
    client=client_qdrant,
    collection_name=COLLECTION_KNOWLEDGE,
    embedding=embeddings
)
retriever_knowledge = vectorstore_knowledge.as_retriever(search_kwargs={"k": 2})

# 2. Vectorstore de Memoria a Largo Plazo de Clientes
if not client_qdrant.collection_exists(COLLECTION_CLIENTS):
    client_qdrant.create_collection(
        collection_name=COLLECTION_CLIENTS,
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
    embedding=embeddings
)

# --- Funciones de Memoria RAG de Clientes ---

def recuperar_memoria_cliente(telefono: str, consulta_actual: str) -> str:
    """Busca antecedentes del cliente en Qdrant filtrando estrictamente por su número."""
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
        resultados = vectorstore_clientes.similarity_search(
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
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0
)

def extraer_y_guardar_hechos_cliente(telefono: str, nombre: str, texto_cliente: str, respuesta_bot: str):
    """Analiza la interacción con Gemini y guarda datos clave del cliente en Qdrant."""
    prompt = f"""Eres un analista de datos comerciales. Analiza esta interacción reciente de WhatsApp y extrae información CLAVE sobre el cliente en 1 frase concisa (por ejemplo: rubro de su negocio, preferencias visuales, requerimientos especiales, presupuesto mencionado).
Si el mensaje solo contiene saludos, despedidas o preguntas genéricas sin información sobre el cliente, responde únicamente la palabra: DESCARTAR.

Cliente ({nombre}): {texto_cliente}
Bot: {respuesta_bot}

Hecho clave extraído:"""

    try:
        resultado = llm_resumen.invoke([HumanMessage(content=prompt)]).content.strip()
        
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


# --- Herramientas del Agente (Tools) ---

@tool
def consultar_servicios_y_politicas(consulta: str) -> str:
    """Consulta la base de conocimientos sobre precios, servicios, tiempos de entrega y políticas del negocio de diseño gráfico."""
    consulta_formateada = f"query: {consulta.strip()}"
    docs = retriever_knowledge.invoke(consulta_formateada)
    if not docs:
        return "No se encontró información específica en los documentos del negocio."
    return "\n\n".join([d.page_content.replace("passage: ", "") for d in docs])

@tool
def registrar_venta_cerrada(nombre_cliente: str, telefono: str, servicio: str, monto: str) -> str:
    """Registra una venta cerrada en la base de datos local y envía una alerta a Discord. Usar solo cuando el cliente confirme el servicio."""
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        conn = sqlite3.connect("ventas.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ventas (fecha, nombre_cliente, telefono, servicio, monto, estado) VALUES (?, ?, ?, ?, ?, ?)",
            (fecha_actual, nombre_cliente, telefono, servicio, monto, "Pendiente Confirmación Anticipo")
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error al escribir en SQLite: {e}")

    # Notificación en Discord
    if DISCORD_WEBHOOK_URL:
        payload = {
            "username": "Bot de Ventas",
            "embeds": [{
                "title": "🎉 ¡Nueva Venta Cerrada!",
                "color": 3066993,
                "fields": [
                    {"name": "👤 Cliente", "value": nombre_cliente, "inline": True},
                    {"name": "📱 Teléfono", "value": telefono, "inline": True},
                    {"name": "🛠 Servicio", "value": servicio, "inline": False},
                    {"name": "💵 Monto", "value": monto, "inline": True},
                    {"name": "🗓 Fecha", "value": fecha_actual, "inline": True}
                ],
                "footer": {"text": "Registrado en ventas.db"}
            }]
        }
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            print(f"Error al notificar a Discord: {e}")

    return "Venta registrada con éxito. Se ha guardado en la base de datos y alertado al equipo."

tools = [consultar_servicios_y_politicas, registrar_venta_cerrada]
tools_by_name = {t.name: t for t in tools}

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
).bind_tools(tools)

# --- Construcción del Grafo LangGraph con Poda de Historial ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SYSTEM_PROMPT = SystemMessage(content="""Eres el asesor comercial de la agencia. Tu objetivo es asesorar a clientes potenciales sobre nuestros servicios digitales y cerrar ventas.

REGLAS DE OPERACIÓN:
1. Solo puedes ofrecer información y tarifas obtenidas a través de `consultar_servicios_y_politicas`. Nunca inventes precios ni descuentos.
2. Solo invoca `registrar_venta_cerrada` si el cliente confirma explícitamente la compra y tienes su nombre. Recuerda siempre el anticipo del 50%.
3. Si en el contexto del mensaje se incluyen "Antecedentes del cliente", úsalos para personalizar tu trato amablemente sin ser invasivo.
4. Respuestas concisas, comerciales y directas, óptimas para WhatsApp.
""")

def call_model(state: AgentState):
    # PODA DE MENSAJES (Ahorro de tokens): conservamos solo los últimos 6 intercambios
    mensajes_recientes = state["messages"][-6:]
    messages = [SYSTEM_PROMPT] + mensajes_recientes
    response = llm.invoke(messages)
    return {"messages": [response]}

def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        output = tool_fn.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    return {"messages": results}

def route_after_model(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_model, {"tools": "tools", "__end__": END})
workflow.add_edge("tools", "agent")

memory_conn = sqlite3.connect("conversations.db", check_same_thread=False)
checkpointer = SqliteSaver(memory_conn)
graph = workflow.compile(checkpointer=checkpointer)

# --- Servidor FastAPI ---
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Programar todos los días a las 20:00 (8 PM)
    scheduler.add_job(
        enviar_reporte_diario_discord,
        trigger="cron",
        hour=20,
        minute=0
    )
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(title="WhatsApp AI Sales Agent", lifespan=lifespan)

def enviar_mensaje_whatsapp(remote_jid: str, mensaje: str):
    url = "http://localhost:3001/send"
    try:
        requests.post(url, json={"remoteJid": remote_jid, "text": mensaje}, timeout=10)
    except Exception as e:
        print(f"Error al enviar mensaje vía WhatsApp Gateway: {e}")

def procesar_mensaje_ia(remote_jid: str, nombre_remitente: str, texto: str, background_tasks: BackgroundTasks):
    telefono = remote_jid.replace("@s.whatsapp.net", "").replace("@lid", "")
    config = {"configurable": {"thread_id": telefono}}
    
    # 1. Recuperar memoria episódica del cliente desde Qdrant
    antecedentes = recuperar_memoria_cliente(telefono, texto)
    
    # 2. Inyectar antecedentes de forma compacta
    prompt_usuario = (
        f"[Antecedentes del cliente en sistema: {antecedentes}]\n"
        f"[Cliente: {nombre_remitente}, Tel: {telefono}]: {texto}"
    )
    
    output = graph.invoke(
        {"messages": [HumanMessage(content=prompt_usuario)]},
        config=config
    )
    
    raw_content = output["messages"][-1].content
    if isinstance(raw_content, list):
        respuesta_final = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_content])
    else:
        respuesta_final = str(raw_content)
        
    # 3. Enviar respuesta por WhatsApp
    enviar_mensaje_whatsapp(remote_jid, respuesta_final)
    
    # 4. Tarea asíncrona: sintetizar y guardar memoria sin bloquear el chat
    background_tasks.add_task(
        extraer_y_guardar_hechos_cliente,
        telefono=telefono,
        nombre=nombre_remitente,
        texto_cliente=texto,
        respuesta_bot=respuesta_final
    )

@app.post("/webhook")
async def recibir_webhook(request: Request, background_tasks: BackgroundTasks):
    datos = await request.json()
    remote_jid = datos.get("remoteJid", "")
    nombre = datos.get("name", "Cliente")
    texto = (datos.get("message") or "").strip()

    if remote_jid and texto:
        background_tasks.add_task(
            procesar_mensaje_ia, 
            remote_jid, 
            nombre, 
            texto, 
            background_tasks
        )

    return {"status": "received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)