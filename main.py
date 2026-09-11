import os
import sqlite3
import requests
from datetime import datetime
from typing import Annotated, Literal
from dotenv import load_dotenv

from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

load_dotenv()

# --- Configuración de Entorno ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "http://localhost:8080")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "clave_secreta_para_tu_api_local")
COLLECTION_NAME = "diseno_grafico_knowledge"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- Inicialización de Base de Datos Local de Ventas ---
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

# --- Conexión RAG Local ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client_qdrant = QdrantClient(url=QDRANT_URL)
vectorstore = QdrantVectorStore(
    client=client_qdrant,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def notificar_venta_discord(nombre: str, telefono: str, servicio: str, monto: float):
    """Envía un Embed enriquecido al canal de Discord vía Webhook."""
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ Advertencia: DISCORD_WEBHOOK_URL no configurado.")
        return

    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    embed = {
        "title": "🎉 ¡Nueva Venta Registrada!",
        "description": "Se ha cerrado un pedido desde el bot de WhatsApp.",
        "color": 3066993,  # Código hexadecimal en decimal (Verde #2ECC71)
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

    payload = {
        "username": "Gestor de Ventas",
        "embeds": [embed]
    }

    try:
        res = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=8)
        if res.status_code in [200, 204]:
            print("✅ Notificación enviada a Discord con éxito.")
        else:
            print(f"❌ Error enviando a Discord (HTTP {res.status_code}): {res.text}")
    except Exception as e:
        print(f"❌ Excepción al conectar con Discord: {e}")


# --- Herramientas del Agente (Tools) ---

@tool
def consultar_servicios_y_politicas(consulta: str) -> str:
    """Consulta la base de conocimientos sobre precios, servicios, tiempos de entrega y políticas del negocio de diseño gráfico."""
    docs = retriever.invoke(consulta)
    if not docs:
        return "No se encontró información específica en los documentos del negocio."
    return "\n\n".join([d.page_content for d in docs])

@tool
def registrar_venta_cerrada(nombre_cliente: str, telefono: str, servicio: str, monto: str) -> str:
    """Registra una venta cerrada en la base de datos local y envía una alerta a Discord. Usar solo cuando el cliente confirme el servicio."""
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Guardar en SQLite
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

    # 2. Notificación en Discord
    if DISCORD_WEBHOOK_URL:
        payload = {
            "username": "Bot de Ventas",
            "embeds": [
                {
                    "title": "🎉 ¡Nueva Venta Cerrada!",
                    "color": 3066993,  # Verde esmeralda
                    "fields": [
                        {"name": "👤 Cliente", "value": nombre_cliente, "inline": True},
                        {"name": "📱 Teléfono", "value": telefono, "inline": True},
                        {"name": "🛠 Servicio", "value": servicio, "inline": False},
                        {"name": "💵 Monto Acordado", "value": monto, "inline": True},
                        {"name": "🗓 Fecha", "value": fecha_actual, "inline": True}
                    ],
                    "footer": {"text": "Registrado en ventas.db local"}
                }
            ]
        }
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            print(f"Error al notificar a Discord: {e}")

    return "Venta registrada con éxito. Se ha alertado al equipo y guardado en la base de datos."

tools = [consultar_servicios_y_politicas, registrar_venta_cerrada]
tools_by_name = {t.name: t for t in tools}

# --- Inicialización del Modelo Gemini ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
).bind_tools(tools)
# --- Construcción del Grafo LangGraph ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SYSTEM_PROMPT = SystemMessage(content="""
Eres el asesor comercial experto de un estudio de diseño gráfico. Tu labor es atender a los clientes por WhatsApp, resolver sus dudas y cerrar ventas.

Reglas operativas obligatorias:
1. INFORMACIÓN ESTRICTA: Usa la herramienta 'consultar_servicios_y_politicas' para verificar precios, tiempos de entrega y condiciones. Nunca inventes tarifas.
2. POLÍTICA DE ANTICIPOS: Recuerda siempre con amabilidad que para iniciar cualquier trabajo se requiere el 50% de anticipo.
3. CIERRE DE VENTA: Si el cliente confirma que desea contratar un servicio, asegúrate de tener:
   - Su nombre (o pregúntaselo).
   - Su teléfono.
   - El servicio específico.
   - El precio correspondiente.
   Una vez confirmados estos datos, ejecuta la herramienta 'registrar_venta_cerrada'.
4. TONO: Cercano, conciso y profesional, apto para WhatsApp (sin textos excesivamente largos).
""")

def call_model(state: AgentState):
    messages = [SYSTEM_PROMPT] + state["messages"]
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

# Memoria de conversación persistente por chat (SQLite)
memory_conn = sqlite3.connect("conversations.db", check_same_thread=False)
checkpointer = SqliteSaver(memory_conn)
graph = workflow.compile(checkpointer=checkpointer)

# --- Servidor Web FastAPI ---
app = FastAPI(title="WhatsApp AI Sales Agent")

def enviar_mensaje_whatsapp(remote_jid: str, mensaje: str):
    url = "http://localhost:3001/send"
    payload = {
        "remoteJid": remote_jid,
        "text": mensaje
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error al enviar mensaje vía Gateway Baileys: {e}")


def procesar_mensaje_ia(remote_jid: str, nombre_remitente: str, texto: str):
    telefono = remote_jid.replace("@s.whatsapp.net", "").replace("@lid", "")
    config = {"configurable": {"thread_id": telefono}}
    prompt_usuario = f"[Cliente: {nombre_remitente}, Tel: {telefono}]: {texto}"
    
    output = graph.invoke(
        {"messages": [HumanMessage(content=prompt_usuario)]},
        config=config
    )
    
    raw_content = output["messages"][-1].content
    
    # Asegurar que siempre sea un string plano
    if isinstance(raw_content, list):
        # Si LangChain devuelve bloques [{"type": "text", "text": "..."}]
        partes = [b.get("text", "") if isinstance(b, dict) else str(b) for b in raw_content]
        respuesta_final = "".join(partes)
    else:
        respuesta_final = str(raw_content)
        
    enviar_mensaje_whatsapp(remote_jid, respuesta_final)



@app.post("/webhook")
async def recibir_webhook(request: Request, background_tasks: BackgroundTasks):
    datos = await request.json()
    remote_jid = datos.get("remoteJid")
    nombre = datos.get("name", "Cliente")
    texto = datos.get("message", "")

    if remote_jid and texto.strip():
        background_tasks.add_task(procesar_mensaje_ia, remote_jid, nombre, texto)

    return {"status": "received"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)