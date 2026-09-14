const { 
    default: makeWASocket, 
    useMultiFileAuthState, 
    DisconnectReason, 
    fetchLatestBaileysVersion,
    Browsers,
    downloadMediaMessage
} = require("@whiskeysockets/baileys");
const qrcode = require("qrcode-terminal");
const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json({ limit: "50mb" }));

const PORT = 3001;
const PYTHON_WEBHOOK = "http://localhost:8000/webhook";

let sock = null;

async function startWhatsApp() {
    const { state, saveCreds } = await useMultiFileAuthState("auth_info_baileys");
    
    // Obtener la versión web oficial más reciente de WhatsApp
    const { version, isLatest } = await fetchLatestBaileysVersion();
    console.log(`Usando WhatsApp Web v${version.join(".")}, ¿Es la última versión?: ${isLatest}`);

    sock = makeWASocket({
        version,
        auth: state,
        printQRInTerminal: false,
        browser: Browsers.ubuntu("Chrome"), // Firma de navegador estándar para evitar rechazos
        syncFullHistory: false // No sincronizar historial antiguo para vincular más rápido
    });

    sock.ev.on("creds.update", saveCreds);

    sock.ev.on("connection.update", (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            console.log("\n📱 Escanea este nuevo código QR con WhatsApp:");
            qrcode.generate(qr, { small: true });
        }

        if (connection === "close") {
            const statusCode = (lastDisconnect?.error)?.output?.statusCode;
            const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
            console.log(`Conexión cerrada (código: ${statusCode}). Reconectando: ${shouldReconnect}`);
            
            if (shouldReconnect) {
                startWhatsApp();
            } else {
                console.log("Sesión cerrada. Borra auth_info_baileys y vuelve a escanear.");
            }
        } else if (connection === "open") {
            console.log("✅ ¡WhatsApp conectado exitosamente!");
        }
    });

    /// Escuchar mensajes entrantes con soporte para @lid y @s.whatsapp.net
    sock.ev.on("messages.upsert", async (m) => {
        try {
            const msg = m.messages[0];
            if (!msg || !msg.message) return;

            // Ignorar mensajes enviados por el propio bot
            if (msg.key.fromMe) return;

            const remoteJid = msg.key.remoteJid;
            console.log(`\n📩 Mensaje detectado desde JID: ${remoteJid}`);

            // Descartar exclusivamente estados (stories) y grupos
            if (remoteJid === "status@broadcast" || remoteJid.endsWith("@g.us")) {
                console.log(`ℹ️ Mensaje ignorado (grupo o estado): ${remoteJid}`);
                return;
            }

            // Detección de contenido multimedia y texto
            const isImage = Boolean(msg.message?.imageMessage);
            const isAudio = Boolean(msg.message?.audioMessage);
            
            let messageType = "text";
            let mediaBase64 = null;
            let mimeType = null;
            let textContent = 
                msg.message?.conversation || 
                msg.message?.extendedTextMessage?.text || 
                msg.message?.imageMessage?.caption || 
                "";

            const senderName = msg.pushName || "Cliente";

            if (isImage) {
                messageType = "image";
                console.log(`📸 Imagen detectada de ${senderName}`);
                try {
                    const buffer = await downloadMediaMessage(msg, "buffer", {});
                    mediaBase64 = buffer.toString("base64");
                    mimeType = msg.message.imageMessage.mimetype || "image/jpeg";
                } catch (mediaErr) {
                    console.error("❌ Error descargando imagen de Baileys:", mediaErr.message);
                }
            } else if (isAudio) {
                messageType = "audio";
                console.log(`🎙️ Nota de voz / Audio detectado de ${senderName}`);
                try {
                    const buffer = await downloadMediaMessage(msg, "buffer", {});
                    mediaBase64 = buffer.toString("base64");
                    mimeType = msg.message.audioMessage.mimetype || "audio/ogg; codecs=opus";
                } catch (mediaErr) {
                    console.error("❌ Error descargando audio de Baileys:", mediaErr.message);
                }
            }

            console.log(`👤 Remitente: ${senderName} | Tipo: ${messageType} | Texto/Caption: "${textContent}"`);

            if (textContent.trim() || mediaBase64) {
                // 1. Activar animación de "escribiendo..." en WhatsApp
                await sock.sendPresenceUpdate('composing', remoteJid);
                console.log("🚀 Enviando payload a FastAPI (http://localhost:8000/webhook)...");

                try {
                    const response = await axios.post(PYTHON_WEBHOOK, {
                        remoteJid: remoteJid,
                        name: senderName,
                        messageType: messageType,
                        message: textContent,
                        mediaBase64: mediaBase64,
                        mimeType: mimeType
                    });
                    console.log(`📡 Respuesta de FastAPI: Código ${response.status}`);
                } catch (apiErr) {
                    await sock.sendPresenceUpdate('paused', remoteJid);
                    throw apiErr;
                }
            } else {
                console.log("⚠️ Mensaje recibido sin contenido procesable.");
            }
        } catch (err) {
            console.error("❌ Error procesando mensaje entrante en Node:", err.message);
        }
    });
}

// Función auxiliar para pausar la ejecución
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

app.post("/send", async (req, res) => {
    let { remoteJid, text } = req.body;
    if (!sock || !remoteJid || !text) {
        return res.status(400).json({ error: "Faltan parámetros o socket no listo" });
    }

    try {
        const mensajeTexto = typeof text === "string" ? text : String(text);

        // 1. Responder de inmediato a FastAPI para no bloquear la tarea en segundo plano
        res.json({ status: "queued", delaySeconds: 20 });

        console.log(`⏳ Simulando presencia humana para ${remoteJid} (20s de espera)...`);

        // 2. Notificar a WhatsApp que el usuario está escribiendo
        await sock.sendPresenceUpdate("composing", remoteJid);

        // 3. Esperar 20 segundos antes del disparo final
        await delay(20000);

        // 4. Pausar el estado de escritura y enviar el mensaje
        await sock.sendPresenceUpdate("paused", remoteJid);
        await sock.sendMessage(remoteJid, { text: mensajeTexto });

        console.log(`📤 Respuesta enviada con éxito tras el retraso a ${remoteJid}`);
    } catch (err) {
        console.error("Error enviando mensaje con retraso:", err.message);
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Gateway de WhatsApp escuchando en http://localhost:${PORT}`);
    startWhatsApp();
});