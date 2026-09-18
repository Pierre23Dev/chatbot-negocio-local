# Guía de Actualización de la Base de Conocimiento RAG (`modify_ingest.md`)

Esta guía explica cómo agregar nuevos productos, modificar políticas de negocio y actualizar la base de datos RAG en Supabase, tanto en desarrollo local como cuando el chatbot esté **desplegado y funcionando en producción en Vercel**.

---

## 📌 Principio Clave: Vercel vs. Supabase

- **Vercel** aloja únicamente el código de la API y el Webhook en un entorno Serverless.
- **Supabase** almacena **toda la información** (los vectores RAG, la memoria de clientes y la tabla de ventas).

> [!TIP]
> **Ventaja de esta arquitectura**: Cada vez que actualices la información con `ingest.py`, los datos se guardan **directamente en Supabase**. **No necesitas volver a desplegar en Vercel**. El bot que corre en Vercel comenzará a responder con los nuevos datos al instante.

---

## 1. Modificar o Agregar Nuevos Datos en Local

### Caso A: Agregar Nuevos Productos o Servicios en Archivos Existentes
1. Edita el archivo `data/servicios.txt` o `data/politicas.txt`.
2. Guarda los cambios.
3. Ejecuta la ingesta especificando el número de teléfono del negocio (tenant):
   ```powershell
   .\.venv_new\Scripts\python.exe ingest.py --tenant "+5215512345678"
   ```

---

### Caso B: Agregar Archivos Nuevos (Ej: `preguntas_frecuentes.txt` o `promociones.txt`)
1. Crea el nuevo archivo en la carpeta `data/` (ej: `data/promociones.txt`).
2. Ejecuta `ingest.py` usando el argumento `--archivos`:
   ```powershell
   .\.venv_new\Scripts\python.exe ingest.py --tenant "+5215512345678" --archivos "data/servicios.txt" "data/politicas.txt" "data/promociones.txt"
   ```

---

### Caso C: Actualizar Precios o Políticas sin Duplicar Información
Si cambiaste un precio (ej: de $50 a $70) o modificaste condiciones, debes limpiar los fragmentos anteriores para que la IA no confunda la información vieja con la nueva:

1. Modifica tus archivos en `data/`.
2. Agrega el parámetro `--limpiar` al ejecutar el script:
   ```powershell
   .\.venv_new\Scripts\python.exe ingest.py --tenant "+5215512345678" --limpiar
   ```

El flag `--limpiar` realizará dos pasos automáticos:
1. Elimina todos los vectores de conocimiento anteriores de ese `tenant`.
2. Lee y convierte los nuevos archivos a vectores en Supabase.

---

## 2. ¿Cómo Actualizar la Información cuando el Proyecto esté en Producción (Vercel)?

Cuando el chatbot esté desplegado en Vercel y atendiendo clientes reales en WhatsApp, existen dos formas de actualizar los conocimientos del bot:

### Método 1: Desde la Terminal (Recomendado y más rápido)
No necesitas tocar Vercel. Asegúrate de tener la variable `DATABASE_URL` de Supabase configurada en tu archivo `.env` local y ejecuta:

```powershell
.\.venv_new\Scripts\python.exe ingest.py --tenant "+5215512345678" --limpiar
```

#### ¿Qué sucede internamente?
1. El script local se conecta directamente a tu base de datos en la nube de **Supabase**.
2. Actualiza los vectores en la tabla `diseno_grafico_knowledge`.
3. El webhook que corre en **Vercel** consulta a Supabase en cada mensaje entrante, por lo que responderá con los nuevos precios/políticas inmediatamente en WhatsApp.

---

### Método 2: Mediante Integración Futura (Panel Web / Dashboard Admin)
Si en el futuro deseas que usuarios o clientes actualicen su catálogo desde un panel web (sin usar la terminal de comandos):
1. Se puede habilitar un endpoint `POST /api/admin/ingest` en FastAPI en `src/app/main.py`.
2. Un panel web sube el archivo de texto y Vercel procesa la ingesta y la guarda en Supabase.

---

## 📋 Resumen de Comandos Rápidos

| Acción | Comando a Ejecutar |
| :--- | :--- |
| **Ingestar archivos por defecto** | `python ingest.py --tenant "+NUMERO"` |
| **Reemplazar datos viejos (Actualizar)** | `python ingest.py --tenant "+NUMERO" --limpiar` |
| **Ingestar archivos adicionales** | `python ingest.py --tenant "+NUMERO" --archivos "data/archivo1.txt" "data/archivo2.txt"` |
