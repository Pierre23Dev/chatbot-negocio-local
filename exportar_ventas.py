import os
import sys
import csv
import argparse
from datetime import datetime
from dotenv import load_dotenv
import psycopg

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("[ERROR] No se encontró la variable DATABASE_URL en el entorno.")
    sys.exit(1)

# Normalizar URL para psycopg si viene con la sintaxis de SQLAlchemy
if "+psycopg" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("+psycopg", "")

def exportar_a_csv(tenant_phone: str = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = tenant_phone if tenant_phone else "todos"
    archivo_salida = f"reporte_ventas_{suffix}_{timestamp}.csv"

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                if tenant_phone:
                    cursor.execute(
                        "SELECT id, tenant_phone, fecha, nombre_cliente, telefono, servicio, monto, estado FROM ventas WHERE tenant_phone = %s ORDER BY id ASC",
                        (tenant_phone,)
                    )
                else:
                    cursor.execute(
                        "SELECT id, tenant_phone, fecha, nombre_cliente, telefono, servicio, monto, estado FROM ventas ORDER BY id ASC"
                    )
                
                filas = cursor.fetchall()
                if not filas:
                    print(f"ℹ️ No hay registros de ventas en Supabase para: {suffix}")
                    return

                columnas = [desc[0] for desc in cursor.description]

                with open(archivo_salida, mode="w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(columnas)
                    writer.writerows(filas)

                print(f"✅ Reporte generado exitosamente: {archivo_salida}")
                print(f"📊 Total de ventas exportadas: {len(filas)}")

    except Exception as e:
        print(f"❌ Error al exportar desde Supabase: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportar ventas desde Supabase PostgreSQL a CSV")
    parser.add_argument("--tenant", help="Número telefónico del tenant para filtrar ventas (opcional)")
    args = parser.parse_args()

    exportar_a_csv(args.tenant)