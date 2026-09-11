import sqlite3
import csv
from datetime import datetime

DB_PATH = "ventas.db"

def exportar_a_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_salida = f"reporte_ventas_{timestamp}.csv"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, fecha, nombre, telefono, servicio, precio, estado FROM ventas")
        filas = cursor.fetchall()
        
        if not filas:
            print("ℹ️ No hay registros de ventas en la base de datos.")
            return

        columnas = [desc[0] for desc in cursor.description]

        with open(archivo_salida, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(columnas)
            writer.writerows(filas)

        conn.close()
        print(f"✅ Reporte generado exitosamente: {archivo_salida}")
        print(f"📊 Total de ventas exportadas: {len(filas)}")

    except sqlite3.OperationalError as e:
        print(f"❌ Error al consultar la base de datos: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    exportar_a_csv()