import os
import socket
import subprocess

def encontrar_puerto_libre(start=5000, end=5100):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError("No hay puertos libres disponibles")

puerto = encontrar_puerto_libre()
print(f"✅ Puerto libre detectado: {puerto}")

comando = f"source ~/chatbot/venv/bin/activate && python ~/chatbot/MiniLM.py --port={puerto}"
proceso = subprocess.Popen(["bash", "-c", comando])

# Guardar PID para control institucional
with open("logs/minilm.pid", "w") as f:
    f.write(str(proceso.pid))

print(f"🚀 Backend RAG lanzado en puerto {puerto}")
print(f"🌐 Endpoint disponible en: http://localhost:{puerto}/bot")
