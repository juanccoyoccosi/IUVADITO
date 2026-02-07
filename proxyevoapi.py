from flask import Flask, request, jsonify
import requests
from datetime import datetime, timezone
import json

app = Flask(__name__)

BASE_EVOLUTION_API_URL = "https://chatbot-evolution-api.mhcwfe.easypanel.host/message/sendText"

@app.route('/proxy-evolution/<instance_name>', methods=['POST'])
def proxy_evolution(instance_name):
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = request.get_json(silent=True)
    apikey = request.headers.get("apikey")

    print(f"[{timestamp}] 🔍 Payload recibido: {json.dumps(payload, indent=2)}")
    print(f"[{timestamp}] 🔐 API Key recibida: {apikey}")

    number = payload.get("number")
    text = payload.get("text")

    if not all([number, text, apikey]):
        print(f"[{timestamp}] ⚠️ Payload incompleto → number: {number}, text: {text}, apikey: {apikey}")
        return jsonify({
            "status": "error",
            "message": "Faltan campos obligatorios",
            "timestamp": timestamp
        }), 400

    if not isinstance(text, str) or not text.strip():
        print(f"[{timestamp}] ⚠️ El campo 'text' está vacío o mal formado")
        return jsonify({
            "status": "error",
            "message": "El campo 'text' debe ser una cadena no vacía",
            "timestamp": timestamp
        }), 400

    full_url = f"{BASE_EVOLUTION_API_URL}/{instance_name}"

    try:
        print(f"[{timestamp}] 📦 Payload final enviado a Evolution API:")
        print(json.dumps({
            "number": number,
            "text": text
        }, indent=2))

        print(f"[{timestamp}] 🚀 Enviando a Evolution API → {full_url}")
        r = requests.post(
            full_url,
            json={
                "number": number,
                "text": text
            },
            headers={
                "apikey": apikey
            },
            timeout=5
        )
        print(f"[{timestamp}] ✅ Respuesta Evolution API: {r.status_code} {r.text}")
        return jsonify({
            "status": "ok",
            "message": "Mensaje enviado a Evolution API",
            "response": r.text,
            "timestamp": timestamp
        }), r.status_code
    except Exception as e:
        print(f"[{timestamp}] ❌ Error enviando a Evolution API: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": timestamp
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)