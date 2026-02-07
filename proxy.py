from flask import Flask, request, jsonify
import requests
import datetime

app = Flask(__name__)

N8N_WEBHOOK_URL = "https://chatbot-n8n.mhcwfe.easypanel.host/webhook/iuvadito"  # o usa la IP pública si no estás en red Docker

@app.route('/proxy-webhook', methods=['POST'])
def proxy_webhook():
    payload = request.get_json()
    timestamp = datetime.datetime.utcnow().isoformat()

    # Respuesta inmediata a Evolution API
    response = {
        "status": "ok",
        "message": "Webhook recibido",
        "timestamp": timestamp
    }

    # Reenvío a n8n
    try:
        requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"[{timestamp}] Error reenviando a n8n: {e}")

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)