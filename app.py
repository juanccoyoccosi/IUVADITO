from flask import Flask, request, jsonify
import datetime

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "service": "chatbot-facturacion",
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/bot', methods=['POST'])
def bot():
    try:
        data = request.get_json() or {}
        user_message = data.get('message', '')
        remote_jid = data.get('key', {}).get('remoteJid', 'unknown')
        
        response = {
            "response": f"Chatbot: Recibí '{user_message}'",
            "status": "success",
            "remoteJid": remote_jid,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Flask iniciado en puerto 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
