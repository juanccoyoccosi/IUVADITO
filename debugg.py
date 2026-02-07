import os
import google.generativeai as genai
from google.genai.errors import APIError

def obtener_respuesta_gemini(mensaje_usuario: str):
    """
    Se conecta a la API de Gemini y obtiene una respuesta para el mensaje dado.

    Args:
        mensaje_usuario: El texto que el usuario quiere enviar a Gemini.
    """
    # --- 1. CONFIGURACIÓN DEL API KEY ---
    # La API Key se lee de la variable de entorno 'GEMINI_API_KEY'.
    # Si no está configurada, el código lanzará un error.
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ ERROR: La variable de entorno 'GEMINI_API_KEY' no está configurada.")
            print("Por favor, establece tu clave API antes de ejecutar el código.")
            return

        # Inicializar el cliente de Gemini
        client = genai.Client(api_key=api_key)

    except Exception as e:
        print(f"❌ Error al inicializar el cliente: {e}")
        return

    # --- 2. SOLICITUD A LA API ---
    try:
        # Usa el modelo `gemini-2.5-flash` que es rápido y capaz
        print(f"-> Enviando mensaje a Gemini: '{mensaje_usuario}'")
        
        # Llama a la API para generar contenido
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=mensaje_usuario
        )

        # --- 3. PROCESAMIENTO DE LA RESPUESTA ---
        # Imprime la respuesta de texto
        print("\n==============================")
        print("🤖 Respuesta de Gemini:")
        print(response.text)
        print("==============================")

    except APIError as e:
        print(f"❌ Error de la API de Gemini (código de estado o clave inválida): {e}")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")

# =================================================================
#                         PUNTO DE EJECUCIÓN
# =================================================================

# 📢 Reemplaza este mensaje con lo que quieras preguntar a Gemini
mensaje =input ("Ingrese el mensaje: ")

# Ejecutar la función
obtener_respuesta_gemini(mensaje)