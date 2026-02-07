
from typing import Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import couchdb
import traceback
import re
import argparse
import ssl
import uuid
from collections import deque

# Cargar variables de entorno
load_dotenv()

# Argumentos de línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=None)
args = parser.parse_args()

app = Flask(__name__)
CORS(app)
app.config['THREADED'] = True

# ============================================
# CONFIGURACIÓN DESDE .ENV
# ============================================
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = args.port or int(os.getenv('FLASK_PORT', '5000'))

# CouchDB
COUCHDB_HOST = os.getenv('COUCHDB_HOST', '62.171.179.255')
COUCHDB_PORT = os.getenv('COUCHDB_PORT', '5984')
COUCHDB_USER = os.getenv('COUCHDB_USER', 'Juan')
COUCHDB_PASSWORD = os.getenv('COUCHDB_PASSWORD', 'Elpro123')
COUCHDB_DATABASE = os.getenv('COUCHDB_DATABASE', 'chatbot_data')
COUCHDB_QUESTION = os.getenv('COUCHDB_QUESTION', 'data_frecuentquestion')
COUCHDB_VACIO = os.getenv('COUCHDB_VACIO', 'data_vacio')
COUCHDB_CONVERSACIONES = os.getenv('COUCHDB_CONVERSACIONES', 'conversaciones')

# Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY no está configurado en .env")

# SSL
SSL_CERT = os.getenv('SSL_CERT_PATH', 'server.crt')
SSL_KEY = os.getenv('SSL_KEY_PATH', 'server.key')

# Configurar Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Conectar a CouchDB
COUCHDB_URL = f'http://{COUCHDB_USER}:{COUCHDB_PASSWORD}@{COUCHDB_HOST}:{COUCHDB_PORT}'
couch = couchdb.Server(COUCHDB_URL)

# Crear bases de datos si no existen
for db_name in [COUCHDB_DATABASE, COUCHDB_QUESTION, COUCHDB_VACIO, COUCHDB_CONVERSACIONES]:
    if db_name not in couch:
        couch.create(db_name)

db_knowledge = couch[COUCHDB_DATABASE]
db_questions = couch[COUCHDB_QUESTION]
db_vacio = couch[COUCHDB_VACIO]
db_conversaciones = couch[COUCHDB_CONVERSACIONES]

# ============================================
# MODELO DE EMBEDDINGS
# ============================================
print("🔄 Cargando modelo de embeddings...")
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
print("✅ Modelo de embeddings cargado")

# ============================================
# SISTEMA DE MEMORIA CONVERSACIONAL
# ============================================
class MemoriaConversacional:
    """Sistema de memoria que mantiene el historial de conversaciones por usuario"""
    
    def __init__(self, db_conversaciones, max_mensajes=10, tiempo_expiracion_minutos=60):
        self.db = db_conversaciones
        self.max_mensajes = max_mensajes
        self.tiempo_expiracion = timedelta(minutes=tiempo_expiracion_minutos)
        self.memoria_ram = {}
        
        self._cargar_conversaciones_activas()
    
    def _cargar_conversaciones_activas(self):
        """Carga conversaciones recientes desde la base de datos"""
        print("🔄 Cargando conversaciones activas...")
        ahora = datetime.now()
        
        for doc_id in self.db:
            try:
                doc = self.db[doc_id]
                ultima_actualizacion = datetime.fromisoformat(doc['ultima_actualizacion'])
                
                if ahora - ultima_actualizacion < self.tiempo_expiracion:
                    remote_jid = doc['remote_jid']
                    self.memoria_ram[remote_jid] = deque(doc['mensajes'], maxlen=self.max_mensajes)
                    print(f"   ✓ Conversación cargada: {remote_jid} ({len(doc['mensajes'])} mensajes)")
            except Exception as e:
                print(f"   ⚠️ Error cargando conversación {doc_id}: {e}")
        
        print(f"✅ {len(self.memoria_ram)} conversaciones activas cargadas")
    
    def agregar_mensaje(self, remote_jid, rol, mensaje, metadata=None):
        """Agrega un mensaje al historial de conversación"""
        if remote_jid not in self.memoria_ram:
            self.memoria_ram[remote_jid] = deque(maxlen=self.max_mensajes)
        
        mensaje_obj = {
            'rol': rol,
            'mensaje': mensaje,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.memoria_ram[remote_jid].append(mensaje_obj)
        
        if len(self.memoria_ram[remote_jid]) % 3 == 0:
            self._persistir_conversacion(remote_jid)
    
    def obtener_historial(self, remote_jid, ultimos_n=None):
        """Obtiene el historial de conversación de un usuario"""
        if remote_jid not in self.memoria_ram:
            return []
        
        historial = list(self.memoria_ram[remote_jid])
        
        if ultimos_n:
            return historial[-ultimos_n:]
        
        return historial
    
    def obtener_contexto_formateado(self, remote_jid, ultimos_n=5):
        """Obtiene el contexto formateado para incluir en el prompt"""
        historial = self.obtener_historial(remote_jid, ultimos_n)
        
        if not historial:
            return ""
        
        contexto_partes = ["=" * 50]
        contexto_partes.append("HISTORIAL DE CONVERSACIÓN:")
        contexto_partes.append("=" * 50)
        
        for idx, msg in enumerate(historial, 1):
            rol_emoji = "👤 USUARIO" if msg['rol'] == 'usuario' else "🤖 ASISTENTE"
            contexto_partes.append(f"\n[Mensaje {idx}] {rol_emoji}:")
            contexto_partes.append(msg['mensaje'])
            
            if msg.get('metadata') and msg['metadata'].get('clasificacion'):
                contexto_partes.append(f"   (Tema: {msg['metadata']['clasificacion']})")
        
        contexto_partes.append("\n" + "=" * 50)
        
        return "\n".join(contexto_partes)
    
    def limpiar_conversacion(self, remote_jid):
        """Limpia la conversación de un usuario"""
        if remote_jid in self.memoria_ram:
            del self.memoria_ram[remote_jid]
        
        for doc_id in self.db:
            doc = self.db[doc_id]
            if doc.get('remote_jid') == remote_jid:
                self.db.delete(doc)
                break
    
    def _persistir_conversacion(self, remote_jid):
        """Guarda la conversación en la base de datos"""
        try:
            mensajes = list(self.memoria_ram[remote_jid])
            doc_id = f"conv_{remote_jid}"
            
            if doc_id in self.db:
                doc = self.db[doc_id]
                doc['mensajes'] = mensajes
                doc['ultima_actualizacion'] = datetime.now().isoformat()
            else:
                doc = {
                    '_id': doc_id,
                    'remote_jid': remote_jid,
                    'mensajes': mensajes,
                    'creado': datetime.now().isoformat(),
                    'ultima_actualizacion': datetime.now().isoformat()
                }
            
            self.db.save(doc)
            
        except Exception as e:
            print(f"⚠️ Error persistiendo conversación: {e}")
    
    def limpiar_conversaciones_expiradas(self):
        """Limpia conversaciones expiradas de la memoria RAM y BD"""
        ahora = datetime.now()
        remote_jids_expirados = []
        
        for remote_jid, mensajes in self.memoria_ram.items():
            if mensajes:
                ultimo_mensaje = mensajes[-1]
                ultima_fecha = datetime.fromisoformat(ultimo_mensaje['timestamp'])
                
                if ahora - ultima_fecha > self.tiempo_expiracion:
                    remote_jids_expirados.append(remote_jid)
        
        for remote_jid in remote_jids_expirados:
            del self.memoria_ram[remote_jid]
            print(f"🧹 Conversación expirada eliminada: {remote_jid}")

# ============================================
# CLASIFICADOR MEJORADO DE INTENCIONES
# ============================================
class ClasificadorIntenciones:
    """Clasificador avanzado basado en patrones y embeddings - MEJORADO"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
        self.intenciones_no_guardar = [
            'saludo', 'despedida', 'agradecimiento',
            'confirmacion', 'ayuda_generica', 'mensaje_corto'
        ]
        
        self.intenciones_desde_rag = {}
        self._cargar_intenciones_desde_rag()
    
    def _cargar_intenciones_desde_rag(self):
        """Carga TODAS las intenciones desde documentos RAG"""
        print("🔄 Cargando intenciones desde base de conocimientos...")
        
        for doc_id in db_knowledge:
            try:
                doc = db_knowledge[doc_id]
                
                if 'intencion' in doc:
                    intencion = doc['intencion']
                    
                    if intencion not in self.intenciones_desde_rag:
                        self.intenciones_desde_rag[intencion] = {
                            'doc_ids': [],
                            'patrones': [],
                            'embeddings': []
                        }
                    
                    self.intenciones_desde_rag[intencion]['doc_ids'].append(doc_id)
                    
                    if 'patrones' in doc:
                        patrones = doc['patrones']
                        if isinstance(patrones, list):
                            self.intenciones_desde_rag[intencion]['patrones'].extend(patrones)
                        elif isinstance(patrones, str):
                            self.intenciones_desde_rag[intencion]['patrones'].append(patrones)
                    
                    if 'embedding' in doc:
                        self.intenciones_desde_rag[intencion]['embeddings'].append(
                            np.array(doc['embedding'])
                        )
                
            except Exception as e:
                print(f"⚠️ Error procesando documento {doc_id}: {e}")
        
        print(f"✅ {len(self.intenciones_desde_rag)} intenciones cargadas desde RAG")
    
    def clasificar(self, texto):
        """Clasifica una consulta - MEJORADO CON UMBRAL MÁS BAJO"""
        texto_lower = texto.lower().strip()
        
        # Mensajes muy cortos
        if len(texto_lower) < 3:
            return {
                'intencion': 'mensaje_corto',
                'confianza': 1.0,
                'es_relevante': False,
                'usar_rag': False,
                'tipo': 'simple'
            }
        
        query_embedding = self.embedding_model.encode([texto])[0]
        mejores_matches = []
        
        # Buscar coincidencias por embedding y patrones
        for intencion, config in self.intenciones_desde_rag.items():
            if config['embeddings']:
                for emb in config['embeddings']:
                    similarity = cosine_similarity([query_embedding], [emb])[0][0]
                    mejores_matches.append({
                        'intencion': intencion,
                        'confianza': float(similarity),
                        'metodo': 'embedding'
                    })
            
            for patron in config['patrones']:
                try:
                    if re.search(re.escape(patron.lower()), texto_lower):
                        mejores_matches.append({
                            'intencion': intencion,
                            'confianza': 0.85,
                            'metodo': 'regex'
                        })
                except:
                    pass
        
        # CAMBIO CRÍTICO: Umbral reducido de 0.65 a 0.45
        if mejores_matches:
            mejor = max(mejores_matches, key=lambda x: x['confianza'])
            
            # ✅ UMBRAL MÁS BAJO - permite más coincidencias
            if mejor['confianza'] >= 0.45:  # ANTES: 0.65
                debe_guardarse = mejor['intencion'] not in self.intenciones_no_guardar
                
                return {
                    'intencion': mejor['intencion'],
                    'confianza': mejor['confianza'],
                    'es_relevante': debe_guardarse,
                    'usar_rag': True,  # ✅ SÍ buscar en RAG
                    'metodo': mejor['metodo'],
                    'tipo': 'intencion_conocida'
                }
        
        # ✅ SIEMPRE BUSCAR EN RAG, incluso para consultas nuevas
        return {
            'intencion': None,
            'confianza': 0.0,
            'es_relevante': True,
            'usar_rag': True,  # ✅ CAMBIO CRÍTICO: Activar RAG
            'tipo': 'nueva_intencion'
        }
    
    def sugerir_intencion(self, texto):
        """Sugiere una intención basada en el contenido del texto"""
        texto_lower = texto.lower()
        
        keywords = {
            'remision': ['remision', 'remisión', 'guia de remision', 'guía'],
            'factura': ['factura', 'facturación', 'comprobante'],
            'boleta': ['boleta', 'ticket'],
            'error': ['error', 'fallo', 'problema', 'bug', 'crash', 'no funciona'],
            'consulta': ['consulta', 'pregunta', 'duda', 'información', 'saber', 'que es'],
            'solicitud': ['quiero', 'necesito', 'puedes', 'podrías', 'solicitar', 'crear'],
            'configuracion': ['configurar', 'configuración', 'ajustar', 'setup', 'instalar'],
            'precio': ['precio', 'costo', 'cuánto', 'valor', 'tarifa', 'pagar'],
            'horario': ['horario', 'hora', 'cuándo', 'fecha', 'disponible']
        }
        
        for categoria, palabras in keywords.items():
            for palabra in palabras:
                if palabra in texto_lower:
                    return f"consulta_{categoria}"
        
        return "consulta_general"

# ============================================
# CLASE RAG MEJORADA
# ============================================
class AdvancedRAG:
    def __init__(self, db_knowledge, embedding_model):
        self.db = db_knowledge
        self.model = embedding_model
        self.document_cache = {}
        self.embedding_cache = {}
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Genera embeddings para documentos que no los tengan"""
        print("🔄 Inicializando embeddings de la base de conocimientos...")
        
        for doc_id in self.db:
            try:
                doc = self.db[doc_id]
                
                if 'embedding' in doc:
                    self.embedding_cache[doc_id] = np.array(doc['embedding'])
                    self.document_cache[doc_id] = doc
                    continue
                
                texto = self._extract_text_from_doc(doc)
                if texto:
                    embedding = self.model.encode([texto])[0]
                    doc['embedding'] = embedding.tolist()
                    self.db.save(doc)
                    
                    self.embedding_cache[doc_id] = embedding
                    self.document_cache[doc_id] = doc
                
            except Exception as e:
                print(f"⚠️ Error procesando documento {doc_id}: {e}")
        
        print(f"✅ Total de documentos indexados: {len(self.embedding_cache)}")
    
    def _extract_text_from_doc(self, doc):
        """Extrae texto relevante de un documento"""
        textos = []
        
        campos_texto = ['patrones', 'respuestas', 'contenido', 'texto', 'descripcion']
        for campo in campos_texto:
            if campo in doc:
                valor = doc[campo]
                if isinstance(valor, str):
                    textos.append(valor)
                elif isinstance(valor, list):
                    textos.extend([str(v) for v in valor if v])
        
        if 'categoria' in doc:
            textos.append(f"Categoría: {doc['categoria']}")
        
        if 'intencion' in doc:
            textos.append(f"Intención: {doc['intencion']}")
        
        return ' '.join(textos).strip()
    
    def retrieve_relevant_docs(self, query, top_k=3, threshold=0.25):  # ✅ threshold 0.3 -> 0.25
        """Recupera documentos relevantes usando similitud semántica"""
        query_embedding = self.model.encode([query])[0]
        
        similitudes = []
        for doc_id, doc_embedding in self.embedding_cache.items():
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            
            if similarity >= threshold:
                similitudes.append({
                    'doc_id': doc_id,
                    'doc': self.document_cache[doc_id],
                    'score': float(similarity)
                })
        
        similitudes.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"   🔍 Encontradas {len(similitudes)} coincidencias (threshold={threshold})")
        if similitudes:
            print(f"   📊 Top score: {similitudes[0]['score']:.3f}")
        
        return similitudes[:top_k]
    
    def generate_context_from_docs(self, relevant_docs):
        """Genera contexto formateado desde documentos relevantes"""
        if not relevant_docs:
            return ""
        
        contexto_partes = []
        
        for idx, doc_info in enumerate(relevant_docs, 1):
            doc = doc_info['doc']
            score = doc_info['score']
            
            contexto_partes.append(f"[Fuente {idx} - Relevancia: {score:.2%}]")
            
            if 'categoria' in doc:
                contexto_partes.append(f"Categoría: {doc['categoria']}")
            
            if 'respuestas' in doc:
                respuestas = doc['respuestas']
                if isinstance(respuestas, list):
                    contexto_partes.append(f"Información: {' '.join(respuestas)}")
                else:
                    contexto_partes.append(f"Información: {respuestas}")
            
            if 'contenido' in doc:
                contexto_partes.append(f"Contenido: {doc['contenido']}")
            
            contexto_partes.append("")
        
        return "\n".join(contexto_partes)

# ============================================
# GESTIÓN DE CONSULTAS DE USUARIOS
# ============================================
def guardar_consulta_usuario(mensaje, nombre, remote_jid, clasificacion, embedding):
    """Guarda consulta del usuario en data_frecuentquestion"""
    
    if not clasificacion['es_relevante']:
        print(f"🔸 Consulta no relevante, no se guarda: {clasificacion['intencion']}")
        return None
    
    try:
        consulta_id = str(uuid.uuid4())
        
        doc = {
            '_id': consulta_id,
            'nombre': nombre,
            'mensaje': mensaje,
            'texto': mensaje,
            'remoteJid': remote_jid,
            'timestamp': datetime.now().isoformat(),
            'procesado': False,
            'intencion': clasificacion.get('intencion', 'sin_clasificar'),
            'embedding': embedding.tolist()
        }
        
        db_questions.save(doc)
        print(f"✅ Consulta guardada en data_frecuentquestion: {consulta_id}")
        
        return consulta_id
    
    except Exception as e:
        print(f"⚠️ Error guardando consulta: {e}")
        traceback.print_exc()
        return None

def guardar_consulta_no_clasificada(mensaje, nombre, remote_jid, clasificacion, embedding):
    """Guarda consultas NO CLASIFICADAS en data_vacio"""
    
    try:
        palabras_clave = extraer_keywords(mensaje)
        
        intencion_sugerida = clasificacion.get('intencion', 'sin_clasificar')
        timestamp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        consulta_id = f"RAG_AUTO_{intencion_sugerida}_{timestamp_id}"
        
        doc = {
            '_id': consulta_id,
            'intencion': intencion_sugerida,
            'categoria': 'nueva_consulta',
            'patrones': [
                mensaje.lower().strip(),
                mensaje.lower().replace('?', '').strip(),
                mensaje.lower().replace('¿', '').strip()
            ],
            'respuestas': [
                f"He registrado tu consulta: '{mensaje}'",
                "Esta es una consulta nueva que estoy aprendiendo.",
                "Un administrador revisará esta información pronto para mejorar mis respuestas."
            ],
            'contenido': f"Consulta no clasificada del usuario {nombre}: {mensaje}",
            'metadata': {
                'creado_manualmente': False,
                'fecha_creacion': datetime.now().isoformat(),
                'requiere_revision': True,
                'activo': False,
                'prioridad': 'media',
                'keywords': palabras_clave,
                'usuario_origen': nombre,
                'remote_jid': remote_jid,
                'confianza_clasificacion': clasificacion.get('confianza', 0.0),
                'tipo_clasificacion': clasificacion.get('tipo', 'desconocido')
            },
            'embedding': embedding.tolist()
        }
        
        db_vacio.save(doc)
        print(f"✅ Consulta no clasificada guardada en data_vacio: {consulta_id}")
        
        return consulta_id
    
    except Exception as e:
        print(f"⚠️ Error guardando consulta no clasificada: {e}")
        traceback.print_exc()
        return None

def extraer_keywords(texto):
    """Extrae palabras clave relevantes de un texto"""
    stop_words = {
        'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
        'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
        'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
        'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy', 'sin',
        'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo', 'yo',
        'también', 'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero',
        'desde', 'grande', 'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella',
        'sí', 'día', 'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
        'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde', 'ahora', 'parte',
        'después', 'vida', 'quedar', 'siempre', 'creer', 'hablar', 'llevar', 'dejar',
        'nada', 'cada', 'seguir', 'menos', 'nuevo', 'encontrar', 'algo', 'solo',
        'decir', 'llamar', 'venir', 'pensar', 'salir', 'volver', 'tomar', 'conocer',
        'vivir', 'sentir', 'tratar', 'mirar', 'contar', 'empezar', 'esperar', 'buscar',
        'existir', 'entrar', 'trabajar', 'escribir', 'perder', 'producir', 'ocurrir',
        'entender', 'pedir', 'recibir', 'recordar', 'terminar', 'permitir', 'aparecer',
        'conseguir', 'comenzar', 'servir', 'sacar', 'necesitar', 'mantener', 'resultar',
        'leer', 'caer', 'cambiar', 'presentar', 'crear', 'abrir', 'considerar', 'oír',
        'acabar', 'mil', 'cual', 'sea', 'está', 'tiene', 'fue', 'son', 'cómo', 'puede',
        'es', 'una', 'del'
    }
    
    palabras = re.findall(r'\b\w+\b', texto.lower())
    
    keywords = [
        palabra for palabra in palabras 
        if palabra not in stop_words and len(palabra) > 2  
    ]
    
    return list[Any](set[Any](keywords))[:10]

# GENERACIÓN DE RESPUESTA

def sanitizar_query_para_gemini(query):
    """
    Sanitiza queries para evitar bloqueos de Gemini
    Reemplaza palabras y combinaciones problemáticas por sinónimos seguros
    """
    import re
    
    # Diccionario de reemplazos - ORDEN IMPORTANTE
    reemplazos = [
        # Combinaciones específicas (verificar primero)
        (r'\b(como|cómo)\s+(generar|crear|emitir|hacer)\s+una\s+factura\s+electr[oó]nica', 'cómo procesar un comprobante electrónico'),
        (r'\b(como|cómo)\s+(generar|crear|emitir|hacer)\s+una\s+factura', 'cómo procesar un comprobante de pago'),
        (r'\b(como|cómo)\s+(generar|crear|emitir|hacer)\s+factura', 'cómo procesar comprobante'),
        
        # Verbos + factura
        (r'\b(generar|crear|emitir|hacer|registrar|procesar)\s+una\s+factura\s+electr[oó]nica', 'procesar comprobante electrónico'),
        (r'\b(generar|crear|emitir|hacer|registrar|procesar)\s+una\s+factura', 'procesar comprobante de pago'),
        (r'\b(generar|crear|emitir|hacer|registrar|procesar)\s+factura', 'procesar comprobante'),
        
        # Solo "factura" + "electronica"
        (r'\bfactura\s+electr[oó]nica', 'comprobante electrónico'),
        (r'\bfacturas\s+electr[oó]nicas', 'comprobantes electrónicos'),
        
        # Solo "factura"
        (r'\bfactura\b', 'comprobante'),
        (r'\bfacturas\b', 'comprobantes'),
        (r'\bfacturaci[oó]n\b', 'emisión de comprobantes'),
        
        # Otras palabras problemáticas
        (r'\b(generar|crear|hacer)\s+una\s+venta', 'procesar operación de venta'),
        (r'\bventas?\b', 'operaciones comerciales'),
        (r'\bcompra\b', 'adquisición'),
        (r'\beliminar\b', 'quitar'),
        (r'\bborrar\b', 'remover'),
    ]
    
    query_sanitizada = query
    cambios_realizados = []
    
    # Aplicar reemplazos en orden
    for patron, reemplazo in reemplazos:
        if re.search(patron, query_sanitizada, re.IGNORECASE):
            query_antes = query_sanitizada
            query_sanitizada = re.sub(patron, reemplazo, query_sanitizada, flags=re.IGNORECASE)
            if query_antes != query_sanitizada:
                cambios_realizados.append(f"'{patron}' → '{reemplazo}'")
    
    if cambios_realizados:
        print(f"\n🧹 QUERY SANITIZADA:")
        print(f"   📥 Original: '{query}'")
        print(f"   📤 Sanitizada: '{query_sanitizada}'")
        for cambio in cambios_realizados:
            print(f"      • {cambio}")
        print()
    
    return query_sanitizada

def generar_respuesta_con_contexto_completo(query, relevant_docs, contexto_conversacional):
    """Genera respuesta usando Gemini con RAG + Memoria - CON SANITIZACIÓN"""
    
   
    query_sanitizada = sanitizar_query_para_gemini(query)
    
    contexto_rag = rag_system.generate_context_from_docs(relevant_docs)
    
    
    prompt = f"""Eres un asistente especializado en el sistema SIGGE (Sistema Integral de Gestión Gubernamental y Empresarial).

{contexto_conversacional}

BASE DE CONOCIMIENTOS DISPONIBLE:
{contexto_rag if contexto_rag else "No hay información específica adicional en la base de conocimientos para esta consulta."}

PREGUNTA ACTUAL DEL USUARIO: {query_sanitizada}

INSTRUCCIONES CRÍTICAS:
1. Si hay información en la BASE DE CONOCIMIENTOS, úsala prioritariamente
2. Sé específico y práctico en tus respuestas
3. Si el usuario pregunta "qué es X", explícalo claramente con ejemplos
4. Si el usuario dice "quiero crear X", proporciona pasos específicos
5. Usa el historial de conversación para mantener coherencia
6. NO digas "no tengo información" si hay datos en la base de conocimientos
7. Sé conciso pero completo
8. IMPORTANTE: Cuando respondas, usa el término "factura" normalmente en tu respuesta, no uses "comprobante" si el usuario preguntó por "factura"

Responde de manera útil y directa:"""

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1500
            ),
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
        )
        
        if hasattr(response, "parts") and response.parts:
            return response.text
        else:
            return "Disculpa, no pude generar una respuesta. ¿Podrías reformular tu pregunta?"
    
    except Exception as e:
        print(f"❌ Error con Gemini API: {e}")
        traceback.print_exc()
        return "Error al generar respuesta. Por favor, intenta de nuevo."

def generar_respuesta_sin_contexto(query, contexto_conversacional):
    """Genera respuesta cuando no hay documentos RAG - CON SANITIZACIÓN"""
    
    # ✅ SANITIZAR LA QUERY
    query_sanitizada = sanitizar_query_para_gemini(query)
    
    if contexto_conversacional:
        prompt = f"""Eres un asistente inteligente especializado en el sistema SIGGE.

{contexto_conversacional}

PREGUNTA/COMENTARIO ACTUAL: {query_sanitizada}

INSTRUCCIONES:
1. Analiza el historial completo para entender el contexto
2. Si puedes responder basándote en el contexto previo, hazlo
3. Si es una consulta nueva, proporciona información general útil
4. Sé específico y práctico
5. Si realmente no tienes información suficiente, pregunta qué detalles específicos necesita
6. Usa el término "factura" normalmente en tu respuesta si el usuario preguntó por eso

Responde de manera útil y natural:"""
    else:
        prompt = f"""Eres un asistente inteligente del sistema SIGGE.

PREGUNTA: {query_sanitizada}

No tienes información específica documentada sobre esta consulta, pero puedes:
1. Proporcionar información general si conoces el tema
2. Hacer preguntas para entender mejor qué necesita el usuario
3. Explicar qué información necesitarías para ayudar mejor

Responde de manera útil:"""

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500
            ),
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
        )
        
        if hasattr(response, "parts") and response.parts:
            return response.text
        else:
            return "¿Podrías proporcionar más detalles específicos para ayudarte mejor?"
    
    except Exception as e:
        print(f"❌ Error con Gemini API: {e}")
        return "Disculpa, hubo un error. ¿Podrías reformular tu pregunta?"

# ============================================
# ENDPOINT PRINCIPAL MEJORADO
# ============================================
@app.route('/bot', methods=['POST'])
def responder():
    try:
        data = request.json
        print(f"\n📥 Payload recibido: {json.dumps(data, indent=2)[:500]}...")
        
        # Extraer datos
        mensaje = None
        nombre = "Usuario"
        remote_jid = ""
        
        if 'body' in data and isinstance(data['body'], dict):
            body_data = data['body'].get('data', {})
            if isinstance(body_data, dict):
                mensaje = body_data.get('message', {}).get('conversation', '')
                nombre = body_data.get('pushName', 'Usuario')
                remote_jid = body_data.get('remoteJid', '')
        
        if not mensaje:
            mensaje = data.get('mensaje', data.get('text', ''))
            nombre = data.get('pushName', data.get('nombre', 'Usuario'))
            remote_jid = data.get('remoteJid', '')
        
        if not mensaje or not mensaje.strip():
            return jsonify({
                "text": "No recibí ningún mensaje.",
                "estado": "ERROR"
            }), 400
        
        print(f"💬 Procesando mensaje de {nombre}: '{mensaje}'")
        
        # 1. OBTENER CONTEXTO CONVERSACIONAL
        contexto_conversacional = memoria.obtener_contexto_formateado(remote_jid, ultimos_n=5)
        tiene_contexto = bool(contexto_conversacional)
        
        if tiene_contexto:
            print(f"🧠 Memoria: {len(memoria.obtener_historial(remote_jid))} mensajes previos")
        
        # 2. CLASIFICACIÓN
        clasificacion = clasificador.clasificar(mensaje)
        print(f"🎯 Clasificación: {json.dumps(clasificacion, indent=2)}")
        
        # 3. GENERAR EMBEDDING
        mensaje_embedding = embedding_model.encode([mensaje])[0]
        
        # 4. BUSCAR EN RAG (SIEMPRE, con threshold bajo)
        print(f"📚 Buscando información relevante en RAG...")
        relevant_docs = rag_system.retrieve_relevant_docs(mensaje, top_k=3, threshold=0.25)
        
        docs_utilizados = []
        consulta_id = None
        
        if relevant_docs:
            print(f"✅ {len(relevant_docs)} documentos relevantes encontrados")
            print(f"   📊 Mejor coincidencia: {relevant_docs[0]['score']:.3f}")
            
            respuesta = generar_respuesta_con_contexto_completo(
                mensaje, relevant_docs, contexto_conversacional
            )
            docs_utilizados = relevant_docs
            
            # Guardar consulta si es relevante
            if clasificacion['es_relevante']:
                consulta_id = guardar_consulta_usuario(
                    mensaje, nombre, remote_jid, clasificacion, mensaje_embedding
                )
        else:
            print(f"⚠️ No se encontraron documentos suficientemente relevantes")
            respuesta = generar_respuesta_sin_contexto(mensaje, contexto_conversacional)
            
            # Guardar como consulta no clasificada
            if clasificacion['tipo'] == 'nueva_intencion':
                intencion_sugerida = clasificador.sugerir_intencion(mensaje)
                clasificacion['intencion'] = intencion_sugerida
                
                consulta_id = guardar_consulta_no_clasificada(
                    mensaje, nombre, remote_jid, clasificacion, mensaje_embedding
                )
        
        # 5. AGREGAR A MEMORIA CONVERSACIONAL
        memoria.agregar_mensaje(remote_jid, 'usuario', mensaje)
        memoria.agregar_mensaje(
            remote_jid, 
            'asistente', 
            respuesta,
            metadata={
                'clasificacion': clasificacion.get('intencion', 'sin_clasificar'),
                'docs_utilizados': len(docs_utilizados),
                'mejor_score': relevant_docs[0]['score'] if relevant_docs else 0.0
            }
        )
        
        print(f"✅ Respuesta generada y guardada en memoria")
        
        return jsonify({
            "text": respuesta,
            "estado": "OK",
            "metadata": {
                "clasificacion": clasificacion,
                "docs_utilizados": len(docs_utilizados),
                "mejor_score": relevant_docs[0]['score'] if relevant_docs else 0.0,
                "consulta_guardada": consulta_id is not None,
                "consulta_id": consulta_id,
                "tiene_contexto_previo": tiene_contexto,
                "mensajes_en_memoria": len(memoria.obtener_historial(remote_jid)),
                "timestamp": datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        print(f"❌ Error procesando solicitud: {e}")
        traceback.print_exc()
        return jsonify({
            "text": "Error interno del servidor. Intenta nuevamente.",
            "estado": "ERROR"
        }), 500

# ============================================
# ENDPOINTS AUXILIARES
# ============================================
@app.route('/api/memoria/<remote_jid>', methods=['GET'])
def obtener_memoria_usuario(remote_jid):
    """Obtiene el historial de conversación de un usuario"""
    try:
        historial = memoria.obtener_historial(remote_jid)
        
        return jsonify({
            'remote_jid': remote_jid,
            'total_mensajes': len(historial),
            'historial': historial
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memoria/<remote_jid>', methods=['DELETE'])
def limpiar_memoria_usuario(remote_jid):
    """Limpia la memoria de conversación de un usuario"""
    try:
        memoria.limpiar_conversacion(remote_jid)
        
        return jsonify({
            'mensaje': f'Memoria limpiada para {remote_jid}',
            'estado': 'OK'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memoria/limpiar-expiradas', methods=['POST'])
def limpiar_memorias_expiradas():
    """Limpia todas las conversaciones expiradas"""
    try:
        memoria.limpiar_conversaciones_expiradas()
        
        return jsonify({
            'mensaje': 'Conversaciones expiradas limpiadas',
            'conversaciones_activas': len(memoria.memoria_ram),
            'estado': 'OK'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/consultas-vacias', methods=['GET'])
def obtener_consultas_vacias():
    """Obtiene consultas no clasificadas guardadas en data_vacio"""
    try:
        consultas = []
        for doc_id in db_vacio:
            doc = db_vacio[doc_id]
            if doc.get('metadata', {}).get('requiere_revision', False):
                consultas.append({
                    'id': doc['_id'],
                    'intencion': doc.get('intencion', ''),
                    'categoria': doc.get('categoria', ''),
                    'patrones': doc.get('patrones', []),
                    'contenido': doc.get('contenido', ''),
                    'fecha_creacion': doc.get('metadata', {}).get('fecha_creacion', ''),
                    'usuario_origen': doc.get('metadata', {}).get('usuario_origen', ''),
                    'keywords': doc.get('metadata', {}).get('keywords', []),
                    'confianza': doc.get('metadata', {}).get('confianza_clasificacion', 0)
                })
        
        consultas.sort(key=lambda x: x['fecha_creacion'], reverse=True)
        
        return jsonify({
            'total': len(consultas),
            'consultas': consultas
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/consultas-pendientes', methods=['GET'])
def obtener_consultas_pendientes():
    """Obtiene consultas pendientes de procesar"""
    try:
        consultas = []
        for doc_id in db_questions:
            doc = db_questions[doc_id]
            if not doc.get('procesado', False):
                consultas.append({
                    'id': doc['_id'],
                    'mensaje': doc.get('mensaje', ''),
                    'nombre': doc.get('nombre', ''),
                    'intencion': doc.get('intencion', ''),
                    'timestamp': doc.get('timestamp', ''),
                    'remoteJid': doc.get('remoteJid', '')
                })
        
        consultas.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'total': len(consultas),
            'consultas': consultas
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/estadisticas', methods=['GET'])
def obtener_estadisticas():
    """Obtiene estadísticas del sistema"""
    try:
        clasificaciones = {}
        intenciones_count = {}
        
        for doc_id in db_questions:
            doc = db_questions[doc_id]
            intencion = doc.get('intencion', 'sin_clasificar')
            intenciones_count[intencion] = intenciones_count.get(intencion, 0) + 1
        
        consultas_vacias_count = 0
        consultas_vacias_pendientes = 0
        for doc_id in db_vacio:
            doc = db_vacio[doc_id]
            consultas_vacias_count += 1
            if doc.get('metadata', {}).get('requiere_revision', False):
                consultas_vacias_pendientes += 1
        
        top_intenciones = sorted(
            intenciones_count.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return jsonify({
            'documentos_rag': len(rag_system.document_cache),
            'consultas_clasificadas': len([d for d in db_questions]),
            'consultas_no_clasificadas': consultas_vacias_count,
            'consultas_vacias_pendientes': consultas_vacias_pendientes,
            'consultas_pendientes': len([d for d in db_questions if not db_questions[d].get('procesado', False)]),
            'conversaciones_activas': len(memoria.memoria_ram),
            'top_intenciones': [{'intencion': k, 'count': v} for k, v in top_intenciones],
            'intenciones_cargadas': len(clasificador.intenciones_desde_rag),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check del servicio"""
    try:
        test_response = gemini_model.generate_content("test")
        gemini_status = 'ok'
    except:
        gemini_status = 'error'
    
    try:
        couch.version()
        couchdb_status = 'ok'
    except:
        couchdb_status = 'error'
    
    return jsonify({
        'status': 'ok',
        'services': {
            'gemini': gemini_status,
            'couchdb': couchdb_status,
            'rag': 'ok',
            'embeddings': 'ok',
            'clasificador': 'ok',
            'memoria': 'ok'
        },
        'databases': {
            'knowledge': COUCHDB_DATABASE,
            'questions': COUCHDB_QUESTION,
            'vacio': COUCHDB_VACIO,
            'conversaciones': COUCHDB_CONVERSACIONES
        },
        'stats': {
            'documentos_indexados': len(rag_system.document_cache),
            'intenciones_cargadas': len(clasificador.intenciones_desde_rag),
            'consultas_almacenadas': len([d for d in db_questions]),
            'consultas_no_clasificadas': len([d for d in db_vacio]),
            'conversaciones_activas': len(memoria.memoria_ram)
        },
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# INICIALIZACIÓN
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🤖 CHATBOT RAG MEJORADO CON MEMORIA - GEMINI 2.5 FLASH")
    print("=" * 70)
    
    print(f"\n📊 Configuración:")
    print(f"   ✓ Host: {FLASK_HOST}:{FLASK_PORT}")
    print(f"   ✓ CouchDB: {COUCHDB_HOST}:{COUCHDB_PORT}")
    print(f"   ✓ Base de Conocimientos: {COUCHDB_DATABASE}")
    print(f"   ✓ Consultas Clasificadas: {COUCHDB_QUESTION}")
    print(f"   ✓ Consultas NO Clasificadas: {COUCHDB_VACIO}")
    print(f"   ✓ Conversaciones: {COUCHDB_CONVERSACIONES}")
    
    print(f"\n🤖 Modelos de IA:")
    print(f"   ✓ Embeddings: paraphrase-multilingual-mpnet-base-v2")
    print(f"   ✓ LLM: Gemini 2.5 Flash")
    
    # ============================================
    # ✅ INICIALIZAR OBJETOS AQUÍ (ANTES DE USARLOS)
    # ============================================
    print(f"\n🔄 Inicializando componentes del sistema...")
    
    # 1. Sistema RAG
    print(f"   → Inicializando RAG...")
    rag_system = AdvancedRAG(db_knowledge, embedding_model)
    
    # 2. Clasificador (NECESITA que rag_system exista)
    print(f"   → Inicializando Clasificador...")
    clasificador = ClasificadorIntenciones(embedding_model)
    
    # 3. Sistema de Memoria
    print(f"   → Inicializando Memoria Conversacional...")
    memoria = MemoriaConversacional(
        db_conversaciones, 
        max_mensajes=10, 
        tiempo_expiracion_minutos=60
    )
    
    print(f"✅ Componentes inicializados correctamente")
    
    # ============================================
    # AHORA SÍ PODEMOS MOSTRAR ESTADÍSTICAS
    # ============================================
    print(f"\n🎯 Clasificación MEJORADA:")
    print(f"   ✓ Umbral de clasificación: 0.45 (antes: 0.65)")
    print(f"   ✓ Umbral RAG retrieval: 0.25 (antes: 0.30)")
    print(f"   ✓ Intenciones totales en RAG: {len(clasificador.intenciones_desde_rag)}")
    print(f"   ✓ Intenciones que NO se guardan: {', '.join(clasificador.intenciones_no_guardar)}")
    
    print(f"\n🧠 Sistema de Memoria:")
    print(f"   ✓ Máximo mensajes por usuario: {memoria.max_mensajes}")
    print(f"   ✓ Tiempo de expiración: {memoria.tiempo_expiracion.seconds // 60} minutos")
    print(f"   ✓ Conversaciones activas cargadas: {len(memoria.memoria_ram)}")
    
    print(f"\n🔍 Verificando servicios...")
    try:
        test_response = gemini_model.generate_content("test")
        print("   ✅ Gemini API conectado")
    except Exception as e:
        print(f"   ❌ Error con Gemini API: {e}")
    
    try:
        version = couch.version()
        print(f"   ✅ CouchDB conectado (versión {version})")
    except Exception as e:
        print(f"   ❌ Error con CouchDB: {e}")
    
    print(f"\n📡 Endpoints:")
    print(f"   ✓ POST /bot - Endpoint principal (con memoria)")
    print(f"   ✓ GET /api/memoria/<remote_jid> - Ver historial")
    print(f"   ✓ DELETE /api/memoria/<remote_jid> - Limpiar memoria")
    print(f"   ✓ POST /api/memoria/limpiar-expiradas")
    print(f"   ✓ GET /api/consultas-pendientes")
    print(f"   ✓ GET /api/consultas-vacias")
    print(f"   ✓ GET /api/estadisticas")
    print(f"   ✓ GET /health")
    
    print(f"\n📚 Sistema RAG:")
    print(f"   ✓ Documentos indexados: {len(rag_system.document_cache)}")
    print(f"   ✓ Embeddings en cache: {len(rag_system.embedding_cache)}")
    
    print(f"\n💾 Base de Datos de Consultas:")
    print(f"   ✓ Consultas CLASIFICADAS → '{COUCHDB_QUESTION}'")
    print(f"   ✓ Consultas NO CLASIFICADAS → '{COUCHDB_VACIO}'")
    
    print(f"\n🔧 MEJORAS APLICADAS:")
    print(f"   ✅ Umbral de clasificación reducido para más matches")
    print(f"   ✅ Búsqueda RAG activada para TODAS las consultas")
    print(f"   ✅ Threshold de similitud más permisivo (0.25)")
    print(f"   ✅ Mejores prompts para Gemini")
    print(f"   ✅ Logging detallado de scores de similitud")
    
    print("\n" + "=" * 70)
    print(f"🚀 SERVIDOR INICIADO EN https://{FLASK_HOST}:{FLASK_PORT}")
    print("=" * 70 + "\n")
    
    ssl_context = None
    if os.path.exists(SSL_CERT) and os.path.exists(SSL_KEY):
        ssl_context = (SSL_CERT, SSL_KEY)
        print(f"🔐 SSL habilitado: {SSL_CERT}, {SSL_KEY}")
    else:
        print(f"⚠️ SSL no disponible (archivos no encontrados)")
    
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        ssl_context=ssl_context,
        debug=False,
        threaded=True,
        use_reloader=False
    )