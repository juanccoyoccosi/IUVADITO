from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    Flowable
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.enums import TA_JUSTIFY
import io

import google.generativeai as genai

import numpy as np
import json
import logging
import os
import requests
from typing import List, Dict, Optional
from datetime import datetime
from collections import Counter, defaultdict

import couchdb
from flask import Flask, request, jsonify, send_file
import argparse
import atexit

from dotenv import load_dotenv
load_dotenv()

from reportlab.lib.pagesizes import A4
import matplotlib.pyplot as plt

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeneradorPDF:
    """Clase para generar PDFs profesionales con gráficas"""
    
    def __init__(self, filename):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=A4,
                                     rightMargin=72, leftMargin=72,
                                     topMargin=72, bottomMargin=18)
        self.story = []
        self.styles = getSampleStyleSheet()
        self._crear_estilos_personalizados()
    
    def _crear_estilos_personalizados(self):
        """Crea estilos personalizados para el PDF"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#283593'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
    
    def agregar_portada(self, titulo, subtitulo, fecha):
        """Agrega una portada profesional"""
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(titulo, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.3*inch))
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        self.story.append(Paragraph(subtitulo, subtitle_style))
        self.story.append(Spacer(1, 0.5*inch))
        
        fecha_style = ParagraphStyle(
            'Fecha',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER
        )
        self.story.append(Paragraph(f"Generado el: {fecha}", fecha_style))
        self.story.append(PageBreak())
    
    def agregar_seccion(self, titulo):
        """Agrega un título de sección"""
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph(titulo, self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.1*inch))
    
    def agregar_parrafo(self, texto):
        """Agrega un párrafo de texto"""
        self.story.append(Paragraph(texto, self.styles['CustomBody']))
    
    def agregar_tabla(self, datos, col_widths=None):
        """Agrega una tabla con estilo"""
        tabla = Table(datos, colWidths=col_widths)
        tabla.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        self.story.append(tabla)
        self.story.append(Spacer(1, 0.2*inch))
    
    def agregar_grafica(self, fig):
        """Agrega una gráfica de matplotlib al PDF"""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        
        img = Image(img_buffer, width=5*inch, height=3*inch)
        self.story.append(img)
        self.story.append(Spacer(1, 0.2*inch))
        plt.close(fig)
    
    def generar(self):
        """Genera el PDF final"""
        self.doc.build(self.story)
        logger.info(f"PDF generado: {self.filename}")


class NotificadorN8N:
    """Clase mejorada para enviar notificaciones a n8n con soporte para Gmail"""
    
    def __init__(self, webhook_url: str = None, n8n_webhook_url: str = None):
        self.webhook_url = webhook_url
        self.n8n_webhook_url = n8n_webhook_url
    
    def enviar_notificacion(self, reporte_data: Dict) -> bool:
        """Envía una notificación POST al webhook configurado"""
        if not self.webhook_url:
            logger.info("No hay webhook configurado, saltando notificación")
            return False
        
        try:
            payload = {
                "evento": "reporte_generado",
                "timestamp": datetime.now().isoformat(),
                "mes": reporte_data.get('mes'),
                "total_consultas": reporte_data.get('total_consultas'),
                "pdf_generado": reporte_data.get('pdf_path'),
                "pdf_url": reporte_data.get('pdf_url'),
                "reporte_id": reporte_data.get('reporte_id'),
                "estadisticas_resumen": {
                    "intenciones_unicas": len(reporte_data.get('estadisticas', {}).get('intenciones', {})),
                    "usuarios_activos": len(reporte_data.get('estadisticas', {}).get('usuarios_frecuentes', [])),
                    "canales": len(reporte_data.get('estadisticas', {}).get('canales', {}))
                }
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code in [200, 201, 204]:
                logger.info(f"✅ Notificación enviada exitosamente al webhook")
                return True
            else:
                logger.warning(f"⚠️ Webhook respondió con código {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error al enviar notificación al webhook: {e}")
            return False
    
    def enviar_a_n8n_gmail(self, reporte_data: Dict, destinatarios: List[str]) -> Dict:
        """Envía los datos a n8n para el flujo de Gmail"""
        if not self.n8n_webhook_url:
            logger.warning("No hay webhook de n8n configurado")
            return {"status": "error", "message": "Webhook de n8n no configurado"}
        
        try:
            payload = {
                "evento": "reporte_generado",
                "timestamp": datetime.now().isoformat(),
                "reporte": {
                    "id": reporte_data.get('reporte_id'),
                    "mes": reporte_data.get('mes'),
                    "total_consultas": reporte_data.get('total_consultas'),
                    "pdf_path": reporte_data.get('pdf_path'),
                    "pdf_url": reporte_data.get('pdf_url'),
                    "estadisticas": reporte_data.get('estadisticas_resumen')
                },
                "email": {
                    "destinatarios": destinatarios,
                    "asunto": f"📊 Reporte Mensual - {reporte_data.get('mes')}",
                    "mensaje": self._generar_mensaje_email(reporte_data)
                }
            }
            
            response = requests.post(
                self.n8n_webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code in [200, 201, 204]:
                logger.info(f"✅ Reporte enviado a n8n para Gmail")
                return {
                    "status": "success",
                    "message": "Reporte enviado al flujo de n8n",
                    "response": response.json() if response.text else {}
                }
            else:
                logger.warning(f"⚠️ n8n respondió con código {response.status_code}")
                return {
                    "status": "error",
                    "message": f"Error en n8n: {response.status_code}",
                    "details": response.text
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error al enviar a n8n: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generar_mensaje_email(self, reporte_data: Dict) -> str:
        """Genera el mensaje HTML para el email"""
        stats = reporte_data.get('estadisticas_resumen', {})
        return f"""
        <h2>📊 Reporte Mensual Generado</h2>
        
        <p><strong>Período:</strong> {reporte_data.get('mes')}</p>
        <p><strong>Fecha de Generación:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        
        <h3>Resumen Ejecutivo</h3>
        <ul>
            <li><strong>Total de Consultas:</strong> {reporte_data.get('total_consultas')}</li>
            <li><strong>Intenciones Únicas:</strong> {stats.get('intenciones_unicas', 0)}</li>
            <li><strong>Usuarios Activos:</strong> {stats.get('usuarios_activos', 0)}</li>
            <li><strong>Canales:</strong> {stats.get('canales', 0)}</li>
        </ul>
        
        <p>El reporte completo está adjunto en formato PDF.</p>
        
        <p><em>Generado automáticamente por el Sistema de Análisis</em></p>
        """


class analisis:
    def __init__(self, db, base, couch_url, gemini_api_key,
                 campo_intencion="intencion", campo_texto="embedding",
                 webhook_url=None, n8n_webhook_url=None, base_url=None):
        self.db = db
        self.base = base
        self.couch_url = couch_url
        self.campo_intencion = campo_intencion
        self.campo_texto = campo_texto
        self.intenciones_embebidas = {}
        self.base_url = base_url or "http://62.171.179.255:5010"
        
        # Configurar notificador
        self.notificador = NotificadorN8N(webhook_url, n8n_webhook_url)
        
        # Modelo ligero (82MB en vez de 420MB)
        logger.info("Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
        self.modelo = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info("Modelo cargado exitosamente")
        
        # Configurar Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Directorio para PDFs
        self.pdf_dir = "reportes_pdf"
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        # Generar embeddings para documentos que no los tienen
        logger.info("Verificando embeddings existentes...")
        self._generar_embeddings()
    
    def _generar_embeddings(self):
        """Genera embeddings para documentos que no los tienen"""
        try:
            respuesta = requests.get(f"{self.couch_url}/{self.base}/_all_docs?include_docs=true")
            docs = respuesta.json()["rows"]
            
            total_docs = len(docs)
            docs_con_embedding = 0
            docs_generados = 0
            
            for fila in docs:
                doc = fila["doc"]
                doc_id = doc["_id"]
                
                # Saltar documentos de sistema
                if doc_id.startswith('_'):
                    continue
                
                # Verificar si ya tiene embedding
                if self.campo_texto in doc:
                    self.intenciones_embebidas[doc_id] = doc[self.campo_texto]
                    docs_con_embedding += 1
                    continue
                
                # Generar embedding si tiene el campo de intención
                if self.campo_intencion in doc:
                    texto = doc.get(self.campo_intencion, "")
                    if texto:
                        embedding = self.modelo.encode([texto])[0].tolist()
                        doc[self.campo_texto] = embedding
                        self.db.save(doc)
                        self.intenciones_embebidas[doc_id] = embedding
                        docs_generados += 1
                        logger.info(f"Embedding generado para {doc_id}")
            
            logger.info(f"Embeddings: {docs_con_embedding} existentes, {docs_generados} generados")
        except Exception as e:
            logger.error(f"Error al generar embeddings: {e}")
    
    def obtener_datos_mes_actual(self) -> List[Dict]:
        """Obtiene solo los documentos del mes actual"""
        ahora = datetime.now()
        inicio_mes = datetime(ahora.year, ahora.month, 1)
        
        respuesta = requests.get(f"{self.couch_url}/{self.base}/_all_docs?include_docs=true")
        docs = respuesta.json()["rows"]
        
        datos_mes = []
        for fila in docs:
            doc = fila["doc"]
            
            # Saltar documentos de sistema
            if doc["_id"].startswith('_') or doc["_id"].startswith('reporte_'):
                continue
            
            # Verificar timestamp
            if "timestamp" in doc:
                try:
                    timestamp = datetime.fromisoformat(doc["timestamp"].replace("Z", "+00:00"))
                    if timestamp >= inicio_mes:
                        datos_mes.append(doc)
                except Exception as e:
                    logger.error(f"Error al procesar timestamp de {doc['_id']}: {e}")
        
        logger.info(f"Se encontraron {len(datos_mes)} documentos del mes actual")
        return datos_mes
    
    def agrupar_por_intencion(self, datos: List[Dict]) -> Dict[str, List[Dict]]:
        """Agrupa los datos por intención"""
        intenciones = defaultdict(list)
        
        for doc in datos:
            if self.campo_intencion in doc:
                intencion = doc[self.campo_intencion]
                intenciones[intencion].append(doc)
        
        return dict(intenciones)
    
    def analizar_similitud_embeddings(self, datos: List[Dict]) -> Dict:
        """Analiza la similitud entre embeddings para detectar patrones"""
        embeddings_por_intencion = defaultdict(list)
    
        for doc in datos:
            if self.campo_texto in doc and self.campo_intencion in doc:
               intencion = doc[self.campo_intencion]
               embedding = doc[self.campo_texto]
            
               if isinstance(embedding, list) and len(embedding) > 0:
                  embeddings_por_intencion[intencion].append({
                     'embedding': embedding,
                     'nombre': doc.get('nombre', 'Desconocido'),
                     'mensaje': doc.get('mensaje', ''),
                     'doc_id': doc['_id']
                   })
    
        analisis_similitud = {}
        for intencion, items in embeddings_por_intencion.items():
            if len(items) > 1:
                try:
                    embeddings = [item['embedding'] for item in items]
                
                    embedding_lengths = [len(emb) for emb in embeddings]
                    if len(set(embedding_lengths)) > 1:
                       logger.warning(f"Embeddings con diferentes dimensiones en '{intencion}': {set(embedding_lengths)}")
                       most_common_length = Counter(embedding_lengths).most_common(1)[0][0]
                       embeddings = [emb for emb in embeddings if len(emb) == most_common_length]
                       logger.info(f"Usando {len(embeddings)} embeddings de dimensión {most_common_length}")
                
                    if len(embeddings) < 2:
                       logger.warning(f"No hay suficientes embeddings válidos para '{intencion}'")
                       continue
                
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    similitudes = cosine_similarity(embeddings_array)
                    promedio_similitud = np.mean(similitudes[np.triu_indices_from(similitudes, k=1)])
                
                    analisis_similitud[intencion] = {
                            'cantidad': len(items),
                            'similitud_promedio': float(promedio_similitud),
                            'usuarios': [item['nombre'] for item in items]
                        }
                except Exception as e:
                        logger.error(f"Error al analizar similitud para '{intencion}': {e}")
                        continue
    
        return analisis_similitud    
    
    def generar_estadisticas(self, datos: List[Dict]) -> Dict:
        """Genera estadísticas generales de los datos"""
        intenciones_agrupadas = self.agrupar_por_intencion(datos)
        
        conteo_intenciones = {k: len(v) for k, v in intenciones_agrupadas.items()}
        
        usuarios = [doc.get('nombre', 'Desconocido') for doc in datos]
        usuarios_frecuentes = Counter(usuarios).most_common(10)
        
        canales = [doc.get('canal', 'Desconocido') for doc in datos]
        canales_frecuentes = Counter(canales).most_common()
        
        mensajes_por_dia = defaultdict(int)
        for doc in datos:
            if 'timestamp' in doc:
                try:
                    fecha = datetime.fromisoformat(doc['timestamp'].replace("Z", "+00:00"))
                    dia = fecha.strftime('%Y-%m-%d')
                    mensajes_por_dia[dia] += 1
                except:
                    pass
        
        return {
            'total_consultas': len(datos),
            'intenciones': conteo_intenciones,
            'top_intenciones': sorted(conteo_intenciones.items(), key=lambda x: x[1], reverse=True)[:10],
            'usuarios_frecuentes': usuarios_frecuentes,
            'canales': dict(canales_frecuentes),
            'mensajes_por_dia': dict(sorted(mensajes_por_dia.items()))
        }
    
    def crear_graficas(self, estadisticas: Dict) -> Dict[str, plt.Figure]:
        """Crea las gráficas para el reporte"""
        graficas = {}
        
        if estadisticas['top_intenciones']:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            top_intenciones = estadisticas['top_intenciones'][:10]
            intenciones_labels = [x[0][:30] + '...' if len(x[0]) > 30 else x[0] for x in top_intenciones]
            intenciones_values = [x[1] for x in top_intenciones]
            
            colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(intenciones_labels)))
            ax1.barh(intenciones_labels, intenciones_values, color=colors_bar)
            ax1.set_xlabel('Número de Consultas', fontsize=12, fontweight='bold')
            ax1.set_title('Top 10 Intenciones Más Frecuentes', fontsize=14, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            graficas['top_intenciones'] = fig1
        
        if estadisticas['mensajes_por_dia']:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            fechas = list(estadisticas['mensajes_por_dia'].keys())
            valores = list(estadisticas['mensajes_por_dia'].values())
            
            ax2.plot(fechas, valores, marker='o', linewidth=2, markersize=6, color='#3f51b5')
            ax2.fill_between(range(len(valores)), valores, alpha=0.3, color='#3f51b5')
            ax2.set_xlabel('Fecha', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Número de Mensajes', fontsize=12, fontweight='bold')
            ax2.set_title('Tendencia de Consultas Diarias', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            graficas['mensajes_por_dia'] = fig2
        
        if estadisticas['canales']:
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            canales_labels = list(estadisticas['canales'].keys())
            canales_values = list(estadisticas['canales'].values())
            
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(canales_labels)))
            ax3.pie(canales_values, labels=canales_labels, autopct='%1.1f%%',
                   startangle=90, colors=colors_pie, textprops={'fontsize': 12})
            ax3.set_title('Distribución por Canales', fontsize=14, fontweight='bold')
            plt.tight_layout()
            graficas['canales'] = fig3
        
        if estadisticas['usuarios_frecuentes']:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            usuarios = [x[0] for x in estadisticas['usuarios_frecuentes'][:10]]
            consultas = [x[1] for x in estadisticas['usuarios_frecuentes'][:10]]
            
            colors_users = plt.cm.plasma(np.linspace(0.3, 0.9, len(usuarios)))
            ax4.bar(usuarios, consultas, color=colors_users)
            ax4.set_xlabel('Usuario', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Número de Consultas', fontsize=12, fontweight='bold')
            ax4.set_title('Top 10 Usuarios Más Activos', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            ax4.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            graficas['usuarios_frecuentes'] = fig4
        
        return graficas
    
    def generar_reporte_con_gemini(self, estadisticas: Dict, analisis_similitud: Dict) -> str:
        """Genera un reporte completo usando Gemini"""
        
        prompt = f"""
Eres un analista de datos experto. Genera un reporte mensual profesional basado en estos datos:

ESTADÍSTICAS:
{json.dumps(estadisticas, indent=2, ensure_ascii=False)}

ANÁLISIS DE SIMILITUD:
{json.dumps(analisis_similitud, indent=2, ensure_ascii=False)}

Genera un reporte con:
1. Resumen Ejecutivo (3-4 párrafos)
2. Análisis de Intenciones Principales
3. Análisis de Usuarios y Comportamiento
4. Tendencias Temporales
5. Problemas Identificados
6. Recomendaciones Estratégicas (mínimo 5)
7. Conclusiones

NO uses markdown, solo texto plano con saltos de línea.
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            logger.info("Reporte generado con Gemini")
            return response.text
        except Exception as e:
            logger.error(f"Error al generar reporte con Gemini: {e}")
            return f"Error al generar reporte: {str(e)}"
    
    def generar_pdf_reporte(self, estadisticas: Dict, analisis_similitud: Dict, reporte_texto: str) -> str:
        """Genera un PDF profesional con el reporte completo"""
        
        fecha_actual = datetime.now()
        nombre_archivo = f"{self.pdf_dir}/reporte_{fecha_actual.strftime('%Y_%m')}.pdf"
        
        pdf = GeneradorPDF(nombre_archivo)
        
        pdf.agregar_portada(
            titulo="REPORTE MENSUAL DE ANÁLISIS",
            subtitulo=f"Análisis de Consultas - {fecha_actual.strftime('%B %Y')}",
            fecha=fecha_actual.strftime('%d de %B de %Y')
        )
        
        pdf.agregar_seccion("📊 RESUMEN EJECUTIVO")
        pdf.agregar_parrafo(f"<b>Total de consultas del mes:</b> {estadisticas['total_consultas']}")
        pdf.agregar_parrafo(f"<b>Intenciones únicas identificadas:</b> {len(estadisticas['intenciones'])}")
        pdf.agregar_parrafo(f"<b>Usuarios activos:</b> {len(estadisticas['usuarios_frecuentes'])}")
        pdf.agregar_parrafo(f"<b>Canales utilizados:</b> {len(estadisticas['canales'])}")
        
        graficas = self.crear_graficas(estadisticas)
        
        if 'top_intenciones' in graficas:
            pdf.agregar_seccion("📈 TOP 10 INTENCIONES MÁS FRECUENTES")
            pdf.agregar_grafica(graficas['top_intenciones'])
        
        pdf.agregar_seccion("📋 DETALLE DE INTENCIONES")
        datos_tabla = [['#', 'Intención', 'Cantidad']]
        for idx, (intencion, cantidad) in enumerate(estadisticas['top_intenciones'][:10], 1):
            intencion_corta = intencion[:50] + '...' if len(intencion) > 50 else intencion
            datos_tabla.append([str(idx), intencion_corta, str(cantidad)])
        pdf.agregar_tabla(datos_tabla, col_widths=[0.5*inch, 3.5*inch, 1*inch])
        
        if 'mensajes_por_dia' in graficas:
            pdf.agregar_seccion("📅 TENDENCIA TEMPORAL")
            pdf.agregar_grafica(graficas['mensajes_por_dia'])
        
        if 'canales' in graficas:
            pdf.agregar_seccion("📱 DISTRIBUCIÓN POR CANALES")
            pdf.agregar_grafica(graficas['canales'])
        
        if 'usuarios_frecuentes' in graficas:
            pdf.agregar_seccion("👥 USUARIOS MÁS ACTIVOS")
            pdf.agregar_grafica(graficas['usuarios_frecuentes'])
        
        pdf.agregar_seccion("🤖 ANÁLISIS DETALLADO CON IA")
        secciones = reporte_texto.split('\n\n')
        for seccion in secciones:
            if seccion.strip():
                if len(seccion) < 100 and seccion.isupper():
                    pdf.agregar_seccion(seccion.strip())
                else:
                    pdf.agregar_parrafo(seccion.strip())
        
        pdf.agregar_seccion("ℹ️ INFORMACIÓN DEL REPORTE")
        pdf.agregar_parrafo(f"<b>Generado por:</b> Sistema de Análisis Automático")
        pdf.agregar_parrafo(f"<b>Fecha:</b> {fecha_actual.strftime('%d/%m/%Y %H:%M:%S')}")
        pdf.agregar_parrafo(f"<b>Período:</b> {fecha_actual.strftime('%B %Y')}")
        
        pdf.generar()
        
        logger.info(f"PDF generado: {nombre_archivo}")
        return nombre_archivo
    
    def ejecutar_analisis_mensual(self, enviar_email: bool = False, destinatarios: List[str] = None) -> Dict:
        """Ejecuta el análisis mensual completo con opción de envío a Gmail"""
        logger.info("Iniciando análisis mensual...")
        
        datos_mes = self.obtener_datos_mes_actual()
        
        if not datos_mes:
            logger.warning("No hay datos del mes actual")
            return {"status": "no_data", "message": "No hay datos del mes actual"}
        
        estadisticas = self.generar_estadisticas(datos_mes)
        analisis_similitud = self.analizar_similitud_embeddings(datos_mes)
        reporte = self.generar_reporte_con_gemini(estadisticas, analisis_similitud)
        pdf_path = self.generar_pdf_reporte(estadisticas, analisis_similitud, reporte)
        
        # Crear URL pública del PDF
        pdf_url = f"{self.base_url}/reportes/{os.path.basename(pdf_path)}"
        
        reporte_doc = {
            "_id": f"reporte_{datetime.now().strftime('%Y_%m')}",
            "tipo": "reporte_mensual",
            "fecha_generacion": datetime.now().isoformat(),
            "mes": datetime.now().strftime('%Y-%m'),
            "estadisticas": estadisticas,
            "reporte": reporte,
            "pdf_path": pdf_path,
            "pdf_url": pdf_url
        }
        
        try:
            self.db.save(reporte_doc)
            logger.info(f"Reporte guardado: {reporte_doc['_id']}")
        except couchdb.http.ResourceConflict:
            doc_existente = self.db[reporte_doc['_id']]
            reporte_doc['_rev'] = doc_existente['_rev']
            self.db.save(reporte_doc)
            logger.info(f"Reporte actualizado: {reporte_doc['_id']}")
        
        resultado = {
            "status": "success",
            "total_consultas": len(datos_mes),
            "reporte_id": reporte_doc['_id'],
            "pdf_path": pdf_path,
            "pdf_url": pdf_url,
            "mes": datetime.now().strftime('%Y-%m'),
            "estadisticas": estadisticas,
            "estadisticas_resumen": {
                "intenciones_unicas": len(estadisticas.get('intenciones', {})),
                "usuarios_activos": len(estadisticas.get('usuarios_frecuentes', [])),
                "canales": len(estadisticas.get('canales', {}))
            }
        }
        
        # Enviar notificación al webhook principal
        self.notificador.enviar_notificacion(resultado)
        
        # Si se solicita envío por Gmail, enviar a n8n
        if enviar_email and destinatarios:
            resultado_gmail = self.notificador.enviar_a_n8n_gmail(resultado, destinatarios)
            resultado["gmail_enviado"] = resultado_gmail
            logger.info(f"Resultado envío Gmail: {resultado_gmail['status']}")
        
        return resultado


# ========== FLASK APP ==========

app = Flask(__name__)

USERNAME = "Juan"
PASSWORD = "Elpro123"
DATABASE = "data_frecuentquestion"
COUCH_URL = f"http://{USERNAME}:{PASSWORD}@62.171.179.255:5984"
GEMINI_API_KEY = "AIzaSyBE535gpiIYwV58b0grSsZ4WH4yuk_r4xQ"
WEBHOOK_URL = os.environ.get('WEBHOOK_URL', None)
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', None)
BASE_URL = os.environ.get('BASE_URL', 'http://62.171.179.255:5010')

server = couchdb.Server(COUCH_URL)
db = server[DATABASE]

analizador = analisis(
    db=db,
    base=DATABASE,
    couch_url=COUCH_URL,
    gemini_api_key=GEMINI_API_KEY,
    webhook_url=WEBHOOK_URL,
    n8n_webhook_url=N8N_WEBHOOK_URL,
    base_url=BASE_URL
)

# ========== SCHEDULER ==========

def job_generar_reporte_mensual():
    """Tarea programada que se ejecuta automáticamente al final de cada mes"""
    try:
        logger.info("🤖 Generando reporte automático de fin de mes...")
        resultado = analizador.ejecutar_analisis_mensual()
        
        if resultado['status'] == 'success':
            logger.info(f"✅ Reporte generado: {resultado['pdf_path']}")
            logger.info(f"📧 Notificación POST enviada automáticamente")
        else:
            logger.error(f"❌ Error en generación automática: {resultado}")
    except Exception as e:
        logger.error(f"❌ Excepción en generación automática: {e}")

scheduler = BackgroundScheduler()

scheduler.add_job(
    job_generar_reporte_mensual,
    CronTrigger(day='last', hour=23, minute=59),
    id='reporte_mensual_automatico',
    replace_existing=True
)

scheduler.start()
logger.info("📅 Scheduler activado - Reportes automáticos configurados")
logger.info("⏰ Próxima ejecución: Último día del mes a las 23:59")

atexit.register(lambda: scheduler.shutdown())

# ========== ENDPOINTS ==========

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar el estado del servicio"""
    try:
        next_run = scheduler.get_job('reporte_mensual_automatico').next_run_time
        next_run_str = next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else 'No programado'
    except:
        next_run_str = 'Error al obtener próxima ejecución'
    
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "scheduler_running": scheduler.running,
        "proxima_ejecucion_automatica": next_run_str,
        "webhook_configurado": WEBHOOK_URL is not None,
        "webhook_url": WEBHOOK_URL if WEBHOOK_URL else "No configurado",
        "n8n_webhook_configurado": N8N_WEBHOOK_URL is not None,
        "n8n_webhook_url": N8N_WEBHOOK_URL if N8N_WEBHOOK_URL else "No configurado"
    })

@app.route('/generar-reporte', methods=['GET', 'POST'])
def generar_reporte():
    """
    Endpoint para generar un reporte manualmente
    
    POST con JSON:
    {
        "enviar_email": true,
        "destinatarios": ["ccoyoccosijuan544@gmail.com"]
    }
    
    GET: Genera el reporte sin enviar email
    """
    try:
        enviar_email = False
        destinatarios = []
        
        if request.method == 'POST':
            data = request.get_json() or {}
            enviar_email = data.get('enviar_email', False)
            destinatarios = data.get('destinatarios', [])
            
            # Validar destinatarios si se solicita envío
            if enviar_email and not destinatarios:
                return jsonify({
                    "status": "error",
                    "message": "Se requiere al menos un destinatario para enviar por email"
                }), 400
        
        resultado = analizador.ejecutar_analisis_mensual(
            enviar_email=enviar_email,
            destinatarios=destinatarios
        )
        
        return jsonify(resultado), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generar-reporte-interactivo', methods=['POST'])
def generar_reporte_interactivo():
    """
    Endpoint interactivo que pregunta si desea enviar por Gmail
    
    Paso 1: Genera el reporte y devuelve información
    Paso 2: Usuario confirma si desea enviar por email
    
    POST con JSON:
    {
        "confirmar_email": false,  // Primera llamada
        "destinatarios": []
    }
    
    O segunda llamada con confirmación:
    {
        "confirmar_email": true,
        "reporte_id": "reporte_2025_11",
        "destinatarios": ["ccoyoccosijuan544@gmail.com"]
    }
    """
    try:
        data = request.get_json() or {}
        confirmar_email = data.get('confirmar_email', False)
        
        # Primera llamada: Generar reporte
        if not confirmar_email:
            resultado = analizador.ejecutar_analisis_mensual(enviar_email=False)
            
            if resultado['status'] != 'success':
                return jsonify(resultado), 500
            
            return jsonify({
                "status": "reporte_generado",
                "message": "Reporte generado exitosamente. ¿Desea enviarlo por Gmail?",
                "reporte": {
                    "id": resultado['reporte_id'],
                    "mes": resultado['mes'],
                    "total_consultas": resultado['total_consultas'],
                    "pdf_url": resultado['pdf_url']
                },
                "pregunta": "¿Enviar este reporte por Gmail?",
                "opciones": {
                    "si": "Enviar POST a /generar-reporte-interactivo con confirmar_email=true",
                    "no": "No hacer nada, el reporte ya está generado"
                }
            }), 200
        
        # Segunda llamada: Confirmar envío por Gmail
        else:
            reporte_id = data.get('reporte_id')
            destinatarios = data.get('destinatarios', [])
            
            if not reporte_id:
                return jsonify({
                    "status": "error",
                    "message": "Se requiere 'reporte_id' para confirmar el envío"
                }), 400
            
            if not destinatarios:
                return jsonify({
                    "status": "error",
                    "message": "Se requiere al menos un destinatario"
                }), 400
            
            # Obtener el reporte de la base de datos
            try:
                doc = db[reporte_id]
            except couchdb.http.ResourceNotFound:
                return jsonify({
                    "status": "error",
                    "message": f"Reporte '{reporte_id}' no encontrado"
                }), 404
            
            # Preparar datos para n8n
            reporte_data = {
                "reporte_id": reporte_id,
                "mes": doc.get('mes'),
                "total_consultas": doc.get('estadisticas', {}).get('total_consultas'),
                "pdf_path": doc.get('pdf_path'),
                "pdf_url": doc.get('pdf_url'),
                "estadisticas_resumen": {
                    "intenciones_unicas": len(doc.get('estadisticas', {}).get('intenciones', {})),
                    "usuarios_activos": len(doc.get('estadisticas', {}).get('usuarios_frecuentes', [])),
                    "canales": len(doc.get('estadisticas', {}).get('canales', {}))
                }
            }
            
            # Enviar a n8n
            resultado_gmail = analizador.notificador.enviar_a_n8n_gmail(reporte_data, destinatarios)
            
            return jsonify({
                "status": "success",
                "message": "Reporte enviado al flujo de n8n para Gmail",
                "reporte_id": reporte_id,
                "destinatarios": destinatarios,
                "n8n_response": resultado_gmail
            }), 200
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/descargar-reporte-actual', methods=['GET'])
def descargar_reporte_actual():
    """Endpoint para descargar el PDF del reporte del mes actual"""
    try:
        mes_actual = datetime.now().strftime('%Y_%m')
        reporte_id = f"reporte_{mes_actual}"
        
        try:
            doc = db[reporte_id]
            pdf_path = doc.get('pdf_path')
            
            if pdf_path and os.path.exists(pdf_path):
                return send_file(pdf_path, mimetype='application/pdf', as_attachment=True)
        except couchdb.http.ResourceNotFound:
            pass
        
        resultado = analizador.ejecutar_analisis_mensual()
        if resultado['status'] == 'success':
            return send_file(resultado['pdf_path'], mimetype='application/pdf', as_attachment=True)
        return jsonify(resultado), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reportes/<path:filename>', methods=['GET'])
def servir_pdf(filename):
    """Sirve archivos PDF del directorio de reportes"""
    try:
        pdf_path = os.path.join(analizador.pdf_dir, filename)
        if os.path.exists(pdf_path):
            return send_file(pdf_path, mimetype='application/pdf')
        else:
            return jsonify({"status": "error", "message": "PDF no encontrado"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/config', methods=['GET', 'POST'])
def configurar_webhook():
    """Endpoint para ver o actualizar la configuración del webhook"""
    global WEBHOOK_URL, N8N_WEBHOOK_URL
    
    if request.method == 'GET':
        return jsonify({
            "webhook_url": WEBHOOK_URL if WEBHOOK_URL else "No configurado",
            "webhook_activo": WEBHOOK_URL is not None,
            "n8n_webhook_url": N8N_WEBHOOK_URL if N8N_WEBHOOK_URL else "No configurado",
            "n8n_webhook_activo": N8N_WEBHOOK_URL is not None
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        nueva_url = data.get('webhook_url')
        nueva_n8n_url = data.get('n8n_webhook_url')
        
        cambios = {}
        
        if nueva_url:
            WEBHOOK_URL = nueva_url
            analizador.notificador.webhook_url = nueva_url
            cambios['webhook_url'] = nueva_url
            logger.info(f"Webhook actualizado: {nueva_url}")
        
        if nueva_n8n_url:
            N8N_WEBHOOK_URL = nueva_n8n_url
            analizador.notificador.n8n_webhook_url = nueva_n8n_url
            cambios['n8n_webhook_url'] = nueva_n8n_url
            logger.info(f"Webhook n8n actualizado: {nueva_n8n_url}")
        
        if cambios:
            return jsonify({
                "status": "success",
                "message": "Webhooks configurados correctamente",
                "cambios": cambios
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Se requiere 'webhook_url' o 'n8n_webhook_url' en el body"
            }), 400

@app.route('/reportes/historial', methods=['GET'])
def listar_reportes():
    """Endpoint para listar todos los reportes generados"""
    try:
        respuesta = requests.get(f"{COUCH_URL}/{DATABASE}/_all_docs?include_docs=true&startkey=\"reporte_\"&endkey=\"reporte_\ufff0\"")
        docs = respuesta.json()["rows"]
        
        reportes = []
        for fila in docs:
            doc = fila["doc"]
            reportes.append({
                "id": doc["_id"],
                "mes": doc.get("mes"),
                "fecha_generacion": doc.get("fecha_generacion"),
                "total_consultas": doc.get("estadisticas", {}).get("total_consultas"),
                "pdf_disponible": os.path.exists(doc.get("pdf_path", "")),
                "pdf_url": doc.get("pdf_url")
            })
        
        return jsonify({
            "status": "success",
            "total_reportes": len(reportes),
            "reportes": reportes
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reportes/<reporte_id>/descargar', methods=['GET'])
def descargar_reporte_especifico(reporte_id):
    """Endpoint para descargar un reporte específico por su ID"""
    try:
        doc = db[reporte_id]
        pdf_path = doc.get('pdf_path')
        
        if pdf_path and os.path.exists(pdf_path):
            return send_file(pdf_path, mimetype='application/pdf', as_attachment=True)
        else:
            return jsonify({
                "status": "error",
                "message": "PDF no encontrado"
            }), 404
    except couchdb.http.ResourceNotFound:
        return jsonify({
            "status": "error",
            "message": f"Reporte '{reporte_id}' no encontrado"
        }), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reportes/<reporte_id>/enviar-gmail', methods=['POST'])
def enviar_reporte_gmail(reporte_id):
    """
    Endpoint para enviar un reporte específico por Gmail a través de n8n
    
    POST con JSON:
    {
        "destinatarios": ["email1@example.com", "email2@example.com"]
    }
    """
    try:
        data = request.get_json() or {}
        destinatarios = data.get('destinatarios', [])
        
        if not destinatarios:
            return jsonify({
                "status": "error",
                "message": "Se requiere al menos un destinatario"
            }), 400
        
        # Obtener el reporte de la base de datos
        try:
            doc = db[reporte_id]
        except couchdb.http.ResourceNotFound:
            return jsonify({
                "status": "error",
                "message": f"Reporte '{reporte_id}' no encontrado"
            }), 404
        
        # Preparar datos para n8n
        reporte_data = {
            "reporte_id": reporte_id,
            "mes": doc.get('mes'),
            "total_consultas": doc.get('estadisticas', {}).get('total_consultas'),
            "pdf_path": doc.get('pdf_path'),
            "pdf_url": doc.get('pdf_url'),
            "estadisticas_resumen": {
                "intenciones_unicas": len(doc.get('estadisticas', {}).get('intenciones', {})),
                "usuarios_activos": len(doc.get('estadisticas', {}).get('usuarios_frecuentes', [])),
                "canales": len(doc.get('estadisticas', {}).get('canales', {}))
            }
        }
        
        # Enviar a n8n
        resultado_gmail = analizador.notificador.enviar_a_n8n_gmail(reporte_data, destinatarios)
        
        return jsonify({
            "status": "success",
            "message": "Reporte enviado al flujo de n8n para Gmail",
            "reporte_id": reporte_id,
            "destinatarios": destinatarios,
            "n8n_response": resultado_gmail
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test-webhook', methods=['POST'])
def test_webhook():
    """Endpoint para probar el envío de notificación al webhook"""
    try:
        data_prueba = {
            "evento": "test",
            "mensaje": "Prueba de webhook",
            "timestamp": datetime.now().isoformat()
        }
        
        exito = analizador.notificador.enviar_notificacion(data_prueba)
        
        if exito:
            return jsonify({
                "status": "success",
                "message": "Notificación de prueba enviada correctamente",
                "webhook_url": WEBHOOK_URL
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Error al enviar notificación de prueba",
                "webhook_url": WEBHOOK_URL
            }), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test-n8n-gmail', methods=['POST'])
def test_n8n_gmail():
    """Endpoint para probar el flujo de Gmail con n8n"""
    try:
        data = request.get_json() or {}
        destinatarios = data.get('destinatarios', ['test@example.com'])
        
        data_prueba = {
            "reporte_id": "test_reporte",
            "mes": "2025-11",
            "total_consultas": 100,
            "pdf_path": "/ruta/test.pdf",
            "pdf_url": f"{BASE_URL}/reportes/test.pdf",
            "estadisticas_resumen": {
                "intenciones_unicas": 10,
                "usuarios_activos": 5,
                "canales": 3
            }
        }
        
        resultado = analizador.notificador.enviar_a_n8n_gmail(data_prueba, destinatarios)
        
        return jsonify({
            "status": "success",
            "message": "Prueba de Gmail enviada a n8n",
            "n8n_webhook_url": N8N_WEBHOOK_URL,
            "resultado": resultado
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ========== MAIN ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5010, help='Puerto del servidor')
    parser.add_argument('--webhook', type=str, default=None, help='URL del webhook para notificaciones POST')
    parser.add_argument('--n8n-webhook', type=str, default=None, help='URL del webhook n8n para Gmail')
    parser.add_argument('--base-url', type=str, default=None, help='URL base del servidor')
    args = parser.parse_args()
    
    if args.webhook:
        WEBHOOK_URL = args.webhook
        analizador.notificador.webhook_url = args.webhook
    
    if args.n8n_webhook:
        N8N_WEBHOOK_URL = args.n8n_webhook
        analizador.notificador.n8n_webhook_url = args.n8n_webhook
    
    if args.base_url:
        BASE_URL = args.base_url
        analizador.base_url = args.base_url
    
    print("=" * 80)
    print("🚀 SISTEMA DE ANÁLISIS MENSUAL CON N8N Y GMAIL")
    print("=" * 80)
    print(f"📅 Generación automática: Último día del mes")
    print(f"📧 Notificación POST: {'✅ ACTIVADA' if WEBHOOK_URL else '❌ NO CONFIGURADA'}")
    if WEBHOOK_URL:
        print(f"🔗 Webhook: {WEBHOOK_URL}")
    print(f"📬 n8n Gmail: {'✅ ACTIVADO' if N8N_WEBHOOK_URL else '❌ NO CONFIGURADO'}")
    if N8N_WEBHOOK_URL:
        print(f"🔗 n8n Webhook: {N8N_WEBHOOK_URL}")
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"📁 PDFs: {analizador.pdf_dir}")
    print(f"🌐 Puerto: {args.port}")
    print("=" * 80)
    print("\n📋 ENDPOINTS:")
    print(f"  GET  /health")
    print(f"  GET  /generar-reporte")
    print(f"  POST /generar-reporte (con destinatarios)")
    print(f"  POST /generar-reporte-interactivo (modo interactivo)")
    print(f"  GET  /descargar-reporte-actual")
    print(f"  GET  /reportes/<filename>")
    print(f"  GET  /reportes/historial")
    print(f"  GET  /reportes/<id>/descargar")
    print(f"  POST /reportes/<id>/enviar-gmail")
    print(f"  GET  /webhook/config")
    print(f"  POST /webhook/config")
    print(f"  POST /test-webhook")
    print(f"  POST /test-n8n-gmail")
    print("=" * 80)
    
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("\n👋 Servidor detenido")