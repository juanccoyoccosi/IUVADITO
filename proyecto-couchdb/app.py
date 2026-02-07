from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import psutil
import platform
import datetime
import os
import pyotp
import qrcode
from io import BytesIO
import base64
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from functools import wraps
import secrets
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'iuvade-secret-key-2024'  # Cambia esto en producción

# Configuración de CouchDB (tus credenciales actuales)
COUCHDB_URL = "http://62.171.179.255:5984"
COUCHDB_USER = "Juan"
COUCHDB_PASS = "Elpro123"

# Cargar modelo miniLM
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor inicia sesión para acceder al sistema'
login_manager.login_message_category = 'warning'

# Base de datos para usuarios (SQLite local)
def init_db():
    """Inicializar base de datos de usuarios"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Tabla de usuarios
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            two_factor_secret TEXT,
            two_factor_enabled BOOLEAN DEFAULT 0,
            backup_codes TEXT,
            role TEXT DEFAULT 'user',
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabla de logs de acceso
    c.execute('''
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            ip_address TEXT,
            user_agent TEXT,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN,
            two_factor_used BOOLEAN,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Crear usuario admin por defecto si no existe
    c.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if c.fetchone()[0] == 0:
        password_hash = generate_password_hash('Admin123!')
        two_factor_secret = pyotp.random_base32()
        c.execute('''
            INSERT INTO users (username, email, password_hash, two_factor_secret, two_factor_enabled, role)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('admin', 'admin@iuvade.com', password_hash, two_factor_secret, 0, 'admin'))
    
    conn.commit()
    conn.close()

# Modelo de Usuario para Flask-Login
class User(UserMixin):
    def __init__(self, id, username, email, role, two_factor_secret=None, two_factor_enabled=False):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.two_factor_secret = two_factor_secret
        self.two_factor_enabled = two_factor_enabled
    
    def get_id(self):
        return str(self.id)
    
    def verify_password(self, password_hash, password):
        return check_password_hash(password_hash, password)
    
    def verify_totp(self, token):
        if not self.two_factor_secret:
            return False
        totp = pyotp.TOTP(self.two_factor_secret)
        return totp.verify(token, valid_window=1)
    
    def generate_backup_codes(self):
        """Generar códigos de respaldo"""
        codes = [secrets.token_hex(4).upper() for _ in range(8)]
        return codes

@login_manager.user_loader
def load_user(user_id):
    """Cargar usuario desde la base de datos"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, username, email, role, two_factor_secret, two_factor_enabled FROM users WHERE id = ?', (user_id,))
    user_data = c.fetchone()
    conn.close()
    
    if user_data:
        return User(*user_data)
    return None

def log_login_attempt(user_id, success, two_factor=False):
    """Registrar intento de login"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO login_logs (user_id, ip_address, user_agent, success, two_factor_used)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, request.remote_addr, request.user_agent.string, success, two_factor))
    conn.commit()
    conn.close()

# Middleware para verificar autenticación
def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# RUTAS DE AUTENTICACIÓN

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Página de login principal"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Buscar usuario en la base de datos
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, username, email, password_hash, role, two_factor_secret, two_factor_enabled FROM users WHERE username = ? AND is_active = 1', (username,))
        user_data = c.fetchone()
        conn.close()
        
        if user_data:
            user = User(*user_data[:6])
            if user.verify_password(user_data[3], password):
                # Verificar si requiere 2FA
                if user.two_factor_enabled and user.two_factor_secret:
                    # Guardar información temporal en sesión
                    session['2fa_user_id'] = user.id
                    session['2fa_remember'] = 'remember' in request.form
                    
                    # Registrar log exitoso
                    log_login_attempt(user.id, True, False)
                    
                    return redirect(url_for('two_factor'))
                else:
                    # Login directo (sin 2FA)
                    login_user(user, remember='remember' in request.form)
                    
                    # Registrar log exitoso
                    log_login_attempt(user.id, True, False)
                    
                    flash('¡Bienvenido al sistema IUVADE!', 'success')
                    return redirect(url_for('dashboard'))
            else:
                log_login_attempt(user_data[0] if user_data else None, False, False)
                flash('Usuario o contraseña incorrectos', 'danger')
        else:
            flash('Usuario o contraseña incorrectos', 'danger')
    
    return render_template('login.html')

@app.route('/two-factor', methods=['GET', 'POST'])
def two_factor():
    """Verificación de dos factores"""
    if '2fa_user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['2fa_user_id']
    user = load_user(user_id)
    
    if not user:
        session.pop('2fa_user_id', None)
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        token = request.form.get('totp_code', '').strip()
        
        if len(token) != 6 or not token.isdigit():
            flash('Por favor ingresa un código de 6 dígitos válido', 'warning')
        elif user.verify_totp(token):
            # Login exitoso con 2FA
            login_user(user, remember=session.get('2fa_remember', False))
            
            # Registrar log exitoso con 2FA
            log_login_attempt(user.id, True, True)
            
            # Limpiar sesión temporal
            session.pop('2fa_user_id', None)
            session.pop('2fa_remember', None)
            
            flash('¡Autenticación exitosa!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Código de verificación incorrecto', 'danger')
    
    # Generar QR para la primera vez si no hay secreto
    if not user.two_factor_secret:
        user.two_factor_secret = pyotp.random_base32()
        
        # Actualizar en la base de datos
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('UPDATE users SET two_factor_secret = ? WHERE id = ?', 
                 (user.two_factor_secret, user.id))
        conn.commit()
        conn.close()
    
    # Generar QR Code
    totp = pyotp.TOTP(user.two_factor_secret)
    uri = totp.provisioning_uri(user.email or user.username, issuer_name="IUVADE System")
    
    # Crear QR Code
    img = qrcode.make(uri)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    qr_code = base64.b64encode(buffered.getvalue()).decode()
    
    return render_template('two_factor.html', qr_code=qr_code, user=user)

@app.route('/setup-2fa', methods=['GET', 'POST'])
@login_required
def setup_2fa():
    """Configurar autenticación de dos factores"""
    if request.method == 'POST':
        enable = request.form.get('enable_2fa') == 'yes'
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        if enable and not current_user.two_factor_secret:
            # Generar nuevo secreto
            new_secret = pyotp.random_base32()
            c.execute('UPDATE users SET two_factor_secret = ?, two_factor_enabled = 1 WHERE id = ?',
                     (new_secret, current_user.id))
            flash('2FA habilitado correctamente', 'success')
        elif not enable:
            c.execute('UPDATE users SET two_factor_enabled = 0 WHERE id = ?', (current_user.id,))
            flash('2FA deshabilitado correctamente', 'info')
        
        conn.commit()
        conn.close()
        
        return redirect(url_for('dashboard'))
    
    return render_template('setup_2fa.html')

@app.route('/logout')
@login_required
def logout():
    """Cerrar sesión"""
    logout_user()
    flash('Has cerrado sesión correctamente', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard principal - redirige a tu aplicación actual"""
    return render_template("template.html")  # Tu template actual

# TUS RUTAS ORIGINALES (protegidas con autenticación)

def generate_smart_context(data):
    """Genera contexto inteligente basado en los datos del JSON"""
    try:
        # Extraer información clave
        text_parts = []
        for key, value in data.items():
            if not key.startswith('_'):
                text_parts.append(f"{key}: {value}")
        
        full_text = " ".join(text_parts)
        
        # Generar embedding y análisis
        embedding = model.encode(full_text)
        
        # Crear contexto descriptivo
        field_count = len([k for k in data.keys() if not k.startswith('_')])
        context = f"Este documento contiene {field_count} campos. "
        
        # Detectar tipo de documento
        keys_lower = [k.lower() for k in data.keys()]
        
        if any(k in keys_lower for k in ['nombre', 'name', 'apellido']):
            context += "Parece ser información de una persona o cliente. "
        elif any(k in keys_lower for k in ['producto', 'product', 'precio', 'price']):
            context += "Parece ser información de un producto o servicio. "
        elif any(k in keys_lower for k in ['fecha', 'date', 'hora', 'time']):
            context += "Contiene información temporal o de eventos. "
        else:
            context += "Documento con información general. "
        
        context += f"Campos principales: {', '.join(list(data.keys())[:5])}"
        
        return context
    except Exception as e:
        return "No se pudo generar contexto automático."

@app.route('/')
def index():
    """Ruta principal - redirige al login"""
    return redirect(url_for('login'))

# Todas tus rutas API originales, ahora protegidas
@app.route('/api/databases')
@login_required
def get_databases():
    try:
        response = requests.get(
            f"{COUCHDB_URL}/_all_dbs",
            auth=(COUCHDB_USER, COUCHDB_PASS)
        )
        databases = response.json()
        databases = [db for db in databases if not db.startswith('_')]
        return jsonify(databases)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<db_name>')
@login_required
def get_documents(db_name):
    try:
        response = requests.get(
            f"{COUCHDB_URL}/{db_name}/_all_docs",
            params={"include_docs": "true"},
            auth=(COUCHDB_USER, COUCHDB_PASS)
        )
        
        data = response.json()
        documents = [row['doc'] for row in data.get('rows', []) if 'doc' in row]
        
        return jsonify({"documents": documents})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-context', methods=['POST'])
@login_required
def generate_context():
    try:
        doc_data = request.json
        context = generate_smart_context(doc_data)
        return jsonify({"context": context})
    except Exception as e:
        return jsonify({"context": "No se pudo generar contexto automático"}), 200

@app.route('/api/generate-embedding', methods=['POST'])
@login_required
def generate_embedding():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No se proporcionó texto"}), 400
        
        # Generar embedding con miniLM
        embedding = model.encode(text)
        embedding_list = embedding.tolist()
        
        return jsonify({
            "embedding": embedding_list,
            "dimension": len(embedding_list),
            "text_length": len(text)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/move-document', methods=['POST'])
@login_required
def move_document():
    try:
        data = request.json
        source_db = data.get('source_db')
        target_db = data.get('target_db')
        doc_id = data.get('doc_id')
        doc_rev = data.get('doc_rev')
        
        # Obtener el documento completo
        get_response = requests.get(
            f"{COUCHDB_URL}/{source_db}/{doc_id}",
            auth=(COUCHDB_USER, COUCHDB_PASS)
        )
        
        if not get_response.ok:
            return jsonify({"error": "No se pudo obtener el documento"}), 400
        
        doc_data = get_response.json()
        
        # Eliminar campos de sistema
        doc_data_clean = {k: v for k, v in doc_data.items() if not k.startswith('_')}
        
        # Crear en la base de datos destino
        create_response = requests.post(
            f"{COUCHDB_URL}/{target_db}",
            json=doc_data_clean,
            auth=(COUCHDB_USER, COUCHDB_PASS)
        )
        
        if not create_response.ok:
            return jsonify({"error": "No se pudo crear el documento en la base de datos destino"}), 400
        
        # Eliminar del origen
        delete_response = requests.delete(
            f"{COUCHDB_URL}/{source_db}/{doc_id}",
            params={"rev": doc_rev},
            auth=(COUCHDB_USER, COUCHDB_PASS)
        )
        
        if not delete_response.ok:
            return jsonify({"error": "El documento se copió pero no se pudo eliminar del origen"}), 400
        
        return jsonify({
            "success": True,
            "message": "Documento movido exitosamente",
            "new_id": create_response.json().get('id')
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/document/<db_name>', methods=['POST'])
@login_required
def create_document(db_name):
    try:
        doc_data = request.json
        
        response = requests.post(
            f"{COUCHDB_URL}/{db_name}",
            json=doc_data,
            auth=(COUCHDB_USER, COUCHDB_PASS)
        )
        
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/document/<db_name>/<doc_id>', methods=['PUT'])
@login_required
def update_document(db_name, doc_id):
    try:
        doc_data = request.json
        
        response = requests.put(
            f"{COUCHDB_URL}/{db_name}/{doc_id}",
            json=doc_data,
            auth=(COUCHDB_USER, COUCHDB_PASS)
        )
        
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/document/<db_name>/<doc_id>', methods=['DELETE'])
@login_required
def delete_document(db_name, doc_id):
    try:
        rev = request.args.get('rev')
        
        response = requests.delete(
            f"{COUCHDB_URL}/{db_name}/{doc_id}",
            params={"rev": rev},
            auth=(COUCHDB_USER, COUCHDB_PASS)
        )
        
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Página de administración de usuarios (solo admin)
@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('No tienes permisos para acceder a esta página', 'danger')
        return redirect(url_for('dashboard'))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, username, email, role, two_factor_enabled, created_at FROM users ORDER BY id')
    users = c.fetchall()
    conn.close()
    
    return render_template('admin_users.html', users=users)

# API para crear nuevos usuarios (solo admin)
@app.route('/api/create-user', methods=['POST'])
@login_required
def create_user():
    if current_user.role != 'admin':
        return jsonify({"error": "No autorizado"}), 403
    
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not username or not password:
        return jsonify({"error": "Username y password requeridos"}), 400
    
    password_hash = generate_password_hash(password)
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        c.execute('''
            INSERT INTO users (username, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, role))
        conn.commit()
        user_id = c.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "El usuario ya existe"}), 400
    finally:
        conn.close()
    
    return jsonify({"success": True, "user_id": user_id})

if __name__ == '__main__':
    # Inicializar base de datos
    init_db()
    
    # Ejecutar aplicación
    app.run(debug=True, host='0.0.0.0', port=5020)