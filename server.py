from flask import Flask, request, jsonify, session, redirect, url_for
import sqlite3
from datetime import datetime
import hashlib
import math
import os
import smtplib
import json
import time
import urllib.parse
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from authlib.integrations.flask_client import OAuth
from werkzeug.exceptions import HTTPException

app = Flask(__name__, 
            static_folder=os.path.dirname(__file__),
            static_url_path='')
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ─── EMAIL CONFIG ──────────────────────────────────────────────────────────────
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", 587))
EMAIL_USER = os.environ.get("EMAIL_USER", "your_gmail@gmail.com")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "your_app_password")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "HydroSense <your_gmail@gmail.com>")
ALERT_THRESHOLD = float(os.environ.get("ALERT_THRESHOLD", 8.0))
ML_MIN_TRAIN_SAMPLES = int(os.environ.get("ML_MIN_TRAIN_SAMPLES", 20))
ML_LEARNING_RATE = float(os.environ.get("ML_LEARNING_RATE", 0.05))
ML_TRAIN_EPOCHS = int(os.environ.get("ML_TRAIN_EPOCHS", 250))
ML_MAX_TRAIN_ROWS = int(os.environ.get("ML_MAX_TRAIN_ROWS", 1500))
ML_TRAIN_MAX_SECONDS = float(os.environ.get("ML_TRAIN_MAX_SECONDS", 2.0))
ML_MODEL_PATH = os.environ.get("ML_MODEL_PATH", "relay_ml_model.json")
ML_STAGE_CLASSES = ["NORMAL FLOW", "LOW WATER ALERT", "DRY RUN ALERT"]

# ─── GOOGLE OAUTH CONFIG ───────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "524761068730-mu8649j0089sqvekrpltg52jlp5b6jor.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET")

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# ─── GLOBAL STATE ──────────────────────────────────────────────────────────────
relay_state = "OFF"
current_liters = 0.0
current_flow_rate = 0.0
current_temperature = None
current_amps = None
current_stage = "SYSTEM READY"
current_ai_stage = "SYSTEM READY"
last_update_time = None
last_relay_state = "OFF"
last_alert_time = None
manual_override = False
auto_decision_reason = "startup"
ml_model = None
ml_model_ready = False

def try_load_ml_model():
    global ml_model, ml_model_ready, auto_decision_reason

    if not os.path.exists(ML_MODEL_PATH):
        return False

    try:
        with open(ML_MODEL_PATH, "r", encoding="utf-8") as model_file:
            data = json.load(model_file)
    except (OSError, json.JSONDecodeError) as exc:
        ml_model = None
        ml_model_ready = False
        auto_decision_reason = f"ML model load failed: {exc}"
        return False

    required_keys = ["weights", "bias", "means", "scales"]
    if not all(key in data for key in required_keys):
        ml_model = None
        ml_model_ready = False
        auto_decision_reason = "ML model file missing required fields"
        return False

    try:
        class_count = len(data["weights"])
        feature_count = len(data["weights"][0]) if class_count else 0
        if class_count != len(ML_STAGE_CLASSES) or feature_count == 0:
            raise ValueError("model shape mismatch")
        if len(data["bias"]) != class_count:
            raise ValueError("bias length mismatch")
        if len(data["means"]) != feature_count or len(data["scales"]) != feature_count:
            raise ValueError("feature stats length mismatch")
    except (TypeError, ValueError) as exc:
        ml_model = None
        ml_model_ready = False
        auto_decision_reason = f"ML model invalid: {exc}"
        return False

    ml_model = data
    ml_model_ready = True
    auto_decision_reason = f"ML model loaded from {ML_MODEL_PATH}"
    return True

def _stage_to_code(stage_text):
    if not isinstance(stage_text, str):
        return 0.0
    text = stage_text.upper()
    if "LOW WATER" in text or "LOW FLOW" in text:
        return 1.0
    if "DRY RUN" in text or "DRY" in text:
        return 2.0
    if "NORMAL" in text or "STARTING" in text or "WAIT SIGNAL" in text:
        return 0.0
    return 0.0

def _stage_text_to_index(stage_text, relay_value=None):
    if isinstance(stage_text, str):
        text = stage_text.upper()
        if "LOW WATER" in text or "LOW FLOW" in text:
            return 1
        if "DRY RUN" in text or "DRY" in text:
            return 2
        if "NORMAL" in text:
            return 0
    if relay_value is not None and str(relay_value).upper() == "OFF":
        return 2
    return 0

def _stage_index_to_text(index):
    if index < 0 or index >= len(ML_STAGE_CLASSES):
        return ML_STAGE_CLASSES[0]
    return ML_STAGE_CLASSES[index]

def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def _sigmoid(value):
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))

def _build_ml_features(liters, flow_rate, temperature, current_value, stage_text):
    return [
        _safe_float(liters, 0.0),
        _safe_float(flow_rate, 0.0),
        _safe_float(temperature, 0.0),
        _safe_float(current_value, 0.0),
        _stage_to_code(stage_text),
    ]

def train_ml_relay_model():
    """Train a 3-class softmax model from historical logs instead of using thresholds."""
    global ml_model, ml_model_ready, auto_decision_reason

    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    try:
        c.execute(
            """SELECT liters, flow_rate, temperature, current, stage, relay
               FROM logs
               WHERE flow_rate IS NOT NULL
               ORDER BY id DESC
               LIMIT ?""",
            (ML_MAX_TRAIN_ROWS,)
        )
    except sqlite3.OperationalError as exc:
        conn.close()
        ml_model = None
        ml_model_ready = False
        auto_decision_reason = f"ML model unavailable: {exc}"
        return False

    rows = c.fetchall()
    conn.close()

    if len(rows) < ML_MIN_TRAIN_SAMPLES:
        ml_model = None
        ml_model_ready = False
        auto_decision_reason = f"ML model waiting for data: {len(rows)} samples"
        return False

    features = []
    labels = []
    for liters, flow_rate, temperature, current_value, stage_text, relay in rows:
        stage_index = _stage_text_to_index(stage_text, relay)
        features.append(_build_ml_features(liters, flow_rate, temperature, current_value, stage_text))
        labels.append(stage_index)

    if len(set(labels)) < 2:
        ml_model = None
        ml_model_ready = False
        auto_decision_reason = f"ML stage model waiting for class diversity: {len(rows)} samples"
        return False

    feature_count = len(features[0])
    means = [0.0] * feature_count
    scales = [1.0] * feature_count

    for index in range(feature_count):
        column = [row[index] for row in features]
        means[index] = sum(column) / len(column)
        variance = sum((value - means[index]) ** 2 for value in column) / max(1, len(column) - 1)
        scales[index] = math.sqrt(variance) if variance > 1e-9 else 1.0

    normalized = []
    for row in features:
        normalized.append([
            (row[index] - means[index]) / scales[index]
            for index in range(feature_count)
        ])

    class_count = len(ML_STAGE_CLASSES)
    weights = [[0.0] * feature_count for _ in range(class_count)]
    bias = [0.0] * class_count

    train_started = time.time()
    epochs_completed = 0
    for _ in range(ML_TRAIN_EPOCHS):
        if (time.time() - train_started) >= ML_TRAIN_MAX_SECONDS:
            break

        for row, label in zip(normalized, labels):
            scores = [sum(weight * value for weight, value in zip(class_weights, row)) + class_bias
                      for class_weights, class_bias in zip(weights, bias)]
            max_score = max(scores)
            exp_scores = [math.exp(score - max_score) for score in scores]
            total_score = sum(exp_scores) or 1.0
            probabilities = [value / total_score for value in exp_scores]

            for class_index in range(class_count):
                target = 1.0 if class_index == label else 0.0
                error = probabilities[class_index] - target
                for feature_index in range(feature_count):
                    weights[class_index][feature_index] -= ML_LEARNING_RATE * error * row[feature_index]
                bias[class_index] -= ML_LEARNING_RATE * error
        epochs_completed += 1

    ml_model = {
        "weights": weights,
        "bias": bias,
        "means": means,
        "scales": scales,
        "samples": len(features),
        "epochs": epochs_completed,
    }
    ml_model_ready = True
    auto_decision_reason = f"ML model trained on {len(features)} samples ({epochs_completed} epochs)"

    try:
        with open(ML_MODEL_PATH, "w", encoding="utf-8") as model_file:
            json.dump(ml_model, model_file)
    except OSError:
        pass

    return True

def predict_ml_stage(liters, flow_rate, temperature, current_value, stage_text):
    global auto_decision_reason

    if not ml_model_ready or not ml_model:
        fallback_stage = _stage_index_to_text(_stage_text_to_index(stage_text))
        auto_decision_reason = f"ML model not trained yet; fallback stage={fallback_stage}"
        return fallback_stage

    values = _build_ml_features(liters, flow_rate, temperature, current_value, stage_text)
    normalized = [
        (values[index] - ml_model["means"][index]) / ml_model["scales"][index]
        for index in range(len(values))
    ]
    scores = [sum(weight * value for weight, value in zip(class_weights, normalized)) + class_bias
              for class_weights, class_bias in zip(ml_model["weights"], ml_model["bias"])]
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total_score = sum(exp_scores) or 1.0
    probabilities = [value / total_score for value in exp_scores]
    best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
    predicted_stage = _stage_index_to_text(best_index)
    auto_decision_reason = (
        f"ML stage probabilities normal={probabilities[0]:.3f}, low={probabilities[1]:.3f}, "
        f"dry={probabilities[2]:.3f}, flow={_safe_float(flow_rate):.3f}"
    )
    return predicted_stage

# ─── DATABASE INIT ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("water.db", timeout=8)
    c = conn.cursor()
    c.execute("PRAGMA busy_timeout = 5000")
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("PRAGMA synchronous = NORMAL")
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            liters REAL,
            flow_rate REAL,
            temperature REAL,
            current REAL,
            relay TEXT,
            stage TEXT,
            time TEXT,
            user_id INTEGER
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT,
            full_name TEXT,
            email TEXT UNIQUE,
            phone TEXT,
            address TEXT,
            google_id TEXT UNIQUE,
            avatar_url TEXT,
            auth_provider TEXT DEFAULT 'local',
            created_at TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            event TEXT,
            message TEXT,
            sent_at TEXT,
            status TEXT
        )
    ''')

    # Safely add missing columns (for existing databases)
    try:
        c.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'local'")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE users ADD COLUMN google_id TEXT UNIQUE")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE logs ADD COLUMN stage TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE logs ADD COLUMN ai_stage TEXT")
    except sqlite3.OperationalError:
        pass

    for column_def in [
        ("flow_rate", "REAL"),
        ("temperature", "REAL"),
        ("current", "REAL"),
    ]:
        try:
            c.execute(f"ALTER TABLE logs ADD COLUMN {column_def[0]} {column_def[1]}")
        except sqlite3.OperationalError:
            pass

    # Default admin user
    pw = hashlib.sha256("admin123".encode()).hexdigest()
    c.execute("""INSERT OR IGNORE INTO users
        (username, password, full_name, email, auth_provider, created_at)
        VALUES (?,?,?,?,?,?)""",
        ("admin", pw, "Admin User", "admin@watermonitor.io", "local", str(datetime.now())))
    
    conn.commit()
    conn.close()

init_db()
if not try_load_ml_model():
    train_ml_relay_model()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_status():
    global last_update_time
    if last_update_time is None:
        return "OFFLINE"
    diff = (datetime.now() - last_update_time).seconds
    return "OFFLINE" if diff > 5 else "ONLINE"

# ─── EMAIL HELPERS ─────────────────────────────────────────────────────────────
def send_email(to_email, subject, html_body):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False

def log_notification(user_id, event, message, status):
    conn = None
    try:
        conn = sqlite3.connect("water.db", timeout=8)
        c = conn.cursor()
        c.execute("PRAGMA busy_timeout = 5000")
        c.execute("INSERT INTO notifications (user_id, event, message, sent_at, status) VALUES (?,?,?,?,?)",
                  (user_id, event, message, str(datetime.now()), status))
        conn.commit()
    except sqlite3.Error as exc:
        print(f"[NOTIFY DB ERROR] {exc}")
    finally:
        if conn is not None:
            conn.close()

def log_notification_for_all_users(event, message, status="sent"):
    conn = None
    try:
        conn = sqlite3.connect("water.db", timeout=8)
        c = conn.cursor()
        c.execute("PRAGMA busy_timeout = 5000")
        c.execute("SELECT id FROM users")
        user_ids = [row[0] for row in c.fetchall()]
        for user_id in user_ids:
            c.execute("INSERT INTO notifications (user_id, event, message, sent_at, status) VALUES (?,?,?,?,?)",
                      (user_id, event, message, str(datetime.now()), status))
        conn.commit()
    except sqlite3.Error as exc:
        print(f"[NOTIFY DB ERROR] {exc}")
    finally:
        if conn is not None:
            conn.close()

def _stage_event_name(stage_text):
    text = str(stage_text or "").upper()
    if "DRY" in text:
        return "stage_dry"
    if "LOW" in text:
        return "stage_low"
    if "START" in text:
        return "stage_starting"
    if "NORMAL" in text:
        return "stage_normal"
    return "stage_ready"

def notify_python_error(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    frame = tb[-1] if tb else None
    if frame:
        location = f"{os.path.basename(frame.filename)}:{frame.lineno}"
        message = f"Python error at {location} - {exc.__class__.__name__}: {exc}"
    else:
        message = f"Python error - {exc.__class__.__name__}: {exc}"
    log_notification_for_all_users("python_error", message, "failed")

def notify_all_users(event, subject, html_body):
    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    c.execute("SELECT id, email FROM users WHERE email IS NOT NULL AND email != ''")
    users = c.fetchall()
    conn.close()
    for uid, email in users:
        ok = send_email(email, subject, html_body)
        log_notification(uid, event, subject, "sent" if ok else "failed")

def email_welcome(user_email, full_name):
    subject = "Welcome to HydroSense 💧"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;background:#060a0f;color:#e8f4f8;padding:32px;border-radius:12px;">
      <h2 style="color:#00d4ff;letter-spacing:2px;">💧 HYDROSENSE</h2>
      <p>Hi <strong>{full_name}</strong>,</p>
      <p>Your account has been created successfully.</p>
      <div style="background:#0c1420;border:1px solid #1e3048;border-radius:8px;padding:16px;margin:20px 0;">
        <p style="margin:0;color:#7fa8c4;">🔐 Keep your credentials safe.<br>
        📊 Dashboard updates every 1.5 seconds.<br>
        🚨 You'll receive alerts when the relay toggles or levels spike.</p>
      </div>
      <p style="color:#4a7090;font-size:12px;">— HydroSense Monitoring System</p>
    </div>"""
    send_email(user_email, subject, html)

def email_relay_change(relay_status, liters):
    subject = f"⚡ Relay turned {relay_status} — HydroSense Alert"
    color = "#00ff88" if relay_status == "ON" else "#ff3366"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;background:#060a0f;color:#e8f4f8;padding:32px;border-radius:12px;">
      <h2 style="color:#00d4ff;">💧 HYDROSENSE — Relay Alert</h2>
      <div style="background:#0c1420;border:1px solid {color};border-radius:8px;padding:20px;text-align:center;margin:16px 0;">
        <p style="font-size:32px;margin:0;color:{color};font-weight:bold;">RELAY {relay_status}</p>
        <p style="color:#7fa8c4;margin-top:8px;">Water flow: <strong>{liters:.3f} L</strong></p>
        <p style="color:#4a7090;font-size:12px;">Triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      </div>
      <p style="color:#4a7090;font-size:12px;">— HydroSense Auto-Alert System</p>
    </div>"""
    notify_all_users(f"relay_{relay_status.lower()}", subject, html)

def email_threshold_alert(liters):
    subject = f"🚨 High Water Level Alert — {liters:.2f}L detected"
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;background:#060a0f;color:#e8f4f8;padding:32px;border-radius:12px;">
      <h2 style="color:#ff3366;">🚨 HIGH LEVEL ALERT</h2>
      <p>Water level has exceeded the alert threshold of <strong>{ALERT_THRESHOLD}L</strong>.</p>
      <div style="background:#0c1420;border:1px solid #ff3366;border-radius:8px;padding:20px;text-align:center;">
        <p style="font-size:40px;color:#ff3366;margin:0;font-weight:bold;">{liters:.2f} L</p>
        <p style="color:#7fa8c4;">Recorded at {datetime.now().strftime('%H:%M:%S')}</p>
      </div>
      <p style="color:#4a7090;font-size:12px;">— HydroSense Monitoring System</p>
    </div>"""
    notify_all_users("threshold_alert", subject, html)

# ─── GOOGLE OAUTH ROUTES ───────────────────────────────────────────────────────
@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    token = google.authorize_access_token()
    user_info = token.get('userinfo')
    if not user_info:
        return redirect('/?error=google_auth_failed')

    google_id = user_info.get('sub')
    email = user_info.get('email')
    full_name = user_info.get('name', '')
    avatar_url = user_info.get('picture', '')
    username = email.split('@')[0] if email else f"user_{google_id[:8]}"

    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    c.execute("SELECT id, username, full_name, email, phone, address FROM users WHERE google_id=? OR email=?",
              (google_id, email))
    user = c.fetchone()

    if user:
        c.execute("UPDATE users SET google_id=?, avatar_url=? WHERE id=?", (google_id, avatar_url, user[0]))
        conn.commit()
        uid = user[0]
    else:
        c.execute("""INSERT INTO users
            (username, password, full_name, email, google_id, avatar_url, auth_provider, created_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (username, None, full_name, email, google_id, avatar_url, 'google', str(datetime.now())))
        conn.commit()
        uid = c.lastrowid
        if email:
            email_welcome(email, full_name or username)

    c.execute("SELECT id, username, full_name, email, phone, address, avatar_url FROM users WHERE id=?", (uid,))
    u = c.fetchone()
    conn.close()

    session['user_id'] = u[0]
    user_dict = {
        "id": u[0], "username": u[1], "full_name": u[2] or "", "email": u[3] or "",
        "phone": u[4] or "", "address": u[5] or "", "avatar_url": u[6] or ""
    }
    user_data = urllib.parse.quote(json.dumps(user_dict))
    return redirect(f'/?oauth_user={user_data}')

# ─── AUTH ROUTES ───────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    c.execute("""SELECT id, username, full_name, email, phone, address, avatar_url
        FROM users WHERE username=? AND password=?""",
        (data.get("username"), hash_pw(data.get("password", ""))))
    user = c.fetchone()
    conn.close()
    if user:
        session['user_id'] = user[0]
        return jsonify({"success": True, "user": {
            "id": user[0], "username": user[1], "full_name": user[2],
            "email": user[3], "phone": user[4], "address": user[5],
            "avatar_url": user[6] or ""
        }})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    try:
        conn = sqlite3.connect("water.db")
        c = conn.cursor()
        c.execute("""INSERT INTO users
            (username, password, full_name, email, phone, address, auth_provider, created_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (data['username'], hash_pw(data['password']), data.get('full_name', ''),
             data.get('email', ''), data.get('phone', ''), data.get('address', ''),
             'local', str(datetime.now())))
        conn.commit()
        conn.close()
        if data.get('email'):
            email_welcome(data['email'], data.get('full_name') or data['username'])
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Username or email already exists"}), 409

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/profile', methods=['GET', 'PUT'])
def profile():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    if request.method == 'GET':
        c.execute("""SELECT id, username, full_name, email, phone, address, created_at, avatar_url, auth_provider
            FROM users WHERE id=?""", (session['user_id'],))
        u = c.fetchone()
        conn.close()
        if not u:
            return jsonify({"error": "User not found"}), 404
        return jsonify({
            "id": u[0], "username": u[1], "full_name": u[2], "email": u[3],
            "phone": u[4], "address": u[5], "created_at": u[6],
            "avatar_url": u[7] or "", "auth_provider": u[8]
        })
    else:
        data = request.get_json()
        c.execute("UPDATE users SET full_name=?, email=?, phone=?, address=? WHERE id=?",
                  (data.get('full_name'), data.get('email'), data.get('phone'),
                   data.get('address'), session['user_id']))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

# ─── NOTIFICATIONS ─────────────────────────────────────────────────────────────
@app.route('/api/notifications')
def get_notifications():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    conn = None
    try:
        conn = sqlite3.connect("water.db", timeout=8)
        c = conn.cursor()
        c.execute("PRAGMA busy_timeout = 5000")
        c.execute("SELECT event, message, sent_at, status FROM notifications WHERE user_id=? ORDER BY id DESC LIMIT 10",
                  (session['user_id'],))
        rows = [{"event": r[0], "message": r[1], "sent_at": r[2], "status": r[3]} for r in c.fetchall()]
        return jsonify(rows)
    except sqlite3.Error as exc:
        print(f"[NOTIFICATION API ERROR] {exc}")
        return jsonify([])
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/test-email', methods=['POST'])
def test_email():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    c.execute("SELECT email, full_name FROM users WHERE id=?", (session['user_id'],))
    u = c.fetchone()
    conn.close()
    if not u or not u[0]:
        return jsonify({"success": False, "message": "No email on file"}), 400
    ok = send_email(u[0], "✅ HydroSense Test Email",
                    f"<p>Hi {u[1]}, your email notifications are working!</p>")
    return jsonify({"success": ok})

@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if isinstance(exc, HTTPException):
        return exc

    try:
        notify_python_error(exc)
    except Exception as notify_exc:
        print(f"[ERROR NOTIFY FAILED] {notify_exc}")

    print(f"[UNHANDLED EXCEPTION] {exc}")
    if request.path.startswith('/api'):
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }), 500
    return jsonify({"error": "Internal server error"}), 500

# ─── ESP32 AUTO UPDATE ─────────────────────────────────────────────────────────
@app.route('/update', methods=['POST'])
def update():
    global relay_state, current_liters, current_flow_rate, current_temperature, current_amps, current_stage, current_ai_stage, last_update_time, last_relay_state, last_alert_time, manual_override

    print("Received from ESP32:", request.get_json())

    if not request.is_json:
        return jsonify({"error": "JSON required"}), 400

    data = request.get_json()
    liters = float(data.get("liters", 0.0))
    flow_rate = data.get("flow_rate")
    if flow_rate is None:
        flow_rate = data.get("flowRate")

    try:
        current_flow_rate = float(flow_rate) if flow_rate is not None else 0.0
    except (TypeError, ValueError):
        current_flow_rate = 0.0

    temperature = data.get("temperature")
    if temperature is None:
        temperature = data.get("temp")
    if temperature is None:
        temperature = data.get("temperature_c")

    try:
        current_temperature = float(temperature) if temperature is not None else None
    except (TypeError, ValueError):
        current_temperature = None

    current_value = data.get("current")
    if current_value is None:
        current_value = data.get("current_a")
    if current_value is None:
        current_value = data.get("current_amps")

    try:
        current_amps = float(current_value) if current_value is not None else None
    except (TypeError, ValueError):
        current_amps = None

    previous_stage = current_stage
    stage = data.get("stage")
    if isinstance(stage, str) and stage.strip():
        current_stage = stage.strip()

    if current_stage != previous_stage and current_stage != "SYSTEM READY":
        stage_message = f"Stage changed to {current_stage} at {current_flow_rate:.3f} L/min"
        log_notification_for_all_users(_stage_event_name(current_stage), stage_message, "sent")

    current_liters = liters
    last_update_time = datetime.now()
    log_stage = current_stage if isinstance(current_stage, str) and current_stage.strip() else "SYSTEM READY"

    ai_stage = predict_ml_stage(current_liters, current_flow_rate, current_temperature, current_amps, log_stage)
    current_ai_stage = ai_stage
    effective_relay = "OFF" if ai_stage == "DRY RUN ALERT" else "ON"
    if manual_override:
        effective_relay = relay_state

    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    c.execute("INSERT INTO logs (liters, flow_rate, temperature, current, relay, stage, ai_stage, time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (liters, current_flow_rate, current_temperature, current_amps, effective_relay, log_stage, ai_stage, str(datetime.now())))
    conn.commit()
    conn.close()

    if effective_relay != last_relay_state:
        email_relay_change(effective_relay, liters)
        last_relay_state = effective_relay

    if liters >= ALERT_THRESHOLD:
        if last_alert_time is None or (datetime.now() - last_alert_time).total_seconds() > 300:
            email_threshold_alert(liters)
            last_alert_time = datetime.now()

    relay_state = effective_relay
    mode = "MANUAL" if manual_override else "AUTO"
    print(f"Relay set to: {relay_state} ({mode}) | AI stage: {ai_stage} | Liters: {liters:.3f} | Flow: {current_flow_rate:.3f} L/min | Temp: {current_temperature} C | Current: {current_amps} A | {auto_decision_reason}")
    return jsonify({
        "relay": relay_state,
        "mode": mode,
        "flow_rate": round(current_flow_rate, 3),
        "ai_stage": ai_stage,
        "auto_reason": auto_decision_reason,
        "auto_model": "AI"
    })

# ─── MANUAL RELAY CONTROL ──────────────────────────────────────────────────────
@app.route('/api/relay', methods=['POST'])
def manual_relay():
    global relay_state, manual_override
    data = request.get_json()
    new_state = data.get("relay", "OFF").upper()

    if new_state not in ["ON", "OFF"]:
        return jsonify({"success": False, "message": "Invalid state"}), 400

    relay_state = new_state
    manual_override = True
    print(f"Manual relay command: relay={relay_state}, mode=MANUAL")
    return jsonify({"success": True, "relay": relay_state, "mode": "MANUAL"})

@app.route('/api/relay/auto', methods=['POST'])
def relay_auto_mode():
    global relay_state, manual_override, current_ai_stage
    manual_override = False
    ai_stage = predict_ml_stage(current_liters, current_flow_rate, current_temperature, current_amps, current_stage)
    current_ai_stage = ai_stage
    relay_state = "OFF" if ai_stage == "DRY RUN ALERT" else "ON"
    print(f"Relay control switched to AUTO. relay={relay_state}, ai_stage={ai_stage} | {auto_decision_reason}")
    return jsonify({"success": True, "relay": relay_state, "mode": "AUTO", "ai_stage": ai_stage})

# ─── DATA ENDPOINTS ────────────────────────────────────────────────────────────
@app.route('/api/data')
def api_data():
    return jsonify({
        "relay": relay_state,
        "liters": round(current_liters, 3),
        "flow_rate": round(current_flow_rate, 3),
        "temperature": round(current_temperature, 2) if current_temperature is not None else None,
        "current": round(current_amps, 3) if current_amps is not None else None,
        "stage": current_stage,
        "ai_stage": current_ai_stage,
        "auto_reason": auto_decision_reason,
        "status": get_status(),
        "mode": "MANUAL" if manual_override else "AUTO",
        "auto_model": "AI"
    })

@app.route('/api/history')
def history():
    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    c.execute("SELECT liters, relay, stage, time FROM logs ORDER BY id DESC LIMIT 20")
    rows = [{"liters": r[0], "relay": r[1], "stage": r[2], "time": r[3]} for r in c.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/stats')
def stats():
    conn = sqlite3.connect("water.db")
    c = conn.cursor()
    c.execute("SELECT SUM(liters), MAX(liters), COUNT(*) FROM logs")
    r = c.fetchone()
    conn.close()
    return jsonify({
        "total": r[0] or 0,
        "peak": r[1] or 0,
        "readings": r[2] or 0
    })

# ─── SERVE FRONTEND ────────────────────────────────────────────────────────────
@app.route('/')
def home():
    # Serve index.html from root directory
    index_path = os.path.join(os.path.dirname(__file__), 'index.html')
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({
            "endpoints": [
                "GET /health",
                "POST /data",
                "GET /api/latest",
                "GET /api/history?limit=120",
                "GET /api/stats",
                "GET /api/report?hours=24"
            ],
            "message": "Backend is running. Use POST /data for ESP32 uploads.",
            "ok": True,
            "service": "mineguard-backend"
        })

if __name__ == '__main__':
    app.run( host='0.0.0.0',port=5000, debug=True)