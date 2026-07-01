import sqlite3
import datetime
import os
from typing import Optional, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "access_management.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            nin TEXT PRIMARY KEY,
            french_name TEXT,
            arabic_name TEXT,
            category TEXT,
            created_at TIMESTAMP
        )
    ''')
    
    # Create Access Logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nin TEXT,
            action TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY(nin) REFERENCES users(nin) ON UPDATE CASCADE ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user(nin: str, french_name: str, arabic_name: str, category: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now()
    cursor.execute(
        "INSERT INTO users (nin, french_name, arabic_name, category, created_at) VALUES (?, ?, ?, ?, ?)",
        (nin, french_name, arabic_name, category, now)
    )
    conn.commit()
    conn.close()

def get_user(nin: str) -> Optional[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE nin = ?", (nin,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_users() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def log_access(nin: str, action: str):
    """action should be 'CHECK_IN' or 'CHECK_OUT'"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now()
    cursor.execute(
        "INSERT INTO access_logs (nin, action, timestamp) VALUES (?, ?, ?)",
        (nin, action, now)
    )
    conn.commit()
    conn.close()

def get_last_action(nin: str) -> Optional[str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT action FROM access_logs WHERE nin = ? ORDER BY timestamp DESC LIMIT 1",
        (nin,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_total_entries_today() -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cursor.execute(
        "SELECT COUNT(*) FROM access_logs WHERE action = 'CHECK_IN' AND timestamp >= ?",
        (today_start,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_user_logs(nin: str) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM access_logs WHERE nin = ? ORDER BY timestamp DESC", (nin,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Initialize database if it doesn't exist
if not os.path.exists(DB_PATH):
    init_db()
else:
    # Always ensure tables exist
    init_db()

def update_user(old_nin: str, new_nin: str, french_name: str, arabic_name: str, category: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute(
        "UPDATE users SET nin = ?, french_name = ?, arabic_name = ?, category = ? WHERE nin = ?",
        (new_nin, french_name, arabic_name, category, old_nin)
    )
    conn.commit()
    conn.close()

def delete_user(nin: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("DELETE FROM users WHERE nin = ?", (nin,))
    conn.commit()
    conn.close()

def delete_log(log_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM access_logs WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()
