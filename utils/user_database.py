# utils/user_database.py

import sqlite3
import os

# Define the path to the SQLite database
DB_PATH = os.path.join(os.getcwd(), 'data', 'users.db')

def initialize_database():
    """
    Initializes the SQLite database with users and models tables.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Create models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            model_data TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_user(username, hashed_password):
    """
    Adds a new user to the database.
    
    Parameters:
    - username (str): The desired username.
    - hashed_password (str): The hashed password.
    
    Returns:
    - bool: True if user added successfully, False otherwise.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists

def verify_user(username, hashed_password):
    """
    Verifies a user's credentials.
    
    Parameters:
    - username (str): The username.
    - hashed_password (str): The hashed password.
    
    Returns:
    - bool: True if credentials are valid, False otherwise.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        stored_password = result[0]
        return stored_password == hashed_password
    return False

def get_user_id(username):
    """
    Retrieves the user ID for a given username.
    
    Parameters:
    - username (str): The username.
    
    Returns:
    - int or None: The user ID if found, else None.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

def save_model(user_id, model_name, model_data):
    """
    Saves a custom model for a user.
    
    Parameters:
    - user_id (int): The user's ID.
    - model_name (str): The name of the model.
    - model_data (str): Serialized model data (e.g., JSON string).
    
    Returns:
    - bool: True if saved successfully, False otherwise.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO models (user_id, model_name, model_data) VALUES (?, ?, ?)',
                       (user_id, model_name, model_data))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def load_models(user_id):
    """
    Loads all saved models for a user.
    
    Parameters:
    - user_id (int): The user's ID.
    
    Returns:
    - list of tuples: Each tuple contains (model_name, model_data).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT model_name, model_data FROM models WHERE user_id = ?', (user_id,))
    results = cursor.fetchall()
    conn.close()
    return results

def load_model_by_name(user_id, model_name):
    """
    Loads a specific model by name for a user.
    
    Parameters:
    - user_id (int): The user's ID.
    - model_name (str): The name of the model.
    
    Returns:
    - str or None: The model data if found, else None.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT model_data FROM models WHERE user_id = ? AND model_name = ?', (user_id, model_name))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None
