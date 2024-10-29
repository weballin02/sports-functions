# utils/auth.py

import bcrypt

def hash_password(plain_password):
    """
    Hashes a plaintext password.
    
    Parameters:
    - plain_password (str): The user's password.
    
    Returns:
    - str: The hashed password.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def check_password(plain_password, hashed_password):
    """
    Checks if a plaintext password matches the hashed password.
    
    Parameters:
    - plain_password (str): The user's input password.
    - hashed_password (str): The stored hashed password.
    
    Returns:
    - bool: True if matches, False otherwise.
    """
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
