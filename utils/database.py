# utils/database.py

import os
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the database URL
DATABASE_URL = "sqlite:///./models.db"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class
Base = declarative_base()

# Define the Model table with the corrected column name
class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    model_data = Column(LargeBinary, nullable=False)
    model_metadata = Column(Text, nullable=True)  # Renamed from 'metadata' to 'model_metadata'

def init_db():
    """
    Initializes the database by creating tables.
    """
    Base.metadata.create_all(bind=engine)

def save_model(user_id, model_name, model_data, model_metadata):
    """
    Saves a trained model to the database.
    
    Parameters:
    - user_id (int): ID of the user (not used in this example).
    - model_name (str): Name of the model.
    - model_data (bytes): Serialized model data.
    - model_metadata (str): Description or metadata of the model.
    """
    session = SessionLocal()
    existing_model = session.query(Model).filter(Model.name == model_name).first()
    if existing_model:
        # Update existing model
        existing_model.model_data = model_data
        existing_model.model_metadata = model_metadata
    else:
        # Create new model entry
        new_model = Model(name=model_name, model_data=model_data, model_metadata=model_metadata)
        session.add(new_model)
    session.commit()
    session.close()

def get_saved_models(user_id):
    """
    Retrieves all saved models for a user.
    
    Parameters:
    - user_id (int): ID of the user (not used in this example).
    
    Returns:
    - list of tuples: List containing (model_name, model_metadata).
    """
    session = SessionLocal()
    models = session.query(Model.name, Model.model_metadata).all()
    session.close()
    return models

def load_model(user_id, model_name):
    """
    Loads a model from the database.
    
    Parameters:
    - user_id (int): ID of the user (not used in this example).
    - model_name (str): Name of the model to load.
    
    Returns:
    - tuple: (model object, scaler object)
    """
    import pickle
    session = SessionLocal()
    model_entry = session.query(Model).filter(Model.name == model_name).first()
    session.close()
    if model_entry:
        try:
            loaded_model, loaded_scaler = pickle.loads(model_entry.model_data)
            return loaded_model, loaded_scaler
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            return None, None
    else:
        return None, None
