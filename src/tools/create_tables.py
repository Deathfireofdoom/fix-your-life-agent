from sqlalchemy import create_engine
from agent.models import Base

# Create the SQLite engine
engine = create_engine("sqlite:///principles.db")

# Create all tables
Base.metadata.create_all(engine)
print("Database tables created.")