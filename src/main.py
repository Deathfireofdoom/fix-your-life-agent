from dotenv import load_dotenv
load_dotenv()

from agent.principle_agent import PrincipleAgent
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///principles.db")
Session = sessionmaker(bind=engine)

if __name__ == '__main__':
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    
    # Create a new session for the PrincipleService
    with Session() as session:
        agent = PrincipleAgent(llm, session)
        agent.ask("add book the lean startup")
