from dotenv import load_dotenv
load_dotenv()

from agent.agent import Agent
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///principles.db")
Session = sessionmaker(bind=engine)

if __name__ == '__main__':
    from agent.services.principle import PrincipleService
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    
    # Create a new session for the PrincipleService
    with Session() as session:
        principle_service = PrincipleService(session=session)

        # Create the Agent
        agent = Agent(principle_service=principle_service, llm=llm)

        # Run your queries
        #query_add_book = "add book the making of a manager"
        #agent.ask(query_add_book)
        query_generate = "generate actions task for me to test in my next meeting"
        agent.ask(query_generate)