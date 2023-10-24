import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv
from sqlalchemy import Boolean
from passlib.context import CryptContext

# from app.logging_config import setup_logger


# Environment and DB setup
load_dotenv(
    os.getenv("ENV_PATH", "/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot/API/.env")
)  # Make sure this path is correct
DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./test.db"
)  # Provide a fallback if the env variable is missing
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
DB_DIR = os.getenv("DB_DIR", "/home/ubuntu/falcon/kubeflow/llm-falcon-chatbot")
DB_SERVICE_URL = os.getenv(
    "DB_SERVICE_URL"
)  # Make sure this is used somewhere in your application


db_path = os.path.join(DB_DIR, "test.db")

# Check if the database file exists
if not os.path.exists(db_path):
    Base = declarative_base()
    engine = create_engine(DATABASE_URL)

DATABASE_URL = f"sqlite:///{db_path}"
# SQLAlchemy base class
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    prompt_token_number = Column(Integer, default=0)
    gen_token_number = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    token_limit = Column(Integer, default=1000)
    role = Column(String, default="User")


# Conversation model
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    conversation_number = Column(Integer)  # Add conversation number column
    content = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    conversation_name = Column(String)


# Create database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


# Create tables if they don't exist
# Database session generator
class Database:
    def __init__(self):
        self.db = SessionLocal()

    def add_conversation(self, input: dict):
        # Check if the user exists by username
        user = self.db.query(User).filter(User.username == input["username"]).first()
        # Retrieve the user object with the new ID

        # Calculate the conversation number for the user
        conversation_number = (
            self.db.query(Conversation).filter(Conversation.user_id == user.id).count()
            + 1
        )

        # Add the conversation
        conversation = Conversation(
            user_id=user.id,
            conversation_number=conversation_number,
            content=input["content"],
            conversation_name=input["conversation_name"],  # Save the name of the conversation
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.close()
        return {"message": "Conversation added"}

    def update_conversation(self, input):
        user = self.db.query(User).filter(User.username == input["username"]).first()

        if not user:
            self.db.close()
            return {f"User {input['username']} not found"}

        # Check if conversation_number is provided in the input
        if "conversation_number" not in input or not input["conversation_number"]:
            # Get the conversation with the highest number for the user
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.user_id == user.id)
                .order_by(Conversation.conversation_number.desc())
                .first()
            )
            # If there's no conversation for the user, create a new one
            if not conversation:
                # Calculate the conversation number for the user
                conversation_number = (
                    self.db.query(Conversation).filter(Conversation.user_id == user.id).count()
                    + 1
                )
                # Add the new conversation
                conversation = Conversation(
                    user_id=user.id,
                    conversation_number=conversation_number,
                    content=input["content"],
                    conversation_name=input.get("conversation_name", ""),  # Use the provided name or an empty string
                )
                self.db.add(conversation)
                self.db.commit()
                self.db.close()
                return {"message": "New conversation added"}
            # Otherwise, continue to update the existing conversation
        else:
            # Fetch the specified conversation using the provided number
            conversation = (
                self.db.query(Conversation)
                .filter(
                    Conversation.conversation_number == input["conversation_number"],
                    Conversation.user_id == user.id,
                )
                .first()
            )

            if not conversation:
                self.db.close()
                return {f"Conversation not found"}

        # Update conversation content
        conversation.content = input["content"]

        # Increment token numbers and check limits
        if input.get("prompt_token_number"):
            user.prompt_token_number += input["prompt_token_number"]

        if input.get("gen_token_number"):
            user.gen_token_number += input["gen_token_number"]

            # Disable user if gen_token_number exceeds token_limit
            if user.gen_token_number > user.token_limit:
                user.disabled = True
                user.token_limit = 0

        self.db.commit()
        self.db.close()
        return {"message": "Conversation updated"}

    def retrieve_conversation(self, input):
        user = self.db.query(User).filter(User.username == input["username"]).first()
        if not user:
            self.db.close()
            return {"error": f"User {input['username']} not found"}

        # Check if conversation_number is provided in the input
        if "conversation_number" in input and input["conversation_number"]:
            conversation = (
                self.db.query(Conversation)
                .filter(
                    Conversation.conversation_number == input["conversation_number"],
                    Conversation.user_id == user.id,
                )
                .first()
            )
        else:
            # Get the latest conversation for the user
            conversation = (
                self.db.query(Conversation)
                .filter(Conversation.user_id == user.id)
                .order_by(Conversation.timestamp.desc())
                .first()
            )

        if not conversation:
            self.db.close()
            return {"error": "Conversation not found"}

        self.db.close()
        return {
            "user_id": conversation.user_id,
            "conversation_id": conversation.conversation_number,
            "content": conversation.content,
            "timestamp": conversation.timestamp,
            "conversation_name": conversation.conversation_name,
        }
