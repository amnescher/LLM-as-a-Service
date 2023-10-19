from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List, Dict
from typing import Optional
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DATABASE_URL = "sqlite:///./test.db"

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    prompt_token_number = Column(Integer, default=0)  
    gen_token_number = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    hashed_password = Column(String)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    conversation_number = Column(Integer)  # Add conversation number column
    content = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    conversation_name = Column(String)


class Input(BaseModel):
    username: Optional[str]
    password: Optional[str]
    content: Optional[str]
    conversation_number: Optional[int]
    conversation_name: Optional[str] 
    user_id: Optional[int]
    prompt_token_number: Optional[int]
    gen_token_number: Optional[int]


app = FastAPI()

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


@app.post("/add_user/")
def add_user(input: Input):
    db = SessionLocal()
    hashed_password = pwd_context.hash(input.password) 
    user = User(username=input.username, hashed_password=hashed_password,prompt_token_number=0, gen_token_number=0) 
    
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return {"user_id": user.id}


@app.post("/add_conversation/")
def add_conversation(input: Input):
    db = SessionLocal()

    # Check if the user exists by username
    user = db.query(User).filter(User.username == input.username).first()
    if not user:
        # If the user doesn't exist, add it using the add_user function
        user_id = add_user(input)
        user = db.query(User).get(
            user_id["user_id"]
        )  # Retrieve the user object with the new ID
    else:
        user_id = user.id

    # Calculate the conversation number for the user
    conversation_number = (
        db.query(Conversation).filter(Conversation.user_id == user.id).count() + 1
    )

    # Add the conversation
    conversation = Conversation(
        user_id=user.id,
        conversation_number=conversation_number,
        content=input.content,
        conversation_name=input.conversation_name  # Save the name of the conversation
    )
    db.add(conversation)
    db.commit()
    db.close()

    return {"message": "Conversation added"}


@app.delete("/delete_user/")
def delete_user(input: Input):
    db = SessionLocal()
    user = db.query(User).filter(User.username == input.username).first()
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

    db.query(Conversation).filter(Conversation.user_id == user.id).delete()
    db.delete(user)
    db.commit()
    db.close()

    return {"message": "User and related content deleted"}


@app.delete("/delete_conversation/")
def delete_conversation(input: Input):
    db = SessionLocal()

    user = db.query(User).filter(User.username == input.username).first()
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.conversation_number == input.conversation_number,
            Conversation.user_id == user.id,
        )
        .first()
    )

    if not conversation:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()
    db.close()
    return {"message": "Conversation deleted"}


@app.get("/get_all_data/")
def get_all_data():
    db = SessionLocal()
    users = db.query(User).all()
    data = []

    for user in users:
        user_data = {
            "user_id": user.id,
            "username": user.username,
            "prompt_token_number": user.prompt_token_number,
            "gen_token_number": user.gen_token_number,
            "conversations": [],
        }

        conversations = (
            db.query(Conversation)
            .filter(Conversation.user_id == user.id)
            .order_by(Conversation.conversation_number)  # Order by conversation number
            .all()
        )
        for conversation in conversations:
            user_data["conversations"].append(
                {
                    "conversation_number": conversation.conversation_number,
                    "content": conversation.content,
                    "timestamp": conversation.timestamp,
                }
            )

        data.append(user_data)

    db.close()
    return data



@app.get("/check_user_existence/")
def check_user_existence(input: Input):
    db = SessionLocal()
    user = db.query(User).filter(User.username == input.username).first()
    db.close()

    if user:
        return {"user_exists": True, "hashed_password": user.hashed_password}  # Return the hashed password
    else:
        return {"user_exists": False}


@app.post("/retrieve_conversation/")
def retrieve_conversation(input: Input):
    db = SessionLocal()
    user = db.query(User).filter(User.username == input.username).first()
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.conversation_number == input.conversation_number,
            Conversation.user_id == user.id,
        )
        .first()
    )

    if not conversation:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.close()
    return {
        "user_id": conversation.user_id,
        "conversation_id": conversation.conversation_number,
        "content": conversation.content,
        "timestamp": conversation.timestamp,
        "conversation_name": conversation.conversation_name,
    }


@app.get("/retrieve_latest_conversation/")
def retrieve_latest_conversation(input: Input):
    db = SessionLocal()

    user = db.query(User).filter(User.username == input.username).first()
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

    conversation = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.timestamp.desc())
        .first()
    )

    db.close()

    if conversation:
        return {
            "user_id": conversation.user_id,
            "conversation_number": conversation.conversation_number,
            "content": conversation.content,
            "timestamp": conversation.timestamp,
        }
    else:
        return {"message": "No conversations found for the user"}


@app.post("/update_conversation/")
def update_conversation(input: Input):
    db = SessionLocal()
    user = db.query(User).filter(User.username == input.username).first()
    
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.conversation_number == input.conversation_number,
            Conversation.user_id == user.id,
        )
        .first()
    )

    if not conversation:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Update conversation content
    conversation.content = input.content

    # Increment prompt_token_number and gen_token_number if provided in the input
    if input.prompt_token_number is not None:
        user.prompt_token_number += input.prompt_token_number

    if input.gen_token_number is not None:
        user.gen_token_number += input.gen_token_number

    db.commit()
    db.close()

    return {"message": "Conversation updated"}

@app.post("/update_conversation_name/")
def update_conversation_name(input: Input):
    db = SessionLocal()

    # Check if the user exists by username
    user = db.query(User).filter(User.username == input.username).first()
    
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

    # Find the conversation for the user with the given conversation_number
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.conversation_number == input.conversation_number,
            Conversation.user_id == user.id,
        )
        .first()
    )

    if not conversation:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Update the name of the conversation
    conversation.conversation_name = input.conversation_name
    db.commit()
    db.close()

    return {"message": "Conversation name updated successfully"}



@app.get("/retrieve_all_conversations/")
def retrieve_all_conversations(input: Input):
    db = SessionLocal()

    # Check if the user exists by username
    user = db.query(User).filter(User.username == input.username).first()

    if not user:
        db.close()
        return None  # User not found, return None

    # Retrieve all conversations for the user
    conversations = db.query(Conversation).filter(Conversation.user_id == user.id).all()

    # Create a dictionary to store conversations (key: conversation_number, value: conversation_content)
    db.close()

    return len(conversations)
@app.get("/get_user_conversations/")
def get_user_conversations(input: Input):
    db = SessionLocal()

    # Check if the user exists by username
    user = db.query(User).filter(User.username == input.username).first()
    if not user:
        db.close()
        raise HTTPException(status_code=404, detail="User not found")

    # Retrieve conversation numbers and names for the user
    conversations = (
        db.query(Conversation.conversation_number, Conversation.conversation_name)
        .filter(Conversation.user_id == user.id)
        .all()
    )

    # Extract the conversation numbers and names from the result and store them in a list of dictionaries
    conversation_data = [{"number": cnv[0], "name": cnv[1]} for cnv in conversations]

    db.close()
    return {"conversations": conversation_data}


# Define other endpoints similarly
# Remember to handle exceptions and error cases

if __name__ == "__main__":
    import uvicorn

    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=5000)