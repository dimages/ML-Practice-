from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from pydantic import BaseModel, Field
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from enum import Enum
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Annotated
from fastapi import Depends
from database import Base


AuthFormData = Annotated[OAuth2PasswordRequestForm, Depends()]

# Определение модели User
class UserModel(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    money = Column(Float, default=300)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None


class UserInDB(User):
    hashed_password: str


class UserCreate(BaseModel):
    username: str
    password: str
    email: str


class Model(Base):
    __tablename__ = "model"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    cost = Column(Float)

    
    
class Prediction(Base):
    __tablename__ = "prediction"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("model.id"))
    user_id = Column(Integer, ForeignKey("user.id"), index=True)
    input_data = Column(String, nullable=False)
    output_data = Column(String, nullable=True)

    

class ModelRepository:
    @staticmethod
    def get_model_by_name(name: str, session: Session) -> Model | None:
        query = select(Model).where(Model.name == name)
        result = (session.scalars(query)).first()
        return result

    @staticmethod
    def get_model_by_id(model_id: int, session: Session) -> Model | None:
        query = select(Model).where(Model.id == model_id)
        result = (session.scalars(query)).first()
        return result

    @staticmethod
    def get_all_models(session: Session) -> list[Model] | None:
        query = select(Model)
        result = (session.scalars(query)).all()
        return result  # type: ignore[return-value]
    

class PredictionRepository:
    @staticmethod
    def get_predictions_by_user_id(user_id: int, session: Session) -> list[Prediction] | None:
        query = select(Prediction).where(Prediction.user_id == user_id)
        result = (session.scalars(query)).all()
        return result  # type: ignore[return-value]

    @staticmethod
    def create_predictions(
        user_id: int, model_id: int, data: list[str], session: Session, result: int
    ) -> list[Prediction]:
        predictions = [
            Prediction(user_id=user_id, model_id=model_id, input_data=prediction_data, output_data=result) for prediction_data in data
        ]
        session.add_all(predictions)
        session.commit()
        return predictions




class LoginResponse(BaseModel):
    access_token: str
    token_type: str


class SingUpRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=50)


class UserScheme(BaseModel):
    username: str
    balance: float


class PredictionItem(BaseModel):
    id: int
    predicted_model_id: int
    result: str | None
    input_data: str


class PredictionScheme(BaseModel):
    predictions: list[PredictionItem]


class ModelScheme(BaseModel):
    id: int
    name: str
    cost: float


class ModelListScheme(BaseModel):
    models: list[ModelScheme]



class RequestPrediction(BaseModel):
    data: list[str] = Field(..., min_length=1)


category_mapping = {
    2615: "CPUs",
    2617: "Digital Cameras",
    2619: "Dishwashers",
    2621: "Freezers",
    2622: "Fridge Freezers",
    2623: "Fridges",
    2618: "Microwaves",
    2612: "Mobile Phones",
    2614: "TVs",
    2620: "Washing Machines"
}


# class AvailableModels(str, Enum):
#     base = "base"
#     logreg_tfidf = "logreg_tfidf"
#     catboost = "catboost"

