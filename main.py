from datetime import datetime, timedelta, timezone
from typing import Optional, Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import UserModel, Base, User, Token, UserCreate, UserInDB, TokenData, Prediction
from database import get_db
from database import Base, engine
from models import ModelRepository
from models import ModelListScheme, ModelScheme, PredictionScheme, PredictionItem, PredictionRepository, RequestPrediction
from fastapi.responses import JSONResponse
import pandas as pd
import queue
from sqlalchemy import update
from tasks import svc_model_predict, rf_model_predict, catboost_model_predict
from models import Model
from models import category_mapping



SECRET_KEY = "a521eb7a9518c46e1b76c2f3c71019f5608c123835ed916d495bd9ae77d783d8"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


Base.metadata.create_all(bind=engine)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db: Session, username: str):
    return db.query(UserModel).filter(UserModel.username == username).first()


# Измененная функция для аутентификации пользователя
def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def subtract_money(user_id: int, cost: float, session: Session) -> None:
    session.execute(update(UserModel).where(UserModel.id == user_id).values(money=UserModel.money - cost))
    session.commit()


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    return current_user


def create_models():
    rf_model = Model(id = 1, name = 'rf_model', cost=70)
    svc_model = Model(id = 2, name ='svc_model', cost=100)
    catboost_model = Model(id = 3, name = 'catboost_model', cost=130)
    with SessionLocal() as db:
        db.add(rf_model)
        db.add(svc_model)
        db.add(catboost_model)
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            print(e)  


create_models()


# Эта функция заменяет предыдущую функцию create_user
def create_user_db(db: Session, user_data: UserCreate):
    hashed_password = get_password_hash(user_data.password)
    db_user = UserModel(username=user_data.username, hashed_password=hashed_password, email=user_data.email) # Используйте UserModel здесь
    db.add(db_user)
    try:
        db.commit()
        return db_user
    except Exception as e:
        db.rollback()
        print(e)
        raise HTTPException(status_code=409, detail="User could not be created") from e
    

# Эндпоинт для регистрации пользователя
@app.post("/users/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    create_user_db(db, user)
    user = authenticate_user(db, user.username, user.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


# Измененный эндпоинт для получения токена
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")



@app.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return current_user


@app.get("/get_models")
async def get_models(session: Session = Depends(get_db)) -> ModelListScheme:
    models = ModelRepository.get_all_models(session)
    if models is None:
        return ModelListScheme(models=[])
    return ModelListScheme(
        models=[
            ModelScheme(
                id=model.id,
                name=model.name,
                cost=model.cost,
            )
            for model in models
        ]
    )


@app.get("/get_predictions")
async def get_user_predictions(user: Annotated[User, Depends(get_current_user)], session: Session = Depends(get_db)):
    predictions = PredictionRepository.get_predictions_by_user_id(user.id, session)
    return [PredictionItem(
                id=prediction.id,
                predicted_model_id=prediction.model_id,
                input_data=prediction.input_data,
                result=prediction.output_data,
            )
            for prediction in predictions
        ]


def check_user_balance(user_id: int, model_cost: float, session: Session):
    user = session.query(UserModel).filter(UserModel.id == user_id).first()
    if user.money < model_cost:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient funds. You need at least {model_cost} to use this model."
        )



@app.post("/predict")
async def predict(
    model_name: str, data: RequestPrediction, user: Annotated[User, Depends(get_current_user)], session: Session = Depends(get_db)
):
    model = ModelRepository.get_model_by_name(model_name, session)
    if model is None:
        raise HTTPException(status_code=400, detail="Model not found")

    check_user_balance(user.id, model.cost, session)

    try:
        subtract_money(user.id, model.cost, session)
        df = pd.DataFrame(data.data, columns=[" Cluster Label"])
        category_id = -1 
        if 'rf_model' == model_name:
            category_id = rf_model_predict(df)[0].item() # Преобразование numpy.int64 в int
        elif 'svc_model' == model_name:
            category_id = svc_model_predict(df)[0].item() # Аналогичное преобразование
        elif 'catboost_model' == model_name:
            category_id = catboost_model_predict(df).item() # Проверьте, что функция возвращает numpy.int64

        category_name = category_mapping.get(category_id, "Unknown Category")
        predictions = PredictionRepository.create_predictions(user.id, model.id, data.data, session, category_name)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"category_id": category_id, "category_name": category_name}




def increment_user_money(session: Session, username: str, amount: float):
    user = session.query(UserModel).filter(UserModel.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.money += amount
    session.commit()
    return user


@app.post("/increment_money")
async def add_money(
    amount: float,
    session: Session = Depends(get_db),
    user: UserModel = Depends(get_current_user),
):
    updated_user = increment_user_money(session, user.username, amount)
    return {
        "id": updated_user.id,
        "username": updated_user.username,
        "money": updated_user.money
    }


@app.get("/get_balance")
async def get_balance(user: UserModel = Depends(get_current_user), session: Session = Depends(get_db)):
    # Тут мы предполагаем, что функция get_current_user возвращает экземпляр UserModel текущего пользователя
    return {"balance": user.money}


CurrentUser = Annotated[User, Depends(get_current_user)]



