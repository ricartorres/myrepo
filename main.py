from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
#import pandas as pd

class ApiInput(BaseModel):
    time: int  # El parámetro de entrada (por ejemplo, el valor de 'time')
    ejection_fraction: float
    serum_creatinine: float

class ApiOutput(BaseModel):
    is_insuficiencia: int  # Resultado de la predicción (si hay insuficiencia)
    #accuracy: float  # Precisión del modelo

app = FastAPI()  # Creamos la aplicación FastAPI

# Cargar el modelo previamente entrenado
model = joblib.load("model.joblib")

# Cargar el dataset desde el repositorio de GitHub
#url = "https://raw.githubusercontent.com/username/repository/branch/insuficiencia_cardiaca.csv"  # Reemplaza con tu URL de GitHub
#df_insuficiencia_cardiaca = pd.read_csv(url)

#Features = ['time']
#x = df_insuficiencia_cardiaca[Features]
#y = df_insuficiencia_cardiaca["DEATH_EVENT"]

# División en entrenamiento y prueba
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Calcular la precisión (accuracy) sobre el conjunto de prueba
#xgb_pred = model.predict(x_test)
#accuracy = accuracy_score(y_test, xgb_pred)

# Guardar la precisión en el estado de la aplicación
#app.state.model_accuracy = accuracy

@app.post("/insuficiencia", response_model=ApiOutput)
async def define_insuficiencia(data: ApiInput) -> ApiOutput:
    # Convertir los datos de entrada a una forma que pueda predecir el modelo
    #features = np.array([data['time']]).reshape(1, -1)  # Ajustar las dimensiones del array para que sea compatible con el modelo
    ###features = np.array([data.time]).reshape(1, -1)
    features = np.array([data.time, data.ejection_fraction, data.serum_creatinine]).reshape(1, -1)

    # Hacer la predicción
    prediction = model.predict(features)#[0]
    #prediction = 30
    #return prediction
    return ApiOutput(is_insuficiencia=int(prediction))
    # Retornar la predicción y la precisión (accuracy) del modelo
    #return ApiOutput(
    #    is_insuficiencia=int(prediction)
        #is_insuficiencia=int(prediction[0])#,  # Convertir la predicción a un entero (0 o 1)
    #)
