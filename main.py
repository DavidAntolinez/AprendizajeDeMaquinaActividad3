from fastapi import FastAPI
from pycaret.classification import load_model
import pandas as pd
import sys
import joblib

def add_imc_and_delete(X):
    X = X.copy()
    X['IMC'] = X['Weight'] / (X['Height'] ** 2)
    X = X.drop(columns=['Height', 'Weight'])
    return X

class modelos:
    
    def __init__(self):
        self.rf_model = joblib.load('RandomForest_model.pkl')
        self.pipeline = joblib.load('pipeline_completa.pkl')
        self.pycaret_model = load_model('pycaret_model')
        self.le = joblib.load('label_encoder.pkl')
        return

    def predcit_rf(self, df):
        return self.le.inverse_transform(self.rf_model.predict(df))[0]
    
    def predict_pipeline(self, df):
        return self.le.inverse_transform(self.pipeline.predict(df))[0]

    def predict_pycaret(self, df):
        return self.pycaret_model.predict(df)[0]

sys.modules['__main__'].add_imc_and_delete = add_imc_and_delete

app = FastAPI()
model = modelos()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict_rf")
def predict(
    Age: int,
    FCVC: float,
    NCP: float,
    CH2O: float,
    FAF: float,
    TUE: float,
    Gender_Male: int,
    family_history_yes: int,
    FAVC_yes: int,
    CAEC_Frequently: int,
    CAEC_Sometimes: int,
    CAEC_no: int,
    SMOKE_yes: int,
    SCC_yes: int,
    CALC_Frequently: int,
    CALC_Sometimes: int,
    CALC_no: int,
    MTRANS_Bike: int,
    MTRANS_Motorbike: int,
    MTRANS_Public_Transportation: int,
    MTRANS_Walking: int,
    IMC: float
):
    df = pd.DataFrame([{
        "Age": Age,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "Gender_Male": Gender_Male,
        "family_history_yes": family_history_yes,
        "FAVC_yes": FAVC_yes,
        "CAEC_Frequently": CAEC_Frequently,
        "CAEC_Sometimes": CAEC_Sometimes,
        "CAEC_no": CAEC_no,
        "SMOKE_yes": SMOKE_yes,
        "SCC_yes": SCC_yes,
        "CALC_Frequently": CALC_Frequently,
        "CALC_Sometimes": CALC_Sometimes,
        "CALC_no": CALC_no,
        "MTRANS_Bike": MTRANS_Bike,
        "MTRANS_Motorbike": MTRANS_Motorbike,
        "MTRANS_Public_Transportation": MTRANS_Public_Transportation,
        "MTRANS_Walking": MTRANS_Walking,
        "IMC": IMC
    }])

    prediccion = model.predcit_rf(df)

    return {"prediccion": prediccion.tolist()}

@app.get("/predict_pipeline")
def predict_pipeline(
    Gender: str,
    Age: int,
    Height: float,
    Weight: float,
    family_history: str,
    FAVC: str,
    FCVC: float,
    NCP: float,
    CAEC: str,
    SMOKE: str,
    CH2O: float,
    SCC: str,
    FAF: float,
    TUE: float,
    CALC: str,
    MTRANS: str
):
    df = pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "family_history": family_history,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "SCC": SCC,
        "FAF": FAF,
        "TUE": TUE,
        "CALC": CALC,
        "MTRANS": MTRANS
    }])

    prediccion = model.predict_pipeline(df)

    return {"prediccion": prediccion}

@app.get("/predict_pycaret")
def predict_pipeline(
    Gender: str,
    Age: int,
    Height: float,
    Weight: float,
    family_history: str,
    FAVC: str,
    FCVC: float,
    NCP: float,
    CAEC: str,
    SMOKE: str,
    CH2O: float,
    SCC: str,
    FAF: float,
    TUE: float,
    CALC: str,
    MTRANS: str
):
    df = pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "family_history": family_history,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "SCC": SCC,
        "FAF": FAF,
        "TUE": TUE,
        "CALC": CALC,
        "MTRANS": MTRANS
    }])

    prediccion = model.predict_pycaret(df)

    return {"prediccion": prediccion}