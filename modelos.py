import joblib
from pycaret.classification import load_model
import pandas as pd

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


x = modelos()
df_rf = pd.DataFrame([{
  "Age": 21,
  "FCVC": 2,
  "NCP": 3,
  "CH2O": 2,
  "FAF": 0,
  "TUE": 1,
  "Gender_Male": 0,
  "family_history_yes": 1,
  "FAVC_yes": 0,
  "CAEC_Frequently": 0,
  "CAEC_Sometimes": 1,
  "CAEC_no": 0,
  "SMOKE_yes": 0,
  "SCC_yes": 0,
  "CALC_Frequently": 0,
  "CALC_Sometimes": 0,
  "CALC_no": 1,
  "MTRANS_Bike": 0,
  "MTRANS_Motorbike": 0,
  "MTRANS_Public_Transportation": 1,
  "MTRANS_Walking": 0,
  "IMC": 24.39
}])

df_raw = pd.DataFrame([{
    "Gender": "Female",
    "Age": 21,
    "Height": 1.62,
    "Weight": 64,
    "family_history": "yes",
    "FAVC": "no",
    "FCVC": 2,
    "NCP": 3,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2,
    "SCC": "no",
    "FAF": 0,
    "TUE": 1,
    "CALC": "no",
    "MTRANS": "Public_Transportation"
}])

print(x.predcit_rf(df_rf))
print(x.predict_pipeline(df_raw))
print(x.predict_pycaret(df_raw))