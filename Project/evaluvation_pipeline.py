import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from joblib import dump, load
from sklearn.metrics import r2_score

def evaluvation_pipeline(x_test_path,y_test_path,model):

    #Read data
    X=pd.read_csv(x_test_path)
    y_true=pd.read_csv(y_test_path)
    
    #seperate numerical columns
    
    numerical_cols=X.select_dtypes(exclude='object')
    
    
    # numerical-- Scalling
    model_scaling=load('standard_scaler.pkl')
    scaled_data= model_scaling.transform(numerical_cols)
    scaled_data=pd.DataFrame(scaled_data,columns=numerical_cols.columns)


    # model testing
    lr=load(model)
    y_pred=pd.DataFrame(lr.predict(scaled_data))
    test_score=r2_score(y_true, y_pred)*100
    
    return y_pred,test_score

