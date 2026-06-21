import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def train_alos_model():
    print("Training ALOS Prediction Model...")
    df = pd.read_csv('data/patients.csv')
    
    # Feature Engineering
    df['department'] = df['department'].astype('category')
    df['condition'] = df['condition'].astype('category')
    
    X = df[['age', 'department', 'condition']]
    y = df['length_of_stay']
    
    # Create DMatrix with enable_categorical
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.1,
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Evaluate
    preds = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"ALOS Model RMSE: {rmse:.2f}")
    print(f"ALOS Model R2: {r2:.2f}")
    
    os.makedirs('models/saved', exist_ok=True)
    model.save_model('models/saved/alos_model.json')
    print("Model saved to models/saved/alos_model.json")

if __name__ == '__main__':
    train_alos_model()
