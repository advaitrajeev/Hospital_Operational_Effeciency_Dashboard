import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import os

def train_readmission_model():
    print("Training Readmission Prediction Model...")
    df = pd.read_csv('data/patients.csv')
    
    df['department'] = df['department'].astype('category')
    df['condition'] = df['condition'].astype('category')
    
    # For readmission, length of stay might be a feature along with age, dept, condition
    X = df[['age', 'department', 'condition', 'length_of_stay']]
    y = df['readmitted_30d']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.1,
        'eval_metric': 'auc'
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    preds_proba = model.predict(dtest)
    preds = (preds_proba > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, preds_proba)
    acc = accuracy_score(y_test, preds)
    
    print(f"Readmission Model AUC: {auc:.2f}")
    print(f"Readmission Model Accuracy: {acc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    
    os.makedirs('models/saved', exist_ok=True)
    model.save_model('models/saved/readmission_model.json')
    print("Model saved to models/saved/readmission_model.json")

if __name__ == '__main__':
    train_readmission_model()
