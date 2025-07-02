import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from data_loader import load_data
from feature_extractor import extract_features

def generate_submission(model_path, data_dir, output_file='submit.csv'):
    """Generate submission file"""
    # Load model
    pipeline = joblib.load(model_path)
    
    # Load data
    _, _, test_ids = load_data(data_dir)
    X_test = [np.load(os.path.join(data_dir, "ppgs", f"{id}.npy")) for id in test_ids]
    
    # Feature extraction
    test_features = [extract_features(signal) for signal in X_test]
    X_test_df = pd.DataFrame(test_features)
    
    # Predict probabilities
    probas = pipeline.predict_proba(X_test_df)[:, 1]
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'ЛПНП': probas
    })
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

# For validation
def validate_model(model_path, X_val, y_val):
    """Validate on holdout set"""
    pipeline = joblib.load(model_path)
    val_features = [extract_features(signal) for signal in X_val]
    probas = pipeline.predict_proba(pd.DataFrame(val_features))[:, 1]
    return roc_auc_score(y_val, probas)

if __name__ == "__main__":
    DATA_DIR = '/path/to/your/data'
    MODEL_PATH = 'model/ldl_classifier.pkl'
    
    # For validation
    # X_train, y_train, _ = load_data(DATA_DIR)
    # X_train, X_val, y_train, y_val = train_test_split(...)
    # auc = validate_model(MODEL_PATH, X_val, y_val)
    
    generate_submission(MODEL_PATH, DATA_DIR)