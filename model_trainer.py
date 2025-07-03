import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from feature_extractor import extract_features

def create_model():
    """Create stacked ensemble model"""
    base_models = [
        ('xgb', XGBClassifier(
            n_estimators=410,
            max_depth=12,
            learning_rate=0.1,
            reg_lambda=1.0,
            eval_metric='logloss'
        )),
        ('svm', SVC(
            C=0.5, 
            kernel='rbf', 
            probability=True,
            random_state=42
        ))
    ]
    
    return StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(
            penalty='l2', 
            C=0.5, 
            solver='liblinear',
            random_state=42
        ),
        cv=5,
        stack_method='predict_proba'
    )

def train_and_save(X_train, y_train, output_dir='model'):
    """Full training pipeline"""
    # Feature extraction
    print("Extracting features...")
    train_features = [extract_features(signal) for signal in X_train]
    X = pd.DataFrame(train_features)
    
    # Create preprocessing pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', create_model())
    ])
    
    # Cross-validation
    print("Cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_train)):
        X_train_fold, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val = y_train[train_idx], y_train[val_idx]
        
        pipeline.fit(X_train_fold, y_train_fold)
        y_pred = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        cv_scores.append(auc)
        print(f"Fold {fold+1} AUC: {auc:.4f}")
    
    print(f"Mean CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Final training
    print("Training final model...")
    pipeline.fit(X, y_train)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_dir, 'ldl_classifier.pkl'))
    print(f"Model saved to {output_dir}/ldl_classifier.pkl")
    
    return pipeline