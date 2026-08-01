
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    base_dir = Path(__file__).resolve().parent.parent
    
    # Load workflow artifacts
    X_train = pd.read_csv(base_dir / "data" / "Xtrain.csv")
    X_test = pd.read_csv(base_dir / "data" / "Xtest.csv")
    y_train = pd.read_csv(base_dir / "data" / "ytrain.csv")["ProdTaken"]
    y_test = pd.read_csv(base_dir / "data" / "ytest.csv")["ProdTaken"]

    # Identify columns by type
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Combine preprocessing with Random Forest Model
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define hyperparameters for tuning
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
    }

    print("Tuning model...")
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters and metrics
    experiment_log = {
        "best_parameters": grid_search.best_params_,
        "test_accuracy": accuracy
    }
    
    with open(base_dir / "model_building" / "experiment_log.json", "w") as f:
        json.dump(experiment_log, f, indent=4)

    print(f"Model trained. Accuracy: {accuracy:.4f}")
    
    # Save the best model to the root of the project for deployment
    joblib.dump(best_model, base_dir / "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_model()
