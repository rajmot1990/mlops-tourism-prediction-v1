

import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_data():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "tourism.csv"
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

    # Clean data
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.drop(columns=["CustomerID"], errors="ignore")
    df = df.dropna()

    # Split features and target
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y 
    )

    # Save to data folder (to be uploaded as artifacts by GitHub Actions)
    X_train.to_csv(base_dir / "data" / "Xtrain.csv", index=False)
    X_test.to_csv(base_dir / "data" / "Xtest.csv", index=False)
    y_train.to_csv(base_dir / "data" / "ytrain.csv", index=False)
    y_test.to_csv(base_dir / "data" / "ytest.csv", index=False)

    print("Data preparation completed. Train/test splits saved.")

if __name__ == "__main__":
    prepare_data()
