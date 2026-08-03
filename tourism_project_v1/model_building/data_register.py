

import pandas as pd
import sys
from pathlib import Path


def register_data():

    file_path = Path("tourism_project_v1/data/tourism.csv")

    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")

    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        sys.exit(1)


    expected_columns = [
        'CustomerID',
        'ProdTaken',
        'Age',
        'TypeofContact',
        'CityTier',
        'Occupation',
        'Gender',
        'NumberOfPersonVisiting',
        'PreferredPropertyStar',
        'MaritalStatus',
        'NumberOfTrips',
        'Passport',
        'OwnCar',
        'NumberOfChildrenVisiting',
        'Designation',
        'MonthlyIncome',
        'PitchSatisfactionScore',
        'ProductPitched',
        'NumberOfFollowups',
        'DurationOfPitch'
    ]


    # Handle unnamed leftover index columns if they exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    missing_cols = [col for col in expected_columns if col not in df.columns]

    if missing_cols:
        print(f"Validation Error: Missing columns: {missing_cols}")
        sys.exit(1)


    print("Column validation passed!")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Columns:")
    print(df.columns.tolist())

    print("\nDataset Summary:")
    print(df.describe(include="all"))


    return df


if __name__ == "__main__":
    register_data()
