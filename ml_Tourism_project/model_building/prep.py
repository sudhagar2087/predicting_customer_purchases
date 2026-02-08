
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/sudhagar2087/predicting-customer-purchases/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column
df.drop(columns=['CustomerID'], inplace=True)

# -----------------------------
# Encode categorical columns
# -----------------------------
label_encoder = LabelEncoder()

categorical_cols = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# -----------------------------
# Define target variable
# -----------------------------
target_col = 'ProdTaken'   # Customer purchased or not

# Split into features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# Train Test Split
# -----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save split data
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# -----------------------------
# Upload files to HuggingFace
# -----------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="sudhagar2087/predicting-customer-purchases",
        repo_type="dataset",
    )
