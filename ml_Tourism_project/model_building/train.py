
# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# for model serialization
import joblib
import os

# HuggingFace
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi()

# -----------------------------
# Load Dataset from HuggingFace
# -----------------------------
Xtrain_path = "hf://datasets/sudhakar2087/predicting-customer-purchases/Xtrain.csv"
Xtest_path = "hf://datasets/sudhakar2087/predicting-customer-purchases/Xtest.csv"
ytrain_path = "hf://datasets/sudhakar2087/predicting-customer-purchases/ytrain.csv"
ytest_path = "hf://datasets/sudhakar2087/predicting-customer-purchases/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).values.ravel()
ytest = pd.read_csv(ytest_path).values.ravel()

print("Dataset Loaded Successfully")

# -----------------------------
# Numeric Columns
# -----------------------------
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)

# -----------------------------
# Handle Class Imbalance
# -----------------------------
class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# -----------------------------
# XGBoost Model
# -----------------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric='logloss'
)

# -----------------------------
# Hyperparameter Grid
# -----------------------------
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__learning_rate': [0.05, 0.1]
}

# -----------------------------
# Pipeline
# -----------------------------
model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Grid Search
# -----------------------------
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

print("Training Started...")
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

print("Best Params:", grid_search.best_params_)

# -----------------------------
# Evaluation
# -----------------------------
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

print("\nTraining Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Report:")
print(classification_report(ytest, y_pred_test))

# -----------------------------
# Save Model
# -----------------------------
model_file = "best_tourism_model_v1.joblib"
joblib.dump(best_model, model_file)

print("Model Saved Successfully")

# -----------------------------
# Upload Model to HuggingFace
# -----------------------------
repo_id = "sudhakar2087/predicting-customer-purchases"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating Repo '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo=model_file,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Model Uploaded Successfully")
