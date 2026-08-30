import os
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression


print("=" * 70)
print("SAGEMAKER TRAINING STARTED")
print("=" * 70)


# ---------------------------------------------------------
# 1. Get SageMaker training input directory
# ---------------------------------------------------------

train_dir = os.environ.get(
    "SM_CHANNEL_TRAIN",
    "/opt/ml/input/data/train"
)

print("Training directory:")
print(train_dir)


# ---------------------------------------------------------
# 2. Training file
# ---------------------------------------------------------

train_file = os.path.join(
    train_dir,
    "train.csv"
)

print("Training file:")
print(train_file)


# ---------------------------------------------------------
# 3. Load training data
# ---------------------------------------------------------

df = pd.read_csv(train_file)

print("Training data shape:")
print(df.shape)

print("Training data:")
print(df)


# ---------------------------------------------------------
# 4. Select features
# ---------------------------------------------------------

X = df[
    [
        "area_sqft",
        "bedrooms",
        "age_years"
    ]
]


# ---------------------------------------------------------
# 5. Select target
# ---------------------------------------------------------

y = df["price"]


# ---------------------------------------------------------
# 6. Create model
# ---------------------------------------------------------

model = LinearRegression()


# ---------------------------------------------------------
# 7. Train model
# ---------------------------------------------------------

print("Starting model training...")

model.fit(X, y)

print("Model training completed.")


# ---------------------------------------------------------
# 8. Display model information
# ---------------------------------------------------------

print("Model coefficients:")
print(model.coef_)

print("Model intercept:")
print(model.intercept_)


# ---------------------------------------------------------
# 9. Get SageMaker model directory
# ---------------------------------------------------------

model_dir = os.environ.get(
    "SM_MODEL_DIR",
    "/opt/ml/model"
)

print("Model directory:")
print(model_dir)


# ---------------------------------------------------------
# 10. Create model directory
# ---------------------------------------------------------

os.makedirs(
    model_dir,
    exist_ok=True
)


# ---------------------------------------------------------
# 11. Save model
# ---------------------------------------------------------

model_path = os.path.join(
    model_dir,
    "model.joblib"
)

joblib.dump(
    model,
    model_path
)

print("Model saved to:")
print(model_path)


# ---------------------------------------------------------
# 12. Verify model file
# ---------------------------------------------------------

print("Files inside model directory:")

for file in os.listdir(model_dir):
    print(file)


print("=" * 70)
print("SAGEMAKER TRAINING COMPLETED")
print("=" * 70)