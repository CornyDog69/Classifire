# train_traditional_models.py
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

# Flatten images for traditional ML models
# Shape: (samples, height, width, channels) -> (samples, height*width*channels)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"Flattened shape: {X_train_flat.shape}")

# 1. Train SVM
print("\n1. Training SVM...")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_flat, y_train)
svm_pred = svm_model.predict(X_test_flat)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
joblib.dump(svm_model, 'models/svm_model.pkl')

# 2. Train Random Forest
print("\n2. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train)
rf_pred = rf_model.predict(X_test_flat)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
joblib.dump(rf_model, 'models/rf_model.pkl')

# 3. Train Logistic Regression
print("\n3. Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_flat, y_train)
lr_pred = lr_model.predict(X_test_flat)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
joblib.dump(lr_model, 'models/lr_model.pkl')

# 4. K-Means Clustering (Unsupervised)
print("\n4. Training K-Means...")
kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(X_train_flat)
joblib.dump(kmeans_model, 'models/kmeans_model.pkl')
print("K-Means trained and saved")

print("\n✓ All traditional models trained and saved!")