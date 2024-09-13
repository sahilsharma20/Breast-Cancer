import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib

# Load dataset
url = 'Breast_Cancer_data.csv'
data = pd.read_csv(url)

# Preprocessing
X = data.drop(columns=['id', 'Unnamed: 32', 'diagnosis'])
y = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Selection
# RFE
estimator = RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)
selected_features_rfe = X.columns[selector.support_]

# PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Mutual Information
mi = mutual_info_classif(X_train, y_train)
mi_scores = pd.Series(mi, index=X.columns)
top_features_mi = mi_scores.nlargest(10).index

# Build models
models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    if hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc:.2f}')
    print()

# Save best model (assuming Random Forest has the highest accuracy for simplicity)
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'model/rf_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
