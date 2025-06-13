import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ------------------- 1. Chargement et équilibrage -------------------
df = pd.read_csv("DrDoS_DNS.csv", low_memory=False)
df[' Label'] = df[' Label'].apply(lambda x: 'BENIGN' if x.strip() == 'BENIGN' else 'ATTACK')

df_attack = df[df[' Label'] == 'ATTACK']
df_benign = df[df[' Label'] == 'BENIGN']

df_attack_downsampled = resample(df_attack,
                                 replace=False,
                                 n_samples=len(df_benign),
                                 random_state=42)

df_balanced = pd.concat([df_attack_downsampled, df_benign])


print("Données équilibrées :")
print(df_balanced[' Label'].value_counts())
print("-" * 30)

# ------------------- 2. Préparation des données -------------------
X = df_balanced.drop(' Label', axis=1)
y = df_balanced[' Label']
X = X.select_dtypes(include=np.number)
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Supprimer les lignes avec NaN
combined = pd.concat([X, y], axis=1)
combined.dropna(inplace=True)
X_clean = combined.drop(' Label', axis=1)
y_clean = combined[' Label']

print(f"Forme de X après nettoyage : {X_clean.shape}")
print(f"Forme de y après nettoyage : {y_clean.shape}")
print("-" * 30)

# ------------------- 3. Division Train/Test -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
)

# ------------------- 4. Pipeline (imputation + standardisation + modèle) -------------------
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ------------------- 5. Entraînement -------------------
pipeline.fit(X_train, y_train)
print("Le modèle s'est entraîné avec succès !")
print("-" * 30)

joblib.dump(pipeline, 'DDosDetectModel.pkl')
print("Modèle sauvegardé dans 'mon_modele_rf.pkl'")

# ------------------- 6. Évaluation -------------------
y_pred = pipeline.predict(X_test)

print("Évaluation du modèle sur l'ensemble de test :")
print("\nMatrice de Confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de Classification :")
print(classification_report(y_test, y_pred))


