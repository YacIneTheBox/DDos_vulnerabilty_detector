import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Charger le modèle
pipeline = joblib.load('DDosDetectModel.pkl')

# Charger le fichier de test
df_new = pd.read_csv("DrDoS_LDAP.csv", low_memory=False)

# 👉 Convertir les labels (ATTACK ou BENIGN)
df_new[' Label'] = df_new[' Label'].apply(lambda x: 'BENIGN' if x.strip() == 'BENIGN' else 'ATTACK')

# Préparer les données numériques
X_new = df_new.select_dtypes(include=np.number)
X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
X_new.dropna(inplace=True)

# Correspondance des labels (même lignes que X_new après nettoyage)
y_new = df_new.loc[X_new.index, ' Label']

# Prédictions
y_new_pred = pipeline.predict(X_new)

# Résultats
print("Prédictions sur le nouveau jeu :")
print(y_new_pred)

print("\nMatrice de confusion :")
print(confusion_matrix(y_new, y_new_pred))

print("\nRapport de classification :")
print(classification_report(y_new, y_new_pred, zero_division=0))
