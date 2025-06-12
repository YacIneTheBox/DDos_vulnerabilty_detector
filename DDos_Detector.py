import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Charger une partie des données
# Remarque : Votre nom de fichier "DrDoS_DNS.csv" est utilisé tel quel.
# Le nom de la colonne ' Label' avec un espace au début est également conservé.
df = pd.read_csv("DrDoS_DNS.csv", low_memory=False)

# ------------------- 1. Équilibrage des classes (inchangé) -------------------
df_majority = df[df[' Label']=='DrDoS_DNS']
df_minority = df[df[' Label']=='BENIGN']

df_majority_downsampled = resample(df_majority,
                                 replace=False,
                                 n_samples=len(df_minority),
                                 random_state=42)

df_balanced = pd.concat([df_majority_downsampled, df_minority])

print("Données équilibrées :")
print(df_balanced[' Label'].value_counts())
print("-" * 30)


# ------------------- 2. Préparation et Nettoyage AVANT la division -------------------

# Séparer les features (X) et la cible (y)
X = df_balanced.drop(' Label', axis=1)
y = df_balanced[' Label']

# Sélectionner uniquement les colonnes numériques
# Votre code le faisait plusieurs fois, une seule fois suffit.
X = X.select_dtypes(include=np.number)

# Remplacer les valeurs infinies (inf) par des valeurs manquantes (NaN)
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# ERREUR CORRIGÉE : Le nettoyage doit se faire en gardant X et y synchronisés.
# Nous allons supprimer les lignes contenant des NaN dans X et les lignes correspondantes dans y.
# On sauvegarde les index de y avant de supprimer les lignes dans X
y = y[X.index] # S'assure que y a les mêmes index que X avant de filtrer

# Supprimer les lignes avec des valeurs NaN de X et y en même temps
# La méthode `dropna()` sur un DataFrame supprime la ligne entière.
# En l'appliquant au DataFrame combiné, on garantit la synchronisation.
combined = pd.concat([X, y], axis=1)
combined.dropna(inplace=True)

# Recréer X et y propres
X_clean = combined.drop(' Label', axis=1)
y_clean = combined[' Label']

# Vérification finale
print(f"Forme de X après nettoyage : {X_clean.shape}")
print(f"Forme de y après nettoyage : {y_clean.shape}")
assert len(X_clean) == len(y_clean), "Erreur : X et y ont des tailles différentes !"
print("-" * 30)


# ------------------- 3. Division Train/Test -------------------
# On utilise les données nettoyées (X_clean, y_clean)
# Ajout de random_state pour la reproductibilité
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean)


# ------------------- 4. Normalisation (Standardisation) -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------- 5. Entraînement du Modèle -------------------
# Ajout de random_state pour la reproductibilité
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

print("Le modèle s'est entraîné avec succès !")
print("-" * 30)


# ------------------- 6. Évaluation (Bonus) -------------------
# Les métriques étaient importées mais non utilisées, voici comment les utiliser.
print("Évaluation du modèle sur l'ensemble de test :")
y_pred = model.predict(X_test_scaled)

print("\nMatrice de Confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de Classification :")
print(classification_report(y_test, y_pred))