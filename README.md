Rapport de Projet : Détection d'Attaques DDoS avec un Modèle de Machine Learning

Auteur : Yacine Ait ALi Yahia
Date : 13/06/2025
Version : 1.0
1. Introduction et Objectif

Ce rapport détaille la création, l'entraînement et l'évaluation d'un modèle de classification destiné à détecter les attaques par déni de service distribué (DDoS), plus spécifiquement les attaques de type DrDoS_DNS.

L'objectif principal est de construire un pipeline de Machine Learning robuste et efficace, capable de distinguer avec une haute précision le trafic réseau légitime (BENIGN) du trafic malveillant (ATTACK). Le modèle final est sauvegardé pour une utilisation ultérieure dans des applications de sécurité réseau.
2. Méthodologie

Le projet a été développé en Python en utilisant des bibliothèques standards de science des données comme Pandas, NumPy et Scikit-learn. Le processus peut être décomposé en plusieurs étapes clés.
2.1. Chargement et Équilibrage des Données

    Source : Les données ont été chargées à partir du fichier DrDoS_DNS.csv.
    Problématique : Le jeu de données initial présentait un fort déséquilibre entre les classes ATTACK et BENIGN, avec une surreprésentation des attaques. Un tel déséquilibre peut biaiser le modèle, le rendant moins sensible à la classe minoritaire.
    Solution : Pour corriger ce biais, une technique de sous-échantillonnage (downsampling) a été appliquée. La classe majoritaire (ATTACK) a été réduite de manière aléatoire pour avoir le même nombre d'échantillons que la classe minoritaire (BENIGN).

Le résultat est un jeu de données parfaitement équilibré, comme le montre le décompte des classes :

Données équilibrées :
 Label
ATTACK    53797
BENIGN    53797
Name: count, dtype: int64

2.2. Préparation et Nettoyage des Données

Avant l'entraînement, les données ont été préparées comme suit :

    Sélection des Caractéristiques : Seules les colonnes de type numérique ont été conservées pour le modèle. La colonne Label a été utilisée comme cible (y).
    Gestion des Valeurs Infinies : Les valeurs infinies (inf et -inf), qui peuvent résulter de calculs comme des divisions par zéro, ont été remplacées par NaN (Not a Number).
    Suppression des Données Manquantes : Toutes les lignes contenant au moins une valeur NaN ont été supprimées (dropna) pour garantir la qualité et l'intégrité des données fournies au modèle.

2.3. Construction du Pipeline de Modélisation

Un Pipeline Scikit-learn a été utilisé pour enchaîner les étapes de prétraitement et de classification. Cette approche garantit que les mêmes transformations sont appliquées de manière cohérente aux données d'entraînement et de test.

Le pipeline est composé de trois étapes :

    SimpleImputer(strategy='median') : Comble les éventuelles valeurs manquantes restantes en utilisant la médiane de chaque colonne. La médiane est un choix robuste face aux valeurs aberrantes.
    StandardScaler() : Standardise les caractéristiques en supprimant la moyenne et en les mettant à l'échelle de la variance de l'unité. Cette étape est cruciale pour la performance de nombreux algorithmes, y compris le Random Forest.
    RandomForestClassifier() : Le modèle de classification choisi est un Random Forest, un algorithme d'ensemble puissant qui construit plusieurs arbres de décision et agrège leurs prédictions. Il est reconnu pour sa haute performance et sa robustesse au surapprentissage.

2.4. Entraînement et Sauvegarde du Modèle

    Division des Données : Le jeu de données nettoyé a été divisé en un ensemble d'entraînement (70%) et un ensemble de test (30%). Le paramètre stratify=y_clean a été utilisé pour s'assurer que la proportion de classes ATTACK et BENIGN soit la même dans les deux ensembles.
    Entraînement : Le pipeline complet a été entraîné sur l'ensemble d'entraînement (X_train, y_train) via la commande pipeline.fit().
    Sauvegarde : Une fois l'entraînement terminé, le modèle a été sérialisé et sauvegardé dans le fichier DDosDetectModel.pkl à l'aide de joblib. Cela permet de recharger et d'utiliser le modèle entraîné sans avoir à répéter tout le processus.

3. Résultats et Évaluation

Le modèle a été évalué sur l'ensemble de test, qui n'a pas été utilisé durant l'entraînement. Les performances sont excellentes et démontrent la grande efficacité du modèle.
3.1. Matrice de Confusion

La matrice de confusion détaille la performance du modèle en comparant les prédictions aux vraies étiquettes.

Matrice de Confusion :
[[16135     5]
 [    0 16140]]

    Vrais Négatifs (TN) : 16 135 (BENIGN prédit BENIGN)
    Faux Positifs (FP) : 5 (BENIGN prédit ATTACK)
    Faux Négatifs (FN) : 0 (ATTACK prédit BENIGN)
    Vrais Positifs (TP) : 16 140 (ATTACK prédit ATTACK)

L'analyse montre un nombre extrêmement faible d'erreurs. Notamment, aucune attaque n'a été manquée (0 Faux Négatifs), ce qui est un résultat critique pour un système de détection.
3.2. Rapport de Classification

Ce rapport fournit des métriques de performance détaillées pour chaque classe.

Rapport de Classification :
              precision    recall  f1-score   support

      ATTACK       1.00      1.00      1.00     16140
      BENIGN       1.00      1.00      1.00     16140

    accuracy                           1.00     32280
   macro avg       1.00      1.00      1.00     32280
weighted avg       1.00      1.00      1.00     32280

    Précision : Presque 100% pour les deux classes. Cela signifie que lorsque le modèle prédit une classe, il a presque toujours raison.
    Rappel (Recall) : Presque 100% pour les deux classes. Le modèle est capable d'identifier la quasi-totalité des instances de chaque classe.
    F1-Score : Un score harmonique de la précision et du rappel. Un score proche de 1.00 indique une performance quasi parfaite.
    Accuracy (Précision globale) : Le modèle a atteint une précision globale de 99.98%, ce qui est un résultat exceptionnel.

4. Conclusion

Le projet a permis de développer avec succès un modèle de détection d'attaques DrDoS_DNS hautement performant. Grâce à une préparation rigoureuse des données, à une stratégie d'équilibrage des classes et à l'utilisation d'un pipeline de modélisation robuste basé sur un Random Forest, le modèle atteint une précision de près de 100% sur les données de test.

Le modèle final, sauvegardé dans DDosDetectModel.pkl, est prêt à être intégré dans un système de surveillance réseau pour identifier et bloquer le trafic malveillant en temps réel, renforçant ainsi significativement la sécurité de l'infrastructure.
