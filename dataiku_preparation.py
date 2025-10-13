import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("🚀 Début de la préparation des données pour Dataiku...")

# Chargement des données
df = pd.read_csv('train.csv')

print(f"📊 Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# 🔧 ÉTAPE 1 : NETTOYAGE DES DONNÉES
print("\n1. 🔧 Nettoyage des valeurs manquantes...")

# Age - Remplacer les valeurs manquantes par la médiane
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)
print(f"   ✅ Age : {df['Age'].isnull().sum()} valeurs manquantes restantes")

# Embarked - Remplacer par le mode (valeur la plus fréquente)
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)
print(f"   ✅ Embarked : {df['Embarked'].isnull().sum()} valeurs manquantes")

# Cabin - Remplacer par 'Unknown'
df['Cabin'].fillna('Unknown', inplace=True)
print(f"   ✅ Cabin : {df['Cabin'].isnull().sum()} valeurs manquantes")

# 🎨 ÉTAPE 2 : CRÉATION DE NOUVELLES VARIABLES (FEATURE ENGINEERING)
print("\n2. 🎨 Création de nouvelles variables...")

# Extraire le titre depuis le nom (Mr, Mrs, Miss, etc.) - CORRIGÉ
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
print(f"   ✅ Titres extraits : {df['Title'].unique()}")

# Taille de la famille
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print(f"   ✅ FamilySize créé (min: {df['FamilySize'].min()}, max: {df['FamilySize'].max()})")

# Est-ce que la personne est seule ?
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print(f"   ✅ IsAlone créé : {df['IsAlone'].sum()} personnes seules")

# Extraire le pont (deck) depuis la cabine
df['Deck'] = df['Cabin'].str[0]  # Première lettre
print(f"   ✅ Deck créé : {df['Deck'].unique()}")

# 🔢 ÉTAPE 3 : ENCODAGE DES VARIABLES CATÉGORIELLES
print("\n3. 🔢 Encodage des variables catégorielles...")

# Encodage du sexe (Female → 0, Male → 1)
le_sex = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
print(f"   ✅ Sex encodé : {dict(zip(le_sex.classes_, range(len(le_sex.classes_))))}")

# Encodage du port d'embarquement
le_embarked = LabelEncoder()
df['Embarked_encoded'] = le_embarked.fit_transform(df['Embarked'])
print(f"   ✅ Embarked encodé : {dict(zip(le_embarked.classes_, range(len(le_embarked.classes_))))}")

# Encodage du deck
le_deck = LabelEncoder()
df['Deck_encoded'] = le_deck.fit_transform(df['Deck'])
print(f"   ✅ Deck encodé : {len(le_deck.classes_)} catégories")

# 📊 ÉTAPE 4 : CRÉATION DE GROUPES (BINS)
print("\n4. 📊 Création de groupes...")

# Groupes d'âge
bins_age = [0, 12, 18, 35, 60, 100]
labels_age = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age)
print(f"   ✅ AgeGroup créé : {df['AgeGroup'].value_counts().to_dict()}")

# Groupes de prix de billet
df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
print(f"   ✅ FareGroup créé")

# 🎯 ÉTAPE 5 : SÉLECTION DES VARIABLES FINALES
print("\n5. 🎯 Sélection des variables pour Dataiku...")

# Liste des variables que nous allons garder
features_finales = [
    'Pclass',           # Classe sociale
    'Sex_encoded',      # Sexe encodé
    'Age',              # Âge
    'SibSp',            # Frères/soeurs + époux
    'Parch',            # Parents/enfants
    'Fare',             # Prix du billet
    'Embarked_encoded', # Port d'embarquement encodé
    'FamilySize',       # Taille famille
    'IsAlone',          # Seul ou non
    'Deck_encoded',     # Pont encodé
    'Title',            # Titre
    'AgeGroup',         # Groupe d'âge
    'FareGroup',        # Groupe de prix
    'Survived'          # Variable cible
]

df_final = df[features_finales].copy()

print(f"   ✅ Dataset final : {df_final.shape[0]} lignes, {df_final.shape[1]} colonnes")

# 💾 ÉTAPE 6 : SAUVEGARDE
print("\n6. 💾 Sauvegarde du fichier pour Dataiku...")

df_final.to_csv('data/titanic_dataiku_ready.csv', index=False)

print("🎉 PRÉPARATION TERMINÉE AVEC SUCCÈS !")
print("=" * 50)
print("📁 Fichier créé : 'data/titanic_dataiku_ready.csv'")
print("📊 Dimensions :", df_final.shape)
print("🔍 Colonnes :", df_final.columns.tolist())
print("\n➡️  Prochaine étape : Importer ce fichier dans Dataiku !")











































































































































































































import os

os.makedirs("data", exist_ok=True)

df.to_csv("data/titanic_dataiku_ready.csv", index=False)
print("✅ Fichier créé : data/titanic_dataiku_ready.csv — shape:", df.shape)

