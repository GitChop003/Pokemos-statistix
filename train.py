import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# 1. Chargement et vérification des données
# 1. Chargement et vérification des données
df = pd.read_csv("Pokemon Data_cleaned.csv")

# 2. Preprocessing (Nettoyage)
# On remplace les types manquants par 'None'
df['Type_2'] = df['Type_2'].fillna('None')

# Sélection des colonnes prédictives (X) et de la cible (y)
# Note : On exclut HP, Attack, Defense, etc. car leur somme est égale au Total (triche)
X = df[['Type_1', 'Type_2', 'Generation', 'isLegendary', 'Color', 'Height_m', 'Weight_kg', 'Body_Style']]
y = df['Total']

# 3. Pipeline de transformation (Gestion du texte -> Nombres)
cat_features = ['Type_1', 'Type_2', 'Color', 'Body_Style']
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[('cat', cat_transformer, cat_features)], 
    remainder='passthrough'
)

# 4. Création du modèle Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 5. Optimisation (GridSearchCV - Requis par votre sujet)
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None]
}

print("Début de l'optimisation du modèle...")
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)

# 6. Sauvegarde du modèle final
joblib.dump(grid_search.best_estimator_, 'pokemon_model.pkl')
print(f"Modèle sauvegardé ! Score de précision (R2) : {grid_search.best_score_:.4f}")