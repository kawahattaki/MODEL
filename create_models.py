# create_models.py

import tensorflow as tf
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression

# Assurez-vous que le dossier models existe
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"Dossier '{MODEL_DIR}' créé.")

# --- 1. Création des Modèles Keras (.h5) ---

# Définir une entrée standard pour les modèles d'images (224x224 pixels, 3 canaux de couleur)
IMAGE_INPUT_SHAPE = (224, 224, 3) 
NUM_CLASSES = 5  # Correspond au nombre de classes de fleurs (daisy, rose, etc.)

# Créer un modèle Keras très simple (pour tm_model.h5 et nb1_model.h5)
try:
    print("Création des modèles Keras (tm_model.h5 et nb1_model.h5)...")
    
    modele_keras = tf.keras.Sequential([
        # Aplatir l'image en un seul vecteur
        tf.keras.layers.Flatten(input_shape=IMAGE_INPUT_SHAPE), 
        # Couche de classification finale (5 sorties pour 5 classes)
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') 
    ])
    
    # Sauvegarde du modèle pour Teachable Machine (tm_model.h5)
    modele_keras.save(os.path.join(MODEL_DIR, 'tm_model.h5'))
    
    # Sauvegarde du modèle pour Notebook 1 (nb1_model.h5)
    modele_keras.save(os.path.join(MODEL_DIR, 'nb1_model.h5')) 
    
    print("Modèles Keras créés avec succès.")
    
except Exception as e:
    print(f"Erreur lors de la création des modèles Keras : {e}")


# --- 2. Création du Modèle Scikit-learn (.pkl) ---

print("\nCréation et sauvegarde de nb3_model.pkl...")
try:
    # Créer un modèle Scikit-learn très simple
    modele_sklearn = LogisticRegression(max_iter=1000)
    
    # Calculer la taille aplatie de l'image (224 * 224 * 3 = 150528)
    flattened_size = np.prod(IMAGE_INPUT_SHAPE) 
    
    # Simuler des données pour donner au modèle une structure (un 'fit' minimum)
    X_dummy = np.random.rand(10, flattened_size) 
    y_dummy = np.random.randint(0, NUM_CLASSES, 10)
    modele_sklearn.fit(X_dummy, y_dummy)
    
    # Sauvegarde du modèle Scikit-learn (Notebook 3)
    joblib.dump(modele_sklearn, os.path.join(MODEL_DIR, 'nb3_model.pkl'))
    
    print("Modèle Scikit-learn (nb3_model.pkl) créé avec succès.")
    
except Exception as e:
    print(f"Erreur lors de la création du modèle Scikit-learn : {e}")

print("\n--- Création de tous les fichiers de modèles terminée ---")