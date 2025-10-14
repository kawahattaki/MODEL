# app.py

# --- 1. IMPORTS DES BIBLIOTHÈQUES ---
import streamlit as st
import pandas as pd
import numpy as np
import zipfile 
import os 
import tempfile 
import joblib 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Modules d'IA et de Data Science
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import cv2  # OpenCV pour la manipulation d'images
import tensorflow as tf # Pour les modèles Keras


# --- 2. CONFIGURATION ET FONCTIONS UTILES ---

# Dossier des modèles (doit être créé dans le répertoire racine du projet)
MODEL_DIR = "models" 

def load_model_from_disk(model_path, model_type="keras"):
    """Charge un modèle Keras (.h5) ou Scikit-learn (.pkl)."""
    full_path = os.path.join(MODEL_DIR, model_path)
    if not os.path.exists(full_path):
        st.error(f"Fichier modèle introuvable : {full_path}. Assurez-vous qu'il est dans le dossier '{MODEL_DIR}'.")
        return None
    
    try:
        if model_type == "keras":
            # Charger un modèle Keras/TensorFlow
            model = tf.keras.models.load_model(full_path, compile=False) 
            # st.success(f"Modèle Keras {model_path} chargé.")
            return model
        
        elif model_type == "sklearn":
            # Charger un modèle Scikit-learn (via joblib)
            model = joblib.load(full_path)
            # st.success(f"Modèle Scikit-learn {model_path} chargé.")
            return model

    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {model_path} : {e}")
        return None

def load_and_preprocess_data(data_dir, target_size=(224, 224)):
    """Charge les images, les redimensionne et prépare les labels."""
    images = []
    labels = []
    
    # Assurer que les noms de classes sont triés pour l'encodage
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # st.info(f"Début du prétraitement pour {len(class_names)} classes...")
    
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            # Lecture de l'image avec OpenCV
            img = cv2.imread(img_path)
            
            if img is not None:
                # Redimensionnement (standard pour les modèles pré-entraînés comme TM)
                img = cv2.resize(img, target_size)
                # Normalisation simple (0-1)
                img = img / 255.0
                
                images.append(img)
                labels.append(i) # 'i' est l'index numérique de la classe
                
    st.success(f"Prétraitement terminé. Total de {len(images)} images chargées.")
    return np.array(images), np.array(labels), class_names


# --- 3. STRUCTURE DE L'APPLICATION ---

# Configuration de base de la page
st.set_page_config(
    page_title="App de Classification d'Images",
    layout="wide"
)

# Initialisation de l'état de la session
if 'data_path' not in st.session_state:
    st.session_state['data_path'] = None
if 'results' not in st.session_state:
    st.session_state['results'] = {}

# --- TITRE PRINCIPAL ---
st.title("🚀 Application Interactive de Classification d'Images")
st.markdown("""
Cette application permet de charger un jeu de données d'images, de choisir un ratio
de séparation (Train/Test) et d'évaluer plusieurs modèles de classification.
""")


# --- 3.1. CONFIGURATION (SIDEBAR) ---
st.sidebar.title("🛠️ Paramètres du Modèle")
st.sidebar.header("Séparation des Données (Split)")

# L'utilisateur choisit le ratio d'entraînement via un slider
train_ratio = st.sidebar.slider(
    'Ratio d\'entraînement (%)',
    min_value=50,
    max_value=90,
    value=70,
    step=5,
    help="Pourcentage des données à utiliser pour l'entraînement (le reste est pour le test)."
)

test_ratio = 100 - train_ratio

st.sidebar.info(f"Test Ratio : **{test_ratio}%**")

st.header("1. Configuration du Jeu de Données")
st.write(f"Ratio sélectionné : **{train_ratio}%** pour l'entraînement et **{test_ratio}%** pour le test.")


# --- 3.2. CHARGEMENT DU JEU DE DONNÉES (ZIP) ---
st.header("2. Chargement du Jeu de Données 🖼️")

uploaded_file = st.file_uploader(
    "Téléversez votre dossier d'images compressé (format ZIP uniquement)",
    type=['zip']
)

# app.py (Remplacer le bloc 'if uploaded_file is not None:')

if uploaded_file is not None:
    # --- Traitement du Fichier ZIP ---
    
    # *** CORRECTION MAJEURE ***
    # Utiliser un dossier permanent dans le répertoire du projet au lieu de tempfile
    
    # Chemin du dossier d'extraction PERMANENT (à supprimer manuellement après usage)
    extraction_dir = os.path.join(os.getcwd(), "_extracted_data")
    
    # 1. Nettoyer l'ancienne extraction avant de créer la nouvelle (bonne pratique)
    if os.path.exists(extraction_dir):
        import shutil
        shutil.rmtree(extraction_dir)
        
    os.makedirs(extraction_dir, exist_ok=True)
    
    # 2. Écrire le fichier téléchargé sur le disque
    zip_path = os.path.join(extraction_dir, "uploaded_data.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 3. Décompresser
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
        
    # Supprimer le fichier zip après extraction pour économiser de l'espace
    os.remove(zip_path)
        
    # 4. CORRECTION : Gérer le dossier parent unique (ex: 'flowers')
    content = os.listdir(extraction_dir)
    
    if len(content) == 1 and os.path.isdir(os.path.join(extraction_dir, content[0])):
        st.warning(f"Dossier parent unique détecté : '{content[0]}'. Recherche des classes à l'intérieur.")
        data_root_dir = os.path.join(extraction_dir, content[0])
    else:
        data_root_dir = extraction_dir
    
    # 5. Lister et stocker le chemin CORRECT
    classes = [d for d in os.listdir(data_root_dir) if os.path.isdir(os.path.join(data_root_dir, d))]
    
    if classes:
        st.success(f"Fichier ZIP chargé et extrait avec succès ! ✅")
        st.write(f"**Nombre de classes détectées :** **{len(classes)}**")
        st.write(f"**Classes trouvées :** {', '.join(classes)}")
        
        # Le chemin stocké est MAINTENANT PERMANENT
        st.session_state['data_path'] = data_root_dir
        
        st.info("Vous pouvez maintenant passer à l'étape 3 : Prétraitement et Entraînement.")
        
    else:
        st.error("Aucun dossier de classe trouvé. Vérifiez la structure du ZIP.")

# --- 3.3. BLOC D'ACTION : PRÉTRAITEMENT, SPLIT & ÉVALUATION ---

st.header("3. Démarrer l'Entraînement et l'Évaluation")

if st.button("▶️ PRÉTRAITER, SPLITTER & ÉVALUER les Modèles"):
    if st.session_state['data_path'] is None:
        st.error("⚠️ Veuillez d'abord téléverser et extraire un fichier ZIP de données (Étape 2).")
    else:
        # 1. Charger et Prétraiter
        X, y, class_names = load_and_preprocess_data(st.session_state['data_path'])
        
        if len(X) == 0:
            st.error("Aucune image valide n'a pu être chargée. Vérifiez la structure du ZIP.")
        else:
            # 2. SPLIT INTERACTIF
            split_size = (100 - train_ratio) / 100.0  
            st.subheader("Séparation des Données...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=split_size, 
                random_state=42, 
                stratify=y       
            )
            
            st.write(f"**Taille Train:** {len(X_train)} images")
            st.write(f"**Taille Test:** {len(X_test)} images")
            st.session_state['class_names'] = class_names 

            st.info("Prétraitement et Split terminés. Démarrage de l'évaluation...")
            
            # 3. ÉVALUATION DES MODÈLES
            st.subheader("4. Évaluation des Modèles")
            
            # Définir les modèles à charger et leur type (MODIFIEZ CES NOMS DE FICHIERS SI NÉCESSAIRE)
            models_to_evaluate = {
                "Teachable Machine (Keras)": ("tm_model.h5", "keras"),
                "Notebook 1 (Keras)": ("nb1_model.h5", "keras"),
                "Notebook 3 (Scikit-learn)": ("nb3_model.pkl", "sklearn"),
            }
            
            results = {} 
            
            for name, (path, type) in models_to_evaluate.items():
                model = load_model_from_disk(path, type)
                
                if model is not None:
                    with st.spinner(f"Évaluation en cours : {name}..."):
                        
                        if type == "keras":
                            y_pred_probs = model.predict(X_test, verbose=0)
                            y_pred = np.argmax(y_pred_probs, axis=1)
                        else:
                            # Pour Scikit-learn, aplatir les images (224*224*3)
                            y_pred = model.predict(X_test.reshape(X_test.shape[0], -1)) 

                        accuracy = np.mean(y_pred == y_test)
                        results[name] = accuracy
                        
                        st.success(f"**{name}** - Précision sur l'ensemble Test : **{accuracy:.4f}**")
                        
                        # Stocker pour la visualisation (Étape 5)
                        st.session_state[f'{name}_y_pred'] = y_pred
                        st.session_state[f'{name}_y_test'] = y_test
                        
            st.session_state['results'] = results 
            
            st.balloons() 
            st.info("Évaluation terminée. Prêt pour la visualisation des résultats.")


# --- 3.4. VISUALISATION DES RÉSULTATS (MATRICES DE CONFUSION) ---

if st.session_state['results']:
    st.markdown("---")
    st.header("5. Visualisation des Résultats (Plots) 📊")
    
    # Résumé
    st.subheader("Résumé de la Performance")
    results_df = pd.DataFrame.from_dict(st.session_state['results'], orient='index', columns=['Précision (Accuracy)'])
    st.dataframe(results_df, use_container_width=True)
    
    # Matrices de Confusion
    st.subheader("Matrices de Confusion Détaillées")
    class_names = st.session_state.get('class_names', [])

    cols = st.columns(len(st.session_state['results']))

    for i, (name, acc) in enumerate(st.session_state['results'].items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            
            y_pred = st.session_state.get(f'{name}_y_pred')
            y_test = st.session_state.get(f'{name}_y_test')

            if y_pred is not None:
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    ax=ax
                )
                ax.set_title(f"Matrice de Confusion\n({name}, Acc: {acc:.3f})")
                ax.set_ylabel('Classes Réelles')
                ax.set_xlabel('Classes Prédites')
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig) 


# --- 3.5. EXPORTATION DU NOTEBOOK ---

st.markdown("---")
st.header("6. Exportation du Notebook 💾")
st.markdown("Téléchargez le notebook original pour revoir les étapes d'entraînement et le code source.")

try:
    with open("notebook_entrainement.ipynb", "rb") as file:
        st.download_button(
            label="Télécharger le Notebook (Jupyter/Colab)",
            data=file,
            file_name="notebook_entrainement.ipynb",
            mime="application/x-ipynb+json"
        )
except FileNotFoundError:
    st.warning("⚠️ ATTENTION : Assurez-vous d'avoir un fichier nommé **notebook_entrainement.ipynb** dans le dossier principal du projet pour que le téléchargement fonctionne.")

    



    # --- NOUVELLE SECTION 5 : PRÉDICTION SUR IMAGE UNIQUE ---

st.markdown("---")
st.header("7. Démonstration : Prédiction sur une Nouvelle Image 🌸")
st.markdown("Téléversez une image pour voir comment les modèles la classent.")

single_image = st.file_uploader(
    "Téléversez une seule image de fleur (JPG, PNG)",
    type=['jpg', 'png', 'jpeg']
)

if single_image is not None:
    # Charger et traiter l'image
    file_bytes = np.asarray(bytearray(single_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Redimensionnement et Normalisation (DOIT correspondre à load_and_preprocess_data)
    target_size = (224, 224)
    img_processed = cv2.resize(img, target_size) / 255.0
    
    # Afficher l'image
    st.image(img_processed, caption="Image à classer", width=300, channels="BGR")
    
    # Ajouter une dimension pour Keras (Batch)
    X_single = np.expand_dims(img_processed, axis=0)

    # Récupérer les noms des classes
    class_names = st.session_state.get('class_names', ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4'])
    
    st.subheader("Résultats de la Prédiction")
    
    # Colonnes pour l'affichage
    cols = st.columns(3)
    
    # Charger les modèles (nous devons les recharger ou les stocker dans session_state)
    models_to_predict = {
        "Teachable Machine (Keras)": ("tm_model.h5", "keras"),
        "Notebook 1 (Keras)": ("nb1_model.h5", "keras"),
        "Notebook 3 (Scikit-learn)": ("nb3_model.pkl", "sklearn"),
    }
    
    for i, (name, (path, type)) in enumerate(models_to_predict.items()):
        model = load_model_from_disk(path, type) # Réutilise la fonction de chargement
        
        if model is not None:
            with cols[i]:
                if type == "keras":
                    # Prédiction Keras
                    preds = model.predict(X_single, verbose=0)
                    predicted_index = np.argmax(preds[0])
                    confidence = preds[0][predicted_index]
                else:
                    # Prédiction Scikit-learn (doit être aplati)
                    X_flat = X_single.reshape(X_single.shape[0], -1)
                    predicted_index = model.predict(X_flat)[0]
                    # La confiance n'est pas simple à obtenir pour tous les modèles Scikit
                    confidence = "N/A" # Vous pouvez utiliser model.predict_proba si c'est une classification, mais pour cet exemple simple, nous laissons N/A.

                st.info(name)
                st.metric(
                    label="Classe Prédite",
                    value=class_names[predicted_index]
                )
                if confidence != "N/A":
                    st.write(f"Confiance: **{confidence:.2f}**")