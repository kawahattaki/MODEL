📁 Application d'Évaluation de Modèles de Classification d'Images

## 🎯 Aperçu et Objectifs du Projet

Ce projet est une application de démonstration interactive développée avec **Streamlit** pour la classification d'images de fleurs (cinq catégories : *daisy*, *dandelion*, *rose*, *sunflower*, *tulip*).

L'objectif principal est de **comparer trois approches d'entraînement de modèles** sur le même jeu de données, en évaluant leurs performances via des métriques et des visualisations détaillées, tout en offrant une interface utilisateur pour la prédiction en temps réel.

### Composants Clés
Le projet est structuré autour des éléments suivants pour garantir l'interactivité, la performance et l'analyse :

Interface Utilisateur : Développée avec Streamlit, elle sert d'application web interactive pour l'évaluation des modèles, la visualisation des métriques et la prédiction en temps réel.

Modèle 1 (Keras - Transfer Learning) : Un modèle de classification d'images basé sur le Transfer Learning (Transfert d'Apprentissage), utilisant potentiellement une architecture pré-entraînée (comme VGG16 ou ResNet), pour tirer parti des connaissances acquises sur de grands jeux de données.

Modèle 2 (Keras - Modèle Séquentiel) : Un modèle Keras classique entraîné à partir de zéro (Custom) sur le jeu de données, permettant de comparer les performances avec l'approche de Transfer Learning.

Modèle 3 (Scikit-learn - Classifieur Traditionnel) : Un modèle d'apprentissage automatique traditionnel (par exemple, Support Vector Machine ou Random Forest) qui opère sur des features (caractéristiques) extraites par un réseau de neurones ou une autre technique de feature engineering.

Historique & Modèles Sérialisés : Utilisation de Git pour le suivi de version du code, et de fichiers Keras H5 et Pickle (.pkl) pour sérialiser et charger rapidement les trois modèles entraînés sans nécessiter un nouvel entraînement.

Documentation Technique : Rédigée avec Sphinx en format reStructuredText, elle fournit un corpus complet sur l'architecture du code, la méthodologie d'entraînement et les résultats détaillés.

## ⚙️ Configuration du Projet et Installation

### Prérequis

  * Python 3.8+
  * Git

### 1\. Clonage du Dépôt

Clonez le projet sur votre machine locale :

```bash
git clone https://github.com/<votre_nom_github>/app_classification.git
cd app_classification
```

### 2\. Configuration de l'Environnement

Nous utilisons un environnement virtuel pour isoler les dépendances.

```bash
# Crée l'environnement virtuel (.venv)
python -m venv .venv

# Active l'environnement virtuel
# Windows PowerShell :
.\.venv\Scripts\activate
# Linux/macOS :
source .venv/bin/activate
```

### 3\. Installation des Dépendances

Installez toutes les bibliothèques requises, y compris TensorFlow, Scikit-learn, et Streamlit :

```bash
pip install -r requirements.txt
```

-----

## 💻 Utilisation de l'Application

### Démarrage

Lancez l'application Streamlit depuis la racine du projet (`app_classification/`) :

```bash
streamlit run app.py
```

L'interface sera accessible via votre navigateur à l'adresse `http://localhost:8501`.

### Fonctionnalités

1.  **Prédiction en Temps Réel** : Permet de glisser-déposer une image. Les trois modèles retournent immédiatement leurs prédictions (classe prédite et niveau de confiance).
2.  **Visualisation de l'Évaluation** : La page principale affiche les matrices de confusion détaillées pour chaque modèle, permettant une analyse rapide des faux positifs et faux négatifs.
3.  **Paramètres du Modèle** : Le panneau latéral permet d'ajuster certains paramètres (comme le ratio de split des données), bien que les modèles finaux soient pré-entraînés pour la démo.

-----

## 🗃️ Structure Détaillée des Données et Modèles

| Chemin | Contenu | Notes de Version |
| :--- | :--- | :--- |
| `app.py` | Logique de l'interface Streamlit. | Gère les chargements des modèles et l'affichage des résultats. |
| `create_models.py` | Script d'entraînement. | Utilisé initialement pour générer les fichiers dans `models/`. |
| `models/tm_model.h5` | Modèle **Keras** (Transfer Learning). | Fichier binaire HDF5. Chargé via `tf.keras.models.load_model()`. |
| `models/notebook1.pkl` | Modèle **Keras Séquentiel** (Custom). | Fichier binaire Pickle. Nécessite l'environnement TensorFlow/Keras pour être utilisé. |
| `models/notebook3.pkl` | Modèle **Scikit-learn**. | Fichier binaire Pickle. (Ex: un classifieur basé sur des *features* extraites par un autre réseau). |
| `.gitignore` | Exclusions Git. | Exclut tous les fichiers modèles lourds (`*.h5`, `*.pkl`) **si** la politique du dépôt l'exige (actuellement, ces fichiers sont exclus de l'historique volumineux initial mais peuvent être présents pour des raisons de déploiement). |

-----

## 📚 Documentation Technique

Une documentation complète a été rédigée avec **Sphinx** pour accompagner ce projet. Elle inclut :

  * L'architecture logicielle de l'application.
  * La méthodologie d'extraction des données.
  * Les détails de l'entraînement et des hyperparamètres pour chaque modèle.
  * Les résultats de performance complets.

La documentation est accessible localement en ouvrant le fichier :

```
docs/_build/html/index.html
```

-----

## 🔒 Gestion des Contribuables (Git)

Ce dépôt est configuré pour un workflow **Fork & Pull Request**.

  * Pour contribuer : Forkez le dépôt, créez une nouvelle branche pour vos modifications, puis soumettez une Pull Request vers la branche `main` du dépôt principal.
  * L'historique Git a été nettoyé des fichiers volumineux pour garantir la rapidité du clonage et du travail collaboratif.

-----

## 👤 Auteur

  * **Auteur** : **hattaki**
  * **Année** : 2025
