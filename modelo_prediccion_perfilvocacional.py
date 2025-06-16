#!/usr/bin/env python
# coding: utf-8

# In[4]:


# 📦 Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    log_loss, make_scorer, roc_curve, auc
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')


# In[5]:


# 📂 Load and prepare dataset
df = pd.read_excel("C:/Users/PC/Documents/DATOS_ENCUESTAS_COMPLETOS.xlsx")
X = df[[f'P{i+1}' for i in range(80)]]
y = df['perfil']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features (for MLP and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


def evaluar_modelo(name, y_true, y_pred):
    print(f"\n📘 Model: {name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


# In[7]:


def validar_modelo(name, model, X_data, y_data):
    scorer = {
        'accuracy': 'accuracy',
        'log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    }
    cv_result = cross_validate(model, X_data, y_data, cv=5, scoring=scorer, return_train_score=True)
    train_acc = cv_result['train_accuracy']
    val_acc = cv_result['test_accuracy']
    train_loss = -cv_result['train_log_loss']
    val_loss = -cv_result['test_log_loss']

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(train_acc, label="Training Accuracy", color='blue')
    ax[0].plot(val_acc, label="Validation Accuracy", color='green')
    ax[0].set_title(f"Accuracy - {name}")
    ax[0].set_xlabel("Fold")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(); ax[0].grid(True)

    ax[1].plot(train_loss, label="Training Loss", color='blue')
    ax[1].plot(val_loss, label="Validation Loss", color='red')
    ax[1].set_title(f"Log Loss - {name}")
    ax[1].set_xlabel("Fold")
    ax[1].set_ylabel("Loss")
    ax[1].legend(); ax[1].grid(True)

    plt.tight_layout(); plt.show()


# In[8]:


def graficar_roc(name, model, X_data, y_data):
    clases = np.unique(y_data)
    y_bin = label_binarize(y_data, classes=clases)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_data, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

    clf = OneVsRestClassifier(model)
    clf.fit(X_train_b, y_train_b)
    y_score = clf.predict_proba(X_test_b)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(clases)):
        fpr[i], tpr[i], _ = roc_curve(y_test_b[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'crimson'])
    for i, color in zip(range(len(clases)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{clases[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[9]:


mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                    solver='adam', max_iter=1500, random_state=42, verbose=True)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
evaluar_modelo("MLP", y_test, y_pred_mlp)
validar_modelo("MLP", mlp, X_train_scaled, y_train)
graficar_roc("MLP", mlp, scaler.fit_transform(X), y)


# In[10]:


rf = RandomForestClassifier(n_estimators=300, max_depth=15,
                            min_samples_split=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluar_modelo("Random Forest", y_test, y_pred_rf)
validar_modelo("Random Forest", rf, X, y)
graficar_roc("Random Forest", rf, X, y)


# In[11]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
evaluar_modelo("KNN", y_test, y_pred_knn)
validar_modelo("KNN", knn, X_train_scaled, y_train)
graficar_roc("KNN", knn, scaler.fit_transform(X), y)


# In[ ]:




