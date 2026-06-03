# -*- coding: utf-8 -*-
# ==========================================
# MODULO DE ENTRENAMIENTO DEL MODELO ML
# Sistema Inteligente de Evaluacion de Creditos
# Cooperativa Cafetalera
# ==========================================
# Algoritmo: Random Forest Classifier
# Libreria: Scikit-Learn
# ==========================================

import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")

# Agregar ruta del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    DATASET_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS,
    MAX_DEPTH,
    FEATURES_MODELO,
    TARGET
)

# ==========================================
# CONFIGURACION VISUAL
# ==========================================
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

COLOR_VERDE = "#2E7D32"
COLOR_ROJO = "#C62828"
COLOR_AZUL = "#1565C0"

# ==========================================
# FUNCION PRINCIPAL DE ENTRENAMIENTO
# ==========================================

def entrenar_modelo():
    """
    Entrena un modelo Random Forest Classifier con el dataset de creditos.

    Realiza:
    1. Carga y validacion del dataset
    2. Division en entrenamiento y prueba
    3. Entrenamiento del modelo
    4. Evaluacion del modelo
    5. Guardado del modelo entrenado
    6. Reporte de metricas

    Returns:
        dict: Diccionario con metricas y el modelo entrenado
    """

    print("=" * 60)
    print("SISTEMA INTELIGENTE DE EVALUACION DE CREDITOS")
    print("Modulo de Entrenamiento - Random Forest Classifier")
    print("=" * 60)

    # ==========================================
    # PASO 1: CARGAR DATASET
    # ==========================================
    print(f"\n[1/5] Cargando dataset desde: {DATASET_PATH}")

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: No se encuentra el dataset en {DATASET_PATH}")
        return None

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nDistribucion de Estado_Credito:")
    print(df[TARGET].value_counts().to_string())
    print(f"  Aprobados (1): {df[TARGET].sum()}")
    print(f"  Rechazados (0): {len(df) - df[TARGET].sum()}")

    # ==========================================
    # PASO 2: PREPARAR DATOS
    # ==========================================
    print(f"\n[2/5] Preparando datos para entrenamiento...")

    X = df[FEATURES_MODELO]
    y = df[TARGET]

    # Verificar valores nulos
    nulos = X.isnull().sum().sum()
    if nulos > 0:
        print(f"ADVERTENCIA: Se encontraron {nulos} valores nulos. Se procedera a limpiarlos.")
        X = X.dropna()
        y = y.loc[X.index]

    print(f"Features ({len(FEATURES_MODELO)}): {FEATURES_MODELO}")
    print(f"Target: {TARGET}")

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nDivision de datos:")
    print(f"  Entrenamiento: {len(X_train)} registros ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Prueba: {len(X_test)} registros ({len(X_test)/len(X)*100:.1f}%)")

    # ==========================================
    # PASO 3: ENTRENAR MODELO
    # ==========================================
    print(f"\n[3/5] Entrenando Random Forest Classifier...")
    print(f"  N_estimators: {N_ESTIMATORS}")
    print(f"  Max_depth: {MAX_DEPTH}")
    print(f"  Random_state: {RANDOM_STATE}")

    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    print(f"  Entrenamiento completado.")

    # ==========================================
    # PASO 4: EVALUAR MODELO
    # ==========================================
    print(f"\n[4/5] Evaluando modelo...")

    # Predicciones
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # Precision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Precision del modelo: {accuracy*100:.2f}%")

    # AUC-ROC
    auc_roc = roc_auc_score(y_test, y_prob)
    print(f"  AUC-ROC: {auc_roc:.4f}")

    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Matriz de Confusion:")
    print(f"  {'':>12} {'Pred 0':>8} {'Pred 1':>8}")
    print(f"  {'Real 0':>12} {cm[0][0]:>8} {cm[0][1]:>8}")
    print(f"  {'Real 1':>12} {cm[1][0]:>8} {cm[1][1]:>8}")

    # Reporte de clasificacion
    print(f"\n  Reporte de Clasificacion:")
    reporte = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred, digits=4))

    # ==========================================
    # PASO 5: GUARDAR MODELO
    # ==========================================
    print(f"\n[5/5] Guardando modelo entrenado...")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Estadisticas del dataset para explicacion de predicciones
    dataset_means = df[FEATURES_MODELO].mean().to_dict()
    dataset_stds = df[FEATURES_MODELO].std().to_dict()
    feature_importances = dict(zip(FEATURES_MODELO, modelo.feature_importances_))

    modelo_data = {
        "modelo": modelo,
        "features": FEATURES_MODELO,
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "fecha_entrenamiento": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_registros": len(df),
        "n_features": len(FEATURES_MODELO),
        "dataset_means": dataset_means,
        "dataset_stds": dataset_stds,
        "feature_importances": feature_importances
    }

    joblib.dump(modelo_data, MODEL_PATH)
    print(f"  Modelo guardado en: {MODEL_PATH}")
    print(f"  Tamano del modelo: {os.path.getsize(MODEL_PATH)/1024:.1f} KB")

    # ==========================================
    # GENERAR GRAFICOS DE EVALUACION
    # ==========================================
    print(f"\nGenerando graficos de evaluacion...")

    # Matriz de Confusion visual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Rechazado", "Aprobado"],
                yticklabels=["Rechazado", "Aprobado"],
                ax=axes[0], cbar=False)
    axes[0].set_title("Matriz de Confusion", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Real")
    axes[0].set_xlabel("Prediccion")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color=COLOR_VERDE, lw=2,
                 label=f"AUC-ROC = {auc_roc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    axes[1].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[1].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    axes[1].set_title("Curva ROC", fontsize=14, fontweight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    ruta_grafico = os.path.join(os.path.dirname(MODEL_PATH), "evaluacion_modelo.png")
    plt.savefig(ruta_grafico, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafico guardado en: {ruta_grafico}")

    # Importancia de variables
    fig, ax = plt.subplots(figsize=(10, 6))
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]

    colores = plt.cm.Greens(np.linspace(0.3, 0.9, len(FEATURES_MODELO)))
    ax.barh(range(len(FEATURES_MODELO)),
            importancias[indices],
            color=colores[indices])
    ax.set_yticks(range(len(FEATURES_MODELO)))
    ax.set_yticklabels([FEATURES_MODELO[i] for i in indices])
    ax.set_xlabel("Importancia Relativa")
    ax.set_title("Importancia de Variables - Random Forest", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    plt.tight_layout()
    ruta_importancia = os.path.join(os.path.dirname(MODEL_PATH), "importancia_variables.png")
    plt.savefig(ruta_importancia, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafico de importancia guardado en: {ruta_importancia}")

    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    print("\n" + "=" * 60)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("=" * 60)
    print(f"  Precision (Accuracy):  {accuracy*100:.2f}%")
    print(f"  AUC-ROC:               {auc_roc:.4f}")
    print(f"  Modelo guardado en:    {MODEL_PATH}")
    print(f"  Fecha:                 {modelo_data['fecha_entrenamiento']}")
    print("=" * 60)

    return {
        "modelo": modelo,
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "matriz_confusion": cm,
        "reporte": reporte,
        "features": FEATURES_MODELO,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

# ==========================================
# PUNTO DE ENTRADA
# ==========================================

if __name__ == "__main__":
    resultados = entrenar_modelo()
