# -*- coding: utf-8 -*-
# ==========================================
# MODULO DE PREDICCION DE CREDITOS
# Sistema Inteligente de Evaluacion de Creditos
# Cooperativa Cafetalera
# ==========================================

import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import MODEL_PATH, FEATURES_MODELO

# ==========================================
# CARGA DEL MODELO
# ==========================================

_modelo_cache = None
_metadata_cache = None

def cargar_modelo():
    """
    Carga el modelo entrenado desde el archivo .pkl.
    Implementa cache para evitar recargar en cada llamada.

    Returns:
        tuple: (modelo, metadata) o (None, None) si hay error
    """
    global _modelo_cache, _metadata_cache

    if _modelo_cache is not None and _metadata_cache is not None:
        return _modelo_cache, _metadata_cache

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No se encuentra el modelo en {MODEL_PATH}")
        print("Ejecute primero models/entrenamiento.py para entrenar el modelo.")
        return None, None

    try:
        modelo_data = joblib.load(MODEL_PATH)

        _modelo_cache = modelo_data.get("modelo")
        _metadata_cache = {
            "accuracy": modelo_data.get("accuracy", 0),
            "auc_roc": modelo_data.get("auc_roc", 0),
            "features": modelo_data.get("features", FEATURES_MODELO),
            "fecha_entrenamiento": modelo_data.get("fecha_entrenamiento", "Desconocida"),
            "n_registros": modelo_data.get("n_registros", 0)
        }

        print(f"Modelo cargado correctamente desde {MODEL_PATH}")
        print(f"Precision: {_metadata_cache['accuracy']*100:.2f}%")

        return _modelo_cache, _metadata_cache

    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None

# ==========================================
# PREDICCION INDIVIDUAL
# ==========================================

def predecir_credito(edad, antiguedad, hectareas, produccion, calidad, monto, mora):
    """
    Evalua una solicitud de credito individual.

    Args:
        edad (int): Edad del solicitante
        antiguedad (int): Anos de antiguedad como socio
        hectareas (float): Hectareas de cafe cultivadas
        produccion (int): Produccion anual en sacos
        calidad (float): Calidad promedio del cafe
        monto (float): Monto solicitado en USD
        mora (int): Historial de mora (0=No, 1=Si)

    Returns:
        dict: Resultado de la evaluacion con:
            - estado (int): 0=Rechazado, 1=Aprobado
            - probabilidad (float): Probabilidad de aprobacion (0-1)
            - confianza (float): Porcentaje de confianza
            - etiqueta (str): Texto del resultado
    """
    modelo, metadata = cargar_modelo()

    if modelo is None:
        return {
            "estado": -1,
            "probabilidad": 0.0,
            "confianza": 0.0,
            "etiqueta": "ERROR: Modelo no disponible"
        }

    # Construir DataFrame de entrada
    datos = pd.DataFrame([{
        "Edad": edad,
        "Antiguedad_Anos": antiguedad,
        "Hectareas_Cafe": hectareas,
        "Produccion_Sacos_Anual": produccion,
        "Calidad_Promedio": calidad,
        "Monto_Solicitado_USD": monto,
        "Historial_Mora": mora
    }])

    # Asegurar orden correcto de columnas
    datos = datos[FEATURES_MODELO]

    # Predecir
    estado = int(modelo.predict(datos)[0])
    probabilidad = float(modelo.predict_proba(datos)[0, 1])

    confianza = probabilidad * 100 if estado == 1 else (1 - probabilidad) * 100
    confianza = round(confianza, 1)

    etiqueta = "CREDITO APROBADO" if estado == 1 else "RIESGO ALTO - REVISION MANUAL"

    # Generar explicacion
    importancias = modelo.feature_importances_
    medios = metadata.get("dataset_means", {})
    desvios = metadata.get("dataset_stds", {})
    factores = []
    valores_entrada = {
        "Edad": edad, "Antiguedad_Anos": antiguedad, "Hectareas_Cafe": hectareas,
        "Produccion_Sacos_Anual": produccion, "Calidad_Promedio": calidad,
        "Monto_Solicitado_USD": monto, "Historial_Mora": mora
    }
    nombres_legibles = {
        "Edad": "Edad", "Antiguedad_Anos": "Antiguedad", "Hectareas_Cafe": "Hectareas",
        "Produccion_Sacos_Anual": "Produccion", "Calidad_Promedio": "Calidad",
        "Monto_Solicitado_USD": "Monto solicitado", "Historial_Mora": "Historial de mora"
    }
    direccion_positiva = {
        "Edad": 1, "Antiguedad_Anos": 1, "Hectareas_Cafe": 1,
        "Produccion_Sacos_Anual": 1, "Calidad_Promedio": 1,
        "Monto_Solicitado_USD": -1, "Historial_Mora": -1
    }
    for i, feat in enumerate(FEATURES_MODELO):
        if feat in medios and feat in desvios:
            valor = valores_entrada[feat]
            media = medios[feat]
            std = desvios[feat]
            if std > 0:
                z_score = (valor - media) / std
                impacto = z_score * importancias[i] * direccion_positiva.get(feat, 1)
                impacto_pct = round(impacto * 100, 1)
                favorable = (z_score > 0 and direccion_positiva[feat] == 1) or (z_score < 0 and direccion_positiva[feat] == -1)
                etiqueta_valor = "Si" if feat == "Historial_Mora" and valor == 1 else "No" if feat == "Historial_Mora" and valor == 0 else str(valor)
                factores.append({
                    "nombre": nombres_legibles.get(feat, feat),
                    "valor": etiqueta_valor,
                    "impacto_pct": impacto_pct,
                    "favorable": favorable
                })

    factores_ordenados = sorted(factores, key=lambda x: abs(x["impacto_pct"]), reverse=True)

    return {
        "estado": estado,
        "probabilidad": round(probabilidad, 4),
        "confianza": confianza,
        "etiqueta": etiqueta,
        "factores": factores_ordenados
    }

# ==========================================
# PREDICCION POR LOTES
# ==========================================

def predecir_lote(df):
    """
    Evalua multiples solicitudes de credito.

    Args:
        df (pd.DataFrame): DataFrame con las mismas columnas del modelo

    Returns:
        pd.DataFrame: DataFrame original con columnas adicionales:
            - Prediccion_Estado
            - Probabilidad_Aprobacion
    """
    modelo, metadata = cargar_modelo()

    if modelo is None:
        return None

    X = df[FEATURES_MODELO]

    df_resultado = df.copy()
    df_resultado["Prediccion_Estado"] = modelo.predict(X)
    df_resultado["Probabilidad_Aprobacion"] = modelo.predict_proba(X)[:, 1]

    return df_resultado

# ==========================================
# VERIFICACION DE MODELO
# ==========================================

def verificar_modelo():
    """Verifica si el modelo existe y esta cargado correctamente."""
    if os.path.exists(MODEL_PATH):
        modelo, metadata = cargar_modelo()
        if modelo is not None:
            return {
                "disponible": True,
                "ruta": MODEL_PATH,
                "precision": metadata["accuracy"],
                "fecha": metadata["fecha_entrenamiento"],
                "features": metadata["features"]
            }
    return {
        "disponible": False,
        "ruta": MODEL_PATH,
        "precision": 0,
        "fecha": "No entrenado",
        "features": FEATURES_MODELO
    }

# ==========================================
# BLOQUE DE PRUEBA
# ==========================================

if __name__ == "__main__":
    print("Verificando modelo...")
    info = verificar_modelo()
    print(f"Modelo disponible: {info['disponible']}")
    print(f"Precision: {info['precision']*100:.2f}%")

    if info["disponible"]:
        print("\n--- Prueba de prediccion ---")
        resultado = predecir_credito(
            edad=35,
            antiguedad=10,
            hectareas=5.0,
            produccion=80,
            calidad=85.0,
            monto=5000.00,
            mora=0
        )
        print(f"Resultado: {resultado['etiqueta']}")
        print(f"Confianza: {resultado['confianza']}%")
        print(f"Probabilidad: {resultado['probabilidad']:.2%}")
