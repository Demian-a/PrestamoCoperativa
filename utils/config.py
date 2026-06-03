# -*- coding: utf-8 -*-
# ==========================================
# CONFIGURACION DEL SISTEMA
# Sistema: Sistema Inteligente de Evaluacion de Creditos
# Cooperativa: Cooperativa Cafetalera
# ==========================================

import os
from pathlib import Path

# ==========================================
# RUTAS PRINCIPALES
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSS_DIR = os.path.join(BASE_DIR, "css")
IMG_DIR = os.path.join(BASE_DIR, "img")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")

# ==========================================
# ARCHIVOS DEL SISTEMA
# ==========================================
DATASET_PATH = os.path.join(DATA_DIR, "credito_caficultores_500.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo.pkl")
CSS_PATH = os.path.join(CSS_DIR, "styles.css")

# ==========================================
# CREDENCIALES DE ACCESO
# ==========================================
USUARIO_ADMIN = "admin"
PASSWORD_ADMIN = "Cooperativa2024"

# ==========================================
# CONFIGURACION DEL MODELO ML
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10

# ==========================================
# VARIABLES DEL MODELO
# ==========================================
FEATURES_MODELO = [
    "Edad",
    "Antiguedad_Anos",
    "Hectareas_Cafe",
    "Produccion_Sacos_Anual",
    "Calidad_Promedio",
    "Monto_Solicitado_USD",
    "Historial_Mora"
]

TARGET = "Estado_Credito"

# ==========================================
# CONFIGURACION VISUAL
# ==========================================
COLOR_PRIMARY = "#2E7D32"
COLOR_SECONDARY = "#4CAF50"
COLOR_CAFE = "#6D4C41"
COLOR_CAFE_CLARO = "#A1887F"
COLOR_BLANCO = "#FFFFFF"
COLOR_FONDO = "#F5F5F5"
COLOR_TEXTO = "#212121"
COLOR_EXITO = "#2E7D32"
COLOR_PELIGRO = "#C62828"
COLOR_ADVERTENCIA = "#F9A825"

# ==========================================
# NOMBRE DE LA COOPERATIVA
# ==========================================
NOMBRE_COOPERATIVA = "Cooperativa Cafetalera Pangoa"
NOMBRE_SISTEMA = "Sistema Inteligente de Evaluacion de Creditos"
VERSION = "1.0.0"
