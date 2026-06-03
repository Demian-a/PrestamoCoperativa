# -*- coding: utf-8 -*-
# ==========================================
# FUNCIONES AUXILIARES
# Sistema Inteligente de Evaluacion de Creditos
# Cooperativa Cafetalera
# ==========================================

import os
import base64
import json
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

# ==========================================
# CARGA DE ARCHIVOS
# ==========================================

def cargar_dataset(ruta):
    """Carga el dataset desde un archivo CSV."""
    try:
        df = pd.read_csv(ruta)
        return df
    except Exception as e:
        print(f"Error al cargar dataset: {e}")
        return None

def cargar_css(ruta):
    """Carga el contenido de un archivo CSS."""
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error al cargar CSS: {e}")
        return ""

# ==========================================
# CODIFICACION DE IMAGENES
# ==========================================

def get_img_base64(ruta_imagen):
    """Convierte una imagen a base64 para incrustar en HTML."""
    try:
        with open(ruta_imagen, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

def get_img_html(ruta_imagen, width="100%", alt=""):
    """Genera un tag HTML img con base64."""
    b64 = get_img_base64(ruta_imagen)
    if b64:
        ext = os.path.splitext(ruta_imagen)[1].lower().replace(".", "")
        mime = "png" if ext == "png" else "jpeg" if ext in ("jpg", "jpeg") else "svg+xml"
        return f'<img src="data:image/{mime};base64,{b64}" width="{width}" alt="{alt}">'
    return ""

# ==========================================
# FORMATEO DE NUMEROS
# ==========================================

def formatear_moneda(valor):
    """Formatea un valor numerico como moneda USD."""
    return f"${valor:,.2f}"

def formatear_porcentaje(valor):
    """Formatea un valor como porcentaje."""
    return f"{valor:.1f}%"

def formatear_numero(valor):
    """Formatea un numero con separadores de miles."""
    return f"{valor:,.0f}"

# ==========================================
# ESTADISTICAS RAPIDAS
# ==========================================

def calcular_estadisticas(df):
    """Calcula estadisticas basicas del dataset."""
    total = len(df)
    aprobados = int(df["Estado_Credito"].sum())
    rechazados = total - aprobados
    morosos = int(df["Historial_Mora"].sum())
    tasa_aprobacion = (aprobados / total * 100) if total > 0 else 0

    return {
        "total": total,
        "aprobados": aprobados,
        "rechazados": rechazados,
        "morosos": morosos,
        "tasa_aprobacion": round(tasa_aprobacion, 1),
        "monto_promedio": round(df["Monto_Solicitado_USD"].mean(), 2),
        "monto_total": round(df["Monto_Solicitado_USD"].sum(), 2),
        "produccion_promedio": round(df["Produccion_Sacos_Anual"].mean(), 1),
        "calidad_promedio": round(df["Calidad_Promedio"].mean(), 1)
    }

# ==========================================
# EXPORTACION DE DATOS
# ==========================================

def df_a_excel(df):
    """Convierte un DataFrame a bytes de Excel."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Creditos")
    return output.getvalue()

def df_a_csv_bytes(df):
    """Convierte un DataFrame a bytes de CSV."""
    return df.to_csv(index=False).encode("utf-8")

# ==========================================
# MANEJO DE SESION
# ==========================================

def inicializar_sesion(st):
    """Inicializa variables de sesion."""
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False
    if "usuario" not in st.session_state:
        st.session_state.usuario = ""
    if "pagina" not in st.session_state:
        st.session_state.pagina = "Inicio"
    if "modelo_cargado" not in st.session_state:
        st.session_state.modelo_cargado = False
    if "historial_predicciones" not in st.session_state:
        st.session_state.historial_predicciones = []
