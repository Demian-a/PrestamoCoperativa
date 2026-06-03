# -*- coding: utf-8 -*-
# ==========================================
# MODULO DASHBOARD - INDICADORES Y METRICAS
# Sistema Inteligente de Evaluacion de Creditos
# Cooperativa Cafetalera
# ==========================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_CAFE,
    COLOR_EXITO,
    COLOR_PELIGRO,
    FEATURES_MODELO
)

# ==========================================
# CONFIGURACION VISUAL
# ==========================================
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

sns.set_style("whitegrid")

COLOR_PALETTE = ["#2E7D32", "#C62828", "#1565C0", "#F9A825", "#6D4C41"]

# ==========================================
# TARJETAS DE INDICADORES
# ==========================================

def mostrar_tarjeta_indicador(label, valor, delta=None, color=COLOR_PRIMARY, icono="", ayuda=""):
    """Muestra una tarjeta de indicador estilizada."""
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border: 1px solid {color}44;
        border-radius: 12px;
        padding: 18px;
        margin: 5px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        backdrop-filter: blur(10px);
    ">
        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
            {icono} {label}
        </div>
        <div style="font-size: 28px; font-weight: 700; color: {color};">
            {valor}
        </div>
        {f'<div style="font-size: 13px; color: {"#2E7D32" if delta is not None else "#C62828"}; margin-top: 5px;">{delta}</div>' if delta is not None else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def mostrar_kpi_metricas(df, precision_modelo=None):
    """Muestra los KPIs principales en una fila de tarjetas."""
    total = len(df)
    aprobados = int(df["Estado_Credito"].sum())
    rechazados = total - aprobados
    morosos = int(df["Historial_Mora"].sum())
    pct_aprobacion = round(aprobados / total * 100, 1)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mostrar_tarjeta_indicador(
            "Total Solicitudes",
            f"{total:,}",
            icono="[clipboard]",
            color="#1565C0"
        )

    with col2:
        mostrar_tarjeta_indicador(
            "Creditos Aprobados",
            f"{aprobados:,}",
            delta=f"{pct_aprobacion}%",
            icono="[check]",
            color=COLOR_EXITO
        )

    with col3:
        mostrar_tarjeta_indicador(
            "Creditos Rechazados",
            f"{rechazados:,}",
            icono="[cruz]",
            color=COLOR_PELIGRO
        )

    with col4:
        mostrar_tarjeta_indicador(
            "Socios con Mora",
            f"{morosos:,}",
            delta=f"{round(morosos/total*100,1)}%",
            icono="[alerta]",
            color="#F9A825"
        )

    with col5:
        if precision_modelo:
            mostrar_tarjeta_indicador(
                "Precision del Modelo",
                f"{precision_modelo:.1f}%",
                icono="[robot]",
                color="#6D4C41"
            )

# ==========================================
# GRAFICO: APROBADOS VS RECHAZADOS
# ==========================================

def grafico_aprobados_rechazados(df):
    """Grafico de barras comparando aprobados vs rechazados."""
    conteo = df["Estado_Credito"].value_counts().reset_index()
    conteo.columns = ["Estado", "Cantidad"]
    conteo["Estado"] = conteo["Estado"].map({0: "Rechazado", 1: "Aprobado"})

    colores = {"Aprobado": COLOR_EXITO, "Rechazado": COLOR_PELIGRO}

    fig = px.bar(
        conteo,
        x="Estado",
        y="Cantidad",
        color="Estado",
        color_discrete_map=colores,
        text="Cantidad",
        title="<b>Distribucion de Creditos: Aprobados vs Rechazados</b>",
        labels={"Cantidad": "Numero de Solicitudes", "Estado": "Resultado"}
    )

    fig.update_traces(textposition="outside", textfont=dict(size=14, color="#333"))
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif"),
        height=380
    )

    return fig

# ==========================================
# GRAFICO: HISTORIAL DE MORA
# ==========================================

def grafico_historial_mora(df):
    """Grafico circular de historial de mora."""
    conteo = df["Historial_Mora"].value_counts().reset_index()
    conteo.columns = ["Mora", "Cantidad"]
    conteo["Mora"] = conteo["Mora"].map({0: "Sin Mora", 1: "Con Mora"})
    conteo["Porcentaje"] = round(conteo["Cantidad"] / len(df) * 100, 1)

    colores = {"Sin Mora": COLOR_EXITO, "Con Mora": COLOR_PELIGRO}

    fig = px.pie(
        conteo,
        values="Cantidad",
        names="Mora",
        color="Mora",
        color_discrete_map=colores,
        title="<b>Historial de Mora entre Socios</b>",
        hole=0.4
    )

    fig.update_traces(
        textinfo="label+percent",
        textfont=dict(size=13),
        marker=dict(line=dict(color="#fff", width=2))
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif"),
        height=380
    )

    return fig

# ==========================================
# GRAFICO: DISTRIBUCION DE MONTOS
# ==========================================

def grafico_distribucion_montos(df):
    """Histograma de distribucion de montos solicitados."""
    fig = px.histogram(
        df,
        x="Monto_Solicitado_USD",
        nbins=20,
        color_discrete_sequence=[COLOR_PRIMARY],
        title="<b>Distribucion de Montos Solicitados (USD)</b>",
        labels={"Monto_Solicitado_USD": "Monto en USD", "count": "Frecuencia"},
        marginal="box"
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif"),
        height=380,
        bargap=0.05
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")

    return fig

# ==========================================
# GRAFICO: PRODUCCION ANUAL
# ==========================================

def grafico_distribucion_produccion(df):
    """Histograma de distribucion de produccion anual."""
    fig = px.histogram(
        df,
        x="Produccion_Sacos_Anual",
        nbins=20,
        color_discrete_sequence=[COLOR_CAFE],
        title="<b>Distribucion de Produccion Anual (Sacos)</b>",
        labels={"Produccion_Sacos_Anual": "Sacos por Ano", "count": "Frecuencia"},
        marginal="box"
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif"),
        height=380,
        bargap=0.05
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")

    return fig

# ==========================================
# GRAFICO: CALIDAD PROMEDIO
# ==========================================

def grafico_distribucion_calidad(df):
    """Histograma de distribucion de calidad promedio."""
    fig = px.histogram(
        df,
        x="Calidad_Promedio",
        nbins=20,
        color_discrete_sequence=["#1565C0"],
        title="<b>Distribucion de Calidad Promedio del Cafe</b>",
        labels={"Calidad_Promedio": "Puntaje de Calidad", "count": "Frecuencia"},
        marginal="box"
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif"),
        height=380,
        bargap=0.05
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")

    return fig

# ==========================================
# GRAFICO: MATRIZ DE CORRELACION
# ==========================================

def grafico_matriz_correlacion(df):
    """Heatmap de correlacion entre variables numericas."""
    vars_correlacion = FEATURES_MODELO + ["Estado_Credito"]
    corr = df[vars_correlacion].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Greens",
        title="<b>Matriz de Correlacion entre Variables</b>",
        labels={"color": "Correlacion"},
        aspect="auto",
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif"),
        height=500
    )

    return fig

# ==========================================
# GRAFICO: IMPORTANCIA DE VARIABLES
# ==========================================

def grafico_importancia_variables(modelo):
    """Grafico de barras horizontales con importancia de variables."""
    importancias = modelo.feature_importances_
    features = FEATURES_MODELO

    df_imp = pd.DataFrame({
        "Variable": features,
        "Importancia": importancias
    }).sort_values("Importancia", ascending=True)

    colores_imp = [COLOR_PRIMARY if i >= df_imp["Importancia"].median()
                   else COLOR_CAFE for i in df_imp["Importancia"]]

    fig = px.bar(
        df_imp,
        y="Variable",
        x="Importancia",
        orientation="h",
        color="Importancia",
        color_continuous_scale="Greens",
        title="<b>Importancia de Variables en la Decision del Modelo</b>",
        labels={"Importancia": "Importancia Relativa", "Variable": ""},
        text_auto=".3f"
    )

    fig.update_traces(textposition="outside", textfont=dict(size=11))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif"),
        height=420,
        xaxis=dict(range=[0, max(importancias) * 1.3])
    )

    return fig

# ==========================================
# DASHBOARD COMPLETO
# ==========================================

def mostrar_dashboard(df, modelo=None):
    """Renderiza el dashboard completo con todos los indicadores y graficos."""
    precision = None
    if modelo is not None:
        import joblib
        import os
        from utils.config import MODEL_PATH
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            precision = data.get("accuracy", 0) * 100

    # KPIs
    mostrar_kpi_metricas(df, precision)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fila 1: Aprobados vs Rechazados | Historial de Mora
    col1, col2 = st.columns(2)
    with col1:
        fig1 = grafico_aprobados_rechazados(df)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = grafico_historial_mora(df)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fila 2: Montos | Produccion
    col1, col2 = st.columns(2)
    with col1:
        fig3 = grafico_distribucion_montos(df)
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = grafico_distribucion_produccion(df)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fila 3: Calidad | Matriz Correlacion
    col1, col2 = st.columns(2)
    with col1:
        fig5 = grafico_distribucion_calidad(df)
        st.plotly_chart(fig5, use_container_width=True)
    with col2:
        fig6 = grafico_matriz_correlacion(df)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Importancia de variables
    if modelo is not None:
        fig7 = grafico_importancia_variables(modelo)
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("[bombillo] Cargue el modelo entrenado para visualizar la importancia de variables.")

    st.markdown("<br>", unsafe_allow_html=True)
