# -*- coding: utf-8 -*-
# ==========================================
# MODULO DE ANALISIS EXPLORATORIO DE DATOS
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

sns.set_style("whitegrid")

# ==========================================
# ANALISIS UNIVARIADO
# ==========================================

def analisis_edad(df):
    """Analiza la distribucion de edades de los socios."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribucion de Edad", "Edad por Estado de Credito"),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )

    fig.add_trace(
        go.Histogram(x=df["Edad"], nbinsx=20, marker_color=COLOR_PRIMARY,
                     name="Edad"),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(y=df[df["Estado_Credito"] == 1]["Edad"],
               name="Aprobado", marker_color=COLOR_EXITO,
               legendgroup="Aprobado"),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=df[df["Estado_Credito"] == 0]["Edad"],
               name="Rechazado", marker_color=COLOR_PELIGRO,
               legendgroup="Rechazado"),
        row=1, col=2
    )

    fig.update_layout(
        title_text="<b>Analisis de Edad de los Solicitantes</b>",
        showlegend=True,
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif")
    )

    return fig

def analisis_antiguedad(df):
    """Analiza la distribucion de antiguedad como socio."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribucion de Antiguedad", "Antiguedad por Estado"),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )

    fig.add_trace(
        go.Histogram(x=df["Antiguedad_Anos"], nbinsx=20,
                     marker_color=COLOR_CAFE, name="Antiguedad"),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(y=df[df["Estado_Credito"] == 1]["Antiguedad_Anos"],
               name="Aprobado", marker_color=COLOR_EXITO),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=df[df["Estado_Credito"] == 0]["Antiguedad_Anos"],
               name="Rechazado", marker_color=COLOR_PELIGRO),
        row=1, col=2
    )

    fig.update_layout(
        title_text="<b>Analisis de Antiguedad como Socio</b>",
        showlegend=True,
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif")
    )

    return fig

def analisis_hectareas(df):
    """Analiza la distribucion de hectareas cultivadas."""
    fig = px.scatter(
        df,
        x="Hectareas_Cafe",
        y="Produccion_Sacos_Anual",
        color="Estado_Credito",
        color_discrete_map={0: COLOR_PELIGRO, 1: COLOR_EXITO},
        size="Monto_Solicitado_USD",
        hover_data=["ID_Socio", "Calidad_Promedio"],
        title="<b>Relacion: Hectareas vs Produccion vs Monto</b>",
        labels={
            "Hectareas_Cafe": "Hectareas de Cafe",
            "Produccion_Sacos_Anual": "Produccion (Sacos/Ano)",
            "Estado_Credito": "Estado"
        },
        opacity=0.7
    )

    fig.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif")
    )

    return fig

def analisis_monto_vs_produccion(df):
    """Relacion entre monto solicitado y produccion."""
    fig = px.scatter(
        df,
        x="Produccion_Sacos_Anual",
        y="Monto_Solicitado_USD",
        color="Estado_Credito",
        color_discrete_map={0: COLOR_PELIGRO, 1: COLOR_EXITO},
        size="Calidad_Promedio",
        hover_data=["ID_Socio", "Antiguedad_Anos"],
        title="<b>Relacion: Produccion vs Monto Solicitado</b>",
        labels={
            "Produccion_Sacos_Anual": "Produccion (Sacos/Ano)",
            "Monto_Solicitado_USD": "Monto Solicitado (USD)",
            "Estado_Credito": "Estado"
        },
        opacity=0.7,
        trendline="ols"
    )

    fig.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif")
    )

    return fig

def analisis_calidad_vs_estado(df):
    """Analiza calidad promedio por estado de credito."""
    fig = px.violin(
        df,
        x="Estado_Credito",
        y="Calidad_Promedio",
        color="Estado_Credito",
        color_discrete_map={0: COLOR_PELIGRO, 1: COLOR_EXITO},
        box=True,
        points="all",
        title="<b>Distribucion de Calidad por Estado de Credito</b>",
        labels={
            "Calidad_Promedio": "Calidad Promedio",
            "Estado_Credito": "Estado"
        },
        hover_data=["ID_Socio"]
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif")
    )

    return fig

# ==========================================
# TABLA DE ESTADISTICAS DESCRIPTIVAS
# ==========================================

def tabla_estadisticas(df):
    """Genera tabla de estadisticas descriptivas."""
    vars_mostrar = FEATURES_MODELO + ["Estado_Credito"]
    stats = df[vars_mostrar].describe().T

    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats.columns = ["Count", "Media", "Std Dev", "Minimo", "Q1", "Mediana", "Q3", "Maximo"]
    stats["Count"] = stats["Count"].astype(int)

    return stats

# ==========================================
# ANALISIS COMPLETO
# ==========================================

def mostrar_analisis(df):
    """Renderiza la seccion de analisis exploratorio."""
    st.markdown("<h2 style='text-align: center; color: #2E7D32;'>[grafico] Analisis Exploratorio de Datos</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 25px;'>
        Visualizacion detallada de las variables del dataset y su relacion con el estado del credito.
    </div>
    """, unsafe_allow_html=True)

    # Estadisticas descriptivas
    with st.expander("[clipboard] Estadisticas Descriptivas", expanded=False):
        stats = tabla_estadisticas(df)
        st.dataframe(
            stats.style.format("{:.2f}", subset=["Media", "Std Dev", "Minimo", "Q1", "Mediana", "Q3", "Maximo"]),
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:
        fig1 = analisis_edad(df)
        st.plotly_chart(fig1, use_container_width=True)

        fig3 = analisis_hectareas(df)
        st.plotly_chart(fig3, use_container_width=True)

        fig5 = analisis_calidad_vs_estado(df)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        fig2 = analisis_antiguedad(df)
        st.plotly_chart(fig2, use_container_width=True)

        fig4 = analisis_monto_vs_produccion(df)
        st.plotly_chart(fig4, use_container_width=True)
