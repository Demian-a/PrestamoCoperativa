# -*- coding: utf-8 -*-
# SISTEMA INTELIGENTE DE EVALUACION DE CREDITOS
# Cooperativa Cafetalera
# Aplicacion Principal - Streamlit

import os
import sys
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="sistema de evaluacion de creditos - cooperativa cafetalera",
    page_icon="\U0001F33F",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    DATASET_PATH, CSS_PATH, IMG_DIR, USUARIO_ADMIN, PASSWORD_ADMIN,
    NOMBRE_SISTEMA, COLOR_EXITO, COLOR_PELIGRO, FEATURES_MODELO
)
from utils.helpers import cargar_dataset, cargar_css, calcular_estadisticas, inicializar_sesion, get_img_base64
from models.prediccion import cargar_modelo, predecir_credito, verificar_modelo
from charts.dashboard import mostrar_dashboard
from charts.analisis import mostrar_analisis
from charts.reportes import mostrar_reportes

inicializar_sesion(st)

css_personalizado = cargar_css(CSS_PATH)
if css_personalizado:
    st.markdown(f"<style>{css_personalizado}</style>", unsafe_allow_html=True)

def mostrar_login():
    st.markdown("""
    <style>
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 50vh;
    }
    .login-card {
        max-width: 420px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    logo_b64 = get_img_base64(os.path.join(IMG_DIR, 'logo.svg'))
    st.markdown(f"""
    <div class="login-card animate-fade-in">
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="data:image/svg+xml;base64,{logo_b64}" width="100" style="margin-bottom: 10px;">
            <h2>Sistema de Evaluacion de Creditos</h2>
            <p>Cooperativa Cafetalera</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.form("login_form"):
        usuario = st.text_input("Usuario", placeholder="Ingrese su usuario")
        password = st.text_input("Contrasena", type="password", placeholder="Ingrese su contrasena")
        submitted = st.form_submit_button("Iniciar Sesion", use_container_width=True, type="primary")
        if submitted:
            if usuario == USUARIO_ADMIN and password == PASSWORD_ADMIN:
                st.session_state.autenticado = True
                st.session_state.usuario = usuario
                st.rerun()
            else:
                st.error("Usuario o contrasena incorrectos.")
    st.markdown('</div>', unsafe_allow_html=True)

def cerrar_sesion():
    st.session_state.autenticado = False
    st.session_state.usuario = ""
    st.rerun()

def mostrar_sidebar():
    with st.sidebar:
        logo_b64 = get_img_base64(os.path.join(IMG_DIR, 'logo.svg'))
        st.markdown(f"""
        <div class="sidebar-logo">
            <img src="data:image/svg+xml;base64,{logo_b64}" width="80" style="margin-bottom: 10px;">
            <h3>Cooperativa Cafetalera</h3>
            <p>Sistema de Evaluacion de Creditos</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Navegacion")
        paginas = ["Inicio", "Dashboard", "Evaluar Credito", "Analisis de Datos", "Reportes"]
        for pagina in paginas:
            if st.button(pagina, use_container_width=True,
                         type="secondary" if st.session_state.pagina != pagina else "primary"):
                st.session_state.pagina = pagina
                st.rerun()
        st.markdown("---")
        st.markdown("### Sistema")
        st.markdown(f"""
        <div style="font-size: 13px; color: rgba(255,255,255,0.7); padding: 10px;">
            <p><strong>Usuario:</strong> {st.session_state.usuario}</p>
            <p><strong>Version:</strong> 1.0.0</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Cerrar Sesion", use_container_width=True):
            cerrar_sesion()
        st.markdown("""
        <div style="font-size: 11px; color: rgba(255,255,255,0.4); text-align: center; margin-top: 30px; padding: 15px;">
            (c) 2024 Cooperativa Cafetalera<br>
            Todos los derechos reservados
        </div>
        """, unsafe_allow_html=True)

def pagina_inicio(df, modelo_info):
    precision_val = modelo_info.get('precision', 0) * 100
    st.markdown(f"""
    <div class="header-banner animate-fade-in">
        <h1>{NOMBRE_SISTEMA}</h1>
        <p>Plataforma inteligente para la evaluacion automatica de solicitudes de credito agricola,
        utilizando Machine Learning supervisado para apoyar la toma de decisiones financieras
        en beneficio de nuestros socios caficultores.</p>
        <span class="badge">Machine Learning - Random Forest - Precision: {precision_val:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

    stats = calcular_estadisticas(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card animate-fade-in"><div class="label">Total Solicitudes</div><div class="value" style="color: #1565C0;">{stats["total"]}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card animate-fade-in"><div class="label">Creditos Aprobados</div><div class="value" style="color: {COLOR_EXITO};">{stats["aprobados"]}</div><div class="delta" style="color: {COLOR_EXITO};">{stats["tasa_aprobacion"]}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card animate-fade-in"><div class="label">Creditos Rechazados</div><div class="value" style="color: {COLOR_PELIGRO};">{stats["rechazados"]}</div><div class="delta" style="color: {COLOR_PELIGRO};">{100-stats["tasa_aprobacion"]}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card animate-fade-in"><div class="label">Socios con Mora</div><div class="value" style="color: #F9A825;">{stats["morosos"]}</div><div class="delta" style="color: #F9A825;">{round(stats["morosos"]/stats["total"]*100,1)}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card card-verde animate-slide">
            <div class="card-title">Como funciona?</div>
            <p style="color: #666; font-size: 14px; line-height: 1.8;">
                El sistema utiliza un modelo de <strong>Random Forest Classifier</strong> entrenado con datos historicos
                de creditos agricolas. Evalua <strong>{len(FEATURES_MODELO)} variables</strong> clave del solicitante
                para predecir la probabilidad de pago exitoso.
            </p>
            <ul style="color: #666; font-size: 14px; line-height: 2; margin-top: 10px;">
                <li>Evaluacion objetiva y automatizada</li>
                <li>Basada en datos historicos reales</li>
                <li>Transparencia en la decision</li>
                <li>Reduccion de riesgos financieros</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card card-cafe animate-slide">
            <div class="card-title">Variables Evaluadas</div>
            <table style="width: 100%; font-size: 14px; color: #666;">
                <tr><td><strong>Edad</strong></td><td>del solicitante</td></tr>
                <tr><td><strong>Antiguedad</strong></td><td>anos como socio</td></tr>
                <tr><td><strong>Hectareas</strong></td><td>de cafe cultivadas</td></tr>
                <tr><td><strong>Produccion</strong></td><td>sacos anuales</td></tr>
                <tr><td><strong>Calidad</strong></td><td>promedio del cafe</td></tr>
                <tr><td><strong>Monto</strong></td><td>solicitado en USD</td></tr>
                <tr><td><strong>Mora</strong></td><td>historial crediticio</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <p><strong>Cooperativa Cafetalera</strong> - Sistema Inteligente de Evaluacion de Creditos v1.0.0</p>
        <p>(c) 2024 - Todos los derechos reservados</p>
    </div>
    """, unsafe_allow_html=True)

def pagina_evaluar_credito(df, modelo, metadata):
    precision_val = metadata.get('accuracy', 0) * 100
    st.markdown(f"""
    <div class="header-banner animate-fade-in" style="background: linear-gradient(135deg, #1565C0, #1E88E5, #42A5F5);">
        <h1>Evaluar Solicitud de Credito</h1>
        <p>Complete el formulario con los datos del socio para obtener una evaluacion automatica
        basada en el modelo de Machine Learning.</p>
        <span class="badge">Precision del modelo: {precision_val:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="form-container animate-fade-in">', unsafe_allow_html=True)
        st.markdown("<h3>Datos del Solicitante</h3>", unsafe_allow_html=True)
        with st.form("formulario_credito"):
            cols = st.columns(2)
            with cols[0]:
                edad = st.number_input("Edad del solicitante", min_value=18, max_value=90, value=35, step=1, help="Edad del socio solicitante (18-90 anos)")
                hectareas = st.number_input("Hectareas de Cafe", min_value=0.5, max_value=100.0, value=5.0, step=0.5, help="Cantidad de hectareas cultivadas")
                calidad = st.number_input("Calidad Promedio", min_value=50.0, max_value=100.0, value=80.0, step=0.5, help="Calidad promedio del cafe (50-100 pts)")
            with cols[1]:
                antiguedad = st.number_input("Antiguedad (anos como socio)", min_value=0, max_value=60, value=5, step=1, help="Anos de antiguedad como socio de la cooperativa")
                produccion = st.number_input("Produccion Anual (sacos)", min_value=0, max_value=5000, value=80, step=1, help="Produccion anual en sacos de cafe")
                monto = st.number_input("Monto Solicitado (USD)", min_value=100.0, max_value=100000.0, value=5000.0, step=100.0, help="Monto de credito solicitado en dolares americanos")
            mora = st.selectbox("Historial de Mora", options=[("No", 0), ("Si", 1)], format_func=lambda x: x[0], help="El socio tiene historial de mora?")
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Evaluar Credito", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if submitted:
            with st.spinner("Analizando solicitud..."):
                resultado = predecir_credito(edad=edad, antiguedad=antiguedad, hectareas=hectareas, produccion=produccion, calidad=calidad, monto=monto, mora=mora[1])
            if resultado["estado"] == 1:
                st.markdown(f"""
                <div class="result-card aprobado animate-fade-in">
                    <div class="icon">APROBADO</div>
                    <div class="title">CREDITO APROBADO</div>
                    <div class="confidence">{resultado['confianza']}%</div>
                    <div class="subtitle">de confianza en la prediccion</div>
                    <div style="margin-top: 15px; padding: 10px; background: rgba(46,125,50,0.1); border-radius: 8px;">
                        <p style="color: #2E7D32; font-size: 14px;">Probabilidad de aprobacion: {resultado['probabilidad']*100:.1f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif resultado["estado"] == 0:
                st.markdown(f"""
                <div class="result-card rechazado animate-fade-in">
                    <div class="icon">RECHAZADO</div>
                    <div class="title">RIESGO ALTO</div>
                    <div style="font-size: 18px; font-weight: 600; color: #C62828; margin-bottom: 5px;">REVISION MANUAL REQUERIDA</div>
                    <div class="confidence">{resultado['confianza']}%</div>
                    <div class="subtitle">de confianza en la prediccion</div>
                    <div style="margin-top: 15px; padding: 10px; background: rgba(198,40,40,0.1); border-radius: 8px;">
                        <p style="color: #C62828; font-size: 14px;">Probabilidad de aprobacion: {resultado['probabilidad']*100:.1f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Error al procesar la solicitud. Verifique que el modelo este entrenado.")

            # Mostrar factores que influyeron en la decision
            factores = resultado.get("factores", [])
            if factores:
                st.markdown("<h4 style='color: #2E7D32; margin-top: 10px;'>Factores que influyeron en la decision:</h4>", unsafe_allow_html=True)
                for f in factores:
                    color = "#2E7D32" if f["favorable"] else "#C62828"
                    signo = "+" if f["favorable"] else ""
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 6px 12px; margin: 3px 0; border-radius: 6px; background: {color}11; border-left: 3px solid {color};">
                        <span style="font-size: 13px;"><strong>{f['nombre']}:</strong> {f['valor']}</span>
                        <span style="font-size: 13px; font-weight: 600; color: {color};">{signo}{f['impacto_pct']:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            if "historial_predicciones" not in st.session_state:
                st.session_state.historial_predicciones = []
            prediccion_registro = {
                "Edad": edad, "Antiguedad_Anos": antiguedad, "Hectareas_Cafe": hectareas,
                "Produccion_Sacos_Anual": produccion, "Calidad_Promedio": calidad,
                "Monto_Solicitado_USD": monto, "Historial_Mora": mora[1],
                "Estado_Predicho": resultado["estado"], "Confianza": resultado["confianza"],
                "Probabilidad": resultado["probabilidad"]
            }
            st.session_state.historial_predicciones.append(prediccion_registro)

            st.markdown(f"""
            <div class="card animate-slide">
                <div class="card-title">Resumen de Variables</div>
                <table style="width: 100%; font-size: 13px; color: #555;">
                    <tr><td>Edad</td><td><strong>{edad}</strong> anos</td></tr>
                    <tr><td>Antiguedad</td><td><strong>{antiguedad}</strong> anos</td></tr>
                    <tr><td>Hectareas</td><td><strong>{hectareas}</strong> has</td></tr>
                    <tr><td>Produccion</td><td><strong>{produccion}</strong> sacos/ano</td></tr>
                    <tr><td>Calidad</td><td><strong>{calidad}</strong> pts</td></tr>
                    <tr><td>Monto</td><td><strong>${monto:,.2f}</strong></td></tr>
                    <tr><td>Historial Mora</td><td><strong>{"Si" if mora[1] == 1 else "No"}</strong></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card animate-fade-in" style="text-align: center; padding: 50px 30px;">
                <div style="font-size: 64px; margin-bottom: 20px;"></div>
                <h4 style="color: #2E7D32; margin-bottom: 10px;">Complete el formulario</h4>
                <p style="color: #666; font-size: 14px;">Ingrese los datos del socio en el formulario y presione<br>
                <strong>"Evaluar Credito"</strong> para obtener el resultado.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #2E7D32;'>Ultimas Solicitudes en el Dataset</h4>", unsafe_allow_html=True)
    df_mostrar = df[["ID_Socio", "Edad", "Antiguedad_Anos", "Hectareas_Cafe", "Monto_Solicitado_USD", "Historial_Mora", "Estado_Credito"]].tail(10)
    df_mostrar = df_mostrar.rename(columns={"ID_Socio": "ID Socio", "Edad": "Edad", "Antiguedad_Anos": "Antiguedad", "Hectareas_Cafe": "Hectare as", "Monto_Solicitado_USD": "Monto (USD)", "Historial_Mora": "Mora", "Estado_Credito": "Estado"})
    df_mostrar["Estado"] = df_mostrar["Estado"].map({0: "Rechazado", 1: "Aprobado"})
    df_mostrar["Mora"] = df_mostrar["Mora"].map({0: "No", 1: "Si"})
    st.dataframe(df_mostrar, use_container_width=True, height=350)

def main():
    if not st.session_state.autenticado:
        mostrar_login()
        return
    mostrar_sidebar()
    df = cargar_dataset(DATASET_PATH)
    if df is None:
        st.error(f"Error: No se pudo cargar el dataset. Verifique que exista en: {DATASET_PATH}")
        st.info("Ejecute primero: python models/entrenamiento.py")
        return
    modelo = None
    metadata = {"accuracy": 0, "fecha_entrenamiento": "No entrenado"}
    modelo_info = verificar_modelo()
    if modelo_info["disponible"]:
        modelo, metadata = cargar_modelo()
    else:
        if st.session_state.pagina in ["Dashboard", "Evaluar Credito"]:
            st.warning("El modelo ML no esta entrenado. Vaya a la seccion 'Inicio' y entrene el modelo.")
    if st.session_state.pagina == "Inicio":
        pagina_inicio(df, modelo_info)
    elif st.session_state.pagina == "Dashboard":
        mostrar_dashboard(df, modelo)
    elif st.session_state.pagina == "Evaluar Credito":
        pagina_evaluar_credito(df, modelo, metadata)
    elif st.session_state.pagina == "Analisis de Datos":
        mostrar_analisis(df)
    elif st.session_state.pagina == "Reportes":
        predicciones_hist = st.session_state.get("historial_predicciones", [])
        predicciones_df = pd.DataFrame(predicciones_hist) if predicciones_hist else None
        mostrar_reportes(df, modelo, predicciones_df)

if __name__ == "__main__":
    main()
