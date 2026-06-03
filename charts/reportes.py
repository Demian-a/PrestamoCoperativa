# -*- coding: utf-8 -*-
# ==========================================
# MODULO DE REPORTES Y EXPORTACION
# Sistema Inteligente de Evaluacion de Creditos
# Cooperativa Cafetalera
# ==========================================

import os
import sys
import io
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_CAFE,
    NOMBRE_COOPERATIVA,
    NOMBRE_SISTEMA,
    FEATURES_MODELO
)

# ==========================================
# GENERACION DE REPORTES EN MEMORIA
# ==========================================

def generar_reporte_excel(df, incluir_predicciones=False, predicciones=None):
    """
    Genera un archivo Excel en memoria con los datos.

    Args:
        df (pd.DataFrame): Datos originales
        incluir_predicciones (bool): Si incluye predicciones
        predicciones (pd.DataFrame): Datos con predicciones

    Returns:
        bytes: Contenido del archivo Excel
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Formato para encabezados
        header_format = workbook.add_format({
            "bold": True,
            "bg_color": "#2E7D32",
            "font_color": "#FFFFFF",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "font_size": 11
        })

        # Hoja de datos principales
        if incluir_predicciones and predicciones is not None:
            df_to_write = predicciones
        else:
            df_to_write = df

        df_to_write.to_excel(writer, sheet_name="Creditos", index=False)
        worksheet = writer.sheets["Creditos"]

        for col_idx, col_name in enumerate(df_to_write.columns):
            worksheet.write(0, col_idx, col_name, header_format)
            col_width = max(len(str(col_name)), 12)
            worksheet.set_column(col_idx, col_idx, col_width + 2)

        # Hoja de estadisticas
        stats = df.describe().T
        stats.to_excel(writer, sheet_name="Estadisticas")
        worksheet_stats = writer.sheets["Estadisticas"]
        for col_idx, col_name in enumerate(stats.columns):
            worksheet_stats.write(0, col_idx, str(col_name), header_format)

        # Hoja de resumen
        resumen = pd.DataFrame({
            "Metrica": [
                "Total Solicitudes",
                "Creditos Aprobados",
                "Creditos Rechazados",
                "Tasa de Aprobacion (%)",
                "Socios con Mora",
                "Tasa de Mora (%)",
                "Monto Total Solicitado (USD)",
                "Monto Promedio (USD)",
                "Produccion Promedio (Sacos)",
                "Calidad Promedio"
            ],
            "Valor": [
                len(df),
                int(df["Estado_Credito"].sum()),
                len(df) - int(df["Estado_Credito"].sum()),
                round(df["Estado_Credito"].mean() * 100, 1),
                int(df["Historial_Mora"].sum()),
                round(df["Historial_Mora"].mean() * 100, 1),
                round(df["Monto_Solicitado_USD"].sum(), 2),
                round(df["Monto_Solicitado_USD"].mean(), 2),
                round(df["Produccion_Sacos_Anual"].mean(), 1),
                round(df["Calidad_Promedio"].mean(), 1)
            ]
        })
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        worksheet_resumen = writer.sheets["Resumen"]
        for col_idx, col_name in enumerate(resumen.columns):
            worksheet_resumen.write(0, col_idx, col_name, header_format)

    return output.getvalue()

def generar_reporte_csv(df):
    """Genera un archivo CSV en memoria."""
    return df.to_csv(index=False).encode("utf-8")

# ==========================================
# GENERACION DE REPORTE TEXTO
# ==========================================

def generar_reporte_texto(df, precision_modelo=None):
    """Genera un reporte en formato texto plano."""
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total = len(df)
    aprobados = int(df["Estado_Credito"].sum())
    rechazados = total - aprobados
    morosos = int(df["Historial_Mora"].sum())

    reporte = f"""
============================================================
{NOMBRE_COOPERATIVA}
{NOMBRE_SISTEMA}
REPORTE DE EVALUACION DE CREDITOS
Fecha: {ahora}
============================================================

RESUMEN EJECUTIVO:
-------------------
Total de Solicitudes:          {total:>6}
Creditos Aprobados:            {aprobados:>6} ({aprobados/total*100:>5.1f}%)
Creditos Rechazados:           {rechazados:>6} ({rechazados/total*100:>5.1f}%)
Socios con Mora:               {morosos:>6} ({morosos/total*100:>5.1f}%)

INDICADORES FINANCIEROS:
-----------------------
Monto Total Solicitado:        ${df['Monto_Solicitado_USD'].sum():>10,.2f}
Monto Promedio Solicitado:     ${df['Monto_Solicitado_USD'].mean():>10,.2f}
Monto Maximo Solicitado:       ${df['Monto_Solicitado_USD'].max():>10,.2f}
Monto Minimo Solicitado:       ${df['Monto_Solicitado_USD'].min():>10,.2f}

INDICADORES PRODUCTIVOS:
-----------------------
Produccion Promedio:           {df['Produccion_Sacos_Anual'].mean():>8.1f} sacos/ano
Hectareas Promedio:            {df['Hectareas_Cafe'].mean():>8.1f} has
Calidad Promedio:              {df['Calidad_Promedio'].mean():>8.1f} pts

RENDIMIENTO DEL MODELO ML:
-------------------------
"""

    if precision_modelo:
        reporte += f"Precision del Modelo:            {precision_modelo*100:>5.1f}%\n"

    reporte += f"""
============================================================
Generado por: {NOMBRE_SISTEMA} v1.0.0
============================================================
"""
    return reporte

# ==========================================
# SECCION DE REPORTES EN STREAMLIT
# ==========================================

def mostrar_reportes(df, modelo=None, predicciones_df=None):
    """Renderiza la seccion de reportes con opciones de exportacion."""
    st.markdown("<h2 style='text-align: center; color: #2E7D32;'>[doc] Reportes y Exportacion</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 25px;'>
        Genere reportes ejecutivos y descargue los datos en diversos formatos.
    </div>
    """, unsafe_allow_html=True)

    precision = None
    if modelo is not None:
        import joblib
        from utils.config import MODEL_PATH
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            precision = data.get("accuracy", 0)

    excel_data = generar_reporte_excel(df, False)
    csv_data = generar_reporte_csv(df)
    texto = generar_reporte_texto(df, precision)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            label=" Descargar Excel",
            data=excel_data,
            file_name=f"reporte_creditos_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            help="Descargar dataset completo en formato Excel"
        )

    with col2:
        st.download_button(
            label=" Descargar CSV",
            data=csv_data,
            file_name=f"creditos_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Descargar dataset en formato CSV"
        )

    with col3:
        st.download_button(
            label=" Reporte Texto",
            data=texto.encode("utf-8"),
            file_name=f"reporte_ejecutivo_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Descargar reporte ejecutivo en texto plano"
        )

    with col4:
        if predicciones_df is not None and len(predicciones_df) > 0:
            st.download_button(
                label=" Predicciones",
                data=generar_reporte_excel(df, True, predicciones_df),
                file_name=f"predicciones_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Descargar predicciones realizadas"
            )
        else:
            st.warning("[alerta] Sin predicciones aun.\nVaya a 'Evaluar Credito' primero.")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # Vista previa de datos
    st.markdown("<h4 style='color: #2E7D32;'>Vista Previa del Dataset</h4>",
                unsafe_allow_html=True)

    cols_mostrar = st.multiselect(
        "Seleccione columnas a visualizar:",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:6]
    )

    if cols_mostrar:
        st.dataframe(
            df[cols_mostrar].head(100),
            use_container_width=True,
            height=300
        )

    st.markdown(f"<p style='text-align: center; color: #999;'>Mostrando {min(100, len(df))} de {len(df)} registros</p>",
                unsafe_allow_html=True)
