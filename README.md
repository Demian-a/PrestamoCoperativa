# 🌿 Sistema Inteligente de Evaluación de Créditos para Cooperativa Cafetalera

Sistema web profesional para la evaluación automática de solicitudes de crédito de socios caficultores utilizando Machine Learning supervisado.

## 📋 Requisitos del Sistema

- Python 3.12 o superior
- pip (gestor de paquetes de Python)
- Git (opcional)

## 🚀 Instalación y Ejecución Local

### 1. Clonar o copiar el proyecto

```bash
cd SISTEMA_CREDITOS_CAFETALEROS
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Entrenar el modelo ML

```bash
python models/entrenamiento.py
```

### 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en http://localhost:8501

## ☁️ Despliegue en Streamlit Cloud

### 1. Subir el proyecto a GitHub

```bash
git init
git add .
git commit -m "Sistema de Evaluación de Créditos v1.0"
git remote add origin https://github.com/TU_USUARIO/SISTEMA_CREDITOS_CAFETALEROS.git
git push -u origin main
```

### 2. Desplegar en Streamlit Cloud

1. Ir a https://share.streamlit.io
2. Iniciar sesión con GitHub
3. Click en "New app"
4. Seleccionar el repositorio: `SISTEMA_CREDITOS_CAFETALEROS`
5. Branch: `main`
6. Main file path: `app.py`
7. Click en "Deploy"

### 3. Configurar el modelo en producción

**Importante:** El modelo entrenado (`models/modelo.pkl`) debe ser generado localmente y subido al repositorio, ya que en Streamlit Cloud no se puede ejecutar el entrenamiento. Siga estos pasos:

```bash
# Entrenar localmente
python models/entrenamiento.py

# Verificar que se creó models/modelo.pkl
# Luego subir a GitHub
git add models/modelo.pkl
git commit -m "Add trained model"
git push
```

## 🔑 Credenciales de Acceso

- **Usuario:** `admin`
- **Contraseña:** `Cooperativa2024`

## 📁 Estructura del Proyecto

```
SISTEMA_CREDITOS_CAFETALEROS/
├── app.py                    # Aplicación principal (Streamlit)
├── data/
│   └── credito_caficultores_500.csv   # Dataset de créditos
├── css/
│   └── styles.css            # Estilos personalizados
├── img/
│   ├── logo.svg              # Logo de la cooperativa
│   ├── banner.svg            # Banner principal
│   └── fondo.svg             # Imagen de fondo
├── models/
│   ├── entrenamiento.py      # Entrenamiento del modelo ML
│   ├── prediccion.py         # Módulo de predicción
│   └── modelo.pkl            # Modelo entrenado (se genera)
├── charts/
│   ├── dashboard.py          # Dashboard con indicadores
│   ├── analisis.py           # Análisis exploratorio
│   └── reportes.py           # Reportes y exportación
├── utils/
│   ├── helpers.py            # Funciones auxiliares
│   └── config.py             # Configuración del sistema
├── .streamlit/
│   └── config.toml           # Configuración de Streamlit
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Este archivo
```

## 🧠 Modelo Machine Learning

- **Algoritmo:** Random Forest Classifier
- **Librería:** Scikit-Learn
- **Variables de entrada (7):**
  - Edad, Antigüedad, Hectáreas de Café
  - Producción Anual, Calidad Promedio
  - Monto Solicitado (USD), Historial de Mora
- **Variable objetivo:** Estado del Crédito (0=Rechazado, 1=Aprobado)
- **Métricas:** Accuracy, AUC-ROC, Matriz de Confusión

## 🎨 Diseño Visual

- Inspirado en https://cacpangoa.com/
- Paleta: Verde agrícola, tonos café, blanco
- Efectos glassmorphism
- Animaciones suaves
- Diseño responsive
- Tarjetas modernas con sombras

## 📊 Funcionalidades

- ✅ Login de administrador
- ✅ Dashboard ejecutivo con KPIs
- ✅ Evaluación automática de créditos
- ✅ Análisis exploratorio de datos
- ✅ Exportación a Excel y CSV
- ✅ Reportes ejecutivos
- ✅ Visualizaciones interactivas (Plotly)

## 🛠️ Tecnologías

| Frontend | Backend | ML | Visualización |
|----------|---------|-----|---------------|
| Streamlit | Python 3.12 | Scikit-Learn | Matplotlib |
| HTML5 | - | RandomForest | Seaborn |
| CSS3 | - | - | Plotly |

## 📝 Notas

- El dataset contiene 500 registros sintéticos de solicitudes de crédito
- La precisión del modelo puede variar según el dataset generado
- Para uso en producción, reemplazar el dataset con datos reales
- La contraseña puede cambiarse en `utils/config.py`

---
Desarrollado con ❤️ para el desarrollo sostenible de nuestros socios caficultores.
