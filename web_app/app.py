import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#---------------------------------------
# Configuración de la página
#---------------------------------------
st.set_page_config(
    page_title="Predicción de Default - Credit Card",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

#---------------------------------------
# Estilos con CSS personalizado
#---------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #4F8BF9;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #555;
        margin-bottom: 2em;
    }
    body {
        background-color: #FAFAFA;
    }
    .main-container {
        background-color: #transparent;
        padding: 2em;
        box-shadow: none;
    }
    h2, h3 {
        color: #4F8BF9;
    }
    .css-1v0mbdj {
        background-color: #F0F4FF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#---------------------------------------
# Encabezado
#---------------------------------------
st.markdown('<p class="main-title">Predicción de Default en Tarjetas de Crédito</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Utilizando datos históricos para anticipar el riesgo de incumplimiento de pago</p>', unsafe_allow_html=True)

#---------------------------------------
# Carga de datos
#---------------------------------------
@st.cache_data
def cargar_datos():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    df = pd.read_excel(url, skiprows=[0])
    return df

df = cargar_datos()

df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df[[df.columns[-1]] + list(df.columns[:-1])]

df.rename(columns={
    'pay_0': 'sept_repayment',
    'pay_2': 'aug_repayment',
    'pay_3': 'july_repayment',
    'pay_4': 'june_repayment',
    'pay_5': 'may_repayment',
    'pay_6': 'april_repayment',
    'bill_amt1': 'sept_bill',
    'bill_amt2': 'aug_bill',
    'bill_amt3': 'july_bill',
    'bill_amt4': 'june_bill',
    'bill_amt5': 'may_bill',
    'bill_amt6': 'april_bill',
    'pay_amt1': 'sept_prevPaid',
    'pay_amt2': 'aug_prevPaid',
    'pay_amt3': 'july_prevPaid',
    'pay_amt4': 'june_prevPaid',
    'pay_amt5': 'may_prevPaid',
    'pay_amt6': 'april_prevPaid'
}, inplace=True)

df['education'] = df['education'].replace({0: 4, 5: 4, 6: 4})

# Remover 'id' si existe
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

X = df.drop(['default_payment_next_month'], axis=1)
y = df['default_payment_next_month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=75)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=140, max_depth=6, random_state=75)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf)
report = classification_report(y_test, y_pred_rf, output_dict=True)
cm = confusion_matrix(y_test, y_pred_rf)

# Calculamos la media para asignar valores por defecto
means = X.mean()

#---------------------------------------
# Tabs
#---------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Introducción", "Análisis", "Resultados del Modelo", "Predicción Interactiva"])

#---------------------------------------
# Tab 1: Introducción
#---------------------------------------
with tab1:
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.subheader("¿De qué trata esta aplicación?")
        st.write("""
        Esta herramienta analiza datos históricos de clientes con tarjetas de crédito para predecir el riesgo de incumplimiento
        de pago en el próximo mes. La información se basa en características del cliente, historial de pago, límite de crédito,
        nivel de educación, entre otros factores.
        
        Puedes explorar el análisis de datos, ver los resultados del modelo y probar la herramienta introduciendo información
        personalizada en la sección de "Predicción Interactiva".
        """)
        st.markdown('</div>', unsafe_allow_html=True)

#---------------------------------------
# Tab 2: Análisis Exploratorio
#---------------------------------------
with tab2:
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.subheader("Vista Previa del Dataset")
        st.write("Dimensiones del dataset:", df.shape)
        st.dataframe(df.head())

        st.subheader("Distribución de la variable objetivo")
        fig_target = px.histogram(df, x='default_payment_next_month', color='default_payment_next_month', 
                                  title="Histograma de Defaults",
                                  labels={'default_payment_next_month': 'Default'})
        st.plotly_chart(fig_target, use_container_width=True)

        st.subheader("Relación entre Edad y Límite de Crédito")
        fig_scatter = px.scatter(df, x='age', y='limit_bal', 
                                 color='default_payment_next_month',
                                 title="Edad vs Límite de Crédito",
                                 labels={'age': 'Edad', 'limit_bal': 'Límite de Crédito'})
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

#---------------------------------------
# Tab 3: Resultados del Modelo
#---------------------------------------
with tab3:
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.subheader("Métricas del Modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Exactitud (Accuracy)", value=f"{accuracy*100:.2f}%")
        precision = report['1']['precision']*100
        recall = report['1']['recall']*100
        col2.metric(label="Precisión (Default=1)", value=f"{precision:.2f}%")
        col3.metric(label="Sensibilidad (Recall)", value=f"{recall:.2f}%")

        st.subheader("Matriz de Confusión")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Predicción")
        ax_cm.set_ylabel("Valor Real")
        st.pyplot(fig_cm)

        st.subheader("Reporte de Clasificación")
        st.dataframe(pd.DataFrame(report).transpose())
        st.markdown('</div>', unsafe_allow_html=True)

#---------------------------------------
# Tab 4: Predicción Interactiva
#---------------------------------------
with tab4:
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.subheader("Prueba el Modelo con tus propios datos")

        st.write("Ajusta los parámetros en la barra lateral:")

        st.sidebar.header("Parámetros del Cliente")

        input_data = {}

        # Definiciones de opciones
        education_options = {1: "Postgrado", 2: "Universidad", 3: "Secundaria", 4: "Otro"}
        repayment_options = list(range(-1,10))  # -1,0,1,...,9

        # Pedir al usuario sólo las variables clave:
        input_data['limit_bal'] = st.sidebar.number_input("Límite de Crédito (NT$):", 
                                                          min_value=1000, 
                                                          max_value=1000000, 
                                                          value=20000, 
                                                          step=5000)

        selected_edu_label = st.sidebar.selectbox("Nivel de Educación:", options=list(education_options.values()))
        input_data['education'] = [k for k,v in education_options.items() if v == selected_edu_label][0]

        input_data['age'] = st.sidebar.number_input("Edad (años):", min_value=18, max_value=100, value=35, step=1)

        input_data['sept_repayment'] = st.sidebar.selectbox("Estado de Pago (Sept):", repayment_options)
        input_data['aug_repayment'] = st.sidebar.selectbox("Estado de Pago (Aug):", repayment_options)
        input_data['july_repayment'] = st.sidebar.selectbox("Estado de Pago (Jul):", repayment_options)
        input_data['june_repayment'] = st.sidebar.selectbox("Estado de Pago (Jun):", repayment_options)
        input_data['may_repayment'] = st.sidebar.selectbox("Estado de Pago (May):", repayment_options)
        input_data['april_repayment'] = st.sidebar.selectbox("Estado de Pago (Abr):", repayment_options)

        # Ahora asignamos valores por defecto (la media) a las variables restantes
        # Primero creamos una copia de la media
        default_data = means.copy()

        # Reemplazamos en default_data las variables que el usuario sí proporcionó
        for k, v in input_data.items():
            default_data[k] = v

        # Crear dataframe con estos valores
        input_df = pd.DataFrame([default_data], columns=X.columns)
        input_scaled = scaler.transform(input_df)

        if st.button("Predecir"):
            prediction = rf.predict(input_scaled)[0]
            probs = rf.predict_proba(input_scaled)[0]
            prob_default = probs[1]*100  # Probabilidad de default en %

            if prediction == 1:
                st.error("**Predicción:** El cliente probablemente incumplirá el pago.")
            else:
                st.success("**Predicción:** El cliente probablemente NO incumplirá el pago.")

            # Gráfico tipo Gauge
            gauge_color = "red" if prob_default > 50 else "green"
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_default,
                title = {'text': "Probabilidad de Default (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightgreen'},
                        {'range': [50, 100], 'color': 'lightcoral'}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
