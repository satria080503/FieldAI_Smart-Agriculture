import streamlit as st
import pandas as pd
from PIL import Image
import joblib

def show_page():
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-color:#FCFCFC;

    }}
    [data-testid="stSidebar"] {{
    background-color:#ffffff;

    }}
    [data-testid="stHeader"] {{
    background-color:#FCFCFC;
    }}
    [data-testid="stToolbar"] {{
    background-color:#FCFCFC;

    }}
    </style>
    """

    st.markdown(page_bg, unsafe_allow_html=True)

    def load_bootstrap():
        return st.markdown("""<link rel="stylesheet"
            href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
            integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm"
            crossorigin="anonymous">""", unsafe_allow_html=True)

    with st.sidebar:
        load_bootstrap()
        df_desc = pd.read_csv('Dataset/Crop_Des.csv', sep=';', encoding='utf-8')
        df = pd.read_csv('Dataset/Crop_recom.csv')

        model_path = 'Model/RDF_model.hdf5'
        rdf_clf = joblib.load(model_path)

        X = df.drop('label', axis=1)
        y = df['label']

    st.markdown("<h3 style='text-align: center;'>Please input the feature values to predict the best crop to plant.</h3><br>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 4, 1, 4, 1, 1], gap='medium')

    with col3:
        n_input = st.number_input('Insert N (kg/ha) value:', min_value=0, max_value=140, help='Insert here the Nitrogen density (kg/ha) from 0 to 140.')
        p_input = st.number_input('Insert P (kg/ha) value:', min_value=5, max_value=145, help='Insert here the Phosphorus density (kg/ha) from 5 to 145.')
        k_input = st.number_input('Insert K (kg/ha) value:', min_value=5, max_value=205, help='Insert here the Potassium density (kg/ha) from 5 to 205.')
        temp_input = st.number_input('Insert Avg Temperature (ºC) value:', min_value=9., max_value=43., step=1., format="%.2f", help='Insert here the Avg Temperature (ºC) from 9 to 43.')

    with col5:
        hum_input = st.number_input('Insert Avg Humidity (%) value:', min_value=15., max_value=99., step=1., format="%.2f", help='Insert here the Avg Humidity (%) from 15 to 99.')
        ph_input = st.number_input('Insert pH value:', min_value=3.6, max_value=9.9, step=0.1, format="%.2f", help='Insert here the pH from 3.6 to 9.9')
        rain_input = st.number_input('Insert Avg Rainfall (mm) value:', min_value=21.0, max_value=298.0, step=0.1, format="%.2f", help='Insert here the Avg Rainfall (mm) from 21 to 298')
        location = st.selectbox('Select location:', ('Java', 'Sumatra', 'Kalimantan', 'Sulawesi', 'Papua', 'Other'))

        # Update logic based on selected location
        if location == 'Java':
            predict_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input, 1, 0, 0, 0, 0]]
        elif location == 'Sumatra':
            predict_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input, 0, 1, 0, 0, 0]]
        elif location == 'Kalimantan':
            predict_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input, 0, 0, 1, 0, 0]]
        elif location == 'Sulawesi':
            predict_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input, 0, 0, 0, 1, 0]]
        elif location == 'Papua':
            predict_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input, 0, 0, 0, 0, 1]]
        else:
            predict_inputs = [[n_input, p_input, k_input, temp_input, hum_input, ph_input, rain_input, 0, 0, 0, 0, 0]]

    with col5:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button('Recommend Crop')
        st.markdown("<br>", unsafe_allow_html=True)

    cola, colb, colc = st.columns([2, 10, 2])
    if predict_btn:
        rdf_predicted_value = rdf_clf.predict(predict_inputs)

        with colb:
            col1, col2 = st.columns([4, 5])

            with col1:
                st.markdown(f"<br><h5 style='text-align: right;'>Best Crop to Plant </h5> <h3 style='text-align: right; line-height: 0'> <b>{rdf_predicted_value[0]} </b></h3>", unsafe_allow_html=True)

            with col2:
                df_desc = df_desc.astype({'label': str, 'image': str})
                df_desc['label'] = df_desc['label'].str.strip()
                df_desc['image'] = df_desc['image'].str.strip()

                df_pred_image = df_desc[df_desc['label'].isin(rdf_predicted_value)]
                df_image = df_pred_image['image'].item()

                st.markdown(f"""<h5 style='text-align: left; height: 300px; object-fit: contain;'> {df_image} </h5>""", unsafe_allow_html=True)

            st.markdown(f"""<h5 style='text-align: center;'>Statistics Summary about NPK and Weather Conditions values for <b> {rdf_predicted_value[0]}
                </b></h5>""", unsafe_allow_html=True)
            df_pred = df[df['label'] == rdf_predicted_value[0]]
            st.dataframe(df_pred.describe(), use_container_width=True)
