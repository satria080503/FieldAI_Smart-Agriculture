import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="FieldAi",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_bootstrap():
    return st.markdown("""<link rel="stylesheet"
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm"
        crossorigin="anonymous">""", unsafe_allow_html=True)

# Load Bootstrap
load_bootstrap()

# Sidebar content
with st.sidebar:
    image = Image.open('FieldAi_Logo bulat.png')
    st.image(image, width=250)
    st.title("Navigation")
    selection = st.radio("Go to", ["Home", "Crop Recommendation System", "Plant Diseases Detection"])

# Main content
if selection == "Home":
    image = Image.open('FieldAi_Logo Full.png')
    st.image(image, width=400)
    st.markdown("""
        <h3>Welcome to FieldAi!</h3>
        <p>This application leverages AI to enhance agricultural practices.</p>
        <h2>Features:</h2>
        <ul>
            <li><strong>Crop Recommendation:</strong> Recommends the best crops to plant based on soil and weather conditions.</li>
            <li><strong>Plant Diseases Detection:</strong> Detects diseases in plants using AI-powered image analysis.</li>
        </ul>
        <p>Select a feature from the navigation menu.</p>
        """, unsafe_allow_html=True)

elif selection == "Crop Recommendation System":
    import croprecommendation
    croprecommendation.show_page()

elif selection == "Plant Diseases Detection":
    import plantdiseasedetect
    plantdiseasedetect.show_page()
