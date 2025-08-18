# -*- coding: utf-8 -*-

import streamlit as st
import requests

st.title("Vérification de documents")

doc_type = st.selectbox("Type de document à vérifier", ["permis", "carte_grise", "certificat_technique", "identite"])
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Vérifier"):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"http://localhost:8000/classify/{doc_type}", files=files)
        result = response.json()
        st.success(f"Categorie : {result['category']}")
