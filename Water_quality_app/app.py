import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pydeck as pdk

# Load model and scaler (you renamed them)
with open("water_potability_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("water_imputer_model.pkl", "rb") as f:
    scaler = pickle.load(f)

# Static state-river-location data
state_river_data = {
    "Andhra Pradesh": [("Godavari", 17.0166, 81.8040), ("Krishna", 16.5728, 80.3575)],
    "Telangana": [("Krishna", 17.1232, 79.2085), ("Manjira", 18.0110, 77.8782)],
    "Maharashtra": [("Godavari", 19.8762, 75.3433), ("Bhima", 18.5204, 73.8567)],
    "Karnataka": [("Cauvery", 12.2958, 76.6394), ("Tungabhadra", 15.1495, 76.9155)],
    "Tamil Nadu": [("Cauvery", 11.1271, 78.6569), ("Vaigai", 9.9252, 78.1198)],
    "Kerala": [("Periyar", 10.8505, 76.2711), ("Bharathapuzha", 10.7452, 76.5004)],
    "Gujarat": [("Narmada", 21.1702, 72.8311), ("Tapi", 21.7645, 72.1519)],
    "Punjab": [("Sutlej", 30.7333, 76.7794), ("Beas", 31.1471, 75.3412)],
    "Uttar Pradesh": [("Ganga", 27.1767, 78.0081), ("Yamuna", 26.8467, 80.9462)],
    "West Bengal": [("Ganga", 22.5726, 88.3639), ("Damodar", 23.6739, 87.6836)],
    "Rajasthan": [("Chambal", 26.9124, 75.7873)],
    "Bihar": [("Ganga", 25.0961, 85.3131)],
    "Jharkhand": [("Subarnarekha", 23.6102, 85.2799)],
    "Madhya Pradesh": [("Narmada", 23.2599, 77.4126)],
    "Odisha": [("Mahanadi", 20.9517, 85.0985)],
    "Chhattisgarh": [("Mahanadi", 21.2787, 81.8661)],
    "Assam": [("Brahmaputra", 26.2006, 92.9376)],
    "Meghalaya": [("Umkhrah", 25.5788, 91.8933)],
    "Tripura": [("Gomati", 23.9408, 91.9882)],
    "Manipur": [("Barak", 24.6637, 93.9063)],
    "Nagaland": [("Dikhu", 26.1584, 94.5624)],
    "Mizoram": [("Tlawng", 23.1645, 92.9376)],
    "Himachal Pradesh": [("Beas", 31.1048, 77.1734)],
    "Uttarakhand": [("Ganga", 30.0668, 79.0193)],
    "Goa": [("Mandovi", 15.2993, 74.1240)],
    "Delhi": [("Yamuna", 28.6139, 77.2090)],
    "Haryana": [("Yamuna", 29.0588, 76.0856)],
    "Sikkim": [("Teesta", 27.5330, 88.5122)],
    "Arunachal Pradesh": [("Siang", 28.2170, 94.7278)]
}

# Streamlit UI
st.set_page_config(page_title="Water Quality & Rivers", layout="wide")
st.title("💧 Water Quality Predictor ")

tab1, tab2 = st.tabs(["🧪 Water Quality Prediction", "🗺️ River Map Explorer"])

# --------- Tab 1 ---------
with tab1:
    st.subheader("Predict if the water is potable based on these parameters:")

    ph = st.slider("pH", 0.0, 14.0, 7.0)
    Chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 6.0)
    Conductivity = st.slider("Conductivity (μS/cm)", 100.0, 1000.0, 400.0)
    Organic_carbon = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 10.0)
    Trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 120.0, 60.0)
    Turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 3.0)

    if st.button("🔍 Predict Water Quality"):
        features = np.array([[ph, Chloramines, 
                              Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            st.success("✅ The water is POTABLE (Safe to Drink)")
        else:
            st.error("🚫 The water is NOT POTABLE (Unsafe for Drinking)")

# --------- Tab 2 ---------
with tab2:
    st.subheader("📍 Select a state to see rivers and their locations on the map")
    state = st.selectbox("Select State", list(state_river_data.keys()))

    if state:
        rivers = state_river_data[state]
        st.markdown(f"### Rivers in {state}:")
        for r in rivers:
            st.markdown(f"- {r[0]}")

        df_map = pd.DataFrame([
            {"river": name, "lat": lat, "lon": lon}
            for name, lat, lon in rivers
        ])

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=df_map["lat"].mean(),
                longitude=df_map["lon"].mean(),
                zoom=5
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position='[lon, lat]',
                    get_color='[0, 0, 255, 160]',
                    get_radius=50000,
                    pickable=True
                )
            ],
            tooltip={"text": "{river}"}
        ))
