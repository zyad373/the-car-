import pickle
import pandas as pd
import streamlit as st

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🚗 Car Value Predictor")
st.write("Enter the vehicle specifications below to predict its value.")

# Input fields
year = st.number_input("Year", min_value=1990, max_value=2030, value=2018)
engine_hp = st.number_input("Engine HP", min_value=50, max_value=1000, value=200)

# Feature lists extracted from model structure
makes = [
    'Acura', 'Alfa Romeo', 'Aston Martin', 'Audi', 'BMW', 'Bentley', 'Bugatti',
    'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ferrari',
    'Ford', 'GMC', 'Genesis', 'HUMMER', 'Honda', 'Hyundai', 'Infiniti', 'Kia',
    'Lamborghini', 'Land Rover', 'Lexus', 'Lincoln', 'Lotus', 'Maserati',
    'Maybach', 'Mazda', 'McLaren', 'Mercedes-Benz', 'Mitsubishi', 'Nissan',
    'Oldsmobile', 'Plymouth', 'Pontiac', 'Porsche', 'Rolls-Royce', 'Saab',
    'Scion', 'Spyker', 'Subaru', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo'
]

vehicle_sizes = ['Compact', 'Large', 'Midsize']
transmission_types = ['AUTOMATED_MANUAL', 'AUTOMATIC', 'DIRECT_DRIVE', 'MANUAL', 'UNKNOWN']

selected_make = st.selectbox("Make", makes)
selected_size = st.selectbox("Vehicle Size", vehicle_sizes)
selected_trans = st.selectbox("Transmission Type", transmission_types)

if st.button("Predict"):
    # Get all feature names expected by the model
    feature_cols = model.feature_names_in_
    
    # Create a zero-filled dictionary for all columns
    input_data = {col: 0 for col in feature_cols}
    
    # Set continuous features
    input_data['year'] = year
    input_data['engine hp'] = engine_hp
    
    # Set one-hot encoded categorical features
    make_col = f"make_{selected_make}"
    size_col = f"vehicle size_{selected_size}"
    trans_col = f"transmission type_{selected_trans}"
    
    if make_col in input_data:
        input_data[make_col] = 1
    if size_col in input_data:
        input_data[size_col] = 1
    if trans_col in input_data:
        input_data[trans_col] = 1
        
    # Convert to DataFrame matching model input format
    input_df = pd.DataFrame([input_data])
    
    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Value: ${prediction:,.2f}")
