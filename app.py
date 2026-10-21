import streamlit as st, joblib, pandas as pd

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.main-title {font-size:42px; font-weight:800; color:#1e88e5; text-align:center;}
.sub-title {text-align:center; color:gray; margin-bottom:25px;}
.price-box {background: linear-gradient(135deg,#1e88e5,#43a047);
            padding:25px; border-radius:15px; text-align:center; color:white; margin-top:20px;}
.price-box h1 {color:white; font-size:48px; margin:0;}
.price-box p {margin:0; font-size:18px;}
.stButton>button {width:100%; background-color:#1e88e5; color:white;
                  font-size:18px; border-radius:10px; height:3em; border:none;}
.stButton>button:hover {background-color:#1565c0; color:white;}
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_files():
    return joblib.load("model.pkl"), joblib.load("columns.pkl")

model, columns = load_files()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app predicts the selling price of a used car using machine learning.")
    st.write("**Dataset:** CarDekho (Kaggle), 301 cars")
    st.write("**Models compared:** Linear Regression and Random Forest")
    st.write("**Best model:** Random Forest (R² = 0.96)")
    st.divider()
    st.caption("Machine Learning Lab Course Project")

# ---------- Header ----------
st.markdown('<div class="main-title">🚗 Used Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter the car details to get an estimated selling price</div>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💰 Predict Price", "📊 Model Insights"])

# ---------- Tab 1: Prediction ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Car details")
        present_price = st.number_input("Showroom price when new (in lakhs)", 0.0, 100.0, 5.0)
        kms = st.number_input("Kilometres driven", 0, 500000, 30000, step=1000)
        age = st.slider("Car age (years)", 0, 30, 5)

    with col2:
        st.subheader("Other details")
        owner = st.selectbox("Previous owners", [0, 1, 2, 3])
        fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG"])
        seller = st.selectbox("Seller type", ["Dealer", "Individual"])
        trans = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("🔍 Predict Price"):
        row = {c: 0 for c in columns}
        row["Present_Price"] = present_price
        row["Kms_Driven"] = kms
        row["Car_Age"] = age
        row["Owner"] = owner
        if "Fuel_Type_" + fuel in row: row["Fuel_Type_" + fuel] = 1
        if "Seller_Type_" + seller in row: row["Seller_Type_" + seller] = 1
        if "Transmission_" + trans in row: row["Transmission_" + trans] = 1

        price = max(model.predict(pd.DataFrame([row]))[0], 0)

        st.markdown(
            f'<div class="price-box"><p>Estimated selling price</p>'
            f'<h1>₹ {price:.2f} lakhs</h1></div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Showroom price", f"₹ {present_price} L")
        m2.metric("Expected drop", f"₹ {max(present_price - price, 0):.2f} L")
        m3.metric("Value retained", f"{(price / present_price * 100) if present_price > 0 else 0:.0f}%")

        st.caption("This is an estimate, not an exact price.")

# ---------- Tab 2: Model insights ----------
with tab2:
    st.subheader("Which details matter most for the price?")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
        st.bar_chart(imp)
        st.write("Taller bars mean that feature affects the price more.")
    else:
        st.info("Feature importance is available only for the Random Forest model.")

    st.subheader("Model comparison")
    st.table(pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R2 Score": [0.849, 0.960],
        "MSE": [3.479, 0.924],
    }))