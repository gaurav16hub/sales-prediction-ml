import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("sales_model.pkl")

model = load_model()

# Dashboard heading
st.title("📊 Sales Prediction Dashboard")
st.write(
    "Predict sales based on TV, Radio, "
    "and Newspaper advertising budgets."
)

st.divider()

# Input section
st.subheader("Enter Advertising Budget")

col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input(
        "TV Advertising Budget",
        min_value=0.0,
        max_value=1000.0,
        value=150.0,
        step=10.0
    )

with col2:
    radio = st.number_input(
        "Radio Advertising Budget",
        min_value=0.0,
        max_value=500.0,
        value=25.0,
        step=5.0
    )

with col3:
    newspaper = st.number_input(
        "Newspaper Advertising Budget",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=5.0
    )

# Prediction button
if st.button("Predict Sales", type="primary"):

    input_data = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=["TV", "Radio", "Newspaper"]
    )

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Predicted Sales",
        value=f"{prediction:.2f}"
    )

    st.info(
        "This is a machine learning estimate, "
        "not a guarantee of actual sales."
    )

    # Display input budget chart
    st.subheader("Advertising Budget Breakdown")

    fig, ax = plt.subplots()

    ax.bar(
        ["TV", "Radio", "Newspaper"],
        [tv, radio, newspaper]
    )

    ax.set_ylabel("Advertising Budget")
    ax.set_title("Budget by Advertising Channel")

    st.pyplot(fig)

st.divider()

st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit."
)