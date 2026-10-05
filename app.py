import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Food Order Cancellation Prediction",
    page_icon="🍔",
    layout="centered"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "food_order_cancellation.joblib"
    )

    return joblib.load(model_path)


    return model


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("🍔 Food Order Cancellation Prediction")

st.write(
    "Enter the food order details to predict "
    "whether the order will be cancelled."
)

st.divider()


# =========================================================
# CUSTOMER & ORDER DETAILS
# =========================================================

st.subheader("📋 Customer & Order Details")


record_id = st.number_input(
    "Record ID",
    min_value=0,
    value=1,
    step=1
)


customer_age = st.number_input(
    "Customer Age",
    min_value=1.0,
    max_value=100.0,
    value=25.0
)


order_value = st.number_input(
    "Order Value",
    min_value=0.0,
    value=500.0,
    step=10.0
)


distance_km = st.number_input(
    "Distance (km)",
    min_value=0.0,
    value=5.0,
    step=0.1
)


delivery_time = st.number_input(
    "Delivery Time (minutes)",
    min_value=0.0,
    value=30.0,
    step=1.0
)


restaurant_rating = st.number_input(
    "Restaurant Rating",
    min_value=0.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)


order_hour = st.slider(
    "Order Hour",
    min_value=0.0,
    max_value=23.0,
    value=12.0,
    step=1.0
)


# =========================================================
# CATEGORICAL DETAILS
# =========================================================

st.subheader("🍽️ Order Information")


food_category = st.text_input(
    "Food Category",
    value="Pizza"
)


payment_method = st.selectbox(
    "Payment Method",
    [
        "UPI",
        "Credit Card",
        "Debit Card",
        "Cash on Delivery"
    ]
)


traffic_level = st.selectbox(
    "Traffic Level",
    [
        "Low",
        "Medium",
        "High"
    ]
)


repeat_customer = st.selectbox(
    "Is Repeat Customer?",
    [
        "True",
        "False"
    ]
)


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Cancellation",
    use_container_width=True
):

    try:

        # Convert repeat customer to string
        # because training data used string type
        repeat_customer = str(repeat_customer)


        # -------------------------------------------------
        # Create DataFrame
        # -------------------------------------------------

        input_data = pd.DataFrame({

            "Record_ID": [record_id],

            "Customer_Age": [customer_age],

            "Order_Value": [order_value],

            "Distance_km": [distance_km],

            "Delivery_Time_Min": [delivery_time],

            "Restaurant_Rating": [restaurant_rating],

            "Order_Hour": [order_hour],

            "Food_Category": [food_category],

            "Payment_Method": [payment_method],

            "Traffic_Level": [traffic_level],

            "Is_Repeat_Customer": [repeat_customer]

        })


        # -------------------------------------------------
        # Make Prediction
        # -------------------------------------------------

        prediction = model.predict(input_data)


        # -------------------------------------------------
        # Prediction Result
        # -------------------------------------------------

        st.subheader("📊 Prediction Result")


        if prediction[0] == 1:

            st.error(
                "❌ The food order is likely to be CANCELLED."
            )

        else:

            st.success(
                "✅ The food order is likely to NOT be cancelled."
            )


        # -------------------------------------------------
        # Prediction Probability
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_data)

            cancellation_probability = probability[0][1] * 100

            not_cancellation_probability = probability[0][0] * 100


            st.write(
                f"**Cancellation Probability:** "
                f"{cancellation_probability:.2f}%"
            )

            st.write(
                f"**Not Cancellation Probability:** "
                f"{not_cancellation_probability:.2f}%"
            )


        # -------------------------------------------------
        # Show Entered Data
        # -------------------------------------------------

        with st.expander("🔎 View Entered Details"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error("⚠️ Prediction Error")

        st.write(
            "Please check the model and input features."
        )

        st.code(str(e))