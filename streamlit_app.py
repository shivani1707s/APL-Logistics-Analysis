import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path

# =================================================
# PAGE CONFIGURATION
# =================================================

st.set_page_config(
    page_title="APL Logistic Analysis",
    page_icon="🚚",
    layout="wide"
)

# ===================================================
# PROJECT PATHS
# ===================================================

PROJECT_ROOT =  Path(__file__).resolve().parent


DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "APL_Logistics_cleaned.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_delay_risk_model.joblib"
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

lightgbm_model = load_model()


# ===========================================================
# TITLE
# ===========================================================

st.title(
    "🚚 Delivery Performance, Delay Risk,"
    " and Logistics Efficiency Analytics"
)

st.markdown(
    """
    **APL Logistics Supply Chain Analytics Dashboard**
    
    Explore delivery performace , delay risk, shipping modes,
    regional patterns, logistics efficiency, and AI-powered
    late-delivery risk prediction.
    """
)


# =================================================================
# FEATURE CALCULATION
# =================================================================

df["Delay_Gap"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔎 Filters")

shipping_modes = st.sidebar.multiselect(
    "Shipping Mode",
    options = sorted(
        df["Shipping Mode"]
        .dropna()
        .unique()
    ),
    default = sorted(
        df["Shipping Mode"]
        .dropna()
        .unique()
    )
)

markets = st.sidebar.multiselect(
    "Market",
    options=sorted(
        df["Market"]
        .dropna()
        .unique()
    ),
    default=sorted(
        df["Market"]
        .dropna()
        .unique()
    )
)


segments = st.sidebar.multiselect(
    "Customer Segment",
    options=sorted(
        df["Customer Segment"]
        .dropna()
        .unique()
    ),
    default=sorted(
        df["Customer Segment"]
        .dropna()
        .unique()
    )
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df[
    df["Shipping Mode"].isin(shipping_modes)
    &
    df["Market"].isin(markets)
    &
    df["Customer Segment"].isin(segments)
].copy()


# ============================================================
# KPI CALCULATIONS
# ============================================================

sla_compliance = (
    filtered_df["Days for shipping (real)"]
    <= filtered_df["Days for shipment (scheduled)"]
).mean() * 100


late_delivery_rate = (
    filtered_df["Delivery Status"]
    .eq("Late delivery")
    .mean() * 100
)


average_delay = filtered_df["Delay_Gap"].mean()

median_delay = filtered_df["Delay_Gap"].median()


late_shipments = (
    filtered_df["Delivery Status"]
    .eq("Late delivery")
    .sum()
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("📋 Dataset Overview")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Total Records",
        f"{len(filtered_df):,}"
    )


with col2:
    st.metric(
        "Total Columns",
        df.shape[1]
    )


with col3:
    st.metric(
        "Missing Values",
        int(filtered_df.isna().sum().sum())
    )


# ============================================================
# DELIVERY PERFORMANCE KPIs
# ============================================================

st.subheader("📊 Delivery Performance KPIs")


col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    st.metric(
        "SLA Compliance",
        f"{sla_compliance:.2f}%"
    )


with col2:
    st.metric(
        "Late Delivery Rate",
        f"{late_delivery_rate:.2f}%"
    )


with col3:
    st.metric(
        "Average Delay",
        f"{average_delay:.2f} days"
    )


with col4:
    st.metric(
        "Median Delay",
        f"{median_delay:.0f} days"
    )


with col5:
    st.metric(
        "Late Shipments",
        f"{late_shipments:,}"
    )


# ============================================================
# SHIPPING MODE ANALYSIS
# ============================================================

st.subheader("🚚 Shipping Mode Performance")


shipping_summary = (
    filtered_df
    .groupby("Shipping Mode")
    .agg(
        Orders=("Shipping Mode", "size"),

        Average_Actual_Days=(
            "Days for shipping (real)",
            "mean"
        ),

        Average_Scheduled_Days=(
            "Days for shipment (scheduled)",
            "mean"
        ),

        Late_Risk=(
            "Late_delivery_risk",
            "mean"
        )
    )
    .reset_index()
)


shipping_summary["Late_Risk"] *= 100


st.dataframe(
    shipping_summary,
    use_container_width=True
)


st.bar_chart(
    shipping_summary.set_index(
        "Shipping Mode"
    )["Late_Risk"]
)


# ============================================================
# MARKET PERFORMANCE
# ============================================================

st.subheader("🌍 Market Performance")


market_summary = (
    filtered_df
    .groupby("Market")
    .agg(
        Orders=("Market", "size"),

        Average_Delay=(
            "Delay_Gap",
            "mean"
        ),

        Late_Risk=(
            "Late_delivery_risk",
            "mean"
        )
    )
    .reset_index()
)


market_summary["Late_Risk"] *= 100


st.dataframe(
    market_summary,
    use_container_width=True
)


st.bar_chart(
    market_summary.set_index(
        "Market"
    )["Late_Risk"]
)


# ============================================================
# CUSTOMER SEGMENT ANALYSIS
# ============================================================

st.subheader("👥 Customer Segment Performance")


segment_summary = (
    filtered_df
    .groupby("Customer Segment")
    .agg(
        Orders=("Customer Segment", "size"),

        Average_Delay=(
            "Delay_Gap",
            "mean"
        ),

        Late_Risk=(
            "Late_delivery_risk",
            "mean"
        )
    )
    .reset_index()
)


segment_summary["Late_Risk"] *= 100


st.dataframe(
    segment_summary,
    use_container_width=True
)


# ============================================================
# PROFITABILITY
# ============================================================

st.subheader("💰 Profitability")


profit_col1, profit_col2, profit_col3 = st.columns(3)


with profit_col1:

    st.metric(
        "Average Sales",
        f"${filtered_df['Sales'].mean():,.2f}"
    )


with profit_col2:

    st.metric(
        "Average Profit",
        f"${filtered_df['Order Profit Per Order'].mean():,.2f}"
    )


with profit_col3:

    negative_profit_rate = (
        filtered_df[
            "Order Profit Per Order"
        ]
        .lt(0)
        .mean()
        * 100
    )

    st.metric(
        "Negative Profit Rate",
        f"{negative_profit_rate:.2f}%"
    )


# ============================================================
# SALES VS PROFIT
# ============================================================

st.subheader("📈 Sales vs Profit")


sales_profit = filtered_df[
    [
        "Sales",
        "Order Profit Per Order"
    ]
].copy()


st.scatter_chart(
    sales_profit,
    x="Sales",
    y="Order Profit Per Order"
)


# ============================================================
# DELIVERY STATUS
# ============================================================

st.subheader("📦 Delivery Status")


delivery_summary = (
    filtered_df[
        "Delivery Status"
    ]
    .value_counts()
    .reset_index()
)


delivery_summary.columns = [
    "Delivery Status",
    "Orders"
]


st.dataframe(
    delivery_summary,
    use_container_width=True
)


st.bar_chart(
    delivery_summary.set_index(
        "Delivery Status"
    )["Orders"]
)


# ============================================================
# DELAY DISTRIBUTION
# ============================================================

st.subheader("⏱️ Delay Gap Distribution")


delay_distribution = (
    filtered_df[
        "Delay_Gap"
    ]
    .value_counts()
    .sort_index()
)


st.bar_chart(
    delay_distribution
)


# ============================================================
# ============================================================
# AI LATE DELIVERY RISK PREDICTION
# ============================================================
# ============================================================

st.divider()

st.header("🤖 AI Late Delivery Risk Prediction")

st.markdown(
    """
    Enter shipment information below to estimate the probability
    of late delivery using the trained LightGBM model.
    
    **Note:** The model uses only features available before the
    delivery outcome. Post-delivery outcome variables are not used.
    """
)


# ============================================================
# MODEL FEATURE SET
# ============================================================

TIMING_SENSITIVE_COLUMNS = [
    "Order Profit Per Order",
    "profit_per_quantity",
    "Benefit per order"
]

TARGET = "Late_delivery_risk"


# Model's internal feature names
model_features = lightgbm_model.feature_name_


# Map LightGBM names back to original dataframe names
feature_mapping = {}

for col in df.columns:

    lightgbm_name = col.replace(" ", "_")

    feature_mapping[lightgbm_name] = col


matched_model_columns = []

for feature in model_features:

    if feature in feature_mapping:

        matched_model_columns.append(
            feature_mapping[feature]
        )


# ============================================================
# PREDICTION FORM
# ============================================================

st.subheader("Shipment Information")


form_col1, form_col2 = st.columns(2)


with form_col1:

    scheduled_days = st.number_input(
        "Scheduled Shipping Days",
        min_value=0,
        max_value=10,
        value=4,
        step=1
    )


    shipping_mode = st.selectbox(
        "Shipping Mode",
        sorted(
            df["Shipping Mode"]
            .dropna()
            .unique()
        )
    )


    customer_segment = st.selectbox(
        "Customer Segment",
        sorted(
            df["Customer Segment"]
            .dropna()
            .unique()
        )
    )


    market = st.selectbox(
        "Market",
        sorted(
            df["Market"]
            .dropna()
            .unique()
        )
    )


    order_region = st.selectbox(
        "Order Region",
        sorted(
            df["Order Region"]
            .dropna()
            .unique()
        )
    )


with form_col2:

    quantity = st.number_input(
        "Order Quantity",
        min_value=1,
        max_value=100,
        value=1,
        step=1
    )


    sales = st.number_input(
        "Sales",
        min_value=0.0,
        value=200.0,
        step=10.0
    )


    product_price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=100.0,
        step=10.0
    )


    discount_rate = st.number_input(
        "Discount Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )


    order_item_total = st.number_input(
        "Order Item Total",
        min_value=0.0,
        value=180.0,
        step=10.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🚀 Predict Late-Delivery Risk",
    type="primary"
)


if predict_button:

    # --------------------------------------------------------
    # Create input using a real dataset row as template
    # --------------------------------------------------------

    input_row = (
        df[
            matched_model_columns
        ]
        .iloc[[0]]
        .copy()
    )


    # --------------------------------------------------------
    # Update user-provided values
    # --------------------------------------------------------

    if "Days for shipment (scheduled)" in input_row.columns:

        input_row[
            "Days for shipment (scheduled)"
        ] = scheduled_days


    if "Shipping Mode" in input_row.columns:

        input_row[
            "Shipping Mode"
        ] = shipping_mode


    if "Customer Segment" in input_row.columns:

        input_row[
            "Customer Segment"
        ] = customer_segment


    if "Market" in input_row.columns:

        input_row[
            "Market"
        ] = market


    if "Order Region" in input_row.columns:

        input_row[
            "Order Region"
        ] = order_region


    if "Order Item Quantity" in input_row.columns:

        input_row[
            "Order Item Quantity"
        ] = quantity


    if "Sales" in input_row.columns:

        input_row[
            "Sales"
        ] = sales


    if "Product Price" in input_row.columns:

        input_row[
            "Product Price"
        ] = product_price


    if "Order Item Product Price" in input_row.columns:

        input_row[
            "Order Item Product Price"
        ] = product_price


    if "Order Item Discount Rate" in input_row.columns:

        input_row[
            "Order Item Discount Rate"
        ] = discount_rate


    if "Order Item Total" in input_row.columns:

        input_row[
            "Order Item Total"
        ] = order_item_total


    # --------------------------------------------------------
    # Recalculate engineered features
    # --------------------------------------------------------

    if "discount_amount_per_unit" in input_row.columns:

        input_row[
            "discount_amount_per_unit"
        ] = (
            (
                product_price
                * discount_rate
            )
            / max(quantity, 1)
        )


    if "sales_per_quantity" in input_row.columns:

        input_row[
            "sales_per_quantity"
        ] = (
            sales
            / max(quantity, 1)
        )


    if "price_discount_interaction" in input_row.columns:

        input_row[
            "price_discount_interaction"
        ] = (
            product_price
            * discount_rate
        )


    if "scheduled_days_bucket" in input_row.columns:

        input_row[
            "scheduled_days_bucket"
        ] = str(scheduled_days)


    if "geo_market_region" in input_row.columns:

        input_row[
            "geo_market_region"
        ] = (
            str(market)
            + " | "
            + str(order_region)
        )


    # --------------------------------------------------------
    # Prepare categorical columns
    # --------------------------------------------------------

    categorical_columns = (
        input_row
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )


    for col in categorical_columns:

        training_values = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
        )

        input_row[col] = pd.Categorical(
            input_row[col].astype(str),
            categories=training_values
        )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        prediction_probability = (
            lightgbm_model
            .predict_proba(input_row)[0, 1]
        )


        prediction_class = (
            int(
                prediction_probability >= 0.50
            )
        )


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.subheader("🎯 Prediction Result")


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Late-Delivery Probability",
                f"{prediction_probability * 100:.2f}%"
            )


        with result_col2:

            if prediction_class == 1:

                st.error(
                    "⚠️ HIGH RISK — "
                    "Potential late delivery"
                )

            else:

                st.success(
                    "✅ LOWER RISK — "
                    "Potential on-time delivery"
                )


        # ----------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------

        st.subheader(
            "🔍 Why did the model make this prediction?"
        )


        try:

            explainer = shap.TreeExplainer(
                lightgbm_model
            )


            shap_result = explainer(
                input_row
            )


            st.markdown(
                """
                The chart below shows which features
                contributed most to this prediction.
                """
            )


            st.pyplot(
                shap.plots.waterfall(
                    shap_result[0],
                    max_display=15,
                    show=False
                ).figure
            )


        except Exception as shap_error:

            st.warning(
                "Prediction succeeded, but the SHAP "
                "explanation could not be generated."
            )

            st.code(
                str(shap_error)
            )


    except Exception as prediction_error:

        st.error(
            "Prediction could not be generated."
        )

        st.code(
            str(prediction_error)
        )


# ============================================================
# DATA PREVIEW
# ============================================================

st.divider()

st.subheader("🔎 Filtered Data Preview")


st.dataframe(
    filtered_df.head(100),
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "APL Logistics — Delivery Performance, Delay Risk, "
    "and Logistics Efficiency Analysis"
)