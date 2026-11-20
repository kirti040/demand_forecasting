import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Retail AI Demand & Inventory Dashboard",
    page_icon="📦",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model = joblib.load("models/best_demand_model.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    benchmark_df = pd.read_csv("models/model_benchmark_results.csv")
    audit_df = pd.read_csv("models/final_inventory_audit.csv")
    forecast_df = pd.read_csv("data/processed/test_forecast_results.csv")
    return model, encoders, benchmark_df, audit_df, forecast_df

model, encoders, benchmark_df, audit_df, forecast_df = load_assets()
best_model_name = encoders.get("best_model_name", "LightGBM")

st.title("📦 AI-Enabled Demand Forecasting & Retail Inventory Optimization")
st.markdown(
    f"Active Forecasting Engine: **{best_model_name}** | Real-time inventory health audit comparing **warehouse stock** "
    "against AI demand forecasts, **Safety Stock**, **Reorder Points (ROP)**, and **Economic Order Quantity (EOQ)**."
)

# Sidebar Policy Simulator
st.sidebar.header("⚙️ Supply Chain Parameters")
lead_time = st.sidebar.slider("Supplier Lead Time (Days)", min_value=1, max_value=21, value=5)
service_level = st.sidebar.selectbox(
    "Target Service Level",
    options=["90% (Z = 1.28)", "95% (Z = 1.645)", "99% (Z = 2.33)"],
    index=1
)
ordering_cost = st.sidebar.number_input("Ordering Cost (₹ / PO)", min_value=50.0, value=500.0, step=50.0)
holding_cost_pct = st.sidebar.slider("Annual Holding Cost (% of Unit Price)", min_value=5, max_value=50, value=20, step=5)
volatility_reduction = st.sidebar.slider("Simulate Demand Volatility Reduction (%)", 0, 50, 0, step=5)

z_map = {"90% (Z = 1.28)": 1.28, "95% (Z = 1.645)": 1.645, "99% (Z = 2.33)": 2.33}
z_score = z_map[service_level]

df_sim = audit_df.copy()
effective_std = df_sim['demand_std'] * (1 - volatility_reduction / 100.0)
effective_holding = np.maximum(5.0, df_sim['unit_price'] * (holding_cost_pct / 100.0))

df_sim['safety_stock'] = np.ceil(z_score * effective_std * np.sqrt(lead_time)).astype(int)
df_sim['reorder_point'] = np.ceil((df_sim['predicted_demand'] * lead_time) + df_sim['safety_stock']).astype(int)
df_sim['eoq'] = np.ceil(
    np.sqrt((2 * (df_sim['predicted_demand'] * 365) * ordering_cost) / effective_holding)
).astype(int)

def classify_stock(row):
    if row['current_stock'] == 0 and row['predicted_demand'] > 0:
        return "CRITICAL STOCKOUT"
    elif row['current_stock'] <= row['reorder_point']:
        return "REORDER NOW"
    elif row['current_stock'] > (row['reorder_point'] + row['eoq'] * 1.5):
        return "OVERSTOCKED"
    return "HEALTHY"

df_sim['stock_status'] = df_sim.apply(classify_stock, axis=1)
df_sim['recommended_order_qty'] = np.where(
    df_sim['stock_status'].isin(["CRITICAL STOCKOUT", "REORDER NOW"]),
    df_sim['eoq'],
    0
)

# Section 1: KPIs & 7-Model Benchmark
st.subheader("1. Warehouse Inventory Health & 7-Model Benchmark")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Tracked SKUs", f"{len(df_sim):,}")
k2.metric("Critical Stockouts", f"{int((df_sim['stock_status'] == 'CRITICAL STOCKOUT').sum()):,}", delta="0 Stock", delta_color="inverse")
k3.metric("Below Reorder Point", f"{int((df_sim['stock_status'] == 'REORDER NOW').sum()):,}", delta="Action Needed", delta_color="inverse")
k4.metric("Healthy Stock SKUs", f"{int((df_sim['stock_status'] == 'HEALTHY').sum()):,}")
k5.metric("Overstocked SKUs", f"{int((df_sim['stock_status'] == 'OVERSTOCKED').sum()):,}")

col_left, col_right = st.columns([1.2, 1])
with col_left:
    st.markdown("**7-Model Validation Benchmark (Top 25 High-Volume SKUs)**")
    st.dataframe(
        benchmark_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2_Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

with col_right:
    st.markdown("**Units to Reorder by Product Category**")
    cat_orders = df_sim.groupby('category')['recommended_order_qty'].sum().sort_values(ascending=False)
    st.bar_chart(cat_orders)

st.markdown("---")

# Section 2: Replenishment Queue
st.subheader("2. Automated Purchase Order (PO) Replenishment Queue")
f1, f2 = st.columns([2, 1])
with f1:
    selected_statuses = st.multiselect(
        "Filter by Stock Status:",
        options=["CRITICAL STOCKOUT", "REORDER NOW", "HEALTHY", "OVERSTOCKED"],
        default=["CRITICAL STOCKOUT", "REORDER NOW"]
    )
with f2:
    all_categories = ["All Categories"] + sorted(df_sim['category'].dropna().unique().tolist())
    selected_cat_filter = st.selectbox("Filter by Category:", all_categories)

filtered_df = df_sim[df_sim['stock_status'].isin(selected_statuses)].copy()
if selected_cat_filter != "All Categories":
    filtered_df = filtered_df[filtered_df['category'] == selected_cat_filter]

filtered_df = filtered_df.sort_values(by=['reorder_point', 'predicted_demand'], ascending=False)

display_cols = [
    'sku', 'category', 'current_stock', 'predicted_demand',
    'safety_stock', 'reorder_point', 'eoq', 'recommended_order_qty', 'stock_status'
]
st.dataframe(
    filtered_df[display_cols].style.format({
        'predicted_demand': '{:.2f}',
        'current_stock': '{:,}',
        'safety_stock': '{:,}',
        'reorder_point': '{:,}',
        'eoq': '{:,}',
        'recommended_order_qty': '{:,}'
    }),
    use_container_width=True,
    height=350
)

st.download_button(
    label="📥 Download Replenishment Purchase Orders (.csv)",
    data=filtered_df[display_cols].to_csv(index=False).encode('utf-8'),
    file_name="retail_replenishment_orders.csv",
    mime="text/csv"
)

st.markdown("---")

# Section 3: SKU-Level Forecast Inspector & Live Predictor
st.subheader("3. SKU-Level Demand Forecast & Stock Inspector")
only_top_25 = st.checkbox("Filter to Top 25 Benchmark SKUs (includes LSTM, ARIMA & SARIMA curves)", value=False)
pool_df = df_sim[df_sim['is_top_benchmark_sku']] if only_top_25 and 'is_top_benchmark_sku' in df_sim.columns else df_sim

c1, c2 = st.columns(2)
with c1:
    inspect_cat = st.selectbox("Select Product Category", sorted(pool_df['category'].dropna().unique()))
with c2:
    cat_skus = sorted(pool_df[pool_df['category'] == inspect_cat]['sku'].unique())
    inspect_sku = st.selectbox("Select SKU to Inspect", cat_skus)

sku_info = df_sim[df_sim['sku'] == inspect_sku].iloc[0]
sku_ts = forecast_df[forecast_df['sku'] == inspect_sku].sort_values('date').copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Warehouse Stock", f"{int(sku_info['current_stock'])} units")
m2.metric("AI Forecasted Demand", f"{sku_info['predicted_demand']:.2f} units/day")
m3.metric("Reorder Point (ROP)", f"{int(sku_info['reorder_point'])} units")
m4.metric("Recommended Order (EOQ)", f"{int(sku_info['recommended_order_qty'])} units")

if not sku_ts.empty:
    curve_map = {
        'total_units_demanded': 'Actual Net Demand',
        'pred_lgbm': 'LightGBM',
        'pred_xgb': 'XGBoost',
        'pred_gb': 'Gradient Boosting',
        'pred_rf': 'Random Forest',
        'pred_lstm': 'LSTM',
        'pred_sarima': 'SARIMA',
        'pred_arima': 'ARIMA'
    }
    valid_cols = [c for c in curve_map if c in sku_ts.columns and not sku_ts[c].isna().all()]
    chart_data = sku_ts[['date'] + valid_cols].rename(columns=curve_map).set_index('date')
    st.line_chart(chart_data)

with st.expander("🔮 Live Model Inference: Predict Next-Day Demand for Custom Lag Inputs"):
    lc1, lc2, lc3, lc4 = st.columns(4)
    in_lag1 = lc1.number_input("Yesterday's Demand (lag_1)", min_value=0.0, value=2.0, step=1.0)
    in_lag2 = lc2.number_input("2 Days Ago Demand (lag_2)", min_value=0.0, value=1.0, step=1.0)
    in_lag7 = lc3.number_input("7 Days Ago Demand (lag_7)", min_value=0.0, value=2.0, step=1.0)
    in_is_weekend = lc4.selectbox("Is Weekend?", options=[0, 1], index=0)

    if st.button("Run Live AI Forecast"):
        le_cat = encoders['le_category']
        le_sku = encoders['le_sku']
        cat_enc = le_cat.transform([inspect_cat])[0] if inspect_cat in le_cat.classes_ else 0
        sku_enc = le_sku.transform([inspect_sku])[0] if inspect_sku in le_sku.classes_ else 0

        input_row = pd.DataFrame([{
            'category_encoded': cat_enc,
            'sku_encoded': sku_enc,
            'year': 2022,
            'month': 6,
            'day': 15,
            'day_of_week': 6 if in_is_weekend == 1 else 2,
            'is_weekend': in_is_weekend,
            'lag_1': in_lag1,
            'lag_2': in_lag2,
            'lag_7': in_lag7,
            'rolling_mean_7': np.mean([in_lag1, in_lag2, in_lag7]),
            'rolling_std_7': np.std([in_lag1, in_lag2, in_lag7])
        }])
        live_pred = max(0.0, float(model.predict(input_row)[0]))
        st.info(f"**{best_model_name} Live Next-Day Demand Prediction for `{inspect_sku}`:** `{live_pred:.2f} units`")
