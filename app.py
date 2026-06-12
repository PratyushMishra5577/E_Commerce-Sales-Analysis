import streamlit as st
import pandas as pd
import joblib
import pickle
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(layout="wide")
st.markdown("""
<style>
/* Main app background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Optional: improve text contrast */
h1, h2, h3, h4, h5, h6, p, label {
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)
# ================= LOGIN =================
def login():
    # 🔥 BANNER / LOGO AT TOP
    col1, col2 = st.columns([1, 6])

    with col1:
        st.image("storelogo.jpg", width=70)   # keep logo in project folder

    with col2:
        st.markdown("""
        <h1 style="color:#00ffcc; margin-bottom:0;">Sales Prediction Dashboard</h1>
        <p style="color:gray;">E-Commerce Data Analysis Project</p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 🔹 HOW TO USE
    st.markdown("""
    <div style="
        background-color:#111827;
        padding:15px;
        border-radius:10px;
        margin-bottom:15px;
    ">
    <h3 style="color:#00ffcc;">📘 How to Use This Dashboard</h3>

    <ul style="font-size:14px;">
    <li>Login using provided credentials</li>
    <li>Select <b>Region</b> and <b>Category</b> to filter data</li>
    <li>Analyze <b>Sales & Profit graphs</b></li>
    <li>Use <b>Prediction</b> for future sales (2026–2028)</li>
    <li>View <b>Forecast</b> for trends</li>
    <li>Download report using <b>PDF button</b></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # 🔹 LOGIN INPUTS
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state["login"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()
# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("Super_Store.csv", encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

    df["Order Month"] = df["Order Date"].dt.month
    df["Order year"] = df["Order Date"].dt.year
    df["Order Day of Week"] = df["Order Date"].dt.dayofweek

    return df

data = load_data()

# ================= LOAD MODEL =================
model = joblib.load("sales_model.pkl")

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# ================= MAIN APP =================
def main_app():

    st.title("📊 Sales Dashboard")

    # FILTER
    col1, col2 = st.columns(2)
    region = col1.selectbox("Region", data["Region"].unique())
    category = col2.selectbox("Category", data["Category"].unique())

    filtered = data[(data["Region"] == region) & (data["Category"] == category)]

    # KPI
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Sales", f"{filtered['Sales'].sum():.0f}")
    k2.metric("Total Profit", f"{filtered['Profit'].sum():.0f}")
    k3.metric("Orders", filtered.shape[0])

    st.markdown("---")

    # ================= GRAPH LOGIC =================
    month_sales = data.groupby("Order Month")["Sales"].sum().reset_index()
    year_sales = data.groupby("Order year")["Sales"].sum().reset_index()

    profit_by_month = data.groupby("Order Month")["Profit"].sum().reset_index()
    profit_by_year = data.groupby("Order year")["Profit"].sum().reset_index()

    region_data = data[data["Region"] == region]

    sales_by_category = region_data.groupby("Category")["Sales"].sum().reset_index()
    sales_by_subcategory = region_data.groupby("Sub-Category")["Sales"].sum().reset_index()

    profit_by_category = region_data.groupby("Category")["Profit"].sum().reset_index()
    profit_by_subcategory = region_data.groupby("Sub-Category")["Profit"].sum().reset_index()

    sales_profit_by_segment = region_data.groupby("Segment").agg({
        "Sales": "sum",
        "Profit": "sum"
    }).reset_index()

    # ================= GRAPHS =================
    st.subheader("Sales Analysis")

    st.plotly_chart(px.line(month_sales, x="Order Month", y="Sales"), use_container_width=True)
    st.plotly_chart(px.line(year_sales, x="Order year", y="Sales"), use_container_width=True)
    st.plotly_chart(px.pie(sales_by_category, names="Category", values="Sales"), use_container_width=True)
    st.plotly_chart(px.bar(sales_by_subcategory, x="Sub-Category", y="Sales"), use_container_width=True)

    st.subheader("Profit Analysis")

    st.plotly_chart(px.line(profit_by_month, x="Order Month", y="Profit"), use_container_width=True)
    st.plotly_chart(px.line(profit_by_year, x="Order year", y="Profit"), use_container_width=True)
    st.plotly_chart(px.pie(profit_by_category, names="Category", values="Profit", hole=0.5), use_container_width=True)
    st.plotly_chart(px.bar(profit_by_subcategory, x="Sub-Category", y="Profit"), use_container_width=True)

    fig9 = go.Figure()
    fig9.add_bar(x=sales_profit_by_segment["Segment"], y=sales_profit_by_segment["Sales"], name="Sales")
    fig9.add_bar(x=sales_profit_by_segment["Segment"], y=sales_profit_by_segment["Profit"], name="Profit")
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("---")

    # ================= PREDICTION =================
    st.subheader("Sales Prediction (2026–2028)")

    month = st.selectbox("Month", list(range(1, 13)))
    year = st.selectbox("Year", [2026, 2027, 2028])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Quantity": 2,
            "Discount": 0.1,
            "Order Month": month,
            "Order year": year,
            "Order Day of Week": 2
        }])

        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        pred = model.predict(input_df)
        st.success(f"Predicted Sales: ₹ {pred[0]:.2f}")

    # ================= WHAT IF =================
    st.subheader("🧪 What-if Scenario")

    q = st.slider("Quantity", 1, 10, 2)
    d = st.slider("Discount", 0.0, 0.5, 0.1)

    inp2 = pd.DataFrame([{
        "Quantity": q,
        "Discount": d,
        "Order Month": month,
        "Order year": year,
        "Order Day of Week": 2
    }]).reindex(columns=model_columns, fill_value=0)

    st.metric("Scenario Prediction", f"₹ {model.predict(inp2)[0]:,.2f}")
     # ================= FEATURE IMPORTANCE =================
    st.subheader("Feature Importance")

    try:
        imp = model.feature_importances_
        feat = pd.DataFrame({"Feature": model_columns, "Importance": imp}).sort_values("Importance", ascending=False).head(10)
        st.plotly_chart(px.bar(feat, x="Importance", y="Feature", orientation='h'))
    except:
        st.info("Not available")

    # ================= PROPHET =================
    st.subheader("Forecast (2026–2028)")

    monthly = data.groupby("Order Date")["Sales"].sum().reset_index()
    monthly = monthly.set_index("Order Date").resample("MS").sum().reset_index()

    prophet_df = monthly.rename(columns={"Order Date": "ds", "Sales": "y"})

    m = Prophet()
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=36, freq="MS")
    forecast = m.predict(future)

    forecast = forecast[
        (forecast["ds"].dt.year >= 2026) &
        (forecast["ds"].dt.year <= 2028)
    ]

    fig = px.line(forecast, x="ds", y="yhat")

    fig.update_layout(xaxis_title=None, yaxis_title=None)

    fig.update_traces(
        hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Sales:</b> ₹ %{y:.2f}<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ================= MODEL COMPARISON =================
    st.subheader("📊 Model Comparison")

    models = ["LR", "DT", "RF", "GB", "XGB"]
    scores = [0.03, 0.51, 0.60, 0.58, 0.53]

    st.plotly_chart(px.bar(x=models, y=scores, title="R2 Score"), use_container_width=True)

    # ================= ACCURACY VISUAL =================
    st.subheader("📈 Accuracy Visualization")

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=models, y=scores, mode="lines+markers"))

    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")
     # ================= EVALUATION =================
    st.subheader("Model Evaluation")

    sample = data.sample(200)
    X_s = pd.get_dummies(sample.drop("Sales", axis=1)).reindex(columns=model_columns, fill_value=0)
    y_t = sample["Sales"]
    y_p = model.predict(X_s)

    st.plotly_chart(px.scatter(x=y_t, y=y_p))

    st.plotly_chart(px.histogram(y_t - y_p))

    # ================= DOWNLOAD PDF =================
    st.subheader("⬇️ Download Report")

    if st.button("Generate PDF"):
        generate_pdf()
        with open("report.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="report.pdf")

   # ================= LIMITATIONS =================
    st.subheader("⚠️ Limitations")

    st.markdown("""
    - Uses historical data  
    - Moderate accuracy (~0.6 R²)  
    - Limited features  
    """)

    st.markdown("---")

    st.markdown("Developed by Pratyush Mishra")

# ================= ROUTING =================
if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()

main_app()