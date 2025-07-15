import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="🚀 AI Supply Chain Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS for Amazing Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .main-header h1 {
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 2.5rem;
        color: white;
        margin: 0;
        text-shadow: 0 0 20px rgba(255,255,255,0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    .prediction-high { 
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
        animation: pulse-red 2s infinite;
    }
    
    .prediction-medium { 
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
        animation: pulse-orange 2s infinite;
    }
    
    .prediction-low { 
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        animation: pulse-green 2s infinite;
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7); }
        50% { box-shadow: 0 0 0 20px rgba(244, 67, 54, 0); }
    }
    
    @keyframes pulse-orange {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.7); }
        50% { box-shadow: 0 0 0 20px rgba(255, 152, 0, 0); }
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        50% { box-shadow: 0 0 0 20px rgba(76, 175, 80, 0); }
    }
    
    .status-online {
        background: linear-gradient(135deg, #4caf50 0%, #81c784 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        animation: glow-green 2s infinite alternate;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #f44336 0%, #ef5350 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
        animation: glow-red 2s infinite alternate;
    }
    
    @keyframes glow-green {
        from { box-shadow: 0 0 5px #4caf50; }
        to { box-shadow: 0 0 20px #4caf50; }
    }
    
    @keyframes glow-red {
        from { box-shadow: 0 0 5px #f44336; }
        to { box-shadow: 0 0 20px #f44336; }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration Data ---
CONFIG = {
    'base_features': [
        'Freight Cost (USD)', 'Weight (Kilograms)', 'Unit quantity', 
        'Shipment Mode', 'Country', 'cost_per_kg', 'warehouse_cost_per_unit',
        'origin_port', 'Scheduled Month', 'Scheduled Day of Week'
    ],
    'newly_engineered_features': [
        'Freight Cost (USD)', 'Weight (Kilograms)', 'Unit quantity',
        'cost_per_kg', 'warehouse_cost_per_unit', 'Scheduled Month',
        'Scheduled Day of Week', 'Shipment Mode_Air', 'Shipment Mode_Ocean',
        'Shipment Mode_Truck', 'Country_Canada', 'Country_China',
        'Country_Germany', 'Country_Japan', 'Country_Mexico',
        'Country_UK', 'Country_USA', 'origin_port_PORTA',
        'origin_port_PORTB', 'origin_port_PORTC'
    ]
}

# --- Initialize Session State ---
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'batch_predictions' not in st.session_state:
    st.session_state.batch_predictions = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

# --- Model Training/Loading ---
@st.cache_resource
def create_and_train_model():
    """Create and train a fast model with sample data."""
    np.random.seed(42)
    n_samples = 5000
    
    # Create synthetic training data
    data = {
        'Freight Cost (USD)': np.random.uniform(100, 5000, n_samples),
        'Weight (Kilograms)': np.random.uniform(10, 2000, n_samples),
        'Unit quantity': np.random.randint(1, 500, n_samples),
        'cost_per_kg': np.random.uniform(0.5, 10, n_samples),
        'warehouse_cost_per_unit': np.random.uniform(0.1, 2, n_samples),
        'Scheduled Month': np.random.randint(1, 13, n_samples),
        'Scheduled Day of Week': np.random.randint(0, 7, n_samples),
        'Shipment Mode': np.random.choice(['Air', 'Ocean', 'Truck'], n_samples),
        'Country': np.random.choice(['USA', 'Canada', 'Mexico', 'Germany', 'UK', 'Japan', 'China'], n_samples),
        'origin_port': np.random.choice(['PORTA', 'PORTB', 'PORTC'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic delay probability
    delay_prob = (
        (df['Freight Cost (USD)'] > 3000) * 0.3 +
        (df['Weight (Kilograms)'] > 1000) * 0.2 +
        (df['Shipment Mode'] == 'Ocean') * 0.25 +
        (df['Shipment Mode'] == 'Truck') * 0.15 +
        (df['Country'].isin(['China', 'Germany'])) * 0.2 +
        (df['Scheduled Month'].isin([12, 1, 2])) * 0.15 +
        np.random.uniform(0, 0.2, n_samples)
    )
    
    df['Delayed'] = (delay_prob > 0.5).astype(int)
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=['Shipment Mode', 'Country', 'origin_port'])
    
    # Prepare features
    feature_cols = [col for col in df_encoded.columns if col != 'Delayed']
    X = df_encoded[feature_cols]
    y = df_encoded['Delayed']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    return model, scaler, feature_cols, metrics

# --- Data Preprocessing ---
def preprocess_input(input_data, feature_cols, scaler):
    """Preprocess input data for prediction."""
    df = pd.DataFrame(input_data)
    df_encoded = pd.get_dummies(df, columns=['Shipment Mode', 'Country', 'origin_port'])
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[feature_cols]
    scaled_data = scaler.transform(df_encoded)
    
    return scaled_data

# --- Generate Dashboard Data ---
@st.cache_data
def generate_dashboard_data():
    """Generate sample data for dashboard visualizations."""
    np.random.seed(42)
    
    # Recent shipments data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    shipments_data = []
    
    for date in dates[-90:]:
        day_of_week = date.weekday()
        week_effect = 1.2 if day_of_week < 5 else 0.8
        seasonal_effect = 1.1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
        
        base_shipments = 25
        n_shipments = int(base_shipments * week_effect * seasonal_effect + np.random.normal(0, 3))
        n_shipments = max(n_shipments, 5)
        
        base_delay_rate = 0.12
        weather_effect = 0.05 * np.sin(2 * np.pi * date.dayofyear / 365 + np.pi)
        delay_rate = max(0.01, min(0.4, base_delay_rate + weather_effect + np.random.normal(0, 0.02)))
        
        shipments_data.append({
            'date': date,
            'total_shipments': n_shipments,
            'delayed_shipments': int(n_shipments * delay_rate),
            'delay_rate': delay_rate,
            'on_time_shipments': n_shipments - int(n_shipments * delay_rate),
            'revenue': n_shipments * np.random.uniform(1500, 3000)
        })
    
    shipments_df = pd.DataFrame(shipments_data)
    
    # Country performance data
    countries = ['USA', 'Canada', 'Mexico', 'Germany', 'UK', 'Japan', 'China']
    country_data = []
    
    for country in countries:
        total = np.random.randint(500, 2000)
        base_delay_rates = {
            'USA': 0.08, 'Canada': 0.06, 'Mexico': 0.15, 'Germany': 0.07,
            'UK': 0.09, 'Japan': 0.05, 'China': 0.18
        }
        delay_rate = base_delay_rates.get(country, 0.12) + np.random.uniform(-0.02, 0.02)
        
        country_data.append({
            'country': country,
            'total_shipments': total,
            'delayed_shipments': int(total * delay_rate),
            'delay_rate': delay_rate,
            'avg_cost': np.random.uniform(800, 3000)
        })
    
    return shipments_df, pd.DataFrame(country_data)

# --- Main Application ---
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI SUPPLY CHAIN COMMAND CENTER</h1>
        <p style="color: white; margin: 0; opacity: 0.9; font-size: 1.2rem;">
            Real-time predictive analytics powered by advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load/Train Model
    with st.spinner('🔄 Initializing AI Engine...'):
        model, scaler, feature_cols, metrics = create_and_train_model()
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols
        st.session_state.model_metrics = metrics
        st.session_state.model_trained = True
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ AI CONTROL PANEL")
        
        # Model Status
        if st.session_state.model_trained:
            st.markdown('<div class="status-online">🟢 AI MODEL: ONLINE</div>', unsafe_allow_html=True)
            st.success(f"✅ Trained on {metrics['training_samples']:,} samples")
            st.info(f"🎯 Accuracy: {metrics['accuracy']:.2%}")
            st.info(f"📊 F1-Score: {metrics['f1_score']:.3f}")
        else:
            st.markdown('<div class="status-offline">🔴 AI MODEL: OFFLINE</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        st.markdown("### 🧭 NAVIGATION")
        page = st.radio("", [
            "🔮 Real-Time Predictions",
            "📊 Analytics Dashboard", 
            "📋 Batch Processing",
            "🎯 AI Insights"
        ])
        
        st.divider()
        
        # Quick Stats
        st.markdown("### 📈 LIVE METRICS")
        if st.session_state.prediction_history:
            recent_predictions = [p['probability'] for p in st.session_state.prediction_history[-10:]]
            avg_risk = np.mean(recent_predictions) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Avg Risk", f"{avg_risk:.1f}%")
            with col2:
                st.metric("📊 Predictions", len(st.session_state.prediction_history))
    
    # Main Content Router
    if page == "🔮 Real-Time Predictions":
        show_prediction_page()
    elif page == "📊 Analytics Dashboard":
        show_dashboard_page()
    elif page == "📋 Batch Processing":
        show_batch_processing_page()
    elif page == "🎯 AI Insights":
        show_insights_page()

def show_prediction_page():
    """Real-time prediction interface."""
    st.markdown("## 🔮 REAL-TIME DELAY PREDICTION")
    
    if not st.session_state.model_trained:
        st.error("❌ AI Model not loaded. Please restart the application.")
        return
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 📝 SHIPMENT DETAILS")
        
        with st.form("prediction_form"):
            st.markdown("**📦 Package Information**")
            freight_cost = st.number_input("💰 Freight Cost (USD)", min_value=0.0, value=1500.0, step=50.0)
            weight = st.number_input("⚖️ Weight (Kg)", min_value=0, value=750, step=25)
            unit_quantity = st.number_input("📦 Unit Quantity", min_value=1, value=150, step=10)
            
            st.markdown("**🚚 Logistics Details**")
            shipment_mode = st.selectbox("🚛 Shipment Mode", options=['Air', 'Ocean', 'Truck'])
            country = st.selectbox("🌍 Destination", options=['USA', 'Canada', 'Mexico', 'Germany', 'UK', 'Japan', 'China'])
            origin_port = st.selectbox("🏭 Origin Port", options=['PORTA', 'PORTB', 'PORTC'])
            
            st.markdown("**💵 Cost Analysis**")
            cost_per_kg = st.number_input("💸 Cost per Kg", min_value=0.0, value=2.0, step=0.1)
            warehouse_cost = st.number_input("🏪 Warehouse Cost/Unit", min_value=0.0, value=0.8, step=0.1)
            
            st.markdown("**📅 Scheduling**")
            delivery_date = st.date_input("📅 Delivery Date", value=datetime.now() + timedelta(days=7))
            
            submitted = st.form_submit_button("🚀 ANALYZE RISK", type="primary", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = {
                    'Freight Cost (USD)': [freight_cost],
                    'Weight (Kilograms)': [weight],
                    'Unit quantity': [unit_quantity],
                    'Shipment Mode': [shipment_mode],
                    'Country': [country],
                    'cost_per_kg': [cost_per_kg],
                    'warehouse_cost_per_unit': [warehouse_cost],
                    'origin_port': [origin_port],
                    'Scheduled Month': [delivery_date.month],
                    'Scheduled Day of Week': [delivery_date.weekday()]
                }
                
                try:
                    # Make prediction
                    processed_data = preprocess_input(input_data, st.session_state.feature_cols, st.session_state.scaler)
                    prediction_proba = st.session_state.model.predict_proba(processed_data)[0][1]
                    prediction_class = st.session_state.model.predict(processed_data)[0]
                    
                    # Store results
                    st.session_state.prediction_proba = prediction_proba
                    st.session_state.prediction_class = prediction_class
                    
                    # Store in history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'probability': prediction_proba,
                        'class': prediction_class,
                        'input_data': input_data,
                        'risk_level': 'HIGH' if prediction_proba > 0.7 else 'MEDIUM' if prediction_proba > 0.4 else 'LOW'
                    })
                    
                    st.success("✅ Analysis Complete!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
    
    with col2:
        st.markdown("### 🎯 RISK ANALYSIS")
        
        if 'prediction_proba' in st.session_state:
            prob = st.session_state.prediction_proba * 100
            
            # Risk display with proper f-string formatting
            if prob > 70:
                risk_html = f"""
                <div class="prediction-high metric-card">
                    <h2>🚨 HIGH RISK ALERT</h2>
                    <h1 style="color: #f44336; font-size: 3rem;">{prob:.1f}%</h1>
                    <p>⚠️ High probability of delay</p>
                    <p><strong>Recommended Action:</strong> Expedite processing</p>
                </div>
                """
            elif prob > 40:
                risk_html = f"""
                <div class="prediction-medium metric-card">
                    <h2>⚠️ MEDIUM RISK</h2>
                    <h1 style="color: #ff9800; font-size: 3rem;">{prob:.1f}%</h1>
                    <p>🟡 Moderate probability of delay</p>
                    <p><strong>Recommended Action:</strong> Monitor closely</p>
                </div>
                """
            else:
                risk_html = f"""
                <div class="prediction-low metric-card">
                    <h2>✅ LOW RISK</h2>
                    <h1 style="color: #4caf50; font-size: 3rem;">{prob:.1f}%</h1>
                    <p>🟢 Low probability of delay</p>
                    <p><strong>Status:</strong> On track for on-time delivery</p>
                </div>
                """
            
            st.markdown(risk_html, unsafe_allow_html=True)
            
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                title = {'text': "Risk Level"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 📊 PREDICTION HISTORY")
        
        if st.session_state.prediction_history:
            # Recent predictions
            recent = st.session_state.prediction_history[-10:]
            
            for i, pred in enumerate(reversed(recent)):
                with st.expander(f"🔍 Prediction {len(recent)-i} - {pred['risk_level']}"):
                    st.write(f"**Time:** {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Risk:** {pred['probability']*100:.1f}%")
                    st.write(f"**Class:** {'Delayed' if pred['class'] == 1 else 'On Time'}")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("📈 No predictions yet. Make your first prediction!")

def show_dashboard_page():
    """Analytics dashboard with visualizations."""
    st.markdown("## 📊 ANALYTICS DASHBOARD")
    
    # Generate sample data
    shipments_df, country_df = generate_dashboard_data()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_shipments = shipments_df['total_shipments'].sum()
        st.metric("📦 Total Shipments", f"{total_shipments:,}", delta="↗️ 5.2%")
    
    with col2:
        avg_delay_rate = shipments_df['delay_rate'].mean()
        st.metric("⏱️ Avg Delay Rate", f"{avg_delay_rate:.1%}", delta="↘️ 2.1%")
    
    with col3:
        total_revenue = shipments_df['revenue'].sum()
        st.metric("💰 Total Revenue", f"${total_revenue:,.0f}", delta="↗️ 8.3%")
    
    with col4:
        on_time_rate = 1 - avg_delay_rate
        st.metric("✅ On-Time Rate", f"{on_time_rate:.1%}", delta="↗️ 1.8%")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Shipment Trends")
        fig = px.line(shipments_df, x='date', y='total_shipments', 
                     title="Daily Shipment Volume")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Delay Rate Trends")
        fig = px.line(shipments_df, x='date', y='delay_rate', 
                     title="Daily Delay Rate")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Country performance
    st.markdown("### 🌍 Country Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(country_df, x='country', y='total_shipments',
                    title="Shipments by Country")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(country_df, x='country', y='delay_rate',
                    title="Delay Rate by Country")
        st.plotly_chart(fig, use_container_width=True)

def show_batch_processing_page():
    """Batch processing interface."""
    st.markdown("## 📋 BATCH PROCESSING")
    
    if not st.session_state.model_trained:
        st.error("❌ AI Model not loaded. Please restart the application.")
        return
    
    st.info("📁 Upload a CSV file with shipment data for batch predictions.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} records found.")
            
            # Display sample data
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head())
            
            # Check required columns
            required_cols = ['Freight Cost (USD)', 'Weight (Kilograms)', 'Unit quantity', 
                           'Shipment Mode', 'Country', 'cost_per_kg', 'warehouse_cost_per_unit', 
                           'origin_port', 'Scheduled Month', 'Scheduled Day of Week']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("Please ensure your CSV contains all required columns.")
            else:
                if st.button("🚀 Process Batch Predictions", type="primary"):
                    with st.spinner("🔄 Processing batch predictions..."):
                    # Prepare batch data
                        batch_data = {
                            'Freight Cost (USD)': df['Freight Cost (USD)'].tolist(),
                            'Weight (Kilograms)': df['Weight (Kilograms)'].tolist(),
                            'Unit quantity': df['Unit quantity'].tolist(),
                            'Shipment Mode': df['Shipment Mode'].tolist(),
                            'Country': df['Country'].tolist(),
                            'cost_per_kg': df['cost_per_kg'].tolist(),
                            'warehouse_cost_per_unit': df['warehouse_cost_per_unit'].tolist(),
                            'origin_port': df['origin_port'].tolist(),
                            'Scheduled Month': df['Scheduled Month'].tolist(),
                            'Scheduled Day of Week': df['Scheduled Day of Week'].tolist()
                        }
                        
                        try:
                            # Make batch predictions
                            processed_data = preprocess_input(batch_data, st.session_state.feature_cols, st.session_state.scaler)
                            predictions_proba = st.session_state.model.predict_proba(processed_data)[:, 1]
                            predictions_class = st.session_state.model.predict(processed_data)
                            
                            # Create results dataframe
                            results_df = df.copy()
                            results_df['Delay_Probability'] = predictions_proba
                            results_df['Predicted_Class'] = ['Delayed' if pred == 1 else 'On Time' for pred in predictions_class]
                            results_df['Risk_Level'] = ['HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.4 else 'LOW' for prob in predictions_proba]
                            
                            # Store batch results
                            st.session_state.batch_predictions = results_df
                            
                            st.success("✅ Batch processing complete!")
                            
                            # Display results
                            st.markdown("### 📊 Batch Results")
                            st.dataframe(results_df[['Freight Cost (USD)', 'Weight (Kilograms)', 'Country', 'Delay_Probability', 'Predicted_Class', 'Risk_Level']])
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                high_risk = len(results_df[results_df['Risk_Level'] == 'HIGH'])
                                st.metric("🚨 High Risk", high_risk)
                            with col2:
                                medium_risk = len(results_df[results_df['Risk_Level'] == 'MEDIUM'])
                                st.metric("⚠️ Medium Risk", medium_risk)
                            with col3:
                                low_risk = len(results_df[results_df['Risk_Level'] == 'LOW'])
                                st.metric("✅ Low Risk", low_risk)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Batch processing failed: {e}")
                            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    # Display previous batch results if available
    if hasattr(st.session_state, 'batch_predictions') and len(st.session_state.batch_predictions) > 0:
        st.markdown("### 📁 Previous Batch Results")
        st.dataframe(st.session_state.batch_predictions.head(10))

def show_insights_page():
    """AI insights and feature importance."""
    st.markdown("## 🎯 AI INSIGHTS")
    
    if not st.session_state.model_trained:
        st.error("❌ AI Model not loaded. Please restart the application.")
        return
    
    # Model performance metrics
    st.markdown("### 📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{st.session_state.model_metrics['accuracy']:.2%}")
    with col2:
        st.metric("📊 Precision", f"{st.session_state.model_metrics['precision']:.2%}")
    with col3:
        st.metric("🔍 Recall", f"{st.session_state.model_metrics['recall']:.2%}")
    with col4:
        st.metric("⚖️ F1-Score", f"{st.session_state.model_metrics['f1_score']:.3f}")
    
    # Feature importance
    st.markdown("### 🔍 Feature Importance")
    
    if hasattr(st.session_state.model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': st.session_state.feature_cols,
            'importance': st.session_state.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Top 10 features
        top_features = feature_importance.head(10)
        
        fig = px.bar(top_features, x='importance', y='feature', 
                    orientation='h', title="Top 10 Most Important Features")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.markdown("### 📋 All Features")
        st.dataframe(feature_importance)
    
    # Prediction insights
    if st.session_state.prediction_history:
        st.markdown("### 📈 Prediction Insights")
        
        # Convert to DataFrame for analysis
        history_df = pd.DataFrame([{
            'timestamp': pred['timestamp'],
            'probability': pred['probability'],
            'risk_level': pred['risk_level']
        } for pred in st.session_state.prediction_history])
        
        # Risk distribution
        risk_counts = history_df['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                    title="Risk Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction trends
        history_df['hour'] = history_df['timestamp'].dt.hour
        hourly_avg = history_df.groupby('hour')['probability'].mean()
        
        fig = px.line(x=hourly_avg.index, y=hourly_avg.values, 
                     title="Average Risk by Hour")
        fig.update_xaxes(title="Hour of Day")
        fig.update_yaxes(title="Average Risk Probability")
        st.plotly_chart(fig, use_container_width=True)
    
    # AI recommendations
    st.markdown("### 🤖 AI Recommendations")
    
    recommendations = [
        "📦 **Shipment Optimization**: Consider consolidating smaller shipments to reduce per-unit costs",
        "🌍 **Route Planning**: Focus on countries with lower historical delay rates",
        "📅 **Scheduling**: Avoid peak season months (December-February) when possible",
        "🚛 **Mode Selection**: Air freight shows lower delay rates for urgent shipments",
        "💰 **Cost Management**: Monitor cost per kg ratios to identify optimization opportunities"
    ]
    
    for rec in recommendations:
        st.markdown(f"• {rec}")
    
    # Export insights
    if st.button("📊 Generate Insights Report"):
        insights_data = {
            'model_metrics': st.session_state.model_metrics,
            'feature_importance': feature_importance.to_dict() if 'feature_importance' in locals() else {},
            'prediction_history': len(st.session_state.prediction_history),
            'recommendations': recommendations
        }
        
        st.json(insights_data)
        st.success("📋 Insights report generated!")

if __name__ == "__main__":
    main()