import streamlit as st
import pandas as pd
from black_scholes_model import BlackScholes
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import log, sqrt, exp
from matplotlib.colors import LinearSegmentedColormap

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styled Call and Put displays
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto;
    margin: 0 auto;
}
.metric-call {
    background-color: #90ee90; /* Light green */
    color: black;
    margin-right: 10px;
    border-radius: 10px;
    padding: 10px;
}
.metric-put {
    background-color: #ffcccb; /* Light red */
    color: black;
    border-radius: 10px;
    padding: 10px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}

</style>
""", unsafe_allow_html=True)

# Define and register custom colormap once
colors = ["red", "white", "green"]
custom_cmap = LinearSegmentedColormap.from_list("custom_pnl", colors, N=256)
# Check if colormap is already registered to avoid re-registration
if "custom_pnl" not in plt.colormaps():
    plt.register_cmap(cmap=custom_cmap)

# Sidebar for User Inputs
with st.sidebar:
    st.write("`Created by:`")
    linkedin_url = "www.linkedin.com/in/anujbhandari2"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Anuj Bhandari`</a>', unsafe_allow_html=True)

    st.title("📊 Black-Scholes Model")
    st.write("Adjust the parameters below to calculate option prices.")

    # Input fields
    current_price = st.number_input("Current Asset Price", min_value=0.01, value=100.0, step=0.01)
    strike = st.number_input("Strike Price", min_value=0.01, value=100.0, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.01)
    volatility = st.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", min_value=0.0, value=0.05, step=0.01)


# Function to generate P&L heatmaps
def plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price):
    call_pnl = np.zeros((len(vol_range), len(spot_range)))
    put_pnl = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            # P&L = Current value - Purchase price
            call_pnl[i, j] = bs_temp.call_price - call_purchase_price if call_purchase_price > 0 else bs_temp.call_price
            put_pnl[i, j] = bs_temp.put_price - put_purchase_price if put_purchase_price > 0 else bs_temp.put_price
    

    # Call P&L Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        call_pnl,
        xticklabels=np.round(spot_range, 2),
        yticklabels=np.round(vol_range, 2),
        annot=True,
        fmt=".2f",
        cmap="custom_pnl",
        center=0,
        ax=ax_call
    )
    ax_call.set_title('Call Profit & Loss')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')

    # Put P&L Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        put_pnl,
        xticklabels=np.round(spot_range, 2),
        yticklabels=np.round(vol_range, 2),
        annot=True,
        fmt=".2f",
        cmap="custom_pnl",
        center=0,
        ax=ax_put
    )
    ax_put.set_title('Put Profit & Loss')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

# Main Page
st.title("Black-Scholes Pricing Model")

# Table of Inputs
st.subheader("Input Parameters")
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data, index=[""])
st.table(input_df)

# Create BlackScholes class with user inputs
bs_model = BlackScholes(
    time_to_maturity=time_to_maturity,
    strike=strike,
    current_price=current_price,
    volatility=volatility,
    interest_rate=interest_rate
)

# Calculate Call and Put prices
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in containers
st.subheader("Option Prices")
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    

# Heatmap Section
st.markdown("---")
st.subheader("Options Profit & Loss - Interactive Heatmap")
st.info("Explore profit and loss based on spot prices and volatility. Enter option contract purchase prices below to see P&L; otherwise, raw option values are shown.")


# Heatmap parameters
st.write("Heatmap Parameters")
col1, col2 = st.columns([1, 1], gap="small")
with col1:
    spot_min = st.number_input("Min Spot Price", min_value=0.01, value=current_price * 0.8, step=0.01)
    vol_min = st.slider("Min Volatility", min_value=0.01, max_value=1.0, value=volatility * 0.5, step=0.01)

with col2:
    spot_max = st.number_input("Max Spot Price", min_value=0.01, value=current_price * 1.2, step=0.01)
    vol_max = st.slider("Max Volatility", min_value=0.01, max_value=1.0, value=volatility * 1.5, step=0.01)


col1, col2 = st.columns([1, 1], gap="small")
with col1:
    call_purchase_price = st.number_input("Call Purchase Price (optional)", min_value=0.0, value=0.0, step=0.01)
with col2:
    put_purchase_price = st.number_input("Put Purchase Price (optional)", min_value=0.0, value=0.0, step=0.01)

# Generate ranges and plot heatmaps
spot_range = np.linspace(spot_min, spot_max, 10)
vol_range = np.linspace(vol_min, vol_max, 10)
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.subheader("Call P&L Heatmap")
    heatmap_fig_call, _ = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price)
    st.pyplot(heatmap_fig_call)
with col2:
    st.subheader("Put P&L Heatmap")
    _, heatmap_fig_put = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price)
    st.pyplot(heatmap_fig_put)


# Display greeks
st.markdown("---")
st.subheader("Greeks")

greeks_data = {

    "Calls": [
        f"{bs_model.call_delta:.4f}",
        f"{bs_model.call_theta:.4f}",
        f"{bs_model.call_rho:.4f}",
        f"{bs_model.call_gamma:.4f}",
        f"{bs_model.vega:.4f}"
    ],
    "Puts": [
        f"{bs_model.put_delta:.4f}",
        f"{bs_model.put_theta:.4f}",
        f"{bs_model.put_rho:.4f}",
        f"{bs_model.call_gamma:.4f}",
        f"{bs_model.vega:.4f}"
    ]
}
col1, col2 = st.columns([1, 1], gap="small")
with col1:
    greeks_df = pd.DataFrame(greeks_data, index=["Delta (Δ)", "Theta (Θ)", "Rho (⍴)", "Gamma (Ɣ)", "Vega (ν)"])
    st.table(greeks_df)
