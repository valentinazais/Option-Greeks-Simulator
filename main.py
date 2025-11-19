import streamlit as st
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

# Updated Black-Scholes function with dividend yield (q)
def black_scholes_option_price_and_greeks(S, K, T, r, q, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        raise ValueError("Time to maturity and volatility must be positive.")
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = math.exp(-q * T) * norm.cdf(d1)
        theta = - (S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2) + q * S * math.exp(-q * T) * norm.cdf(d1)
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        delta = -math.exp(-q * T) * norm.cdf(-d1)
        theta = - (S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2) - q * S * math.exp(-q * T) * norm.cdf(-d1)
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")
    
    gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# Streamlit app
st.title("Black-Scholes Option Strategy Dashboard (with Dividend Yield)")

# Sidebar for shared parameters
st.sidebar.header("Shared Parameters")
S = st.sidebar.slider("Current Underlying Price (S)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
T = st.sidebar.slider("Time to Maturity (T)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
r = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
q = st.sidebar.slider("Dividend Yield (q)", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
sigma = st.sidebar.slider("Volatility (sigma)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

# Manage option legs using session state
if 'legs' not in st.session_state:
    st.session_state.legs = []  # List of dicts: {'type': 'call/put', 'strike': float, 'position': 1 (long) or -1 (short)}

# Form to add a new leg
st.sidebar.header("Add Option Leg")
new_type = st.sidebar.selectbox("Option Type", ["call", "put"], key="new_type")
new_strike = st.sidebar.number_input("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
new_position = st.sidebar.selectbox("Position", ["Long (+)", "Short (-)"], key="new_position")
if st.sidebar.button("Add Leg"):
    position_sign = 1 if new_position == "Long (+)" else -1
    st.session_state.legs.append({'type': new_type, 'strike': new_strike, 'position': position_sign})
    st.rerun()  # Refresh to show updated legs

# Display current legs with remove buttons
st.sidebar.header("Current Strategy Legs")
for i, leg in enumerate(st.session_state.legs):
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.write(f"{leg['type'].capitalize()} K={leg['strike']:.0f} ({'Long' if leg['position'] == 1 else 'Short'})")
    with col2:
        if st.button("Remove", key=f"remove_{i}"):
            del st.session_state.legs[i]
            st.rerun()

# Select which metrics to plot (Greeks + Payoff)
st.sidebar.header("Select to Plot")
plot_options = ["Delta", "Gamma", "Theta", "Vega", "Rho", "Payoff"]
selected_plots = st.sidebar.multiselect("Choose Greeks/Payoff (Each on Separate Chart)", plot_options, default=["Payoff"])

# Compute combined results
try:
    if not st.session_state.legs:
        st.warning("Add at least one option leg to compute.")
    else:
        # Compute current values (at fixed S)
        combined_results = {'price': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        combined_payoff = 0  # Payoff at expiration assuming S_T = current S
        
        for leg in st.session_state.legs:
            res = black_scholes_option_price_and_greeks(S, leg['strike'], T, r, q, sigma, leg['type'])
            sign = leg['position']
            for key in combined_results:
                combined_results[key] += sign * res.get(key, 0)
            
            # Payoff for this leg
            if leg['type'] == 'call':
                payoff_leg = max(S - leg['strike'], 0)
            else:
                payoff_leg = max(leg['strike'] - S, 0)
            combined_payoff += sign * payoff_leg
        
        # Display numerical outputs
        st.header("Combined Strategy Values (at Current S)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Net Premium (Total Price)", f"{combined_results['price']:.4f}")
            st.metric("Net Payoff (at Expiration, S_T = Current S)", f"{combined_payoff:.4f}")
        with col2:
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                st.metric(greek.capitalize(), f"{combined_results[greek]:.4f}")
        
        # Generate data for plots (vary S)
        S_range = np.linspace(max(50, S - 50), S + 50, 100)
        
        # Dictionary to hold data for each plot
        plot_data = {plot_name: [] for plot_name in selected_plots}
        
        for s in S_range:
            for plot_name in selected_plots:
                combined_value = 0
                for leg in st.session_state.legs:
                    sign = leg['position']
                    if plot_name == 'Payoff':
                        if leg['type'] == 'call':
                            value = max(s - leg['strike'], 0)
                        else:
                            value = max(leg['strike'] - s, 0)
                    else:
                        res = black_scholes_option_price_and_greeks(s, leg['strike'], T, r, q, sigma, leg['type'])
                        value = res.get(plot_name.lower(), 0)
                    combined_value += sign * value
                plot_data[plot_name].append(combined_value)
        
        # Display each selected plot in its own separate chart with its own scale
        if selected_plots:
            st.header("Combined Plots vs. Underlying Price (S)")
            for plot_name in selected_plots:
                with st.expander(f"{plot_name} vs. S (Click to Expand/Collapse)", expanded=True):
                    df = pd.DataFrame({plot_name: plot_data[plot_name]}, index=S_range)
                    st.line_chart(df, use_container_width=True, color="#0000FF")  # Fixed color: blue for consistency
                    st.caption(f"Combined {plot_name} for the strategy (own scale for visibility).")
except ValueError as e:
    st.error(f"Error: {e}")

# Instructions for deployment
st.sidebar.markdown("### Deployment Notes")
st.sidebar.markdown("Save this as `app.py` (or `main.py`). Create `requirements.txt` with:")
st.sidebar.code("streamlit\nnumpy\nscipy\npandas")  # Removed matplotlib since we're using st.line_chart
st.sidebar.markdown("Upload to GitHub and deploy on Streamlit Cloud.")
