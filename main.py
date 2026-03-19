## Black-Scholes Option Strategy Dashboard
# link : https://option-greeks.streamlit.app/
# Interactive dashboard for pricing option strategies using Black-Scholes model (with dividend yield). 
# Computes prices, Greeks (Delta, Gamma, Theta, Vega, Rho), payoffs, time value, and premiums. 
# Plots combined or separate metrics vs. underlying price (S).



import streamlit as st
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Apply a nicer style to Matplotlib to make it look more like Streamlit's aesthetic
plt.style.use('ggplot')

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

# ─── Page config ───
st.set_page_config(page_title="Option Greeks Simulator", layout="wide")

# ─── Custom CSS for cleaner look ───
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 8px;
        padding: 12px 16px;
    }
    section[data-testid="stSidebar"] { width: 320px; }
</style>
""", unsafe_allow_html=True)

# ─── Title ───
st.title("Option Greeks Simulator")
st.caption("Black-Scholes pricing · All Greeks · Strategy builder")

# ─── Session-state defaults ───
if 'S' not in st.session_state:
    st.session_state['S'] = 100.0
if 'T' not in st.session_state:
    st.session_state['T'] = 1.0
if 'r' not in st.session_state:
    st.session_state['r'] = 0.05
if 'q' not in st.session_state:
    st.session_state['q'] = 0.0
if 'sigma' not in st.session_state:
    st.session_state['sigma'] = 0.2

# ─── DEFAULT option leg: 1 Long Call K=100 ───
if 'legs' not in st.session_state:
    st.session_state.legs = [{'type': 'call', 'strike': 100.0, 'position': 1}]

if 'single_plots' not in st.session_state:
    st.session_state.single_plots = []

# ─── Sidebar ──────────────────────────────────────────────

# Callbacks to update session state and sync widgets
def update_S():
    st.session_state['S'] = st.session_state['num_S']
    st.session_state['slider_S'] = st.session_state['num_S']

def update_slider_S():
    st.session_state['S'] = st.session_state['slider_S']
    st.session_state['num_S'] = st.session_state['slider_S']

def update_T():
    st.session_state['T'] = st.session_state['num_T']
    st.session_state['slider_T'] = st.session_state['num_T']

def update_slider_T():
    st.session_state['T'] = st.session_state['slider_T']
    st.session_state['num_T'] = st.session_state['slider_T']

def update_r():
    st.session_state['r'] = st.session_state['num_r']
    st.session_state['slider_r'] = st.session_state['num_r']

def update_slider_r():
    st.session_state['r'] = st.session_state['slider_r']
    st.session_state['num_r'] = st.session_state['slider_r']

def update_q():
    st.session_state['q'] = st.session_state['num_q']
    st.session_state['slider_q'] = st.session_state['num_q']

def update_slider_q():
    st.session_state['q'] = st.session_state['slider_q']
    st.session_state['num_q'] = st.session_state['slider_q']

def update_sigma():
    st.session_state['sigma'] = st.session_state['num_sigma']
    st.session_state['slider_sigma'] = st.session_state['num_sigma']

def update_slider_sigma():
    st.session_state['sigma'] = st.session_state['slider_sigma']
    st.session_state['num_sigma'] = st.session_state['slider_sigma']

st.sidebar.header("Market Parameters")

# Current Underlying Price (S)
st.sidebar.slider("Underlying Price (S)", min_value=50.0, max_value=150.0, value=st.session_state['S'], step=1.0, key='slider_S', on_change=update_slider_S)
st.sidebar.number_input("S", min_value=50.0, max_value=150.0, value=st.session_state['S'], step=1.0, key='num_S', on_change=update_S, label_visibility="collapsed")
S = st.session_state['S']

# Time to Maturity (T)
st.sidebar.slider("Time to Maturity (T in years)", min_value=0.01, max_value=5.0, value=st.session_state['T'], step=0.01, key='slider_T', on_change=update_slider_T)
st.sidebar.number_input("T", min_value=0.01, max_value=5.0, value=st.session_state['T'], step=0.01, key='num_T', on_change=update_T, label_visibility="collapsed")
T = st.session_state['T']

# Risk-Free Rate (r)
st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=st.session_state['r'], step=0.01, key='slider_r', on_change=update_slider_r)
st.sidebar.number_input("r", min_value=0.0, max_value=0.2, value=st.session_state['r'], step=0.01, key='num_r', on_change=update_r, label_visibility="collapsed")
r = st.session_state['r']

# Dividend Yield (q)
st.sidebar.slider("Dividend Yield (q)", min_value=0.0, max_value=0.2, value=st.session_state['q'], step=0.01, key='slider_q', on_change=update_slider_q)
st.sidebar.number_input("q", min_value=0.0, max_value=0.2, value=st.session_state['q'], step=0.01, key='num_q', on_change=update_q, label_visibility="collapsed")
q = st.session_state['q']

# Volatility (sigma)
st.sidebar.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=st.session_state['sigma'], step=0.01, key='slider_sigma', on_change=update_slider_sigma)
st.sidebar.number_input("sigma", min_value=0.01, max_value=1.0, value=st.session_state['sigma'], step=0.01, key='num_sigma', on_change=update_sigma, label_visibility="collapsed")
sigma = st.session_state['sigma']

st.sidebar.divider()

# ─── Add option leg ───
st.sidebar.header("Add Option Leg")
new_type = st.sidebar.selectbox("Option Type", ["call", "put"], key="new_type")
new_strike = st.sidebar.number_input("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
new_position = st.sidebar.selectbox("Position", ["Long (+)", "Short (-)"], key="new_position")
if st.sidebar.button("Add Leg", use_container_width=True):
    position_sign = 1 if new_position == "Long (+)" else -1
    st.session_state.legs.append({'type': new_type, 'strike': new_strike, 'position': position_sign})
    st.rerun()

# ─── Current legs ───
st.sidebar.divider()
st.sidebar.header("Strategy Legs")
if not st.session_state.legs:
    st.sidebar.info("No legs yet — add one above.")
for i, leg in enumerate(st.session_state.legs):
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        pos_label = "Long" if leg['position'] == 1 else "Short"
        st.write(f"**{leg['type'].capitalize()}** K={leg['strike']:.0f} ({pos_label})")
    with col2:
        if st.button("✕", key=f"remove_{i}"):
            del st.session_state.legs[i]
            st.rerun()

st.sidebar.divider()

# ─── Plot options ───
st.sidebar.header("Plot Settings")
plot_options = ["Payoff", "Delta", "Gamma", "Theta", "Vega", "Rho", "Time Value", "Premium"]
selected_plots = st.sidebar.multiselect("Overlay on Combined Graph", plot_options, default=["Payoff"])
show_separate = st.sidebar.checkbox("Show separate graph per metric (with Payoff)")

st.sidebar.divider()
st.sidebar.header("Add Individual Graph")
single_metric = st.sidebar.selectbox("Metric", ["Delta", "Gamma", "Theta", "Vega", "Rho", "Time Value", "Premium"], key="single_metric")
if st.sidebar.button("Add Graph", use_container_width=True):
    if single_metric not in st.session_state.single_plots:
        st.session_state.single_plots.append(single_metric)
    st.rerun()

if st.session_state.single_plots:
    st.sidebar.caption("Active individual graphs:")
    for i, metric in enumerate(st.session_state.single_plots):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.write(f"{metric} + Payoff")
        with col2:
            if st.button("✕", key=f"remove_single_{i}"):
                del st.session_state.single_plots[i]
                st.rerun()

# ─── Fixed colors for each metric ───
colors = {
    'Payoff': '#1a1a1a',
    'Time Value': '#006666',
    'Premium': '#556B2F',
    'Delta': '#00008b',
    'Gamma': '#006400',
    'Theta': '#8b0000',
    'Vega': '#4b0082',
    'Rho': '#cc6600'
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN CONTENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    if not st.session_state.legs:
        st.warning("Add at least one option leg to compute.")
    else:
        # ── Compute current values ──
        combined_results = {'price': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        combined_payoff = 0
        combined_time_value = 0
        
        for leg in st.session_state.legs:
            res = black_scholes_option_price_and_greeks(S, leg['strike'], T, r, q, sigma, leg['type'])
            sign = leg['position']
            for key in combined_results:
                combined_results[key] += sign * res.get(key, 0)
            
            if leg['type'] == 'call':
                payoff_leg = max(S - leg['strike'], 0)
            else:
                payoff_leg = max(leg['strike'] - S, 0)
            combined_payoff += sign * payoff_leg
            
            time_value_leg = res['price'] - payoff_leg
            combined_time_value += sign * time_value_leg

        # ── Display Greeks & values — 2 per row, clean grid ──
        st.header("Strategy Values")

        # Row 1: Premium & Payoff
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Net Premium", f"{combined_results['price']:.4f}")
        r1c2.metric("Payoff (Intrinsic)", f"{combined_payoff:.4f}")

        # Row 2: Time Value & Delta
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Time Value", f"{combined_time_value:.4f}")
        r2c2.metric("Delta (Δ)", f"{combined_results['delta']:.4f}")

        # Row 3: Gamma & Theta
        r3c1, r3c2 = st.columns(2)
        r3c1.metric("Gamma (Γ)", f"{combined_results['gamma']:.4f}")
        r3c2.metric("Theta (Θ)", f"{combined_results['theta']:.4f}")

        # Row 4: Vega & Rho
        r4c1, r4c2 = st.columns(2)
        r4c1.metric("Vega (ν)", f"{combined_results['vega']:.4f}")
        r4c2.metric("Rho (ρ)", f"{combined_results['rho']:.4f}")

        st.divider()
        
        # ── Generate data for plots (vary S) ──
        S_range = np.linspace(max(50, S - 50), S + 50, 100)
        
        plot_data = {plot_name: [] for plot_name in plot_options}
        
        for s in S_range:
            for plot_name in plot_options:
                if plot_name == 'Payoff':
                    combined_value = 0
                    for leg in st.session_state.legs:
                        sign = leg['position']
                        if leg['type'] == 'call':
                            value = max(s - leg['strike'], 0)
                        else:
                            value = max(leg['strike'] - s, 0)
                        combined_value += sign * value
                elif plot_name == 'Time Value':
                    combined_value = 0
                    for leg in st.session_state.legs:
                        res = black_scholes_option_price_and_greeks(s, leg['strike'], T, r, q, sigma, leg['type'])
                        price = res['price']
                        if leg['type'] == 'call':
                            intrinsic = max(s - leg['strike'], 0)
                        else:
                            intrinsic = max(leg['strike'] - s, 0)
                        time_value_leg = price - intrinsic
                        combined_value += leg['position'] * time_value_leg
                elif plot_name == 'Premium':
                    combined_value = 0
                    for leg in st.session_state.legs:
                        res = black_scholes_option_price_and_greeks(s, leg['strike'], T, r, q, sigma, leg['type'])
                        combined_value += leg['position'] * res['price']
                else:
                    combined_value = 0
                    for leg in st.session_state.legs:
                        res = black_scholes_option_price_and_greeks(s, leg['strike'], T, r, q, sigma, leg['type'])
                        value = res.get(plot_name.lower(), 0)
                        combined_value += leg['position'] * value
                plot_data[plot_name].append(combined_value)
        
        # ── Combined plot ──
        if selected_plots:
            st.header("Combined Strategy Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]
            lines = []
            
            for i, plot_name in enumerate(selected_plots):
                if i == 0:
                    line, = ax.plot(S_range, plot_data[plot_name], color=colors[plot_name], label=plot_name)
                    ax.set_ylabel(plot_name, color=colors[plot_name])
                    ax.tick_params(axis='y', colors=colors[plot_name])
                else:
                    new_ax = ax.twinx()
                    new_ax.spines['right'].set_position(('axes', 1.0 + 0.1 * (i - 1)))
                    line, = new_ax.plot(S_range, plot_data[plot_name], color=colors[plot_name], label=plot_name)
                    new_ax.set_ylabel(plot_name, color=colors[plot_name])
                    new_ax.tick_params(axis='y', colors=colors[plot_name])
                    axes.append(new_ax)
                lines.append(line)
            
            ax.set_xlabel('Underlying Price (S)')
            ax.set_title('Combined Metrics (Each with Own Y-Scale)')
            ax.grid(True)
            ax.legend(lines, [line.get_label() for line in lines], loc='upper left')
            st.pyplot(fig)
        
        # ── Separate graphs ──
        if show_separate:
            metrics_selected = [p for p in selected_plots if p != "Payoff"]
            if not metrics_selected:
                st.info("Select metrics besides Payoff to generate separate plots.")
            else:
                st.header("Separate Metric Graphs")
                # 2 charts per row
                for idx in range(0, len(metrics_selected), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if idx + j < len(metrics_selected):
                            metric = metrics_selected[idx + j]
                            with col:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.plot(S_range, plot_data["Payoff"], color=colors['Payoff'], label='Payoff')
                                ax.set_ylabel('Payoff', color=colors['Payoff'])
                                ax.tick_params(axis='y', colors=colors['Payoff'])
                                ax2 = ax.twinx()
                                ax2.plot(S_range, plot_data[metric], color=colors[metric], label=metric)
                                ax2.set_ylabel(metric, color=colors[metric])
                                ax2.tick_params(axis='y', colors=colors[metric])
                                ax.set_xlabel('S')
                                ax.set_title(f'{metric} & Payoff')
                                ax.grid(True)
                                all_lines = ax.get_lines() + ax2.get_lines()
                                ax.legend(all_lines, [l.get_label() for l in all_lines], loc='upper left', fontsize=8)
                                st.pyplot(fig)
        
        # ── Individual added graphs ──
        if st.session_state.single_plots:
            st.header("Individual Metric Graphs")
            for idx in range(0, len(st.session_state.single_plots), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if idx + j < len(st.session_state.single_plots):
                        metric = st.session_state.single_plots[idx + j]
                        with col:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(S_range, plot_data["Payoff"], color=colors['Payoff'], label='Payoff')
                            ax.set_ylabel('Payoff', color=colors['Payoff'])
                            ax.tick_params(axis='y', colors=colors['Payoff'])
                            ax2 = ax.twinx()
                            ax2.plot(S_range, plot_data[metric], color=colors[metric], label=metric)
                            ax2.set_ylabel(metric, color=colors[metric])
                            ax2.tick_params(axis='y', colors=colors[metric])
                            ax.set_xlabel('S')
                            ax.set_title(f'{metric} & Payoff')
                            ax.grid(True)
                            all_lines = ax.get_lines() + ax2.get_lines()
                            ax.legend(all_lines, [l.get_label() for l in all_lines], loc='upper left', fontsize=8)
                            st.pyplot(fig)
except ValueError as e:
    st.error(f"Error: {e}")
