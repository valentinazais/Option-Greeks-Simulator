# Black-Scholes Option Strategy Dashboard

Live app: [https://option-greeks.streamlit.app/](https://option-greeks.streamlit.app/)

## Overview
Interactive dashboard for pricing option strategies using Black-Scholes model (with dividend yield). Computes prices, Greeks (Delta, Gamma, Theta, Vega, Rho), payoffs, time value, and premiums. Plots combined or separate metrics vs. underlying price (S).

## Features
- **Shared Parameters**: S (spot), T (time to maturity), r (risk-free rate), q (dividend yield), σ (volatility). Dual input (number/slider) with sync.
- **Strategy Builder**: Add/remove legs (call/put, strike K, long/short).
- **Metrics**: Net premium, payoff (intrinsic), time value, Greeks at current S.
- **Plots**:
  - Overlay multiple metrics (each own y-scale) vs. S range.
  - Separate metric + payoff graphs.
  - Add/remove single metric + payoff graphs.
- Fixed colors per metric for consistency.

## Usage
1. **Sidebar > Shared Parameters**: Adjust S, T, r, q, σ.
2. **Sidebar > Add Option Leg**: Select type (call/put), K, position (long/short). Click "Add Leg".
3. **Sidebar > Current Strategy Legs**: View/remove legs.
4. **Sidebar > Select to Plot**: Choose metrics (e.g., Payoff, Delta) for overlay.
5. **Sidebar > Display Options**: Check "Show Separate Graphs" for metric + payoff pairs.
6. **Sidebar > Add Single Metric + Payoff Graph**: Add specific metric graphs.
7. **Main Area**: View metrics table + plots.

## Example Strategies
- **Covered Call**: Long stock (simulate via deep ITM call), short OTM call.
- **Straddle**: Long ATM call + long ATM put.
- **Iron Condor**: Short OTM call/put + long further OTM call/put.

## Model Details
- Black-Scholes-Merton (continuous dividend q).
- Greeks formulas included.
- Payoff: At expiration (T=0).
- Time Value: Premium - Intrinsic Value.
- S_range: [max(50, S-50), S+50] (100 points).

## Requirements
Streamlit app. Run locally: `streamlit run app.py`.
Dependencies: `streamlit`, `numpy`, `pandas`, `scipy`, `matplotlib`.

## Limitations
- European options only.
- Constant parameters (no smile/skew).
- No transaction costs.
