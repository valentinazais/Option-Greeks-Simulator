# Black-Scholes Option Strategy Dashboard

Live App: https://option-greeks.streamlit.app/

Interactive dashboard for pricing option strategies using the Black-Scholes model
with dividend yield. Computes prices, Greeks, payoffs, time value, and premiums.

---

## Model Formulas

### Black-Scholes $d_1$ and $d_2$

$$
d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)\,T}{\sigma\sqrt{T}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

| Symbol | Definition |
|--------|-----------|
| $S$ | Current underlying price |
| $K$ | Strike price |
| $T$ | Time to maturity (years) |
| $r$ | Risk-free rate |
| $q$ | Continuous dividend yield |
| $\sigma$ | Volatility |

---

### Option Prices

$$
C = S\,e^{-qT}\,\Phi(d_1) - K\,e^{-rT}\,\Phi(d_2)
$$

$$
P = K\,e^{-rT}\,\Phi(-d_2) - S\,e^{-qT}\,\Phi(-d_1)
$$

---

### Greeks

#### Delta

$$
\Delta_C = e^{-qT}\,\Phi(d_1)
$$

$$
\Delta_P = -e^{-qT}\,\Phi(-d_1)
$$

#### Gamma

$$
\Gamma = \frac{e^{-qT}\,\phi(d_1)}{S\,\sigma\sqrt{T}}
$$

#### Theta

$$
\Theta_C =
-\frac{S\,e^{-qT}\,\phi(d_1)\,\sigma}{2\sqrt{T}}
- rKe^{-rT}\Phi(d_2)
+ qSe^{-qT}\Phi(d_1)
$$

$$
\Theta_P =
-\frac{S\,e^{-qT}\,\phi(d_1)\,\sigma}{2\sqrt{T}}
+ rKe^{-rT}\Phi(-d_2)
- qSe^{-qT}\Phi(-d_1)
$$

#### Vega

$$
\nu = S\,e^{-qT}\,\phi(d_1)\,\sqrt{T}
$$

#### Rho

$$
\rho_C = KTe^{-rT}\Phi(d_2)
$$

$$
\rho_P = -KTe^{-rT}\Phi(-d_2)
$$

Where:

- $\phi(\cdot)$ = standard normal PDF  
- $\Phi(\cdot)$ = standard normal CDF

---

### Time Value

$$
\text{Time Value} = \text{Option Price} - \text{Intrinsic Value}
$$

$$
\text{Intrinsic}_C = \max(S - K,\, 0)
$$

$$
\text{Intrinsic}_P = \max(K - S,\, 0)
$$

---

### Strategy Aggregation

For $n$ legs with position sign $\delta_i \in \{+1, -1\}$:

$$
\text{Net Greek} =
\sum_{i=1}^{n} \delta_i \cdot \text{Greek}_i
$$

---

## Features

### Market Parameters
- Underlying Price ($S$)
- Time to Maturity ($T$, years)
- Risk-Free Rate ($r$)
- Dividend Yield ($q$)
- Volatility ($\sigma$)

### Strategy Builder
- Arbitrary number of call/put legs
- Long or short position per leg
- Independent strike per leg
- Live removal of legs

### Output Analytics
- Net Premium (Black-Scholes price)
- Payoff / Intrinsic Value
- Time Value
- Delta, Gamma, Theta, Vega, Rho

---

## Visualizations

### Combined Strategy Plot
Overlays selected metrics versus underlying price $S$ with twin-axis scaling.

### Separate Metric Graphs
One chart per metric, each paired with Payoff on a secondary axis, two-column grid.

### Individual Graphs
User-added single-metric charts stored in session state, displayed in a two-column grid.

---

## Architecture
streamlit (Python)
│
app.py
│
├── Black-Scholes engine (scipy.stats.norm)
├── Session-state strategy builder
├── Sidebar parameter controls
├── Matplotlib figure rendering
└── Streamlit metrics grid
System properties:
- Python backend, Streamlit frontend
- Server-side computation via SciPy
- Stateful multi-leg builder via `st.session_state`
- Deployed on Streamlit Cloud

---

## Numerical Implementation

- `scipy.stats.norm.cdf` for $\Phi(\cdot)$
- `scipy.stats.norm.pdf` for $\phi(\cdot)$
- `math.log`, `math.exp`, `math.sqrt` for scalar operations
- `numpy.linspace` for $S$-range sweep
- `matplotlib` with `ggplot` style and twin-axis layout

---

## Technology

- Python 3
- Streamlit
- Matplotlib
- SciPy
- NumPy
- Streamlit Cloud

---

## Result

A browser-accessible options pricing terminal for exploring:
- Black-Scholes Greeks
- Multi-leg strategy payoffs
- Time value decay
- Sensitivity across the underlying price range

All directly in the browser without local installation.
