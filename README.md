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
