# Option Greeks Simulator

Interactive Black-Scholes option strategy dashboard built with Streamlit.

## Features

- **Black-Scholes pricing** with dividend yield support
- **All Greeks**: Delta, Gamma, Theta, Vega, Rho
- **Strategy builder**: combine multiple option legs (calls/puts, long/short)
- **Payoff**, time value, and premium analysis
- **Interactive plots**: combined overlay or separate per-metric graphs

## Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Deploy on Streamlit Cloud

1. Fork or clone this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point to `main.py` as the entry file
