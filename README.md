\documentclass[12pt, a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath, amssymb}
\usepackage{hyperref}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{tcolorbox}
\usepackage{fontenc}
\usepackage{parskip}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    urlcolor=blue!70!black
}

\titleformat{\section}{\large\bfseries}{}{0em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}

\begin{document}

% ── Title Block ──────────────────────────────────────────
\begin{center}
    {\LARGE\bfseries Black-Scholes Option Strategy Dashboard}\\[0.6em]
    {\normalsize Live App: \href{https://option-greeks.streamlit.app/}
    {\texttt{https://option-greeks.streamlit.app/}}}
\end{center}

\vspace{0.5em}

% ── Model Formulas ───────────────────────────────────────
\section{Model Formulas}

\subsection{Black-Scholes $d_1$ and $d_2$}

\[
d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)\,T}{\sigma\sqrt{T}}
\qquad
d_2 = d_1 - \sigma\sqrt{T}
\]

Where:
\begin{itemize}[noitemsep]
    \item $S$ = current underlying price
    \item $K$ = strike price
    \item $T$ = time to maturity (years)
    \item $r$ = risk-free rate
    \item $q$ = continuous dividend yield
    \item $\sigma$ = volatility
\end{itemize}

\subsection{Option Prices}

\[
C = S\,e^{-qT}\,\Phi(d_1) - K\,e^{-rT}\,\Phi(d_2)
\]
\[
P = K\,e^{-rT}\,\Phi(-d_2) - S\,e^{-qT}\,\Phi(-d_1)
\]

\subsection{Greeks}

\textbf{Delta} $(\Delta)$
\[
\Delta_C = e^{-qT}\,\Phi(d_1)
\qquad
\Delta_P = -e^{-qT}\,\Phi(-d_1)
\]

\textbf{Gamma} $(\Gamma)$
\[
\Gamma = \frac{e^{-qT}\,\phi(d_1)}{S\,\sigma\sqrt{T}}
\]

\textbf{Theta} $(\Theta)$
\[
\Theta_C = -\frac{S\,e^{-qT}\,\phi(d_1)\,\sigma}{2\sqrt{T}}
           - r K e^{-rT}\Phi(d_2)
           + q S e^{-qT}\Phi(d_1)
\]
\[
\Theta_P = -\frac{S\,e^{-qT}\,\phi(d_1)\,\sigma}{2\sqrt{T}}
           + r K e^{-rT}\Phi(-d_2)
           - q S e^{-qT}\Phi(-d_1)
\]

\textbf{Vega} $(\nu)$
\[
\nu = S\,e^{-qT}\,\phi(d_1)\,\sqrt{T}
\]

\textbf{Rho} $(\rho)$
\[
\rho_C = K\,T\,e^{-rT}\,\Phi(d_2)
\qquad
\rho_P = -K\,T\,e^{-rT}\,\Phi(-d_2)
\]

Where $\phi(\cdot)$ and $\Phi(\cdot)$ denote the standard normal PDF and CDF respectively.

\subsection{Time Value}

\[
\text{Time Value} = \text{Option Price} - \text{Intrinsic Value}
\]
\[
\text{Intrinsic}_C = \max(S - K,\, 0)
\qquad
\text{Intrinsic}_P = \max(K - S,\, 0)
\]

\subsection{Strategy Aggregation}

For a multi-leg strategy with legs $i = 1, \ldots, n$, each with position sign
$\delta_i \in \{+1, -1\}$:

\[
\text{Net Greek} = \sum_{i=1}^{n} \delta_i \cdot \text{Greek}_i
\]

% ── Features ─────────────────────────────────────────────
\section{Features}

\subsection{Market Parameters}
\begin{itemize}[noitemsep]
    \item Underlying Price ($S$)
    \item Time to Maturity ($T$, years)
    \item Risk-Free Rate ($r$)
    \item Dividend Yield ($q$)
    \item Volatility ($\sigma$)
\end{itemize}

\subsection{Strategy Builder}
\begin{itemize}[noitemsep]
    \item Arbitrary number of call/put legs
    \item Long or short position per leg
    \item Independent strike per leg
    \item Live removal of legs
\end{itemize}

\subsection{Output Analytics}
\begin{itemize}[noitemsep]
    \item Net Premium (BS price)
    \item Payoff / Intrinsic Value
    \item Time Value
    \item Delta, Gamma, Theta, Vega, Rho
\end{itemize}

\subsection{Visualizations}

\textbf{Combined Strategy Plot}\\
Overlays any selected metrics versus underlying price $S$.
Each metric uses its own $y$-axis scale (twin-axis).

\textbf{Separate Metric Graphs}\\
Renders one chart per selected metric, each paired with Payoff
on a secondary axis. Displayed in a two-column grid.

\textbf{Individual Graphs}\\
User-added single-metric charts, persistently stored in session state,
also rendered in a two-column grid.

% ── Architecture ─────────────────────────────────────────
\section{Architecture}

\begin{tcolorbox}[colback=gray!8, colframe=gray!40, fontupper=\small\ttfamily,
                  title=Application Stack]
streamlit (Python)\\
\quad \textbar\\
\quad \texttt{app.py}\\
\quad \quad \textbar\\
\quad \quad Black-Scholes engine (scipy.stats.norm)\\
\quad \quad Session-state strategy builder\\
\quad \quad Sidebar parameter controls\\
\quad \quad Matplotlib figure rendering\\
\quad \quad Streamlit metrics grid
\end{tcolorbox}

System properties:
\begin{itemize}[noitemsep]
    \item Python backend, Streamlit frontend
    \item Fully server-side computation via SciPy
    \item Stateful multi-leg builder via \texttt{st.session\_state}
    \item Deployed as a Streamlit Cloud application
\end{itemize}

% ── Numerical Implementation ─────────────────────────────
\section{Numerical Implementation}

\begin{itemize}[noitemsep]
    \item \texttt{scipy.stats.norm.cdf} for $\Phi(\cdot)$
    \item \texttt{scipy.stats.norm.pdf} for $\phi(\cdot)$
    \item \texttt{math.log}, \texttt{math.exp}, \texttt{math.sqrt} for scalar operations
    \item \texttt{numpy.linspace} for $S$-range sweep
    \item \texttt{matplotlib} with \texttt{ggplot} style and twin-axis layout
\end{itemize}

% ── Technology ───────────────────────────────────────────
\section{Technology}

\begin{itemize}[noitemsep]
    \item Python 3 pricing engine
    \item Streamlit for UI and interactivity
    \item Matplotlib for visualization
    \item SciPy / NumPy for numerical computation
    \item Streamlit Cloud for static deployment
\end{itemize}

% ── Result ───────────────────────────────────────────────
\section{Result}

A browser-accessible options pricing terminal for interactively exploring
Black-Scholes Greeks, multi-leg strategy payoffs, time value decay,
and metric sensitivity across the full underlying price range,
without requiring local installation.

\end{document}
