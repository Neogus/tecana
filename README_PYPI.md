# Tecana

**Author:** Gustavo Rabino

**Version:** 2.0.0

Tecana is a high-performance Python library for technical analysis of financial markets, optimized for speed and efficiency.

## Features

- **90+ technical indicators** with optimized vectorized implementations
- **160+ trading signals** (momentum, zone, trend, volatility families)
- **Full type annotations** on all 243 public methods
- Simple, consistent API with pandas integration
- Minimal dependencies (just numpy and pandas — no TA-Lib required)
- Safe division helpers preventing ±inf from zero denominators

## What's New in v2.0.0

- **20 new indicators**: DEMA, TEMA, T3, TRIMA, MOM, APO, BOP, ADOSC, NATR, TRANGE, STOCHF, AROONOSC, ADXR, IMI, ACCBANDS, MIDPOINT, MIDPRICE, LINEARREG_SLOPE, ROCP, STDDEV
- **27 new signal methods** for the above indicators
- Type annotations on all public methods
- Wilder RMA helper for correct ATR/RSI smoothing

## Quick Start

```python
import tecana as ta
import pandas as pd

df = pd.read_csv('your_data.csv')
df = ta.rsi(df)           # Add RSI
df = ta.dema(df)          # NEW: Double EMA
df = ta.natr_v(df)        # NEW: Volatility flag
```

## Signal Convention

All signals return int8: **-1** (buy), **0** (neutral), **+1** (sell).

## Links

- **GitHub:** https://github.com/Neogus/tecana
- **Demo:** https://colab.research.google.com/github/Neogus/tecana/blob/main/demo/tecana_demo.ipynb

## License

Tecana is licensed under a modified MIT License that allows free use, modification, and distribution, **except that selling or redistributing the library as a standalone technical indicator library without significant modification is prohibited**. Full license text is included in the package's `LICENSE` file.

## Disclaimer

This software is provided "as-is" without any express or implied warranty. The technical indicators and trading signals are based on mathematical formulas applied to historical price data. **There is no guarantee that calculations are free of errors, bugs, or inaccuracies.** The output is for informational and educational purposes only and should not be construed as financial advice. The author is not responsible for any financial losses or damages arising from the use of this software. Trading involves substantial risk of loss.
