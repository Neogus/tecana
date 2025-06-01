## Tecana - Technical Analysis Library

A high-performance Python library for technical analysis of financial markets, optimized for speed and efficiency.

## Features

- 60+ technical indicators with optimized implementations
- Trading signals derived from indicators (momentum, zone, trend, volatility)
- Simple, consistent API with pandas integration
- Minimal dependencies (just numpy and pandas)
- Comprehensive test coverage

## Installation

```bash
pip install tecana

Quick Start
import tecana as ta
import pandas as pd

# Load your OHLCV data
df = pd.read_csv('your_data.csv')

# Calculate single indicators
df = ta.rsi(df)
df = ta.macd(df)

# Apply multiple indicators efficiently in one call
df = ta.custom(df,
    ['rsi', 14],              # With specific parameter
    ['macd', 12, 26, 9],      # With multiple parameters
    ['bb', {'window': 20}],   # With keyword arguments
    ['atr']                   # With default parameters
)

# Calculate trading signals
df = ta.rsi_m(df)  # Momentum signal
df = ta.rsi_z(df)  # Zone signal (overbought/oversold)
df = ta.rsi_t(df)  # Trend signal

## Demo

You can test the library in this notebook:

https://colab.research.google.com/drive/1BT6Utx_AelOxjPkMqUpFptMX3CH0WUdl?usp=sharing


Indicator Categories
Trend Indicators
SMA, EMA, DEMA, TEMA, KAMA, MACD, Bollinger Bands, Parabolic SAR, Ichimoku Cloud, and more.

Momentum Indicators
RSI, Stochastic Oscillator, CCI, Williams %R, ADX, TRIX, Ultimate Oscillator, and more.

Volatility Indicators
ATR, Bollinger Bands Width, Keltner Channel Width, Choppiness Index, and more.

Volume Indicators
OBV, Chaikin Money Flow, MFI, Volume Price Trend, Ease of Movement, and more.

Signal Types
Momentum Signals (_m): Identify potential reversals or continuations
Zone Signals (_z): Identify overbought/oversold conditions
Trend Signals (_t): Identify trend direction
Volatility Signals (_v): Identify periods of high or low volatility
Disclaimer
This software is provided 'as-is', without any express or implied warranty. The calculations and indicators provided by this library are for informational purposes only and should not be construed as financial advice. The author is not responsible for any errors, inaccuracies, or misuse of this library. Trading and investing involve risk, and you should always conduct your own research before making financial decisions. In no event will the author be held liable for any financial losses or damages arising from the use of this software.

## License

This project is licensed under a custom license that allows free use including commercial applications, 
but prohibits selling the software itself or derivatives. See the LICENSE file for details.

## Support Development
If you find Tecana useful and would like to support its development:

- Bitcoin (Netowork: BTC - SegWit): bc1q496gksyalywftwg4q0hjqs4nuexgxpe638h6lu
- Ethereum (Netowork: ETH - ERC20): 0x4ed38015d1cf0f4cea2010f5aea3f34f9878d0d3
- Solana (Netowork: SOL - Solana): 534JQmpyuA9a4WZvdRW33aon3n69r2od61SQcX8EfSxn
- USDC (Netowork: ETH - ERC20): 0x4ed38015d1cf0f4cea2010f5aea3f34f9878d0d3