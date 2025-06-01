import numpy as np
import pandas as pd


class Tecana:
    """
    Technical analysis library with optimized indicators and signals for financial data.
    """

    def _prepare_df(self, df, required_cols):
        """
        Internal helper to standardize columns to lowercase and check required columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing market data columns.
        required_cols : list
            List of required column names (lowercase) to validate.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with lowercase columns.
        """
        # Lowercase columns
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        # Check required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for indicator calculation: {missing}")

        return df

    def custom(self, df, *args):
        """
        Apply multiple indicators to a dataframe in a single call.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing market data.
        *args : lists or tuples
            Variable number of lists/tuples where each contains:
            - First element: indicator function name as string (e.g., 'rsi', 'macd')
            - Remaining elements (optional): positional parameters for the function
            - Optional: keyword arguments as a dict at the end

        Returns
        -------
        pd.DataFrame
            Dataframe with all requested indicators applied.

        Examples
        --------
        >>> df = tec.custom(df,
        ...     ['rsi'],               # Uses default parameters
        ...     ['macd', 12, 26, 9],   # With multiple parameters
        ...     ['bb', 20],            # With single parameter
        ...     ['atr', {'window': 14}] # With keyword arguments
        ... )
        """
        result_df = df.copy()

        for i, arg in enumerate(args, 1):
            if not arg:
                continue

            # Get function name
            func_name = arg[0]

            # Handle function reference - convert to string name if needed
            if callable(func_name):
                func_name = func_name.__name__
            elif not isinstance(func_name, str):
                raise TypeError(f"Item {i}: Function reference must be a string or callable, not {type(func_name)}")

            # Check if function exists
            if not hasattr(self, func_name):
                raise ValueError(f"Item {i}: Unknown indicator function: '{func_name}'")

            # Get the actual function
            func = getattr(self, func_name)

            # Process parameters
            kwargs = {}
            params = []

            if len(arg) > 1:
                if isinstance(arg[-1], dict):
                    params = arg[1:-1]
                    kwargs = arg[-1]
                else:
                    params = arg[1:]

            try:
                # Apply the function with parameters
                result_df = func(result_df, *params, **kwargs)
            except TypeError as e:
                # Generate helpful error message with function signature
                import inspect
                sig = inspect.signature(func)
                param_info = list(sig.parameters.items())[1:]  # Skip 'self'

                expected_params = []
                for name, param in param_info:
                    if param.default is not inspect.Parameter.empty:
                        expected_params.append(f"{name}={param.default}")
                    else:
                        expected_params.append(name)

                error_msg = f"Error in item {i} ({func_name}):\n"
                error_msg += f"Expected: {func_name}(df, {', '.join(expected_params)})\n"
                error_msg += f"Received: {len(params)} positional args and {len(kwargs)} keyword args"

    def adi(self, df, period=14):
        """
        Accumulation/Distribution Index (ADI)

        A volume-based indicator designed to measure the cumulative flow of money into and out of a security.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', 'low', and 'volume' columns.
        - period (int): Period over which to calculate the ADI.

        Returns:
        - DataFrame with an added 'adi' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low', 'volume'])

        # Calculate Money Flow Multiplier
        df['money flow multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (
                    df['high'] - df['low'])
        df['money flow volume'] = df['money flow multiplier'] * df['volume']
        df['adi'] = df['money flow volume'].rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

        return df.drop(['money flow multiplier', 'money flow volume'], axis=1)

    def ai(self, df, window=25):
        """
        Aaron Indicator

        Identifies trend changes in price and strength of trends. Consists of two lines - Aaron Up (AIU) and Aaron Down (AID).

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - window (int): Period over which to calculate the Aaron Indicator.

        Returns:
        - DataFrame with added 'aiu' and 'aid' columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        # Calculate rolling max index
        rolling_max_index_high = df['high'].rolling(window=window, min_periods=1).max().shift(-(window - 1))
        rolling_max_index_low = df['low'].rolling(window=window, min_periods=1).max().shift(-(window - 1))

        # Check if rolling max index equals window size
        up_days = (rolling_max_index_high == df['high']).astype(int)
        down_days = (rolling_max_index_low == df['low']).astype(int)

        # Calculate AI_Up and AI_Down
        df['aiu'] = (window - up_days * window) / window * 100
        df['aiu'] = np.where(up_days > 0, df['aiu'], 0)
        df['aid'] = (window - down_days * window) / window * 100
        df['aid'] = np.where(down_days > 0, df['aid'], 0)

        return df

    def ao(self, df, short_period=5, long_period=20):
        """
        Aaron Oscillator

        Measures the difference between short-term and long-term EMAs, indicating momentum.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - short_period (int): Short-term period for EMA calculation.
        - long_period (int): Long-term period for EMA calculation.

        Returns:
        - DataFrame with an added 'ao' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate short and long EMAs
        df['ema_short'] = df['close'].ewm(span=short_period, min_periods=short_period, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=long_period, min_periods=long_period, adjust=False).mean()

        # Calculate Aaron Oscillator
        df['ao'] = df['ema_short'] - df['ema_long']

        return df.drop(['ema_short', 'ema_long'], axis=1)

    def atr(self, df, window=14):
        """
        Average True Range (ATR)

        Measures market volatility by decomposing the entire range of an asset price.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', and 'close' columns.
        - window (int): Period over which to calculate ATR.

        Returns:
        - DataFrame with an added 'atr' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close'])

        # Calculate True Range components
        high_low_diff = df['high'] - df['low']
        high_close_prev_diff = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev_diff = np.abs(df['low'] - df['close'].shift(1))

        # Get the maximum of the three
        df['r'] = np.maximum.reduce([high_low_diff, high_close_prev_diff, low_close_prev_diff])

        # Calculate ATR
        df['atr'] = df['r'].rolling(window=window).mean()

        return df.drop(['r'], axis=1)

    def awo(self, df, period1=5, period2=34):
        """
        Awesome Oscillator

        Measures market momentum by showing the difference between a 5-period and 34-period simple moving average.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - period1 (int): Short-term period for the median price SMA.
        - period2 (int): Long-term period for the median price SMA.

        Returns:
        - DataFrame with an added 'awo' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        df['median price'] = (df['high'] + df['low']) / 2
        df['awo'] = df['median price'].rolling(window=period1).mean() - df['median price'].rolling(
            window=period2).mean()

        return df.drop(['median price'], axis=1)

    def bb(self, df, window=20, num_std=2):
        """
        Bollinger Bands

        A volatility indicator that plots upper and lower bands based on the standard deviation of price
        around a moving average. It helps identify overbought or oversold conditions.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - window (int): Period for the simple moving average and standard deviation.
        - num_std (int or float): Number of standard deviations for the upper and lower bands.

        Returns:
        - DataFrame with added 'bbh', 'bbl', and 'bbv' columns:
            - 'bbh': Bollinger Band High (upper band)
            - 'bbl': Bollinger Band Low (lower band)
            - 'bbv': Bollinger Band Width (bbh - bbl)
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['sma'] = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()

        df['bbh'] = df['sma'] + num_std * std
        df['bbl'] = df['sma'] - num_std * std
        df['bbv'] = df['bbh'] - df['bbl']

        return df.drop(['sma'], axis=1)

    def cc(self, df, short_roc_period=11, long_roc_period=14, smoothing_period=10, r=3):
        """
        Coppock Curve

        A momentum indicator that identifies long-term buying opportunities in the market.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - short_roc_period (int): Short-term rate of change period.
        - long_roc_period (int): Long-term rate of change period.
        - smoothing_period (int): Period for smoothing the ROC values.
        - r (int): Moving average period for the signal line.

        Returns:
        - DataFrame with added 'cc' and 'ccs' columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate short-term and long-term Rate of Change
        df['short_roc'] = (df['close'] - df['close'].shift(short_roc_period)) / df['close'].shift(short_roc_period)
        df['long_roc'] = (df['close'] - df['close'].shift(long_roc_period)) / df['close'].shift(long_roc_period)

        # Calculate Coppock Indicator and signal
        df['cc'] = df['short_roc'].rolling(window=smoothing_period).sum() + df['long_roc'].rolling(
            window=smoothing_period).sum()
        df['ccs'] = df['cc'].rolling(r).mean()

        return df.drop(['short_roc', 'long_roc'], axis=1)

    def cci(self, df, window=20):
        """
        Commodity Channel Index

        Measures the current price level relative to an average price level over a given period.
        Used to identify cyclical trends and extremes that might indicate reversals.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', and 'close' columns.
        - window (int): Period over which to calculate the CCI.

        Returns:
        - DataFrame with an added 'cci' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close'])

        # Calculate typical price
        tp = (df['high'] + df['low'] + df['close']) / 3

        # Calculate SMA and Mean Absolute Deviation
        sma = tp.rolling(window=window).mean()
        mad = abs(tp - sma).rolling(window=window).mean()

        # Calculate CCI
        df['cci'] = (tp - sma) / (0.015 * mad)

        return df

    def ce(self, df, period=22, multiplier=3):
        """
        Chandelier Exit

        A volatility-based indicator that sets trailing stops for long and short positions.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - period (int): Lookback period for highest high/lowest low calculation.
        - multiplier (float): ATR multiplier for stop distance.

        Returns:
        - DataFrame with added 'cel' (long exit) and 'ces' (short exit) columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        # Calculate ATR range
        df['atrr'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()

        # Calculate Chandelier Exit levels
        df['cel'] = df['high'].rolling(window=period).max() - df['atrr'] * multiplier
        df['ces'] = df['low'].rolling(window=period).min() + df['atrr'] * multiplier

        return df.drop(['atrr'], axis=1)

    def ci(self, df, period=14):
        """
        Choppiness Index

        Measures the market's trendiness or choppiness, identifying whether the market
        is consolidating or trending.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low' columns.
        - period (int): Period over which to calculate the Choppiness Index.

        Returns:
        - DataFrame with an added 'ci' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        # Calculate True Range
        df['tro'] = df['high'] - df['low']

        # Calculate ATR and High-Low range
        df['atro'] = df['tro'].rolling(window=period).mean()
        df['hl'] = df['high'].rolling(window=period).max() - df['low'].rolling(window=period).min()

        # Calculate Choppiness Index
        df['ci'] = 100 * np.log10(df['atro'] / df['hl']) / np.log10(period)

        return df.drop(['tro', 'atro', 'hl'], axis=1)

    def cmf(self, df, window=14):
        """
        Chaikin Money Flow (CMF)

        A volume-weighted average of accumulation/distribution over a specified period.
        Used to measure the amount of Money Flow Volume over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', 'low', and 'volume' columns.
        - window (int): Period over which to calculate the CMF.

        Returns:
        - DataFrame with an added 'cmf' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low', 'volume'])

        df['mf_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['mf_volume'] = df['mf_multiplier'] * df['volume']
        df['cmf'] = df['mf_volume'].rolling(window=window).sum() / df['volume'].rolling(window=window).sum()

        return df.drop(['mf_multiplier', 'mf_volume'], axis=1)

    def cmo(self, df, period=14):
        """
        Chande Momentum Oscillator (CMO)

        A momentum oscillator that captures the momentum of an instrument by analyzing
        the sum of recent gains versus the sum of recent losses.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period over which to calculate the CMO.

        Returns:
        - DataFrame with an added 'cmo' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate the difference between today's close and yesterday's close
        diff = df['close'].diff()

        # Calculate the absolute value of up and down movements
        up = diff.where(diff > 0, 0)
        down = -diff.where(diff < 0, 0)

        # Calculate the sum of up and down movements over the specified period
        sum_up = up.rolling(window=period, min_periods=period).sum()
        sum_down = down.rolling(window=period, min_periods=period).sum()

        # Calculate the Chande Momentum Oscillator (CMO)
        df['cmo'] = ((sum_up - sum_down) / (sum_up + sum_down)) * 100

        return df

    def dc(self, df, period=20):
        """
        Donchian Channels

        An indicator that defines the high and low of a market
        using the highest high and lowest low over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - period (int): Period over which to calculate the channels.

        Returns:
        - DataFrame with added 'dch' (upper), 'dcl' (lower), and 'dcm' (middle) columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        df['dch'] = df['high'].rolling(window=period).max()
        df['dcl'] = df['low'].rolling(window=period).min()
        df['dcm'] = (df['dch'] + df['dcl']) / 2

        return df

    def di(self, df, window=10):
        """
        Disparity Index

        Measures the relative position of an asset's price to its moving average.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - window (int): Period over which to calculate the moving average.

        Returns:
        - DataFrame with an added 'di' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['mas'] = df['close'].rolling(window=window).mean()
        df['di'] = ((df['close'] - df['mas']) / df['mas']) * 100

        return df.drop(['mas'], axis=1)

    def dma(self, df, period=20, displacement=5):
        """
        Displaced Moving Average

        A moving average that is shifted forward or backward in time.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period over which to calculate the moving average.
        - displacement (int): Number of periods to shift the moving average.

        Returns:
        - DataFrame with an added 'dma' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['dma'] = df['close'].rolling(window=period).mean().shift(displacement)

        return df

    def dpo(self, df, period=20):
        """
        Detrended Price Oscillator

        An indicator designed to remove trend from price to better identify cycles.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period over which to calculate the DPO.

        Returns:
        - DataFrame with an added 'dpo' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['dpo'] = df['close'] - df['close'].shift(period // 2 + 1).rolling(window=period).mean()

        return df

    def dx(self, df, period=14):
        """
        Directional Movement Index

        A set of indicators including ADX that helps determine if a market is trending
        and the strength of that trend.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', and 'close' columns.
        - period (int): Period over which to calculate the indicators.

        Returns:
        - DataFrame with added 'dip' (DI+), 'din' (DI-), and 'adx' columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close'])

        # Calculate True Range (TR)
        df['high-low'] = df['high'] - df['low']
        df['high-prevclose'] = abs(df['high'] - df['close'].shift())
        df['low-prevclose'] = abs(df['low'] - df['close'].shift())
        df['trs'] = df[['high-low', 'high-prevclose', 'low-prevclose']].max(axis=1)

        # Calculate Directional Movement (+DM and -DM)
        df['dm+'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                             df['high'] - df['high'].shift(1), 0)
        df['dm-'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                             df['low'].shift(1) - df['low'], 0)

        # Calculate True Range (TR) and Directional Movement (+DM and -DM) for the period
        tr_period = df['trs'].rolling(window=period).sum()
        dm_plus_period = df['dm+'].rolling(window=period).sum()
        dm_minus_period = df['dm-'].rolling(window=period).sum()

        # Calculate DI+ and DI-
        df['dip'] = (dm_plus_period / tr_period) * 100
        df['din'] = (dm_minus_period / tr_period) * 100

        # Calculate Directional Movement Index (DX)
        df['dx'] = (abs(df['dip'] - df['din']) / (df['dip'] + df['din'])) * 100

        # Calculate Average Directional Index (ADX)
        df['adx'] = df['dx'].rolling(window=period).mean()

        return df.drop(['trs', 'dx', 'dm+', 'dm-', 'high-low', 'high-prevclose', 'low-prevclose'], axis=1)

    def ema(self, df, w=20, ws=10):
        """
        Exponential Moving Average

        A type of moving average that places a greater weight on the most recent data points.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - w (int): Primary EMA window period.
        - ws (int): Secondary EMA window period.

        Returns:
        - DataFrame with added 'ema' and 'emas' columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['ema'] = df['close'].ewm(span=w, adjust=False).mean()
        df['emas'] = df['close'].ewm(span=ws, adjust=False).mean()

        return df

    def eom(self, df):
        """
        Ease of Movement

        A momentum indicator that relates market volume to price change
        to identify the strength of a trend.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', and 'volume' columns.

        Returns:
        - DataFrame with an added 'eom' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'volume'])

        high_low_diff = (df['high'] - df['low']).diff(1)
        volume_ratio = ((df['volume'] / 10000) / ((df['high'] - df['low']))).shift(1)
        df['eom'] = high_low_diff / volume_ratio

        return df

    def eri(self, df, period=13):
        """
        Elder Ray Index

        Measures the buying and selling pressure using the relationship
        between current prices and a moving average.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', and 'close' columns.
        - period (int): Period over which to calculate the EMA.

        Returns:
        - DataFrame with added 'erbup' (Bull Power) and 'erbep' (Bear Power) columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close'])

        df['emas'] = df['close'].ewm(span=period, min_periods=period).mean()
        df['erbup'] = df['high'] - df['emas']  # Bull Power
        df['erbep'] = df['low'] - df['emas']  # Bear Power

        return df.drop(['emas'], axis=1)

    def fi(self, df, period=13):
        """
        Force Index

        Measures the force of price movement by considering both price change and volume.

        Parameters:
        - df (DataFrame): Input data containing 'close' and 'volume' columns.
        - period (int): Period over which to calculate price change.

        Returns:
        - DataFrame with an added 'fi' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'volume'])

        # Calculate price change
        df['price change'] = df['close'].diff(periods=period)

        # Calculate Force Index
        df['fi'] = df['price change'] * df['volume']

        return df.drop(['price change'], axis=1)

    def fr(self, df, w1=25):
        """
        Fibonacci Retracement

        Identifies potential support and resistance levels based on the Fibonacci sequence.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - w1 (int): Period over which to calculate the highest high and lowest low.

        Returns:
        - DataFrame with added Fibonacci retracement level columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        # Calculate highest high and lowest low within the period range
        df['highest high'] = df['high'].rolling(w1).max()
        df['lowest low'] = df['low'].rolling(w1).min()

        # Calculate Fibonacci retracement levels
        highest_high = df['highest high'].iloc[-1]
        lowest_low = df['lowest low'].iloc[-1]
        range_high_low = highest_high - lowest_low
        fibonacci_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        retracement_levels = highest_high - (range_high_low * np.array(fibonacci_levels))

        # Add Fibonacci retracement levels to the DataFrame
        for i, level in enumerate(fibonacci_levels):
            df[f'fr{int(level * 100)}'] = retracement_levels[i]

        return df.drop(['highest high', 'lowest low'], axis=1)

    def ha(self, df):
        """
        Heikin-Ashi Candlesticks

        A variation of candlestick charts that uses average price data to filter out market noise.

        Parameters:
        - df (DataFrame): Input data containing 'open', 'high', 'low', and 'close' columns.

        Returns:
        - DataFrame with added 'hac' (HA close), 'hao' (HA open), 'hah' (HA high), and 'hal' (HA low) columns.
        """
        df = self._prepare_df(df, required_cols=['open', 'high', 'low', 'close'])

        # Calculate Heikin-Ashi close
        df['hac'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # Calculate Heikin-Ashi open
        df['hao'] = df['open'].shift(1)
        df['hao'] = df['hao'].fillna((df['hao'].shift(1) + df['hac'].shift(1)) / 2)

        # Calculate Heikin-Ashi high
        df['hah'] = df[['hao', 'hac', 'high']].max(axis=1)

        # Calculate Heikin-Ashi low
        df['hal'] = df[['hao', 'hac', 'low']].min(axis=1)

        return df

    def ic(self, df, window1=9, window2=26, window3=52):
        """
        Ichimoku Cloud

        A comprehensive indicator that defines support and resistance, identifies trend direction,
        gauges momentum, and provides trading signals.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - window1 (int): Period for Tenkan-sen calculation.
        - window2 (int): Period for Kijun-sen calculation.
        - window3 (int): Period for Senkou Span B calculation.

        Returns:
        - DataFrame with added 'ict' (Tenkan-sen), 'ick' (Kijun-sen), 'icssa' (Senkou Span A),
          and 'icssb' (Senkou Span B) columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        # Calculate Tenkan-sen (Conversion Line)
        df['ict'] = (df['high'].rolling(window=window1).max() + df['low'].rolling(window=window1).min()) / 2

        # Calculate Kijun-sen (Base Line)
        df['ick'] = (df['high'].rolling(window=window2).max() + df['low'].rolling(window=window2).min()) / 2

        # Calculate Senkou Span A (Leading Span A)
        df['icssa'] = ((df['ict'] + df['ick']) / 2).shift(window2)

        # Calculate Senkou Span B (Leading Span B)
        df['icssb'] = ((df['high'].rolling(window=window3).max() + df['low'].rolling(window=window3).min()) / 2).shift(
            window2)

        return df

    def kama(self, df, period=10, fast=2, slow=30):
        """
        Kaufman's Adaptive Moving Average

        An indicator that adapts to price volatility, moving faster in trending markets
        and slower in ranging markets.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period over which to calculate the efficiency ratio.
        - fast (int): Fast EMA period for the smoothing constant.
        - slow (int): Slow EMA period for the smoothing constant.

        Returns:
        - DataFrame with an added 'kama' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate the Efficiency Ratio (ER)
        change = df['close'].diff(period).abs()
        delta = df['close'].diff().abs().rolling(window=period).sum()
        efficiency_ratio = change / delta

        # Calculate the Smoothing Constant (SC)
        smoothing_constant_fast = 2 / (fast + 1)
        smoothing_constant_slow = 2 / (slow + 1)
        smoothing_constant = (efficiency_ratio * (
                    smoothing_constant_fast - smoothing_constant_slow)) + smoothing_constant_slow

        # Initialize KAMA with the first value as the initial close
        df['kama'] = df['close']

        # Calculate the KAMA values using cumulative operations
        kama_shift = df['kama'].shift(1)  # Shift KAMA for previous value reference
        kama_values = kama_shift + smoothing_constant * (df['close'] - kama_shift)

        # Fill the first KAMA value with the first Close price
        df['kama'].iloc[0] = df['close'].iloc[0]

        # Use cumulative product and fill in the KAMA values
        df['kama'].iloc[1:] = kama_values.iloc[1:]

        return df

    def kc(self, df, window=20, atr_window=10, multiplier=2):
        """
        Keltner Channels

        A volatility-based indicator that creates bands above and below a moving average
        using Average True Range (ATR).

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - window (int): Period for the middle line calculation.
        - atr_window (int): Period for the ATR calculation.
        - multiplier (float): Multiplier for the ATR to set channel width.

        Returns:
        - DataFrame with added 'kcm' (middle), 'kch' (upper), 'kcl' (lower),
          and 'kcv' (width) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate ATR using rolling standard deviation
        atrs = df['close'].rolling(window=atr_window).std()

        # Calculate Keltner Channel middle line
        kcm = df['close'].rolling(window=window).mean()

        # Calculate upper and lower bands
        kch = kcm + (multiplier * atrs)
        kcl = kcm - (multiplier * atrs)

        # Calculate Keltner Channel width
        kcv = kch - kcl

        # Assign calculated values to DataFrame
        df['kcm'] = kcm
        df['kch'] = kch
        df['kcl'] = kcl
        df['kcv'] = kcv

        return df

    def kst(self, df, r1=10, r2=15, r3=20, r4=30, s1=10, s2=10, s3=10, s4=15, sp=9):
        """
        Know Sure Thing (KST) Indicator

        A momentum oscillator based on the weighted average of four rate-of-change values.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - r1, r2, r3, r4 (int): Periods for rate of change calculations.
        - s1, s2, s3, s4 (int): Smoothing periods for the ROC values.
        - sp (int): Signal line period.

        Returns:
        - DataFrame with added 'kst', 'ksts' (signal), and 'ksth' (histogram) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate rate of change for different periods
        df['roc1'] = df['close'].pct_change(r1) * 100
        df['roc2'] = df['close'].pct_change(r2) * 100
        df['roc3'] = df['close'].pct_change(r3) * 100
        df['roc4'] = df['close'].pct_change(r4) * 100

        # Calculate smoothed ROCs
        df['sma1'] = df['roc1'].pct_change(s1) * 100
        df['sma2'] = df['roc2'].pct_change(s2) * 100
        df['sma3'] = df['roc3'].pct_change(s3) * 100
        df['sma4'] = df['roc4'].pct_change(s4) * 100

        # Calculate KST
        df['kst'] = df['sma1'] + df['sma2'] + df['sma3'] + df['sma4']

        # Calculate signal line
        df['ksts'] = df['kst'].rolling(window=sp).mean()

        # Calculate histogram
        df['ksth'] = df['kst'] - df['ksts']

        return df.drop(['roc1', 'roc2', 'roc3', 'roc4', 'sma1', 'sma2', 'sma3', 'sma4'], axis=1)

    def lr(self, df, period=14):
        """
        Linear Regression Indicator

        Calculates the slope of the best-fit linear regression line over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period over which to calculate the linear regression.

        Returns:
        - DataFrame with an added 'lr' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        n = len(df)
        idx = np.arange(1, n + 1)  # Use 1-based index for rolling window

        # Calculate cumulative sums for linear regression
        x_sum = np.cumsum(df['close'])
        y_sum = np.cumsum(df['close'] * idx)
        xy_sum = y_sum - x_sum * idx
        xx_sum = np.cumsum(df['close'] ** 2)

        # Calculate window sums
        x_window_sum = x_sum[period:] - x_sum[:-period]
        y_window_sum = y_sum[period:] - y_sum[:-period]
        xy_window_sum = xy_sum[period:] - xy_sum[:-period]
        xx_window_sum = xx_sum[period:] - xx_sum[:-period]

        # Calculate slope
        slope = (period * xy_window_sum - x_window_sum * y_window_sum) / (period * xx_window_sum - x_window_sum ** 2)

        # Adjust the length of the slope array to match the DataFrame index
        slope_values = np.concatenate((np.full(period - 1, np.nan), slope))

        # Assign the calculated values to the DataFrame column
        df['lr'] = slope_values[:n]

        return df

    def macd(self, df, short_window=12, long_window=26, signal_window=9):
        """
        Moving Average Convergence Divergence

        A trend-following momentum indicator that shows the relationship between two moving averages.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - short_window (int): Period for the shorter-term EMA.
        - long_window (int): Period for the longer-term EMA.
        - signal_window (int): Period for the signal line EMA.

        Returns:
        - DataFrame with added 'macd', 'macds' (signal), and 'macdh' (histogram) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate short and long EMAs
        ema_short = df['close'].ewm(span=short_window, adjust=False).mean()
        ema_long = df['close'].ewm(span=long_window, adjust=False).mean()

        # Calculate MACD line
        df['macd'] = ema_short - ema_long

        # Calculate signal line
        df['macds'] = df['macd'].ewm(span=signal_window, adjust=False).mean()

        # Calculate histogram
        df['macdh'] = df['macd'] - df['macds']

        return df

    def mae(self, df, window=20, percent=5):
        """
        Moving Average Envelope

        Creates bands above and below a moving average with a specified percentage.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - window (int): Period for the moving average calculation.
        - percent (float): Percentage distance for the bands.

        Returns:
        - DataFrame with added 'maeu' (upper band) and 'maed' (lower band) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate simple moving average
        df['mae'] = df['close'].rolling(window=window).mean()

        # Calculate upper and lower bands
        df['maeu'] = df['mae'] * (1 + percent / 100)
        df['maed'] = df['mae'] * (1 - percent / 100)

        return df.drop(['mae'], axis=1)

    def mfi(self, df, window=14):
        """
        Money Flow Index

        A momentum indicator that incorporates both price and volume to measure buying and selling pressure.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', 'close', and 'volume' columns.
        - window (int): Period over which to calculate MFI.

        Returns:
        - DataFrame with an added 'mfi' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close', 'volume'])

        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Calculate raw money flow
        df['raw_money_flow'] = df['typical_price'] * df['volume']

        # Calculate money flow sums
        rolling_raw_money_flow_sum = df['raw_money_flow'].rolling(window=window).sum()
        rolling_raw_money_flow_shift_sum = df['raw_money_flow'].rolling(window=window).sum().shift()

        # Calculate MFI with handling for null values
        mfi_values = 100 - (100 / (1 + (rolling_raw_money_flow_sum / rolling_raw_money_flow_shift_sum)))
        mfi_values[rolling_raw_money_flow_shift_sum.isna()] = 50  # Set MFI to 50 when the shifted sum is null
        df['mfi'] = mfi_values

        return df.drop(['typical_price', 'raw_money_flow'], axis=1)

    def mi(self, df, window=25, s1=9, s2=9, p1=9, p2=9):
        """
        Mass Index

        Uses the high-low range to identify trend reversals based on range expansions.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - window (int): Period for the mass index calculation.
        - s1 (int): First EMA period.
        - s2 (int): Second EMA period.
        - p1, p2 (int): Minimum periods for the EMAs.

        Returns:
        - DataFrame with an added 'mi' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        # Calculate price range
        df['range'] = df['high'] - df['low']

        # Calculate EMAs
        df['single ema'] = df['range'].ewm(span=s1, min_periods=p1, adjust=False).mean()
        df['double ema'] = df['single ema'].ewm(span=s2, min_periods=p2, adjust=False).mean()

        # Calculate Mass Index
        df['mi'] = df['single ema'] / df['double ema']
        df['mi'] = df['mi'].rolling(window=window).sum()

        return df.drop(['range', 'single ema', 'double ema'], axis=1)

    def mp(self, df):
        """
        Median Price Indicator

        The arithmetic mean of the high and low prices for a given period.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.

        Returns:
        - DataFrame with an added 'mp' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        df['mp'] = (df['high'] + df['low']) / 2

        return df

    def nvi(self, df, r=14):
        """
        Negative Volume Index

        An indicator that uses volume to confirm price trends or warn of weak price movements.

        Parameters:
        - df (DataFrame): Input data containing 'close' and 'volume' columns.
        - r (int): Period for the signal line calculation.

        Returns:
        - DataFrame with added 'nvi' and 'nvis' columns.
        """
        df = self._prepare_df(df, required_cols=['close', 'volume'])

        # Initialize NVI
        df['nvi'] = 1000

        # Identify days with decreased volume
        decrease_volume_mask = df['volume'] < df['volume'].shift(1)

        # Update NVI only on days with decreased volume
        df.loc[decrease_volume_mask, 'nvi'] = df['nvi'].shift(1) * (
                    1 + (df['close'] - df['close'].shift(1)) / df['close'].shift(1))

        # Calculate signal line
        df['nvis'] = df['close'].rolling(r).mean()

        return df

    def obv(self, df):
        """
        On-Balance Volume

        A momentum indicator that uses volume flow to predict changes in stock price.

        Parameters:
        - df (DataFrame): Input data containing 'close' and 'volume' columns.

        Returns:
        - DataFrame with an added 'obv' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'volume'])

        # Calculate the direction of the day's movement
        direction = np.where(df['close'] > df['close'].shift(1), 1, np.where(df['close'] < df['close'].shift(1), -1, 0))

        # Calculate the change in OBV
        obv_change = direction * df['volume']

        # Calculate the cumulative sum of OBV change
        df['obv'] = obv_change.cumsum()

        return df

    def pc(self, df, period=13):
        """
        Price Change

        Measures the absolute change in price over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period over which to calculate the price change.

        Returns:
        - DataFrame with an added 'pc' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['pc'] = df['close'].diff(periods=period)

        return df

    def pp(self, df):
        """
        Pivot Points

        Price levels of significance derived from the previous period's high, low, and close values.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', and 'close' columns.

        Returns:
        - DataFrame with added 'pp', 'r1', 's1', 'r2', and 's2' columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close'])

        # Calculate pivot point
        df['pp'] = (df['high'] + df['low'] + df['close']) / 3

        # Calculate first resistance and support levels
        df['r1'] = 2 * df['pp'] - df['low']
        df['s1'] = 2 * df['pp'] - df['high']

        # Calculate second resistance and support levels
        df['r2'] = df['pp'] + (df['high'] - df['low'])
        df['s2'] = df['pp'] - (df['high'] - df['low'])

        return df

    def ppo(self, df, short_period=12, long_period=26, signal_period=9):
        """
        Percentage Price Oscillator

        Shows the difference between two moving averages as a percentage, providing insights
        into momentum and trend direction.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - short_period (int): Period for the shorter-term EMA.
        - long_period (int): Period for the longer-term EMA.
        - signal_period (int): Period for the signal line.

        Returns:
        - DataFrame with added 'ppo', 'ppos' (signal), and 'ppoh' (histogram) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate short and long EMAs
        short_ema = df['close'].ewm(span=short_period, min_periods=short_period).mean()
        long_ema = df['close'].ewm(span=long_period, min_periods=long_period).mean()

        # Calculate PPO line
        ppo_line = ((short_ema - long_ema) / long_ema) * 100

        # Calculate signal line
        signal_line = ppo_line.ewm(span=signal_period, min_periods=signal_period).mean()

        # Assign calculated values
        df['ppo'] = ppo_line
        df['ppos'] = signal_line
        df['ppoh'] = df['ppo'] - df['ppos']

        return df

    def proc(self, df, window=14):
        """
        Price Rate of Change

        Measures the percentage change in price over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - window (int): Period over which to calculate the rate of change.

        Returns:
        - DataFrame with an added 'proc' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['proc'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window) * 100

        return df

    def psar(self, df, step=0.02, max_acceleration=0.2):
        """
        Parabolic Stop and Reverse (PSAR)

        A trend-following indicator that places dots above or below price bars to indicate
        potential entry and exit points.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', and 'close' columns.
        - step (float): Acceleration factor step.
        - max_acceleration (float): Maximum acceleration factor.

        Returns:
        - DataFrame with an added 'psar' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close'])

        # Initialize SAR, AF, and EP
        df['sar'] = df['high'].rolling(window=2).min()
        df['acceleration_factor'] = step
        df['extreme_point'] = df['high'].rolling(window=2).max()

        # Calculate AF
        df['acceleration_factor'] = np.where(df['close'] > df['sar'].shift(1),
                                             np.minimum(df['acceleration_factor'].shift(1) + step, max_acceleration),
                                             step)

        # Calculate EP
        df['extreme_point'] = np.where((df['high'].shift(1) > df['high'].shift(2)) & (df['low'] < df['sar'].shift(1)),
                                       df['high'].shift(1),
                                       np.where((df['low'].shift(1) < df['low'].shift(2)) & (
                                                   df['high'] > df['sar'].shift(1)),
                                                df['low'].shift(1),
                                                df['extreme_point'].shift(1)))

        # Calculate PSAR
        df['psar'] = np.where(df['sar'].shift(1) + df['acceleration_factor'].shift(1) * (
                    df['extreme_point'].shift(1) - df['sar'].shift(1)) > df['low'],
                              df['high'],
                              df['sar'].shift(1) + df['acceleration_factor'].shift(1) * (
                                          df['extreme_point'].shift(1) - df['sar'].shift(1)))

        return df.drop(['sar', 'acceleration_factor', 'extreme_point'], axis=1)

    def pvo(self, df, short_period=12, long_period=26, signal_period=9):
        """
        Percentage Volume Oscillator

        Shows the difference between two volume moving averages as a percentage,
        indicating potential momentum shifts.

        Parameters:
        - df (DataFrame): Input data containing 'volume' column.
        - short_period (int): Period for the shorter-term volume EMA.
        - long_period (int): Period for the longer-term volume EMA.
        - signal_period (int): Period for the signal line.

        Returns:
        - DataFrame with added 'pvo', 'pvos' (signal), and 'pvoh' (histogram) columns.
        """
        df = self._prepare_df(df, required_cols=['volume'])

        # Calculate short and long EMAs of volume
        short_ema = df['volume'].ewm(span=short_period, min_periods=short_period).mean()
        long_ema = df['volume'].ewm(span=long_period, min_periods=long_period).mean()

        # Calculate PVO line
        pvo_line = ((short_ema - long_ema) / long_ema) * 100

        # Calculate signal line
        signal_line = pvo_line.ewm(span=signal_period, min_periods=signal_period).mean()

        # Assign calculated values
        df['pvo'] = pvo_line
        df['pvos'] = signal_line
        df['pvoh'] = df['pvo'] - df['pvos']

        return df

    def roc(self, df, period=14, r=6):
        """
        Rate of Change

        Measures the percentage change in price between the current price and the price a
        specified number of periods ago.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period over which to calculate the rate of change.
        - r (int): Period for the signal line calculation.

        Returns:
        - DataFrame with added 'roc' and 'rocs' (signal) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['roc'] = (df['close'] / df['close'].shift(period)) * 100 - 100
        df['rocs'] = df['roc'].rolling(window=r).mean()

        return df

    def rsi(self, df, w1=14):
        """
        Relative Strength Index

        A momentum oscillator that measures the speed and change of price movements,
        ranging from 0 to 100.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - w1 (int): Period over which to calculate RSI.

        Returns:
        - DataFrame with an added 'rsi' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        # Calculate price changes
        delta = df['close'].diff(1)

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=w1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=w1).mean()

        # Calculate relative strength
        rs = gain / loss

        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def sma(self, df, w=20, ws=10):
        """
        Simple Moving Average

        Calculates the arithmetic mean of price data over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - w (int): Primary SMA window period.
        - ws (int): Secondary SMA window period.

        Returns:
        - DataFrame with added 'sma' and 'smas' columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['sma'] = df['close'].rolling(window=w).mean()
        df['smas'] = df['close'].rolling(window=ws).mean()

        return df

    def so(self, df, kw=14, dw=3):
        """
        Stochastic Oscillator

        A momentum indicator comparing a security's closing price to its price range
        over a specific period, helping identify overbought and oversold conditions.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', and 'low' columns.
        - kw (int): %K calculation period.
        - dw (int): %D moving average period.

        Returns:
        - DataFrame with added 'sok' (%K line) and 'sod' (%D line) columns.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low'])

        df['sok'] = ((df['close'] - df['low'].rolling(window=kw).min()) /
                     (df['high'].rolling(window=kw).max() - df['low'].rolling(window=kw).min())) * 100
        df['sod'] = df['sok'].rolling(window=dw).mean()

        return df

    def sroc(self, df, period=14, smoothing=3):
        """
        Smoothed Rate of Change (SROC)

        A variation of the Rate of Change indicator that uses smoothing to reduce noise
        while measuring price momentum.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period for the ROC calculation.
        - smoothing (int): Period for the EMA smoothing.

        Returns:
        - DataFrame with an added 'sroc' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['rocs'] = df['close'].pct_change(period) * 100
        df['sroc'] = df['rocs'].ewm(span=smoothing, min_periods=smoothing).mean()

        return df.drop('rocs', axis=1)

    def srsi(self, df, w1=14, w2=14, smooth_k=3, smooth_d=3):
        """
        Stochastic RSI

        An indicator that applies the stochastic oscillator formula to RSI values,
        creating a more sensitive indicator for overbought and oversold conditions.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - w1 (int): Period for the RSI calculation.
        - w2 (int): Period for the Stochastic calculation.
        - smooth_k (int): Smoothing period for %K line.
        - smooth_d (int): Smoothing period for %D line.

        Returns:
        - DataFrame with added 'srsik' (%K line) and 'srsid' (%D line) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        delta = df['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=w1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=w1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        rsi_min = rsi.rolling(window=w2).min()
        rsi_max = rsi.rolling(window=w2).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)

        df['srsik'] = stoch_rsi.rolling(window=smooth_k).mean()
        df['srsid'] = df['srsik'].rolling(window=smooth_d).mean()

        return df

    def stc(self, df, p1=23, p2=50, p3=10, f=0.5):
        """
        Schaff Trend Cycle (STC)

        A cyclical oscillator that combines elements of MACD and stochastic oscillator
        to generate more reliable trading signals.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - p1 (int): Short-term EMA period.
        - p2 (int): Long-term EMA period.
        - p3 (int): Signal line period.
        - f (float): Factor for signal calculation.

        Returns:
        - DataFrame with added 'stc' and 'stcs' (signal) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        p1 = int(p1)
        p2 = int(p2)
        p3 = int(p3)

        df['ema1'] = df['close'].ewm(span=p1, min_periods=p1).mean()
        df['ema2'] = df['close'].ewm(span=p2, min_periods=p2).mean()
        df['macds'] = df['ema1'] - df['ema2']
        df['stcs'] = df['macds'].ewm(span=p3, min_periods=p3).mean()
        df['stc'] = df['stcs'] + f * (df['stcs'] - df['macds'])

        return df.drop(['ema1', 'ema2', 'macds'], axis=1)

    def sz(self, df, period=14, multiplier=1):
        """
        Safe Zone Indicator

        A volatility-based indicator that creates dynamic support and resistance
        levels around the price based on average price movement.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period for the ATR-like calculation.
        - multiplier (float): Multiplier for the zones.

        Returns:
        - DataFrame with added 'szu' (upper) and 'szl' (lower) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['atrs'] = df['close'].diff().abs().rolling(window=period).mean()
        df['szu'] = df['close'] + multiplier * df['atrs']
        df['szl'] = df['close'] - multiplier * df['atrs']

        return df.drop(['atrs'], axis=1)

    def tmf(self, df, period=21):
        """
        Twiggs Money Flow (TMF)

        A volume-based indicator that identifies buying and selling pressure,
        improving on the Chaikin Money Flow indicator.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', 'low', and 'volume' columns.
        - period (int): Period over which to calculate the TMF.

        Returns:
        - DataFrame with an added 'tmf' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low', 'volume'])

        df['mfs'] = ((2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])).fillna(0)
        df['mfvs'] = df['mfs'] * df['volume']
        df['tmf'] = df['mfvs'].rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

        return df.drop(['mfs', 'mfvs'], axis=1)

    def tmo(self, df, short_period=10, long_period=21):
        """
        Twiggs Momentum Oscillator

        A momentum indicator that measures the rate of change in price over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - short_period (int): Period for the momentum calculation.
        - long_period (int): Period for the moving average smoothing.

        Returns:
        - DataFrame with an added 'tmo' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['momentum'] = df['close'].diff(short_period)
        df['tmo'] = df['momentum'].rolling(window=long_period).mean()

        return df.drop('momentum', axis=1)

    def tp(self, df):
        """
        Typical Price

        A simple average of the high, low, and closing prices for a security.

        Parameters:
        - df (DataFrame): Input data containing 'high', 'low', and 'close' columns.

        Returns:
        - DataFrame with an added 'tp' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close'])

        df['tp'] = (df['high'] + df['low'] + df['close']) / 3

        return df

    def trix(self, df, w=15, sw=7):
        """
        Triple Exponential Moving Average (TRIX)

        A momentum oscillator that displays the percentage rate of change of a triple
        exponentially smoothed moving average, filtering out minor price movements.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - w (int): Period for the triple EMA calculation.
        - sw (int): Period for the signal line calculation.

        Returns:
        - DataFrame with added 'trix' and 'trixs' (signal) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['ema1'] = df['close'].ewm(span=w, adjust=False).mean()
        df['ema2'] = df['ema1'].ewm(span=w, adjust=False).mean()
        df['ema3'] = df['ema2'].ewm(span=w, adjust=False).mean()
        df['trix'] = (df['ema3'] - df['ema3'].shift()) / df['ema3'].shift() * 100
        df['trixs'] = df['trix'].ewm(span=sw, adjust=False).mean()

        return df.drop(['ema1', 'ema2', 'ema3'], axis=1)

    def tsi(self, df, short_period=13, long_period=25, tsi_p=5, ema_period=7):
        """
        True Strength Index

        A momentum oscillator that helps identify trends and reversals by tracking
        the momentum of price changes using double smoothing.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - short_period (int): Short-term smoothing period.
        - long_period (int): Long-term smoothing period.
        - tsi_p (int): Final TSI smoothing period.
        - ema_period (int): Signal line EMA period.

        Returns:
        - DataFrame with added 'tsi' and 'tsis' (signal) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        price_momentum = df['close'].diff(1)
        short_term_momentum = price_momentum.rolling(window=short_period).mean()
        long_term_momentum = price_momentum.rolling(window=long_period).mean()

        df['tsi'] = (short_term_momentum / long_term_momentum).rolling(window=tsi_p).mean() * 100
        df['tsis'] = df['tsi'].ewm(span=ema_period, adjust=False).mean()

        return df

    def tti(self, df, period=21):
        """
        Twiggs Trend Index

        An indicator that compares the close price to a moving average to determine
        the trend direction and strength.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period for the moving average calculation.

        Returns:
        - DataFrame with an added 'tti' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['smas'] = df['close'].rolling(window=period).mean()
        df['tti'] = ((df['close'] - df['smas']) / df['smas']) * 100

        return df.drop('smas', axis=1)

    def tv(self, df, period=20):
        """
        Twiggs Volatility

        A volatility indicator that measures the average price range over a specified period
        using a weighted calculation.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - period (int): Period for the volatility calculation.

        Returns:
        - DataFrame with an added 'tv' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        df['trs'] = high_max - low_min

        df['atrs'] = df['trs'].rolling(window=period).mean()
        df['tv'] = df['atrs'] * (2 ** 0.5)

        return df.drop(['trs', 'atrs'], axis=1)

    def ui(self, df, w1=14, w2=14):
        """
        Ulcer Index

        A volatility indicator that measures downside risk by quantifying the depth and
        duration of drawdowns from previous highs.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - w1 (int): Period for the drawdown calculation.
        - w2 (int): Period for the squared drawdown average.

        Returns:
        - DataFrame with an added 'ui' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['drawdown'] = (df['close'] - df['close'].rolling(window=w1).max()) / df['close'].rolling(
            window=w1).max() * 100
        df['squared_drawdown'] = df['drawdown'] ** 2
        df['ui'] = np.sqrt(df['squared_drawdown'].rolling(window=w2).mean())

        return df.drop(['drawdown', 'squared_drawdown'], axis=1)

    def uo(self, df, window1=7, window2=14, window3=28):
        """
        Ultimate Oscillator

        A momentum oscillator that incorporates three different time periods to reduce
        volatility and false signals common to other indicators.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', and 'low' columns.
        - window1 (int): Short-term period.
        - window2 (int): Medium-term period.
        - window3 (int): Long-term period.

        Returns:
        - DataFrame with an added 'uo' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low'])

        df['bps'] = df['close'] - df['low'].rolling(window=window1).min()
        df['trs'] = df['high'].rolling(window=window1).max() - df['low'].rolling(window=window1).min()
        df['uo'] = (df['bps'].rolling(window=window1).sum() / df['trs'].rolling(window=window1).sum() * 4 +
                    df['close'].rolling(window=window2).sum() / (df['high'].rolling(window=window2).max() -
                                                                 df['low'].rolling(window=window2).min()) * 2 +
                    df['close'].rolling(window=window3).sum() / (df['high'].rolling(window=window3).max() -
                                                                 df['low'].rolling(window=window3).min()) * 1) / (
                               4 + 2 + 1) * 100

        return df.drop(['bps', 'trs'], axis=1)

    def vhf(self, df, period=28):
        """
        Vertical Horizontal Filter

        An indicator that identifies trending and ranging markets by measuring the ratio
        of the largest price move to the sum of absolute price movements.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', and 'low' columns.
        - period (int): Period over which to calculate the VHF.

        Returns:
        - DataFrame with an added 'vhf' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low'])

        df['price change'] = df['close'].diff()
        df['abs price change'] = df['price change'].abs()
        df['sum abs price change'] = df['abs price change'].rolling(window=period).sum()

        df['highest high'] = df['high'].rolling(window=period).max()
        df['lowest low'] = df['low'].rolling(window=period).min()
        df['abs hh - ll'] = (df['highest high'] - df['lowest low']).abs()

        df['vhf'] = df['abs hh - ll'] / df['sum abs price change']

        return df.drop(['price change', 'abs price change', 'sum abs price change',
                        'highest high', 'lowest low', 'abs hh - ll'], axis=1)

    def vi(self, df, period=14):
        """
        Vortex Indicator

        A pair of oscillators that capture positive and negative trend movement based on
        the highs and lows of a security over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', and 'low' columns.
        - period (int): Period over which to calculate the indicator.

        Returns:
        - DataFrame with added 'vip' (positive) and 'vin' (negative) columns.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low'])

        df['trs'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
        df['vmplus'] = abs(df['high'].shift(1) - df['low'])
        df['vmneg'] = abs(df['low'].shift(1) - df['high'])

        tr_period = df['trs'].rolling(window=period).sum()
        vm_plus_period = df['vmplus'].rolling(window=period).sum()
        vm_minus_period = df['vmneg'].rolling(window=period).sum()

        df['vip'] = (vm_plus_period / tr_period) * 100
        df['vin'] = (vm_minus_period / tr_period) * 100

        return df.drop(['trs', 'vmplus', 'vmneg'], axis=1)

    def vo(self, df, short_period=14, long_period=28):
        """
        Volume Oscillator

        A volume-based indicator that shows the difference between two moving averages
        of volume as a percentage.

        Parameters:
        - df (DataFrame): Input data containing 'volume' column.
        - short_period (int): Period for the short-term moving average.
        - long_period (int): Period for the long-term moving average.

        Returns:
        - DataFrame with an added 'vo' column.
        """
        df = self._prepare_df(df, required_cols=['volume'])

        df['smas'] = df['volume'].rolling(window=short_period).mean()
        df['lmas'] = df['volume'].rolling(window=long_period).mean()
        df['vo'] = (df['smas'] - df['lmas']) / df['lmas'] * 100

        return df.drop(['smas', 'lmas'], axis=1)

    def vpt(self, df, sp=5, lp=20):
        """
        Volume Price Trend

        An indicator that combines price and volume to identify price trends and momentum.

        Parameters:
        - df (DataFrame): Input data containing 'close' and 'volume' columns.
        - sp (int): Short-term period.
        - lp (int): Long-term period.

        Returns:
        - DataFrame with added 'vpts' (short-term) and 'vptl' (long-term) columns.
        """
        df = self._prepare_df(df, required_cols=['close', 'volume'])

        vpt_short = df['volume'] * ((df['close'] - df['close'].shift(sp)) / df['close'].shift(sp))
        vpt_long = df['volume'] * ((df['close'] - df['close'].shift(lp)) / df['close'].shift(lp))
        df['vpts'] = vpt_short.cumsum()
        df['vptl'] = vpt_long.cumsum()

        return df

    def vr(self, df, period=20):
        """
        Volatility Ratio

        A measure of relative volatility that compares the current volatility to
        the average volatility over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - period (int): Period over which to calculate returns standard deviation.

        Returns:
        - DataFrame with an added 'vr' column.
        """
        df = self._prepare_df(df, required_cols=['close'])

        df['returns'] = df['close'].pct_change()
        df['std dev returns'] = df['returns'].rolling(window=period).std()
        avg_std_dev_returns = df['std dev returns'].mean()
        df['vr'] = df['std dev returns'] / avg_std_dev_returns

        return df.drop(['returns', 'std dev returns'], axis=1)

    def vroc(self, df, short_period=12, long_period=26):
        """
        Volume Rate of Change

        An indicator that compares the difference between short and long-term
        volume moving averages as a percentage.

        Parameters:
        - df (DataFrame): Input data containing 'volume' column.
        - short_period (int): Period for the short-term moving average.
        - long_period (int): Period for the long-term moving average.

        Returns:
        - DataFrame with an added 'vroc' column.
        """
        df = self._prepare_df(df, required_cols=['volume'])

        short_ma = df['volume'].rolling(window=short_period).mean()
        long_ma = df['volume'].rolling(window=long_period).mean()
        df['vroc'] = ((short_ma - long_ma) / long_ma) * 100

        return df

    def vs(self, df, period=14, multiplier=2):
        """
        Volatility Stops

        A volatility-based stop-loss indicator that helps determine potential
        exit points based on market volatility.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', and 'low' columns.
        - period (int): Period over which to calculate the volatility.
        - multiplier (float): Multiplier for the stops distance.

        Returns:
        - DataFrame with added 'vsh' (high stop) and 'vsl' (low stop) columns.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low'])

        df['high-low'] = df['high'] - df['low']
        df['high-prevclose'] = np.abs(df['high'] - df['close'].shift())
        df['low-prevclose'] = np.abs(df['low'] - df['close'].shift())
        df['trv'] = df[['high-low', 'high-prevclose', 'low-prevclose']].max(axis=1)

        df['atrv'] = df['trv'].rolling(window=period).mean()
        df['vsh'] = df['high'].rolling(window=period).max() - multiplier * df['atrv']
        df['vsl'] = df['low'].rolling(window=period).min() + multiplier * df['atrv']

        return df.drop(['trv', 'atrv', 'high-low', 'high-prevclose', 'low-prevclose'], axis=1)

    def vwap(self, df, window=14):
        """
        Volume Weighted Average Price

        A benchmark that represents the average price a security has traded at throughout
        a period, based on both volume and price.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', 'low', and 'volume' columns.
        - window (int): Period over which to calculate the VWAP.

        Returns:
        - DataFrame with an added 'vwap' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low', 'volume'])

        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close'])) / (
                df['volume'].rolling(window=window).sum() * 3)

        return df

    def wad(self, df, p=14):
        """
        Williams Accumulation/Distribution

        A volume-based indicator that determines if a stock is being accumulated or distributed.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'open', 'high', 'low', and 'volume' columns.
        - p (int): Smoothing period for the signal line.

        Returns:
        - DataFrame with added 'wad' and 'wads' (signal) columns.
        """
        df = self._prepare_df(df, required_cols=['close', 'open', 'high', 'low', 'volume'])

        df['wad'] = df['close'] - df['low'].shift(1).where(df['close'] > df['open'],
                                                           df['high'].shift(1).where(df['close'] < df['open'],
                                                                                     df['close'].shift(1)))
        df['wad'] = df['wad'] + df['volume'] * (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
        df['wads'] = df['wad'].ewm(span=p, min_periods=p).mean()

        return df

    def wma(self, df, p1=20, p2=9):
        """
        Weighted Moving Average

        A moving average that gives more weight to recent prices and less to older ones,
        responding more quickly to price changes than a simple moving average.

        Parameters:
        - df (DataFrame): Input data containing 'close' column.
        - p1 (int): Period for the WMA calculation.
        - p2 (int): Period for the signal line.

        Returns:
        - DataFrame with added 'wma' and 'wmas' (signal) columns.
        """
        df = self._prepare_df(df, required_cols=['close'])

        w = np.arange(p1) + 1
        w_s = w.sum()
        swv = np.lib.stride_tricks.sliding_window_view(df['close'].values.flatten(), window_shape=p1)
        sw = (swv * w).sum(axis=1) / w_s

        df['wma'] = np.concatenate((np.full(p1 - 1, np.nan), sw))
        df['wmas'] = df['wma'].rolling(p2).mean()

        return df

    def wr(self, df, window=14):
        """
        Williams %R

        A momentum indicator that measures overbought and oversold levels by comparing
        the close price to the high-low range over a specified period.

        Parameters:
        - df (DataFrame): Input data containing 'close', 'high', and 'low' columns.
        - window (int): Look-back period.

        Returns:
        - DataFrame with an added 'wr' column.
        """
        df = self._prepare_df(df, required_cols=['close', 'high', 'low'])

        df['wr'] = ((df['high'].rolling(window=window).max() - df['close']) / (
                df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min())) * -100

        return df

    def null(self, df):
        """
        Null Indicator

        A placeholder indicator that simply adds a zero-filled column.
        Useful for testing or as a baseline.

        Parameters:
        - df (DataFrame): Input dataframe.

        Returns:
        - DataFrame with an added 'null' column containing zeros.
        """
        df = self._prepare_df(df, required_cols=[])

        df['null'] = 0

        return df

    '''
    ---------------------------- Signals --------------------------------
    '''

    # Momentum Signals

    def adi_m(self, df, r1=7, r2=21):
        """
        Accumulation Distribution Index Momentum Signal

        Generates a signal based on MACD-style crossover of the ADI indicator.
        Signal: 1 (sell) when ADI MACD crosses below signal line, -1 (buy) when it crosses above.

        Parameters:
        - df (DataFrame): Input data with ADI column or required price and volume data.
        - r1 (int): Short-term EMA period.
        - r2 (int): Long-term EMA period.

        Returns:
        - DataFrame with an added 'adi_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['adi'])

        # Calculate MACD-style signals on ADI
        df['adimacd'] = df['adi'].ewm(span=r2, adjust=False).mean() - df['adi'].ewm(span=r1, adjust=False).mean()
        df['adimacds'] = df['adimacd'].rolling(r2).mean()

        # Generate signals: 1 for sell, -1 for buy
        df.loc[(df['adimacd'].shift(1) > df['adimacds'].shift(1)) & (df['adimacd'] < df['adimacds']), 'adi_m'] = 1
        df.loc[(df['adimacd'].shift(1) < df['adimacds'].shift(1)) & (df['adimacd'] > df['adimacds']), 'adi_m'] = -1

        return df.drop(['adimacd', 'adimacds'], axis=1)

    def ai_m(self, df):
        """
        Aroon Indicator Momentum Signal

        Generates a signal based on crossovers of the Aroon Up and Down indicators.
        Signal: 1 (sell) when Aroon Up crosses below Aroon Down, -1 (buy) for opposite.

        Parameters:
        - df (DataFrame): Input data with Aroon indicators or OHLC data.

        Returns:
        - DataFrame with an added 'ai_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['aiu', 'aid'])

        # Generate signals based on crossovers
        df.loc[(df['aiu'].shift(1) > df['aid'].shift(1)) & (df['aiu'] < df['aid']), 'ai_m'] = 1
        df.loc[(df['aiu'].shift(1) < df['aid'].shift(1)) & (df['aiu'] > df['aid']), 'ai_m'] = -1

        return df

    def atr_m(self, df, d=2):
        """
        Average True Range Momentum Signal

        Generates a signal based on price movements relative to ATR values.
        Signal: 1 (sell) when price drops by more than ATR/d, -1 (buy) when price rises.

        Parameters:
        - df (DataFrame): Input data with 'close' and ATR indicator or OHLC data.
        - d (float): Divisor for the ATR for signal threshold calculation.

        Returns:
        - DataFrame with an added 'atr_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'atr'])

        # Generate signals based on price movements relative to ATR
        df.loc[(df['close'].shift(1) >= df['close'].shift(2) - df['atr'].shift(2) / d) &
               (df['close'] < df['close'].shift(1) - df['atr'].shift(1) / d), 'atr_m'] = 1
        df.loc[(df['close'].shift(1) <= df['close'].shift(2) + df['atr'].shift(2) / d) &
               (df['close'] > df['close'].shift(1) - df['atr'].shift(1) / d), 'atr_m'] = -1

        return df

    def awo_m(self, df):
        """
        Awesome Oscillator Momentum Signal

        Generates a signal based on zero line crossovers and slope changes.
        Signal: 1 (sell) when crossing below zero or slope turns negative,
        -1 (buy) for opposite conditions.

        Parameters:
        - df (DataFrame): Input data with Awesome Oscillator or price data.

        Returns:
        - DataFrame with an added 'awo_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['awo'])

        # Generate signals based on zero line crossovers and slope changes
        df.loc[((df['awo'].shift(1) >= 0) & (df['awo'] < 0)) |
               ((df['awo'] < 0) & (df['awo'].shift(2) <= df['awo'].shift(1)) &
                (df['awo'].shift(1) > df['awo'])), 'awo_m'] = 1

        df.loc[((df['awo'].shift(1) <= 0) & (df['awo'] > 0)) |
               ((df['awo'] > 0) & (df['awo'].shift(2) >= df['awo'].shift(1)) &
                (df['awo'].shift(1) < df['awo'])), 'awo_m'] = -1

        return df

    def cc_m(self, df):
        """
        Coppock Curve Momentum Signal

        Generates a signal based on the Coppock Curve crossing above/below zero.
        Signal: 1 (sell) when crossing below zero, -1 (buy) when crossing above.

        Parameters:
        - df (DataFrame): Input data with Coppock Curve or price data.

        Returns:
        - DataFrame with an added 'cc_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['ccs'])

        # Generate signals based on zero line crossovers
        df.loc[(df['ccs'].shift(1) >= 0) & (df['ccs'] < 0), 'cc_m'] = 1
        df.loc[(df['ccs'].shift(1) <= 0) & (df['ccs'] > 0), 'cc_m'] = -1

        return df

    def cci_m(self, df, bz=100, sz=100):
        """
        Commodity Channel Index Momentum Signal

        Generates a signal based on CCI crossing key threshold levels.
        Signal: 1 (sell) when CCI crosses below upper threshold,
        -1 (buy) when crossing above lower threshold.

        Parameters:
        - df (DataFrame): Input data with CCI indicator or OHLC data.
        - bz (float): Lower boundary threshold for oversold condition.
        - sz (float): Upper boundary threshold for overbought condition.

        Returns:
        - DataFrame with an added 'cci_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['cci'])

        # Generate signals based on threshold crossings
        df.loc[(df['cci'].shift(1) > sz) & (df['cci'] < sz), 'cci_m'] = 1
        df.loc[(df['cci'].shift(1) < -bz) & (df['cci'] > -bz), 'cci_m'] = -1

        return df

    def dc_m(self, df):
        """
        Donchian Channel Momentum Signal

        Generates a signal based on price crossing the Donchian channel boundaries.
        Signal: 1 (sell) when price breaks below lower channel,
        -1 (buy) when price breaks above upper channel.

        Parameters:
        - df (DataFrame): Input data with Donchian Channels or OHLC data.

        Returns:
        - DataFrame with an added 'dc_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'dch', 'dcl'])

        # Generate signals based on price breakouts
        df.loc[(df['close'].shift(1) >= df['dcl'].shift(1)) & (df['close'] < df['dcl']), 'dc_m'] = 1
        df.loc[(df['close'].shift(1) <= df['dch'].shift(1)) & (df['close'] > df['dch']), 'dc_m'] = -1

        return df

    def di_m(self, df):
        """
        Disparity Index Momentum Signal

        Generates a signal based on Disparity Index crossing zero.
        Signal: 1 (sell) when DI crosses from positive to negative,
        -1 (buy) when DI crosses from negative to positive.

        Parameters:
        - df (DataFrame): Input data with Disparity Index or price data.

        Returns:
        - DataFrame with an added 'di_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['di'])

        # Generate signals based on zero line crossings
        df.loc[(df['di'].shift(1) >= 0) & (df['di'] < 0), 'di_m'] = 1
        df.loc[(df['di'].shift(1) <= 0) & (df['di'] > 0), 'di_m'] = -1

        return df

    def dpo_m(self, df):
        """
        Detrended Price Oscillator Momentum Signal

        Generates a signal based on DPO crossing zero.
        Signal: 1 (sell) when DPO crosses from negative to positive,
        -1 (buy) when DPO crosses from positive to negative.

        Parameters:
        - df (DataFrame): Input data with DPO indicator or price data.

        Returns:
        - DataFrame with an added 'dpo_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['dpo'])

        # Generate signals based on zero line crossings (note: reversed from typical since DPO is inverted)
        df.loc[(df['dpo'].shift(1) < 0) & (df['dpo'] > 0), 'dpo_m'] = -1
        df.loc[(df['dpo'].shift(1) > 0) & (df['dpo'] < 0), 'dpo_m'] = 1

        return df

    def ema_m(self, df):
        """
        Exponential Moving Average Momentum Signal

        Generates a signal based on crossover of fast and slow EMAs.
        Signal: 1 (sell) when fast EMA crosses below slow EMA,
        -1 (buy) when fast EMA crosses above slow EMA.

        Parameters:
        - df (DataFrame): Input data with EMA indicators or price data.

        Returns:
        - DataFrame with an added 'ema_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['ema', 'emas'])

        # Generate signals based on EMA crossovers
        df.loc[(df['emas'].shift(1) > df['ema'].shift(1)) & (df['emas'] < df['ema']), 'ema_m'] = 1
        df.loc[(df['emas'].shift(1) < df['ema'].shift(1)) & (df['emas'] > df['ema']), 'ema_m'] = -1

        return df

    def eom_m(self, df):
        """
        Ease of Movement Momentum Signal

        Generates a signal based on EOM crossing zero.
        Signal: 1 (sell) when EOM crosses below zero,
        -1 (buy) when EOM crosses above zero.

        Parameters:
        - df (DataFrame): Input data with EOM indicator or price and volume data.

        Returns:
        - DataFrame with an added 'eom_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['eom'])

        # Generate signals based on zero line crossings
        df.loc[(df['eom'].shift(1) >= 0) & (df['eom'] < 0), 'eom_m'] = 1
        df.loc[(df['eom'].shift(1) <= 0) & (df['eom'] > 0), 'eom_m'] = -1

        return df

    def fr_m(self, df, p=14):
        """
        Fibonacci Retracement Momentum Signal

        Generates a signal based on price crossing the 50% Fibonacci level
        with directional confirmation from DX indicator.

        Parameters:
        - df (DataFrame): Input data with Fibonacci levels or OHLC data.
        - p (int): Period for the Directional Movement Index calculation.

        Returns:
        - DataFrame with an added 'fr_m' signal column.
        """

        # Store column count to check if DX is added later
        a = len(df.columns)

        df = self._prepare_df(df, required_cols=['close', 'fr50', 'dip', 'din'])

        # Generate signals based on price crossing 50% level with DX confirmation
        df.loc[(df['close'].shift(1) > df['fr50'].shift(1)) & (df['close'] < df['fr50']) &
               (df['dip'] < df['din']), 'fr_m'] = 1
        df.loc[(df['close'].shift(1) < df['fr50'].shift(1)) & (df['close'] > df['fr50']) &
               (df['dip'] > df['din']), 'fr_m'] = -1

        # Drop DX columns if they were added by this method
        if a + 1 < len(df.columns):
            df = df.drop(['adx', 'dip', 'din'], axis=1)

        return df

    def ha_m(self, df):
        """
        Heikin-Ashi Momentum Signal

        Generates a signal based on Heikin-Ashi candlestick patterns.
        Signal: 1 (sell) when HA low decreases and close is below open,
        -1 (buy) when HA low increases and close is above open.

        Parameters:
        - df (DataFrame): Input data with Heikin-Ashi values or OHLC data.

        Returns:
        - DataFrame with an added 'ha_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['hal', 'hao', 'hac'])

        # Generate signals based on Heikin-Ashi patterns
        df.loc[(df['hal'].shift(1) > df['hal']) & (df['hac'] <= df['hao']), 'ha_m'] = 1
        df.loc[(df['hal'].shift(1) < df['hal']) & (df['hac'] >= df['hao']), 'ha_m'] = -1

        return df

    def ic_m(self, df):
        """
        Ichimoku Cloud Momentum Signal

        Generates a signal based on Ichimoku Tenkan-sen/Kijun-sen crossovers
        with cloud confirmation.

        Parameters:
        - df (DataFrame): Input data with Ichimoku indicators or OHLC data.

        Returns:
        - DataFrame with an added 'ic_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['high', 'low', 'ict', 'ick', 'icssa', 'icssb'])

        # Generate signals based on Ichimoku crossovers with cloud confirmation
        df.loc[(df['ict'].shift(1) > df['ick'].shift(1)) & (df['ict'] < df['ick']) &
               (df['high'] > df['icssa']), 'ic_m'] = 1
        df.loc[(df['ict'].shift(1) < df['ick'].shift(1)) & (df['ict'] > df['ick']) &
               (df['low'] < df['icssb']), 'ic_m'] = -1

        return df

    def kama_m(self, df):
        """
        Kaufman's Adaptive Moving Average Momentum Signal

        Generates a signal based on KAMA slope direction.
        Signal: 1 (sell) when KAMA slope remains negative,
        -1 (buy) when KAMA slope remains positive.

        Parameters:
        - df (DataFrame): Input data with KAMA indicator or price data.

        Returns:
        - DataFrame with an added 'kama_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['kama'])

        # Generate signals based on consecutive KAMA slope direction
        df.loc[(df['kama'].diff().shift(1) > 0) & (df['kama'].diff() > 0), 'kama_m'] = -1
        df.loc[(df['kama'].diff().shift(1) < 0) & (df['kama'].diff() < 0), 'kama_m'] = 1

        return df

    def kc_m(self, df, w=14, q=1.35):
        """
        Keltner Channel Momentum Signal

        Generates a signal based on price breaking channel boundaries with ATR confirmation.
        Signal: 1 (sell) when price breaks below lower channel with high volatility,
        -1 (buy) when price breaks above upper channel with high volatility.

        Parameters:
        - df (DataFrame): Input data with Keltner Channels or OHLC data.
        - w (int): Period for ATR calculation.
        - q (float): ATR threshold for volatility confirmation.

        Returns:
        - DataFrame with an added 'kc_m' signal column.
        """

        # Store column count to check if ATR is added later
        a = len(df.columns)

        # Calculate ATR if not present
        if 'atr' not in df.columns:
            w = int(w)
            df = self.atr(df, w)

        df = self._prepare_df(df, required_cols=['close', 'kcl', 'kch', 'atr'])

        # Generate signals based on price breaks with ATR confirmation
        df.loc[(df['close'].shift(1) >= df['kcl'].shift(1)) & (df['close'] < df['kcl']) &
               (df['atr'] > q), 'kc_m'] = 1
        df.loc[(df['close'].shift(1) <= df['kch'].shift(1)) & (df['close'] > df['kch']) &
               (df['atr'] > q), 'kc_m'] = -1

        # Drop ATR if it was added by this method
        if a + 1 < len(df.columns):
            df = df.drop(['atr'], axis=1)

        return df

    def kst_m(self, df, bz=50, sz=50):
        """
        Know Sure Thing Momentum Signal

        Generates a signal based on KST crossing its signal line with level confirmation.
        Signal: 1 (sell) when KST crosses below signal line while KST > sz,
        -1 (buy) when KST crosses above signal line while KST < -bz.

        Parameters:
        - df (DataFrame): Input data with KST indicator or price data.
        - bz (float): Lower boundary for buy signal confirmation.
        - sz (float): Upper boundary for sell signal confirmation.

        Returns:
        - DataFrame with an added 'kst_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['kst', 'ksts'])

        # Generate signals based on crossovers with level confirmation
        df.loc[(df['kst'].shift(1) > df['ksts'].shift(1)) & (df['kst'] < df['ksts']) &
               (df['kst'] > sz), 'kst_m'] = 1
        df.loc[(df['kst'].shift(1) < df['ksts'].shift(1)) & (df['kst'] > df['ksts']) &
               (df['kst'] < -bz), 'kst_m'] = -1

        return df

    def macd_m(self, df):
        """
        Moving Average Convergence Divergence Momentum Signal

        Generates a signal based on MACD line crossing its signal line.
        Signal: 1 (sell) when MACD crosses below signal line,
        -1 (buy) when MACD crosses above signal line.

        Parameters:
        - df (DataFrame): Input data with MACD indicators or price data.

        Returns:
        - DataFrame with an added 'macd_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['macd', 'macds'])

        # Generate signals based on MACD crossovers
        df.loc[(df['macd'].shift(1) > df['macds'].shift(1)) & (df['macd'] < df['macds']), 'macd_m'] = 1
        df.loc[(df['macd'].shift(1) < df['macds'].shift(1)) & (df['macd'] > df['macds']), 'macd_m'] = -1

        return df

    def mae_m(self, df):
        """
        Moving Average Envelope Momentum Signal

        Generates a signal based on price crossing the envelope bands.
        Signal: 1 (sell) when price crosses below upper band,
        -1 (buy) when price crosses above lower band.

        Parameters:
        - df (DataFrame): Input data with Moving Average Envelope or price data.

        Returns:
        - DataFrame with an added 'mae_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'maeu', 'maed'])

        # Generate signals based on envelope crossings
        df.loc[(df['close'].shift(1) >= df['maeu'].shift(1)) & (df['close'] < df['maeu']), 'mae_m'] = 1
        df.loc[(df['close'].shift(1) <= df['maed'].shift(1)) & (df['close'] > df['maed']), 'mae_m'] = -1

        return df

    def obv_m(self, df, r1=7, r2=21):
        """
        On-Balance Volume Momentum Signal

        Generates a signal based on MACD-style analysis of OBV and ADI.
        Signal: 1 (sell) when OBV MACD crosses below signal line,
        -1 (buy) when OBV MACD crosses above signal line.

        Parameters:
        - df (DataFrame): Input data with OBV or price and volume data.
        - r1 (int): Short-term EMA period.
        - r2 (int): Long-term EMA period.

        Returns:
        - DataFrame with an added 'obv_m' signal column.
        """

        # Check if ADI needs to be calculated
        a = 0
        if 'adi' not in df.columns:
            df = self.adi(df)
            a = 1

        df = self._prepare_df(df, required_cols=['obv', 'adi'])

        # Calculate MACD-style indicators for OBV
        df['obvmacd'] = df['obv'].ewm(span=r2, adjust=False).mean() - df['adi'].ewm(span=r1, adjust=False).mean()
        df['obvmacds'] = df['obvmacd'].rolling(r2).mean()

        # Generate signals based on OBV MACD crossovers
        df.loc[(df['obvmacd'].shift(1) > df['obvmacds'].shift(1)) & (df['obvmacd'] < df['obvmacds']), 'obv_m'] = 1
        df.loc[(df['obvmacd'].shift(1) < df['obvmacds'].shift(1)) & (df['obvmacd'] > df['obvmacds']), 'obv_m'] = -1

        # Drop ADI if it was added by this method
        if a == 1:
            df = df.drop(['adi'], axis=1)

        return df.drop(['obvmacd', 'obvmacds'], axis=1)

    def pp_m(self, df, p=14):
        """
        Pivot Points Momentum Signal

        Generates a signal based on price position relative to pivot point support/resistance
        with directional movement confirmation.

        Parameters:
        - df (DataFrame): Input data with Pivot Points or OHLC data.
        - p (int): Period for the Directional Movement Index calculation.

        Returns:
        - DataFrame with an added 'pp_m' signal column.
        """

        # Store column count to check if DX is added later
        a = len(df.columns)

        # Calculate Directional Movement Index if missing
        if 'dip' not in df.columns or 'din' not in df.columns:
            df = self.dx(df, p)

        df = self._prepare_df(df, required_cols=['close', 's2', 'r2', 'dip', 'din'])

        # Generate signals based on pivot points with directional movement confirmation
        df.loc[(df['close'] < df['s2']) & (df['dip'] < df['din']), 'pp_m'] = 1
        df.loc[(df['close'] > df['r2']) & (df['dip'] > df['din']), 'pp_m'] = -1

        # Drop DX columns if they were added by this method
        if a + 1 < len(df.columns):
            df = df.drop(['adx', 'dip', 'din'], axis=1)

        return df

    def ppo_m(self, df):
        """
        Percentage Price Oscillator Momentum Signal

        Generates a signal based on PPO crossing its signal line.
        Signal: 1 (sell) when PPO crosses below signal line,
        -1 (buy) when PPO crosses above signal line.

        Parameters:
        - df (DataFrame): Input data with PPO indicators or price data.

        Returns:
        - DataFrame with an added 'ppo_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['ppo', 'ppos'])

        # Generate signals based on PPO crossovers
        df.loc[(df['ppo'].shift(1) >= df['ppos'].shift(1)) & (df['ppo'] < df['ppos']), 'ppo_m'] = 1
        df.loc[(df['ppo'].shift(1) <= df['ppos'].shift(1)) & (df['ppo'] > df['ppos']), 'ppo_m'] = -1

        return df

    def proc_m(self, df):
        """
        Price Rate of Change Momentum Signal

        Generates a signal based on PROC crossing zero.
        Signal: 1 (sell) when PROC crosses from positive to negative,
        -1 (buy) when PROC crosses from negative to positive.

        Parameters:
        - df (DataFrame): Input data with PROC indicator or price data.

        Returns:
        - DataFrame with an added 'proc_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['proc'])

        # Generate signals based on zero line crossings
        df.loc[(df['proc'].shift(1) >= 0) & (df['proc'] < 0), 'proc_m'] = 1
        df.loc[(df['proc'].shift(1) <= 0) & (df['proc'] > 0), 'proc_m'] = -1

        return df

    def psar_m(self, df):
        """
        Parabolic SAR Momentum Signal

        Generates a signal based on price crossing the PSAR value.
        Signal: 1 (sell) when price crosses below PSAR,
        -1 (buy) when price crosses above PSAR.

        Parameters:
        - df (DataFrame): Input data with PSAR indicator or price data.

        Returns:
        - DataFrame with an added 'psar_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'psar'])

        # Generate signals based on PSAR crossovers
        df.loc[(df['close'].shift(1) >= df['psar'].shift(1)) & (df['close'] < df['psar']), 'psar_m'] = 1
        df.loc[(df['close'].shift(1) <= df['psar'].shift(1)) & (df['close'] > df['psar']), 'psar_m'] = -1

        return df

    def pvo_m(self, df):
        """
        Percentage Volume Oscillator Momentum Signal

        Generates a signal based on PVO crossing its signal line.
        Signal: 1 (sell) when PVO crosses below signal line,
        -1 (buy) when PVO crosses above signal line.

        Parameters:
        - df (DataFrame): Input data with PVO indicators or price and volume data.

        Returns:
        - DataFrame with an added 'pvo_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['pvo', 'pvos'])

        # Generate signals based on PVO crossovers
        df.loc[(df['pvo'].shift(1) >= df['pvos'].shift(1)) & (df['pvo'] < df['pvos']), 'pvo_m'] = 1
        df.loc[(df['pvo'].shift(1) <= df['pvos'].shift(1)) & (df['pvo'] > df['pvos']), 'pvo_m'] = -1

        return df

    def roc_m(self, df):
        """
        Rate of Change Momentum Signal

        Generates a signal based on ROC crossing its signal line.
        Signal: 1 (sell) when ROC crosses below signal line,
        -1 (buy) when ROC crosses above signal line.

        Parameters:
        - df (DataFrame): Input data with ROC indicators or price data.

        Returns:
        - DataFrame with an added 'roc_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['roc', 'rocs'])

        # Generate signals based on ROC crossovers
        df.loc[(df['roc'].shift(1) >= df['rocs'].shift(1)) & (df['roc'] < df['rocs']), 'roc_m'] = 1
        df.loc[(df['roc'].shift(1) <= df['rocs'].shift(1)) & (df['roc'] > df['rocs']), 'roc_m'] = -1

        return df

    def rsi_m(self, df, bz=30, sz=70):
        """
        Relative Strength Index Momentum Signal

        Generates a signal based on RSI crossing overbought/oversold thresholds.
        Signal: 1 (sell) when RSI crosses below overbought after being above,
        -1 (buy) when RSI crosses above oversold after being below.

        Parameters:
        - df (DataFrame): Input data with RSI indicator or price data.
        - bz (int): Lower boundary (oversold) threshold.
        - sz (int): Upper boundary (overbought) threshold.

        Returns:
        - DataFrame with an added 'rsi_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['rsi'])

        # Generate signals based on RSI threshold crossovers with confirmation
        df.loc[(df['rsi'].shift(1) > sz) & (df['rsi'] < sz) & (df['rsi'].shift(2) > sz), 'rsi_m'] = 1
        df.loc[(df['rsi'].shift(1) < bz) & (df['rsi'] > bz) & (df['rsi'].shift(2) < bz), 'rsi_m'] = -1

        return df

    def sma_m(self, df):
        """
        Simple Moving Average Momentum Signal

        Generates a signal based on crossover of fast and slow SMAs.
        Signal: 1 (sell) when fast SMA crosses below slow SMA,
        -1 (buy) when fast SMA crosses above slow SMA.

        Parameters:
        - df (DataFrame): Input data with SMA indicators or price data.

        Returns:
        - DataFrame with an added 'sma_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['sma', 'smas'])

        # Generate signals based on SMA crossovers
        df.loc[(df['smas'].shift(1) > df['sma'].shift(1)) & (df['smas'] < df['sma']), 'sma_m'] = 1
        df.loc[(df['smas'].shift(1) < df['sma'].shift(1)) & (df['smas'] > df['sma']), 'sma_m'] = -1

        return df

    def so_m(self, df):
        """
        Stochastic Oscillator Momentum Signal

        Generates a signal based on %K crossing %D.
        Signal: 1 (sell) when %K crosses below %D,
        -1 (buy) when %K crosses above %D.

        Parameters:
        - df (DataFrame): Input data with Stochastic Oscillator indicators or price data.

        Returns:
        - DataFrame with an added 'so_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['sok', 'sod'])

        # Generate signals based on Stochastic Oscillator crossovers
        df.loc[(df['sok'].shift(1) >= df['sod'].shift(1)) & (df['sok'] < df['sod']), 'so_m'] = 1
        df.loc[(df['sok'].shift(1) <= df['sod'].shift(1)) & (df['sok'] > df['sod']), 'so_m'] = -1

        return df

    def srsi_m(self, df):
        """
        Stochastic RSI Momentum Signal

        Generates a signal based on %K crossing %D.
        Signal: 1 (sell) when %K crosses below %D,
        -1 (buy) when %K crosses above %D.

        Parameters:
        - df (DataFrame): Input data with Stochastic RSI indicators or price data.

        Returns:
        - DataFrame with an added 'srsi_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['srsik', 'srsid'])

        # Generate signals based on Stochastic RSI crossovers
        df.loc[(df['srsik'].shift(1) > df['srsid'].shift(1)) & (df['srsik'] < df['srsid']), 'srsi_m'] = 1
        df.loc[(df['srsik'].shift(1) < df['srsid'].shift(1)) & (df['srsik'] > df['srsid']), 'srsi_m'] = -1

        return df

    def stc_m(self, df, bz=25, sz=75):
        """
        Schaff Trend Cycle Momentum Signal

        Generates a signal based on STC crossing overbought/oversold thresholds.
        Signal: 1 (sell) when STC crosses below overbought threshold,
        -1 (buy) when STC crosses above oversold threshold.

        Parameters:
        - df (DataFrame): Input data with STC indicator or price data.
        - bz (int): Lower boundary (oversold) threshold.
        - sz (int): Upper boundary (overbought) threshold.

        Returns:
        - DataFrame with an added 'stc_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['stc'])

        # Generate signals based on STC threshold crossovers
        df.loc[(df['stc'].shift(1) > sz) & (df['stc'] < sz), 'stc_m'] = 1
        df.loc[(df['stc'].shift(1) < bz) & (df['stc'] > bz), 'stc_m'] = -1

        return df

    def tmo_m(self, df, bz=20, sz=20):
        """
        Twiggs Momentum Oscillator Momentum Signal

        Generates a signal based on TMO crossing threshold levels.
        Signal: 1 (sell) when TMO crosses below upper threshold,
        -1 (buy) when TMO crosses above lower threshold.

        Parameters:
        - df (DataFrame): Input data with TMO indicator or price data.
        - bz (float): Lower boundary threshold.
        - sz (float): Upper boundary threshold.

        Returns:
        - DataFrame with an added 'tmo_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['tmo'])

        # Generate signals based on TMO threshold crossovers
        df.loc[(df['tmo'].shift(1) >= sz) & (df['tmo'] < sz), 'tmo_m'] = 1
        df.loc[(df['tmo'].shift(1) <= -bz) & (df['tmo'] > -bz), 'tmo_m'] = -1

        return df

    def trix_m(self, df):
        """
        Triple Exponential Moving Average Momentum Signal

        Generates a signal based on TRIX crossing its signal line.
        Signal: 1 (sell) when TRIX crosses below signal line,
        -1 (buy) when TRIX crosses above signal line.

        Parameters:
        - df (DataFrame): Input data with TRIX indicators or price data.

        Returns:
        - DataFrame with an added 'trix_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['trix', 'trixs'])

        # Generate signals based on TRIX crossovers
        df.loc[(df['trix'].shift(1) > df['trixs'].shift(1)) & (df['trix'] < df['trixs']), 'trix_m'] = 1
        df.loc[(df['trix'].shift(1) < df['trixs'].shift(1)) & (df['trix'] > df['trixs']), 'trix_m'] = -1

        return df

    def tsi_m(self, df):
        """
        True Strength Index Momentum Signal

        Generates a signal based on TSI crossing its signal line.
        Signal: 1 (sell) when TSI crosses below signal line,
        -1 (buy) when TSI crosses above signal line.

        Parameters:
        - df (DataFrame): Input data with TSI indicators or price data.

        Returns:
        - DataFrame with an added 'tsi_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['tsi', 'tsis'])

        # Generate signals based on TSI crossovers
        df.loc[(df['tsi'].shift(1) >= df['tsis'].shift(1)) & (df['tsi'] < df['tsis']), 'tsi_m'] = 1
        df.loc[(df['tsi'].shift(1) <= df['tsis'].shift(1)) & (df['tsi'] > df['tsis']), 'tsi_m'] = -1

        return df

    def uo_m(self, df, r1=3):
        """
        Ultimate Oscillator Momentum Signal

        Generates a signal based on UO crossing its moving average.
        Signal: 1 (sell) when UO crosses below its MA,
        -1 (buy) when UO crosses above its MA.

        Parameters:
        - df (DataFrame): Input data with Ultimate Oscillator or price data.
        - r1 (int): Period for the UO moving average.

        Returns:
        - DataFrame with an added 'uo_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['uo'])

        # Calculate UO moving average
        df['uos'] = df['uo'].rolling(r1).mean()

        # Generate signals based on UO crossing its MA
        df.loc[(df['uo'].shift(1) >= df['uos'].shift(1)) & (df['uo'] < df['uos']), 'uo_m'] = 1
        df.loc[(df['uo'].shift(1) <= df['uos'].shift(1)) & (df['uo'] > df['uos']), 'uo_m'] = -1

        return df.drop(['uos'], axis=1)

    def vi_m(self, df):
        """
        Vortex Indicator Momentum Signal

        Generates a signal based on VI+ crossing VI-.
        Signal: 1 (sell) when VI+ crosses below VI-,
        -1 (buy) when VI+ crosses above VI-.

        Parameters:
        - df (DataFrame): Input data with Vortex Indicators or OHLC data.

        Returns:
        - DataFrame with an added 'vi_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['vip', 'vin'])

        # Generate signals based on VI crossovers
        df.loc[(df['vip'].shift(1) > df['vin'].shift(1)) & (df['vip'] < df['vin']), 'vi_m'] = 1
        df.loc[(df['vip'].shift(1) < df['vin'].shift(1)) & (df['vip'] > df['vin']), 'vi_m'] = -1

        return df

    def vpt_m(self, df):
        """
        Volume Price Trend Momentum Signal

        Generates a signal based on short-term VPT crossing long-term VPT.
        Signal: -1 (buy) for both crossover conditions due to implementation.

        Parameters:
        - df (DataFrame): Input data with VPT indicators or price and volume data.

        Returns:
        - DataFrame with an added 'vpt_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['vpts', 'vptl'])

        # Generate signals based on VPT crossovers
        df.loc[(df['vpts'].shift(1) >= df['vptl'].shift(1)) & (df['vpts'] < df['vptl']), 'vpt_m'] = -1
        df.loc[(df['vpts'].shift(1) <= df['vptl'].shift(1)) & (df['vpts'] > df['vptl']), 'vpt_m'] = -1

        return df

    def vs_m(self, df):
        """
        Volatility Stops Momentum Signal

        Generates a signal based on price crossing the volatility stop levels.
        Signal: 1 (sell) when price drops below high stop,
        -1 (buy) when price rises above low stop.

        Parameters:
        - df (DataFrame): Input data with Volatility Stops or OHLC data.

        Returns:
        - DataFrame with an added 'vs_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'vsh', 'vsl'])

        # Generate signals based on price crossing volatility stops
        df.loc[(df['close'].shift(1) >= df['vsh'].shift(1)) & (df['close'] < df['vsh']), 'vs_m'] = 1
        df.loc[(df['close'].shift(1) <= df['vsl'].shift(1)) & (df['close'] > df['vsl']), 'vs_m'] = -1

        return df

    def wma_m(self, df):
        """
        Weighted Moving Average Momentum Signal

        Generates a signal based on crossover of fast and slow WMAs.
        Signal: 1 (sell) when fast WMA crosses below slow WMA,
        -1 (buy) when fast WMA crosses above slow WMA.

        Parameters:
        - df (DataFrame): Input data with WMA indicators or price data.

        Returns:
        - DataFrame with an added 'wma_m' signal column.
        """

        df = self._prepare_df(df, required_cols=['wma', 'wmas'])

        # Generate signals based on WMA crossovers
        df.loc[(df['wmas'].shift(1) >= df['wma'].shift(1)) & (df['wmas'] < df['wma']), 'wma_m'] = 1
        df.loc[(df['wmas'].shift(1) <= df['wma'].shift(1)) & (df['wmas'] > df['wma']), 'wma_m'] = -1

        return df

    def null_m(self, df):
        """
        Null Momentum Signal

        A placeholder signal that adds a zero-filled column without generating actual signals.
        Useful for testing or as a baseline comparison.

        Parameters:
        - df (DataFrame): Input dataframe.

        Returns:
        - DataFrame with an added 'null_m' column containing zeros.
        """
        df = self._prepare_df(df, required_cols=[])

        df['null_m'] = 0

        return df

    # Zone Signals

    def bb_z(self, df):
        """
        Bollinger Bands Zone Signal

        Identifies overbought/oversold conditions based on price position relative to Bollinger Bands.
        Signal: 1 (overbought) when price is above upper band,
        -1 (oversold) when price is below lower band.

        Parameters:
        - df (DataFrame): Input data with Bollinger Bands or price data.

        Returns:
        - DataFrame with an added 'bb_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'bbh', 'bbl'])

        # Generate zone signals
        df.loc[(df['close'].shift(1) >= df['bbh'].shift(1)) & (df['close'] > df['bbh']), 'bb_z'] = 1
        df.loc[(df['close'].shift(1) <= df['bbl'].shift(1)) & (df['close'] < df['bbl']), 'bb_z'] = -1

        return df

    def cc_z(self, df):
        """
        Coppock Curve Zone Signal

        Identifies bullish/bearish zones based on the Coppock Curve value.
        Signal: 1 (bearish zone) when CC < 0, -1 (bullish zone) when CC >= 0.

        Parameters:
        - df (DataFrame): Input data with Coppock Curve or price data.

        Returns:
        - DataFrame with an added 'cc_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['cc'])

        # Generate zone signals
        df.loc[(df['cc'] < 0), 'cc_z'] = 1
        df.loc[(df['cc'] >= 0), 'cc_z'] = -1

        return df

    def cci_z(self, df, bz=100, sz=100):
        """
        Commodity Channel Index Zone Signal

        Identifies overbought/oversold conditions based on CCI values.
        Signal: 1 (overbought) when CCI > sz, -1 (oversold) when CCI < -bz.

        Parameters:
        - df (DataFrame): Input data with CCI indicator or OHLC data.
        - bz (float): Lower boundary for oversold condition.
        - sz (float): Upper boundary for overbought condition.

        Returns:
        - DataFrame with an added 'cci_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['cci'])

        # Generate zone signals based on threshold values
        df.loc[df['cci'] > sz, 'cci_z'] = 1
        df.loc[df['cci'] < -bz, 'cci_z'] = -1

        return df

    def dc_z(self, df, m=2):
        """
        Donchian Channel Zone Signal

        Identifies extreme price zones based on multipliers of the Donchian Channel.
        Signal: 1 (extremely high zone) when price > upper channel * m,
        -1 (extremely low zone) when price < lower channel * m.

        Parameters:
        - df (DataFrame): Input data with Donchian Channels or OHLC data.
        - m (float): Multiplier for the channel values.

        Returns:
        - DataFrame with an added 'dc_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'dch', 'dcl'])

        # Generate zone signals based on multiplied channel values
        df.loc[(df['close'] > df['dch'] * m), 'dc_z'] = 1
        df.loc[(df['close'] < df['dcl'] * m), 'dc_z'] = -1

        return df

    def di_z(self, df):
        """
        Disparity Index Zone Signal

        Identifies overbought/oversold conditions based on Disparity Index value.
        Signal: 1 (bearish zone) when DI < 0, -1 (bullish zone) when DI >= 0.

        Parameters:
        - df (DataFrame): Input data with Disparity Index or price data.

        Returns:
        - DataFrame with an added 'di_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['di'])

        # Generate zone signals based on DI value
        df.loc[(df['di'] < 0), 'di_z'] = 1
        df.loc[(df['di'] >= 0), 'di_z'] = -1

        return df

    def dpo_z(self, df, qs=0.85, qb=-0.85):
        """
        Detrended Price Oscillator Zone Signal

        Identifies overbought/oversold zones based on DPO threshold values.
        Signal: 1 (overbought zone) when DPO > qs,
        -1 (oversold zone) when DPO < qb.

        Parameters:
        - df (DataFrame): Input data with DPO indicator or price data.
        - qs (float): Upper threshold for overbought condition.
        - qb (float): Lower threshold for oversold condition.

        Returns:
        - DataFrame with an added 'dpo_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['dpo'])

        # Create filtered DPO values for negative and positive readings
        df['dpon'] = df[df['dpo'] < 0]['dpo']
        df['dpop'] = df[df['dpo'] > 0]['dpo']

        # Generate zone signals based on threshold values
        df.loc[(df['dpo'] > qs), 'dpo_z'] = 1
        df.loc[(df['dpo'] < qb), 'dpo_z'] = -1

        return df.drop(['dpon', 'dpop'], axis=1)

    def dx_z(self, df, q=35):
        """
        Directional Movement Index Zone Signal

        Identifies strong directional trends based on ADX and DI values.
        Signal: 1 (strong downward trend) when DI+ < DI- with high ADX,
        -1 (strong upward trend) when DI+ >= DI- with high ADX.

        Parameters:
        - df (DataFrame): Input data with DX indicators or OHLC data.
        - q (float): Threshold for ADX to confirm strong trend.

        Returns:
        - DataFrame with an added 'dx_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['dip', 'din', 'adx'])

        # Generate zone signals based on DI crossovers with ADX confirmation
        df.loc[(df['dip'] < df['din']) & (df['adx'] > q), 'dx_z'] = 1
        df.loc[(df['dip'] >= df['din']) & (df['adx'] > q), 'dx_z'] = -1

        return df

    def ema_z(self, df):
        """
        Exponential Moving Average Zone Signal

        Identifies price zones relative to the EMA.
        Signal: 1 (above EMA) when price > EMA, -1 (below EMA) when price < EMA.

        Parameters:
        - df (DataFrame): Input data with EMA indicator or price data.

        Returns:
        - DataFrame with an added 'ema_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'ema'])

        # Generate zone signals based on price position relative to EMA
        df.loc[(df['close'] > df['ema']), 'ema_z'] = 1
        df.loc[(df['close'] < df['ema']), 'ema_z'] = -1

        return df

    def eom_z(self, df, qb=-13000, qs=13000):
        """
        Ease of Movement Zone Signal

        Identifies extreme zones in the Ease of Movement indicator.
        Signal: 1 (overbought) when EOM > qs, -1 (oversold) when EOM < qb.

        Parameters:
        - df (DataFrame): Input data with EOM indicator or price and volume data.
        - qb (float): Lower threshold for oversold condition.
        - qs (float): Upper threshold for overbought condition.

        Returns:
        - DataFrame with an added 'eom_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['eom'])

        # Generate zone signals based on threshold values
        df.loc[(df['eom'] > qs), 'eom_z'] = 1
        df.loc[(df['eom'] < qb), 'eom_z'] = -1

        return df

    def fi_z(self, df, qb=-1500, qs=1500):
        """
        Force Index Zone Signal

        Identifies extreme zones in the Force Index.
        Signal: 1 (overbought) when FI > qs, -1 (oversold) when FI < qb.

        Parameters:
        - df (DataFrame): Input data with Force Index or price and volume data.
        - qb (float): Lower threshold for oversold condition.
        - qs (float): Upper threshold for overbought condition.

        Returns:
        - DataFrame with an added 'fi_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['fi'])

        # Generate zone signals based on threshold values
        df.loc[df['fi'] > qs, 'fi_z'] = 1
        df.loc[df['fi'] < qb, 'fi_z'] = -1

        return df

    def fr_z(self, df, p=14):
        """
        Fibonacci Retracement Zone Signal

        Identifies price zones between Fibonacci retracement levels with directional confirmation.
        Signal: 1 (resistance zone) when price is between 38.2% and 61.8% with bearish bias,
        -1 (support zone) when price is between levels with bullish bias.

        Parameters:
        - df (DataFrame): Input data with Fibonacci levels or OHLC data.
        - p (int): Period for the Directional Movement Index calculation.

        Returns:
        - DataFrame with an added 'fr_z' signal column.
        """

        # Store column count to check if DX is added later
        a = len(df.columns)

        # Calculate DX if not present
        if 'dip' not in df.columns or 'din' not in df.columns:
            df = self.dx(df, p)

        df = self._prepare_df(df, required_cols=['close', 'fr38', 'fr61', 'dip', 'din'])

        # Generate zone signals based on price position and directional bias
        df.loc[(df['close'] < df['fr61']) & (df['close'] > df['fr38']) &
               (df['dip'] < df['din']), 'fr_z'] = 1
        df.loc[(df['close'] < df['fr38']) & (df['close'] > df['fr61']) &
               (df['dip'] > df['din']), 'fr_z'] = -1

        # Drop DX columns if they were added by this method
        if a + 1 < len(df.columns):
            df = df.drop(['adx', 'dip', 'din'], axis=1)

        return df

    def ic_z(self, df):
        """
        Ichimoku Cloud Zone Signal

        Identifies bullish/bearish zones based on Tenkan-sen and Kijun-sen relationship.
        Signal: 1 (bullish zone) when Tenkan-sen > Kijun-sen,
        -1 (bearish zone) when Tenkan-sen < Kijun-sen.

        Parameters:
        - df (DataFrame): Input data with Ichimoku indicators or OHLC data.

        Returns:
        - DataFrame with an added 'ic_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['ict', 'ick'])

        # Generate zone signals based on line relationships
        df.loc[(df['ict'] > df['ick']), 'ic_z'] = 1
        df.loc[(df['ict'] < df['ick']), 'ic_z'] = -1

        return df

    def kama_z(self, df, qb=500, qs=3500):
        """
        Kaufman's Adaptive Moving Average Zone Signal

        Identifies extreme price deviations from the KAMA baseline trend.
        Signal: 1 (overbought) when price is significantly above KAMA,
        -1 (oversold) when price is significantly below KAMA.

        Parameters:
        - df (DataFrame): Input data with KAMA indicator or price data.
        - qb (float): Lower threshold for price deviation from KAMA.
        - qs (float): Upper threshold for price deviation from KAMA.

        Returns:
        - DataFrame with an added 'kama_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'kama'])

        # Calculate KAMA trend line
        d = (df['kama'].iloc[-1] - df['kama'].iloc[0]) / len(df['kama'])
        df['kaman'] = df['kama'] - df['kama'].iloc[0] - d

        # Generate zone signals based on significant deviations from KAMA
        df.loc[(df['close'] > df['kama'] + qs), 'kama_z'] = 1
        df.loc[(df['close'] < df['kama'] + qb), 'kama_z'] = -1

        return df.drop(['kaman'], axis=1)

    def kc_z(self, df, bz=0.5, sz=0.5):
        """
        Keltner Channel Zone Signal

        Identifies volatility zones based on Keltner Channel width.
        Signal: 1 (low volatility) when KC width < -sz,
        -1 (high volatility) when KC width > bz.

        Parameters:
        - df (DataFrame): Input data with Keltner Channels or OHLC data.
        - bz (float): Threshold for high volatility condition.
        - sz (float): Threshold for low volatility condition.

        Returns:
        - DataFrame with an added 'kc_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['kcv'])

        # Generate zone signals based on channel width
        df.loc[(df['kcv'] < -sz), 'kc_z'] = 1
        df.loc[(df['kcv'] > bz), 'kc_z'] = -1

        return df

    def mae_z(self, df):
        """
        Moving Average Envelope Zone Signal

        Identifies price zones relative to Moving Average Envelope bands.
        Signal: 1 (overbought) when price > upper envelope,
        -1 (oversold) when price < lower envelope.

        Parameters:
        - df (DataFrame): Input data with Moving Average Envelope or price data.

        Returns:
        - DataFrame with an added 'mae_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'maeu', 'maed'])

        # Generate zone signals based on price position relative to envelope
        df.loc[(df['close'] > df['maeu']), 'mae_z'] = 1
        df.loc[(df['close'] < df['maed']), 'mae_z'] = -1

        return df

    def mfi_z(self, df, bz=20, sz=80):
        """
        Money Flow Index Zone Signal

        Identifies overbought/oversold conditions based on MFI values.
        Signal: 1 (overbought) when MFI > sz, -1 (oversold) when MFI < bz.

        Parameters:
        - df (DataFrame): Input data with MFI indicator or OHLC and volume data.
        - bz (int): Lower boundary for oversold condition.
        - sz (int): Upper boundary for overbought condition.

        Returns:
        - DataFrame with an added 'mfi_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['mfi'])

        # Generate zone signals based on threshold values
        df.loc[(df['mfi'] > sz), 'mfi_z'] = 1
        df.loc[(df['mfi'] < bz), 'mfi_z'] = -1

        return df

    def pp_z(self, df):
        """
        Pivot Points Zone Signal

        Identifies price zones relative to pivot point resistance and support levels.
        Signal: 1 (resistance zone) when price is between R1 and R2,
        -1 (support zone) when price is between S1 and R2.

        Parameters:
        - df (DataFrame): Input data with Pivot Points or OHLC data.

        Returns:
        - DataFrame with an added 'pp_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'r1', 'r2', 's1'])

        # Generate zone signals based on price position relative to pivot points
        df.loc[(df['close'] > df['r1']) & (df['close'] < df['r2']), 'pp_z'] = 1
        df.loc[(df['close'] < df['s1']) & (df['close'] > df['r2']), 'pp_z'] = -1

        return df

    def proc_z(self, df):
        """
        Price Rate of Change Zone Signal

        Identifies bullish/bearish zones based on PROC sign.
        Signal: 1 (bearish zone) when PROC < 0, -1 (bullish zone) when PROC >= 0.

        Parameters:
        - df (DataFrame): Input data with PROC indicator or price data.

        Returns:
        - DataFrame with an added 'proc_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['proc'])

        # Generate zone signals based on PROC sign
        df.loc[(df['proc'] < 0), 'proc_z'] = 1
        df.loc[(df['proc'] >= 0), 'proc_z'] = -1

        return df

    def roc_z(self, df, qb=-0.2, qs=0.2):
        """
        Rate of Change Zone Signal

        Identifies extreme zones based on ROC threshold values.
        Signal: 1 (overbought) when ROC > qs, -1 (oversold) when ROC < qb.

        Parameters:
        - df (DataFrame): Input data with ROC indicator or price data.
        - qb (float): Lower threshold for oversold condition.
        - qs (float): Upper threshold for overbought condition.

        Returns:
        - DataFrame with an added 'roc_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['roc'])

        # Generate zone signals based on threshold values
        df.loc[(df['roc'] > qs), 'roc_z'] = 1
        df.loc[(df['roc'] < qb), 'roc_z'] = -1

        return df

    def rsi_z(self, df, bz=30, sz=70):
        """
        Relative Strength Index Zone Signal

        Identifies overbought/oversold conditions based on RSI values.
        Signal: 1 (overbought) when RSI > sz, -1 (oversold) when RSI < bz.

        Parameters:
        - df (DataFrame): Input data with RSI indicator or price data.
        - bz (int): Lower boundary for oversold condition.
        - sz (int): Upper boundary for overbought condition.

        Returns:
        - DataFrame with an added 'rsi_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['rsi'])

        # Generate zone signals based on threshold values
        df.loc[(df['rsi'] > sz), 'rsi_z'] = 1
        df.loc[(df['rsi'] < bz), 'rsi_z'] = -1

        return df

    def sma_z(self, df):
        """
        Simple Moving Average Zone Signal

        Identifies price zones relative to the SMA.
        Signal: 1 (bullish zone) when price > SMA,
        -1 (bearish zone) when price < SMA.

        Parameters:
        - df (DataFrame): Input data with SMA indicator or price data.

        Returns:
        - DataFrame with an added 'sma_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'sma'])

        # Generate zone signals based on price position relative to SMA
        df.loc[(df['close'] > df['sma']), 'sma_z'] = 1
        df.loc[(df['close'] < df['sma']), 'sma_z'] = -1

        return df

    def so_z(self, df, bz=30, sz=70):
        """
        Stochastic Oscillator Zone Signal

        Identifies overbought/oversold conditions based on %K values.
        Signal: 1 (overbought) when %K > sz, -1 (oversold) when %K < bz.

        Parameters:
        - df (DataFrame): Input data with Stochastic Oscillator or OHLC data.
        - bz (int): Lower boundary for oversold condition.
        - sz (int): Upper boundary for overbought condition.

        Returns:
        - DataFrame with an added 'so_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['sok'])

        # Generate zone signals based on threshold values
        df.loc[(df['sok'] > sz), 'so_z'] = 1
        df.loc[(df['sok'] < bz), 'so_z'] = -1

        return df

    def srsi_z(self, df, bz=20, sz=80):
        """
        Stochastic RSI Zone Signal

        Identifies overbought/oversold conditions based on Stochastic RSI %K values.
        Signal: 1 (overbought) when %K > sz, -1 (oversold) when %K < bz.

        Parameters:
        - df (DataFrame): Input data with Stochastic RSI or price data.
        - bz (int): Lower boundary for oversold condition.
        - sz (int): Upper boundary for overbought condition.

        Returns:
        - DataFrame with an added 'srsi_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['srsik'])

        # Generate zone signals based on threshold values
        df.loc[(df['srsik'] > sz), 'srsi_z'] = 1
        df.loc[(df['srsik'] < bz), 'srsi_z'] = -1

        return df

    def stc_z(self, df, bz=25, sz=75):
        """
        Schaff Trend Cycle Zone Signal

        Identifies overbought/oversold conditions based on STC values.
        Signal: 1 (overbought) when STC > sz, -1 (oversold) when STC < bz.

        Parameters:
        - df (DataFrame): Input data with STC indicator or price data.
        - bz (int): Lower boundary for oversold condition.
        - sz (int): Upper boundary for overbought condition.

        Returns:
        - DataFrame with an added 'stc_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['stc'])

        # Generate zone signals based on threshold values
        df.loc[df['stc'] > sz, 'stc_z'] = 1
        df.loc[df['stc'] < bz, 'stc_z'] = -1

        return df

    def sz_z(self, df):
        """
        Safe Zone Indicator Zone Signal

        Identifies price zones relative to Safe Zone bands.
        Signal: 1 (upper zone) when price > upper band,
        -1 (lower zone) when price < lower band.

        Parameters:
        - df (DataFrame): Input data with Safe Zone indicator or price data.

        Returns:
        - DataFrame with an added 'sz_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'szu', 'szl'])

        # Generate zone signals based on price position relative to bands
        df.loc[(df['close'] > df['szu']), 'sz_z'] = 1
        df.loc[(df['close'] < df['szl']), 'sz_z'] = -1

        return df

    def tmo_z(self, df, bz=20, sz=20):
        """
        Twiggs Momentum Oscillator Zone Signal

        Identifies extreme momentum zones based on TMO values.
        Signal: 1 (strong bullish momentum) when TMO > sz,
        -1 (strong bearish momentum) when TMO < -bz.

        Parameters:
        - df (DataFrame): Input data with TMO indicator or price data.
        - bz (float): Lower threshold for bearish momentum.
        - sz (float): Upper threshold for bullish momentum.

        Returns:
        - DataFrame with an added 'tmo_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['tmo'])

        # Generate zone signals based on threshold values
        df.loc[(df['tmo'] > sz), 'tmo_z'] = 1
        df.loc[(df['tmo'] < -bz), 'tmo_z'] = -1

        return df

    def tsi_z(self, df, qb=6, qs=190):
        """
        True Strength Index Zone Signal

        Identifies extreme momentum zones based on TSI values.
        Signal: 1 (overbought) when TSI > qs, -1 (oversold) when TSI < qb.

        Parameters:
        - df (DataFrame): Input data with TSI indicator or price data.
        - qb (float): Lower threshold for oversold condition.
        - qs (float): Upper threshold for overbought condition.

        Returns:
        - DataFrame with an added 'tsi_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['tsi'])

        # Generate zone signals based on threshold values
        df.loc[(df['tsi'] > qs), 'tsi_z'] = 1
        df.loc[(df['tsi'] < qb), 'tsi_z'] = -1

        return df

    def tv_z(self, df, qb=24, qs=4):
        """
        Twiggs Volatility Zone Signal

        Identifies volatility zones based on TV values.
        Signal: 1 (low volatility) when TV < qs,
        -1 (high volatility) when TV > qb.

        Parameters:
        - df (DataFrame): Input data with TV indicator or OHLC data.
        - qb (float): Threshold for high volatility.
        - qs (float): Threshold for low volatility.

        Returns:
        - DataFrame with an added 'tv_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['tv'])

        # Generate zone signals based on threshold values
        df.loc[(df['tv'] < qs), 'tv_z'] = 1
        df.loc[(df['tv'] > qb), 'tv_z'] = -1

        return df

    def uo_z(self, df, bz=30, sz=70):
        """
        Ultimate Oscillator Zone Signal

        Identifies overbought/oversold conditions based on UO values.
        Signal: 1 (overbought) when UO > sz, -1 (oversold) when UO < bz.

        Parameters:
        - df (DataFrame): Input data with Ultimate Oscillator or OHLC data.
        - bz (int): Lower boundary for oversold condition.
        - sz (int): Upper boundary for overbought condition.

        Returns:
        - DataFrame with an added 'uo_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['uo'])

        # Generate zone signals based on threshold values
        df.loc[(df['uo'] > sz), 'uo_z'] = 1
        df.loc[(df['uo'] < bz), 'uo_z'] = -1

        return df

    def vs_z(self, df):
        """
        Volatility Stops Zone Signal

        Identifies price zones relative to volatility stop levels.
        Signal: 1 (upper zone) when price > upper stop,
        -1 (lower zone) when price < lower stop.

        Parameters:
        - df (DataFrame): Input data with Volatility Stops or OHLC data.

        Returns:
        - DataFrame with an added 'vs_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'vsh', 'vsl'])

        # Generate zone signals based on price position relative to stops
        df.loc[(df['close'] > df['vsh']), 'vs_z'] = 1
        df.loc[(df['close'] < df['vsl']), 'vs_z'] = -1

        return df

    def wad_z(self, df):
        """
        Williams Accumulation/Distribution Zone Signal

        Identifies bullish/bearish zones based on WAD relationship to its signal line.
        Signal: 1 (bearish zone) when WAD < signal line,
        -1 (bullish zone) when WAD >= signal line.

        Parameters:
        - df (DataFrame): Input data with Williams AD indicators or OHLC data.

        Returns:
        - DataFrame with an added 'wad_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['wad', 'wads'])

        # Generate zone signals based on WAD position relative to signal
        df.loc[(df['wad'] < df['wads']), 'wad_z'] = 1
        df.loc[(df['wad'] >= df['wads']), 'wad_z'] = -1

        return df

    def wma_z(self, df):
        """
        Weighted Moving Average Zone Signal

        Identifies price zones relative to the WMA.
        Signal: 1 (bullish zone) when price > WMA,
        -1 (bearish zone) when price <= WMA.

        Parameters:
        - df (DataFrame): Input data with WMA indicator or price data.

        Returns:
        - DataFrame with an added 'wma_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['close', 'wma'])

        # Generate zone signals based on price position relative to WMA
        df.loc[(df['close'] > df['wma']), 'wma_z'] = 1
        df.loc[(df['close'] <= df['wma']), 'wma_z'] = -1

        return df

    def wr_z(self, df, bz=80, sz=20):
        """
        Williams %R Zone Signal

        Identifies overbought/oversold conditions based on Williams %R values.
        Signal: 1 (overbought) when WR > -sz, -1 (oversold) when WR < -bz.

        Parameters:
        - df (DataFrame): Input data with Williams %R indicator or OHLC data.
        - bz (int): Lower boundary value for oversold condition.
        - sz (int): Upper boundary value for overbought condition.

        Returns:
        - DataFrame with an added 'wr_z' signal column.
        """

        df = self._prepare_df(df, required_cols=['wr'])

        # Generate zone signals based on threshold values
        df.loc[df['wr'] > -sz, 'wr_z'] = 1
        df.loc[df['wr'] < -bz, 'wr_z'] = -1

        return df

    def null_z(self, df):
        """
        Null Zone Signal

        A placeholder signal that adds a zero-filled column without generating actual signals.
        Useful for testing or as a baseline comparison.

        Parameters:
        - df (DataFrame): Input dataframe.

        Returns:
        - DataFrame with an added 'null_z' column containing zeros.
        """
        df = self._prepare_df(df, required_cols=[])

        df['null_z'] = 0

        return df

    # Trend Signals

    def adi_t(self, df, window=14):
        """
        Accumulation/Distribution Index (ADI) Trading Signal

        Generates trading signals based on the ADI moving average crossover.
        Buy signal (-1) when ADIM crosses below ADI, Sell signal (1) when ADIM crosses above ADI.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing the 'adi' column
        window : int, default 14
            Rolling window period for the moving average

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'adi_t' signal column
        """
        df = self._prepare_df(df, required_cols=['adi'])

        df['adim'] = df['adi'].rolling(window).mean()
        df.loc[(df['adim'].shift(1) > df['adim']), 'adi_t'] = 1
        df.loc[(df['adim'].shift(1) < df['adim']), 'adi_t'] = -1
        return df.drop(['adim'], axis=1)

    def ai_t(self, df):
        """
        Aroon Indicator Trading Signal

        Generates trading signals based on price position relative to Aroon bands.
        Buy signal (-1) when price crosses above upper band,
        Sell signal (1) when price crosses below lower band.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close', 'aid' (lower band), and 'aiu' (upper band) columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'ai_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'aid', 'aiu'])

        df.loc[(df['close'] < df['aid']), 'ai_t'] = 1
        df.loc[(df['close'] > df['aiu']), 'ai_t'] = -1
        return df

    def atr_t(self, df, divisor=3):
        """
        Average True Range (ATR) Trading Signal

        Generates trading signals based on price movements relative to ATR bands.
        Buy signal (-1) when price moves above previous close plus ATR/divisor,
        Sell signal (1) when price moves below previous close minus ATR/divisor.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close' and 'atr' columns
        divisor : int, default 3
            Divisor for ATR bands calculation

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'atr_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'atr'])

        df.loc[(df['close'] < df['close'].shift(1) - df['atr'].shift(1) / divisor), 'atr_t'] = 1
        df.loc[(df['close'] > df['close'].shift(1) + df['atr'].shift(1) / divisor), 'atr_t'] = -1
        return df

    def awo_t(self, df):
        """
        Awesome Oscillator Trading Signal

        Generates trading signals based on the Awesome Oscillator value.
        Buy signal (-1) when AWO is positive, Sell signal (1) when AWO is negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'awo' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'awo_t' signal column
        """
        df = self._prepare_df(df, required_cols=['awo'])

        df.loc[(df['awo'] < 0), 'awo_t'] = 1
        df.loc[(df['awo'] >= 0), 'awo_t'] = -1
        return df

    def bb_t(self, df, period=14):
        """
        Bollinger Bands Trading Signal

        Generates trading signals based on Bollinger Bands and Directional Movement.
        Buy signal (-1) when price touches lower band and DI+ < DI-
        Sell signal (1) when price touches upper band and DI+ > DI-

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close', 'bbh', 'bbl' columns
        period : int, default 14
            Period for directional movement calculation

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'bb_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'bbh', 'bbl'])

        initial_cols = len(df.columns)
        df = dx(df, period)
        df.loc[(df['close'] >= df['bbh']) & (df['dip'] > df['din']), 'bb_t'] = 1
        df.loc[(df['close'] <= df['bbl']) & (df['dip'] < df['din']), 'bb_t'] = -1
        if initial_cols + 1 < len(df.columns):
            df = df.drop(['adx', 'dip', 'din'], axis=1)
        return df

    def cmf_t(self, df):
        """
        Chaikin Money Flow Trading Signal

        Generates trading signals based on CMF crossing zero line.
        Buy signal (-1) when CMF is positive, Sell signal (1) when CMF is negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'cmf' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'cmf_t' signal column
        """
        df = self._prepare_df(df, required_cols=['cmf'])

        df.loc[(df['cmf'] > 0), 'cmf_t'] = 1
        df.loc[(df['cmf'] <= 0), 'cmf_t'] = -1
        return df

    def cmo_t(self, df):
        """
        Chande Momentum Oscillator Trading Signal

        Generates trading signals based on CMO crossing zero line.
        Buy signal (-1) when CMO is positive, Sell signal (1) when CMO is negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'cmo' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'cmo_t' signal column
        """
        df = self._prepare_df(df, required_cols=['cmo'])

        df.loc[(df['cmo'] < 0), 'cmo_t'] = 1
        df.loc[(df['cmo'] >= 0), 'cmo_t'] = -1
        return df

    def dc_t(self, df, period=14):
        """
        Donchian Channel Trading Signal

        Generates trading signals based on price position relative to middle band and DI.
        Buy signal (-1) when price above middle band and DI+ > DI-
        Sell signal (1) when price below middle band and DI+ < DI-

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close', 'dcm' columns
        period : int, default 14
            Period for directional movement calculation

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'dc_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'dcm'])

        initial_cols = len(df.columns)
        df = dx(df, period)
        df.loc[(df['close'] > df['dcm']) & (df['dip'] > df['din']), 'dc_t'] = -1
        df.loc[(df['close'] < df['dcm']) & (df['dip'] < df['din']), 'dc_t'] = 1
        if initial_cols + 1 < len(df.columns):
            df = df.drop(['adx', 'dip', 'din'], axis=1)
        return df

    def dma_t(self, df):
        """
        Dickson Moving Average Trading Signal

        Generates trading signals based on price position relative to DMA.
        Buy signal (-1) when price is above DMA, Sell signal (1) when price is below DMA.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close' and 'dma' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'dma_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'dma'])

        df.loc[(df['close'] < df['dma']), 'dma_t'] = 1
        df.loc[(df['close'] >= df['dma']), 'dma_t'] = -1
        return df

    def dx_t(self, df):
        """
        Directional Movement Index Trading Signal

        Generates trading signals based on DI+ and DI- comparison.
        Buy signal (-1) when DI+ >= DI-, Sell signal (1) when DI+ < DI-.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'dip' and 'din' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'dx_t' signal column
        """
        df = self._prepare_df(df, required_cols=['dip', 'din'])

        df.loc[(df['dip'] < df['din']), 'dx_t'] = 1
        df.loc[(df['dip'] >= df['din']), 'dx_t'] = -1
        return df

    def eom_t(self, df):
        """
        Ease of Movement Trading Signal

        Generates trading signals based on EOM crossing zero line.
        Buy signal (-1) when EOM is positive, Sell signal (1) when EOM is negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'eom' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'eom_t' signal column
        """
        df = self._prepare_df(df, required_cols=['eom'])

        df.loc[(df['eom'] < 0), 'eom_t'] = 1
        df.loc[(df['eom'] >= 0), 'eom_t'] = -1
        return df

    def eri_t(self, df):
        """
        Elder Ray Index Trading Signal

        Generates trading signals based on Bull and Bear Power indicators.
        Buy signal (-1) when Bull Power > 0 and Bear Power < 0
        Sell signal (1) when Bull Power < 0 and Bear Power > 0

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'erbup' and 'erbep' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'eri_t' signal column
        """
        df = self._prepare_df(df, required_cols=['erbup', 'erbep'])

        df.loc[(df['erbup'] < 0) & (df['erbep'] > 0), 'eri_t'] = 1
        df.loc[(df['erbup'] > 0) & (df['erbep'] < 0), 'eri_t'] = -1
        return df

    def fi_t(self, df):
        """
        Force Index Trading Signal

        Generates trading signals based on Force Index crossing zero line.
        Buy signal (-1) when FI is positive, Sell signal (1) when FI is negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'fi' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'fi_t' signal column
        """
        df = self._prepare_df(df, required_cols=['fi'])

        df.loc[df['fi'] > 0, 'fi_t'] = 1
        df.loc[df['fi'] <= 0, 'fi_t'] = -1
        return df

    def ha_t(self, df):
        """
        Heikin Ashi Trading Signal

        Generates trading signals based on Heikin Ashi candle comparison.
        Buy signal (-1) when HA close >= HA open
        Sell signal (1) when HA close < HA open

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'hac' (HA close) and 'hao' (HA open) columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'ha_t' signal column
        """
        df = self._prepare_df(df, required_cols=['hac', 'hao'])

        df.loc[(df['hac'] < df['hao']), 'ha_t'] = 1
        df.loc[(df['hac'] >= df['hao']), 'ha_t'] = -1
        return df

    def ic_t(self, df):
        """
        Ichimoku Cloud Trading Signal

        Generates trading signals based on price position relative to Ichimoku Cloud.
        Buy signal (-1) when price above cloud, Sell signal (1) when price below cloud.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close' and 'ick' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'ic_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'ick'])

        df.loc[(df['close'] < df['ick']), 'ic_t'] = 1
        df.loc[(df['close'] > df['ick']), 'ic_t'] = -1
        return df

    def kama_t(self, df, window=3):
        """
        Kaufman Adaptive Moving Average Trading Signal

        Generates trading signals based on KAMA slope.
        Buy signal (-1) when KAMA is rising, Sell signal (1) when KAMA is falling.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'kama' column
        window : int, default 3
            Rolling window for KAMA smoothing

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'kama_t' signal column
        """
        df = self._prepare_df(df, required_cols=['kama'])

        df['kamas'] = df['kama'].rolling(window=window).mean()
        df.loc[df['kamas'] > df['kamas'].shift(1), 'kama_t'] = 1
        df.loc[df['kamas'] < df['kamas'].shift(1), 'kama_t'] = -1
        return df.drop(['kamas'], axis=1)

    def kc_t(self, df):
        """
        Keltner Channel Trading Signal

        Generates trading signals based on price position relative to middle Keltner Channel.
        Buy signal (-1) when price is above middle channel, Sell signal (1) when below.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close' and 'kcm' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'kc_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'kcm'])

        df.loc[df['close'] < df['kcm'], 'kc_t'] = 1
        df.loc[df['close'] >= df['kcm'], 'kc_t'] = -1
        return df

    def kst_t(self, df):
        """
        Know Sure Thing Trading Signal

        Generates trading signals based on KST histogram.
        Buy signal (-1) when KST histogram is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'ksth' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'kst_t' signal column
        """
        df = self._prepare_df(df, required_cols=['ksth'])

        df.loc[df['ksth'] > 0, 'kst_t'] = -1
        df.loc[df['ksth'] < 0, 'kst_t'] = 1
        return df

    def lr_t(self, df):
        """
        Linear Regression Trading Signal

        Generates trading signals based on price position relative to linear regression line.
        Buy signal (-1) when price is above LR line, Sell signal (1) when below.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close' and 'lr' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'lr_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'lr'])

        df.loc[(df['close'] < df['lr']), 'lr_t'] = 1
        df.loc[(df['close'] >= df['lr']), 'lr_t'] = -1
        return df

    def macd_t(self, df):
        """
        MACD Trading Signal

        Generates trading signals based on MACD histogram.
        Buy signal (-1) when histogram is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'macdh' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'macd_t' signal column
        """
        df = self._prepare_df(df, required_cols=['macdh'])

        df.loc[(df['macdh'] < 0), 'macd_t'] = 1
        df.loc[(df['macdh'] >= 0), 'macd_t'] = -1
        return df

    def mfi_t(self, df):
        """
        Money Flow Index Trading Signal

        Generates trading signals based on MFI crossing zero line.
        Buy signal (-1) when MFI is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'mfi' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'mfi_t' signal column
        """
        df = self._prepare_df(df, required_cols=['mfi'])

        df.loc[(df['mfi'] < 0), 'mfi_t'] = 1
        df.loc[(df['mfi'] >= 0), 'mfi_t'] = -1
        return df

    def nvi_t(self, df):
        """
        Negative Volume Index Trading Signal

        Generates trading signals based on normalized NVI and its signal line crossover.
        Buy signal (-1) when NVI crosses above signal, Sell signal (1) when crosses below.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'nvi' and 'nvis' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'nvi_t' signal column
        """
        df = self._prepare_df(df, required_cols=['nvi', 'nvis'])

        df['nvi'] = (df['nvi'] - df['nvi'].mean()) / df['nvi'].std()
        df['nvis'] = (df['nvis'] - df['nvis'].mean()) / df['nvis'].std()
        df.loc[(df['nvi'] < df['nvis']), 'nvi_t'] = 1
        df.loc[(df['nvi'] > df['nvis']), 'nvi_t'] = -1
        return df

    def obv_t(self, df, r1=7, r2=21):
        """
        On Balance Volume Trading Signal

        Generates trading signals based on OBV MACD crossover.
        Buy signal (-1) when OBV MACD crosses above signal, Sell signal (1) when crosses below.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'obv' column
        r1 : int, default 7
            Short-term period
        r2 : int, default 21
            Long-term period

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'obv_t' signal column
        """
        df = self._prepare_df(df, required_cols=['obv'])

        add_adi = False
        if 'adi' not in df.columns:
            df = adi(df)
            add_adi = True

        df['obvmacd'] = df['obv'].ewm(span=r2, adjust=False).mean() - df['adi'].ewm(span=r1, adjust=False).mean()
        df['obvmacds'] = df['obvmacd'].rolling(r2).mean()
        df.loc[df['obvmacd'] < df['obvmacds'], 'obv_t'] = 1
        df.loc[df['obvmacd'] > df['obvmacds'], 'obv_t'] = -1

        if add_adi:
            df = df.drop(['adi'], axis=1)
        return df.drop(['obvmacd', 'obvmacds'], axis=1)

    def pc_t(self, df):
        """
        Price Channel Trading Signal

        Generates trading signals based on Price Channel momentum.
        Buy signal (-1) when PC is negative, Sell signal (1) when PC is positive.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'pc' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'pc_t' signal column
        """
        df = self._prepare_df(df, required_cols=['pc'])

        df.loc[df['pc'] >= 0, 'pc_t'] = -1
        df.loc[df['pc'] < 0, 'pc_t'] = 1
        return df

    def ppo_t(self, df):
        """
        Percentage Price Oscillator Trading Signal

        Generates trading signals based on PPO histogram.
        Buy signal (-1) when histogram is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'ppoh' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'ppo_t' signal column
        """
        df = self._prepare_df(df, required_cols=['ppoh'])

        df.loc[(df['ppoh'] < 0), 'ppo_t'] = 1
        df.loc[(df['ppoh'] >= 0), 'ppo_t'] = -1
        return df

    def psar_t(self, df):
        """
        Parabolic SAR Trading Signal

        Generates trading signals based on price position relative to PSAR.
        Buy signal (-1) when price is above PSAR, Sell signal (1) when below.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close' and 'psar' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'psar_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'psar'])

        df.loc[(df['close'] < df['psar']), 'psar_t'] = 1
        df.loc[(df['close'] >= df['psar']), 'psar_t'] = -1
        return df

    def pvo_t(self, df):
        """
        Percentage Volume Oscillator Trading Signal

        Generates trading signals based on PVO histogram.
        Buy signal (-1) when histogram is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'pvoh' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'pvo_t' signal column
        """
        df = self._prepare_df(df, required_cols=['pvoh'])

        df.loc[(df['pvoh'] < 0), 'pvo_t'] = 1
        df.loc[(df['pvoh'] >= 0), 'pvo_t'] = -1
        return df

    def roc_t(self, df):
        """
        Rate of Change Trading Signal

        Generates trading signals based on ROC crossing zero line.
        Buy signal (-1) when ROC is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'roc' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'roc_t' signal column
        """
        df = self._prepare_df(df, required_cols=['roc'])

        df.loc[(df['roc'] < 0), 'roc_t'] = 1
        df.loc[(df['roc'] >= 0), 'roc_t'] = -1
        return df

    def so_t(self, df, period=14):
        """
        Stochastic Oscillator Trading Signal

        Generates trading signals based on price and Stochastic Oscillator movements.
        Buy/Sell signals based on price minimums and SO minimums relationships.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close' and 'sok' columns
        period : int, default 14
            Look-back period for calculations

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'so_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'sok'])

        df.loc[(df['close'].rolling(window=period).min() < df['close'].rolling(window=period).min().shift(1)) &
               (df['sok'].rolling(window=period).min() > df['sok'].rolling(window=period).min().shift(1)), 'so_t'] = -1
        df.loc[(df['close'].rolling(window=period).min() > df['close'].rolling(window=period).min().shift(1)) &
               (df['sok'].rolling(window=period).min() < df['sok'].rolling(window=period).min().shift(1)), 'so_t'] = 1
        return df

    def sroc_t(self, df):
        """
        Smoothed Rate of Change Trading Signal

        Generates trading signals based on SROC crossing zero line.
        Buy signal (-1) when SROC is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'sroc' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'sroc_t' signal column
        """
        df = self._prepare_df(df, required_cols=['sroc'])

        df.loc[(df['sroc'] < 0), 'sroc_t'] = 1
        df.loc[(df['sroc'] >= 0), 'sroc_t'] = -1
        return df

    def tmf_t(self, df):
        """
        Twiggs Money Flow Trading Signal

        Generates trading signals based on TMF crossing zero.
        Buy signal (-1) when TMF is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'tmf' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'tmf_t' signal column
        """
        df = self._prepare_df(df, required_cols=['tmf'])

        df.loc[(df['tmf'] < 0), 'tmf_t'] = 1
        df.loc[(df['tmf'] >= 0), 'tmf_t'] = -1
        return df

    def trix_t(self, df):
        """
        TRIX Trading Signal

        Generates trading signals based on TRIX crossing zero.
        Buy signal (-1) when TRIX is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'trix' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'trix_t' signal column
        """
        df = self._prepare_df(df, required_cols=['trix'])

        df.loc[df['trix'] < 0, 'trix_t'] = 1
        df.loc[df['trix'] >= 0, 'trix_t'] = -1
        return df

    def tsi_t(self, df):
        """
        True Strength Index Trading Signal

        Generates trading signals based on TSI crossing zero.
        Buy signal (-1) when TSI is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'tsi' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'tsi_t' signal column
        """
        df = self._prepare_df(df, required_cols=['tsi'])

        df.loc[(df['tsi'] < 0), 'tsi_t'] = 1
        df.loc[(df['tsi'] >= 0), 'tsi_t'] = -1
        return df

    def tti_t(self, df):
        """
        Trend Tracking Index Trading Signal

        Generates trading signals based on TTI crossing zero.
        Buy signal (-1) when TTI is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'tti' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'tti_t' signal column
        """
        df = self._prepare_df(df, required_cols=['tti'])

        df.loc[(df['tti'] < 0), 'tti_t'] = 1
        df.loc[(df['tti'] >= 0), 'tti_t'] = -1
        return df

    def uo_t(self, df, window=3):
        """
        Ultimate Oscillator Trading Signal

        Generates trading signals based on UO level and direction.
        Buy signal (-1) when UO > 50 and rising, Sell signal (1) when UO < 50 and falling.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'uo' column
        window : int, default 3
            Smoothing window for UO

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'uo_t' signal column
        """
        df = self._prepare_df(df, required_cols=['uo'])

        df['uos'] = df['uo'].rolling(window).mean()
        df.loc[(df['uo'] > 50) & (df['uos'].shift(1) < df['uos']), 'uo_t'] = -1
        df.loc[(df['uo'] < 50) & (df['uos'].shift(1) > df['uos']), 'uo_t'] = 1
        return df.drop(['uos'], axis=1)

    def vi_t(self, df):
        """
        Vortex Indicator Trading Signal

        Generates trading signals based on comparing VI+ and VI- lines.
        Buy signal (-1) when VI+ crosses above VI-, Sell signal (1) when VI+ crosses below VI-.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'vip' and 'vin' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'vi_t' signal column
        """
        df = self._prepare_df(df, required_cols=['vip', 'vin'])

        df.loc[(df['vip'] < df['vin']), 'vi_t'] = 1
        df.loc[(df['vip'] >= df['vin']), 'vi_t'] = -1
        return df

    def vpt_t(self, df):
        """
        Volume Price Trend Trading Signal

        Generates trading signals based on VPT signal line crossovers.
        Buy signal (-1) when VPT crosses above signal line, Sell signal (1) when crosses below.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'vpts' and 'vptl' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'vpt_t' signal column
        """
        df = self._prepare_df(df, required_cols=['vpts', 'vptl'])

        df.loc[df['vpts'] < df['vptl'], 'vpt_t'] = 1
        df.loc[df['vpts'] >= df['vptl'], 'vpt_t'] = -1
        return df

    def vroc_t(self, df):
        """
        Volume Rate of Change Trading Signal

        Generates trading signals based on VROC crossing zero.
        Buy signal (-1) when VROC is positive, Sell signal (1) when negative.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'vroc' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'vroc_t' signal column
        """
        df = self._prepare_df(df, required_cols=['vroc'])

        df.loc[(df['vroc'] < 0), 'vroc_t'] = 1
        df.loc[(df['vroc'] >= 0), 'vroc_t'] = -1
        return df

    def vwap_t(self, df):
        """
        Volume Weighted Average Price Trading Signal

        Generates trading signals based on price position relative to VWAP.
        Buy signal (-1) when price is above VWAP, Sell signal (1) when below.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'close' and 'vwap' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'vwap_t' signal column
        """
        df = self._prepare_df(df, required_cols=['close', 'vwap'])

        df.loc[(df['close'] < df['vwap']), 'vwap_t'] = 1
        df.loc[(df['close'] >= df['vwap']), 'vwap_t'] = -1
        return df

    def wr_t(self, df):
        """
        Williams %R Trading Signal

        Generates trading signals based on Williams %R momentum.
        Buy signal (-1) when WR is rising, Sell signal (1) when falling.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'wr' column

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'wr_t' signal column
        """
        df = self._prepare_df(df, required_cols=['wr'])

        df.loc[df['wr'].diff() > 0, 'wr_t'] = -1  # Uptrend
        df.loc[df['wr'].diff() <= 0, 'wr_t'] = 1  # Downtrend
        return df

    def null_t(self, df):
        """
        Null Trading Signal

        Creates a placeholder trading signal column filled with zeros.
        Useful for testing or as a neutral baseline.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (no specific columns required)

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'null_t' signal column filled with zeros
        """
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        df['null_t'] = 0
        return df

    def atr_v(self, df, threshold=1.35):
        """
        ATR Volatility Signal

        Identifies periods of high volatility based on ATR value.
        Signal value 1 indicates ATR above the threshold (high volatility).

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'atr' column
        threshold : float, default 1.35
            ATR threshold for volatility detection

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'atr_v' signal column
        """
        df = self._prepare_df(df, required_cols=['atr'])

        df.loc[df['atr'] > threshold, 'atr_v'] = 1
        return df

    def bb_v(self, df, threshold=5):
        """
        Bollinger Band Volatility Signal

        Identifies periods of high volatility based on Bollinger Band width.
        Signal value 1 indicates BB width above the threshold (high volatility).

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'bbv' column
        threshold : float, default 5
            BB width threshold for volatility detection

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'bb_v' signal column
        """
        df = self._prepare_df(df, required_cols=['bbv'])

        df.loc[(df['bbv'] > threshold), 'bb_v'] = 1
        return df

    def ci1_v(self, df, threshold=50):
        """
        Choppiness Index High Signal

        Identifies choppy market conditions (high volatility, no clear trend).
        Signal value 1 indicates CI above the threshold (choppy market).

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'ci' column
        threshold : float, default 50
            CI threshold for choppy market detection

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'ci1_v' signal column
        """
        df = self._prepare_df(df, required_cols=['ci'])

        df.loc[(df['ci'] > threshold), 'ci1_v'] = 1  # Choppy Market (High Volatility and No trend)
        return df

    def ci2_v(self, df, threshold=50):
        """
        Choppiness Index Low Signal

        Identifies trending market conditions (clear directional movement).
        Signal value 1 indicates CI below the threshold (trending market).

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'ci' column
        threshold : float, default 50
            CI threshold for trending market detection

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'ci2_v' signal column
        """
        df = self._prepare_df(df, required_cols=['ci'])

        df.loc[(df['ci'] < threshold), 'ci2_v'] = 1  # Trendy Market
        return df

    def ui_v(self, df, threshold=0.2):
        """
        Ulcer Index Volatility Signal

        Identifies periods of high downside volatility based on Ulcer Index.
        Signal value 1 indicates UI above the threshold (high downside risk).

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'ui' column
        threshold : float, default 0.2
            UI threshold for volatility detection

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'ui_v' signal column
        """
        df = self._prepare_df(df, required_cols=['ui'])

        df.loc[(df['ui'] > threshold), 'ui_v'] = 1
        return df

    def vhf1_v(self, df, threshold=0.45):
        """
        Vertical Horizontal Filter High Signal

        Identifies trending market conditions based on VHF.
        Signal value 1 indicates VHF above the threshold (trending market).

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'vhf' column
        threshold : float, default 0.45
            VHF threshold for trend detection

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'vhf1_v' signal column
        """
        df = self._prepare_df(df, required_cols=['vhf'])

        df.loc[(df['vhf'] > threshold), 'vhf1_v'] = 1  # Trendy Market
        return df

    def vhf2_v(self, df, threshold=0.3):
        """
        Vertical Horizontal Filter Low Signal

        Identifies ranging market conditions based on VHF.
        Signal value 1 indicates VHF below the threshold (ranging market).

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'vhf' column
        threshold : float, default 0.3
            VHF threshold for range detection

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'vhf2_v' signal column
        """
        df = self._prepare_df(df, required_cols=['vhf'])

        df.loc[(df['vhf'] < threshold), 'vhf2_v'] = 1  # Ranging Market (Fixed Volatility and No trend)
        return df

    def vo_v(self, df, threshold=30):
        """
        Volatility Oscillator Signal

        Identifies periods of high volatility based on Volatility Oscillator.
        Signal value 1 indicates VO above the threshold (high volatility).

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing 'vo' column
        threshold : float, default 30
            VO threshold for volatility detection

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'vo_v' signal column
        """
        df = self._prepare_df(df, required_cols=['vo'])

        df.loc[(df['vo'] > threshold), 'vo_v'] = 1
        return df

    def null_v(self, df):
        """
        Null Volatility Signal

        Creates a placeholder volatility signal column filled with zeros.
        Useful for testing or as a neutral baseline.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (no specific columns required)

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'null_v' signal column filled with zeros
        """
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        df['null_v'] = 0
        return df