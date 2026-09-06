import numpy as np
import pandas as pd


class Tecana:
    """
    Technical analysis library with optimized indicators and signals for financial data.

    Signal Convention
    -----------------
    All signal methods (``_m``, ``_z``, ``_t``, ``_v`` suffixes) return an int8
    column with values from ``{-1, 0, +1}``:

    * **-1** = **BUY** signal (bullish condition detected)
    * **+1** = **SELL** signal (bearish condition detected)
    * **0**  = no signal (neutral / outside trigger zone)

    Volatility signals (``_v`` suffix) are unidirectional flags:

    * **1** = high-volatility regime detected
    * **0** = normal volatility

    .. warning::

       Signals are mathematical pattern detections, **not** trading
       recommendations.  They may contain errors or produce false positives.
       Always validate with your own research before acting on any signal.
    """

    def _prepare_df(self, df: pd.DataFrame, required_cols) -> pd.DataFrame:
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


    @staticmethod
    def _safe_div(numerator, denominator, fill=0.0):
        """Element-wise division returning *fill* where *denominator* is zero.

        Works with both :class:`pd.Series` and :class:`np.ndarray`.

        Parameters
        ----------
        numerator : array-like
            Dividend.
        denominator : array-like
            Divisor.
        fill : float, default 0.0
            Value to substitute where *denominator* == 0.

        Returns
        -------
        pd.Series or np.ndarray
        """
        denom = np.where(np.asarray(denominator) == 0, np.nan, denominator)
        result = numerator / denom
        if isinstance(result, pd.Series):
            return result.fillna(fill)
        return np.nan_to_num(result, nan=fill)

    @staticmethod
    def _wilder_rma(series: pd.Series, window: int) -> pd.Series:
        """Wilder's Running Moving Average (RMA / SMMA).

        Equivalent to ``ewm(alpha=1/window)`` seeded with the first SMA.

        Parameters
        ----------
        series : pd.Series
        window : int

        Returns
        -------
        pd.Series
        """
        return series.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    @staticmethod
    def _sliding_argmax(series: pd.Series, window: int) -> pd.Series:
        """Rolling argmax: 0-based position of the maximum within each window.

        Parameters
        ----------
        series : pd.Series
        window : int

        Returns
        -------
        pd.Series
        """
        return series.rolling(window=window).apply(np.argmax, raw=True)

    @staticmethod
    def _sliding_argmin(series: pd.Series, window: int) -> pd.Series:
        """Rolling argmin: 0-based position of the minimum within each window.

        Parameters
        ----------
        series : pd.Series
        window : int

        Returns
        -------
        pd.Series
        """
        return series.rolling(window=window).apply(np.argmin, raw=True)

    @staticmethod
    def _finalize_signal(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Coerce a signal column to the compact int8 ``{-1, 0, +1}`` encoding.

        Replaces ``NaN`` (no-signal) with ``0`` and casts to ``int8``.

        Parameters
        ----------
        df : pd.DataFrame
        col : str
            Name of the signal column to finalize.

        Returns
        -------
        pd.DataFrame
        """
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(np.int8)
        return df

    def custom(self, df: pd.DataFrame, *args) -> pd.DataFrame:
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

        df = tec.custom(df,
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
                raise TypeError(error_msg) from e

        return result_df

    def adi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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
        high_low_range = df['high'] - df['low']
        mfm = np.where(high_low_range != 0,
                       ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_range,
                       0)

        # Calculate Money Flow Volume
        mfv = mfm * df['volume']

        # Calculate ADI as cumulative sum (not average)
        df['adi'] = pd.Series(mfv).cumsum()

        return df

    def ai(self, df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
        """
        Aroon Indicator

        Identifies trend changes in price by tracking days since highest high and lowest low.
        Consists of two lines - Aroon Up (AIU) and Aroon Down (AID).

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - window (int): Period over which to calculate the Aroon Indicator.

        Returns:
        - DataFrame with added 'aiu' (Aroon Up) and 'aid' (Aroon Down) columns.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        # Create rolling windows for high and low
        high_roll = df['high'].rolling(window=window, min_periods=1)
        low_roll = df['low'].rolling(window=window, min_periods=1)

        # Get indices of max high and min low within each window
        # Note: idxmax and idxmin return the index labels, not the positions
        high_idx = high_roll.apply(lambda x: x.index[-1] - x.idxmax() if len(x) > 0 else 0, raw=False)
        low_idx = low_roll.apply(lambda x: x.index[-1] - x.idxmin() if len(x) > 0 else 0, raw=False)

        # Convert time deltas to number of periods
        high_periods = high_idx / pd.Timedelta('1D')
        low_periods = low_idx / pd.Timedelta('1D')

        # If index is not datetime, we need an alternative approach
        if not isinstance(df.index, pd.DatetimeIndex):
            # Use rolling apply with custom function
            high_idx = high_roll.apply(lambda x: np.argmax(x) if len(x) > 0 else 0, raw=True)
            low_idx = low_roll.apply(lambda x: np.argmin(x) if len(x) > 0 else 0, raw=True)
            high_periods = window - 1 - high_idx
            low_periods = window - 1 - low_idx

        # Calculate Aroon indicators
        df['aiu'] = 100 * (window - high_periods) / window
        df['aid'] = 100 * (window - low_periods) / window

        return df

    def ao(self, df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.DataFrame:
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

    def atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
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

    def awo(self, df: pd.DataFrame, period1: int = 5, period2: int = 34) -> pd.DataFrame:
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

        # Calculate median price
        median_price = (df['high'] + df['low']) / 2

        # Calculate SMAs of median price (not EMAs of close)
        sma1 = median_price.rolling(window=period1).mean()
        sma2 = median_price.rolling(window=period2).mean()

        # Calculate Awesome Oscillator
        df['awo'] = sma1 - sma2

        return df

    def bb(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
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

    def cc(self, df: pd.DataFrame, short_roc_period: int = 11, long_roc_period: int = 14, smoothing_period: int = 10, r: int = 3) -> pd.DataFrame:
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

    def cci(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
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

    def ce(self, df: pd.DataFrame, period: int = 22, multiplier: float = 3) -> pd.DataFrame:
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

    def ci(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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

    def cmf(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
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

    def cmo(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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

    def dc(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
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

    def di(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
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

    def dma(self, df: pd.DataFrame, period: int = 20, displacement: int = 5) -> pd.DataFrame:
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

    def dpo(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
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

    def dx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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

    def ema(self, df: pd.DataFrame, w: int = 20, ws: int = 10) -> pd.DataFrame:
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

    def eom(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def eri(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
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

    def fi(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
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

    def fr(self, df: pd.DataFrame, w1: int = 25) -> pd.DataFrame:
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

    def ha(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def ic(self, df: pd.DataFrame, window1: int = 9, window2: int = 26, window3: int = 52) -> pd.DataFrame:
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

    def kama(self, df: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30) -> pd.DataFrame:
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

        # Calculate Efficiency Ratio (ER)
        change = np.abs(df['close'] - df['close'].shift(period))
        volatility = df['close'].diff().abs().rolling(window=period).sum()

        # Calculate ER with vectorized operations
        er = np.where(volatility != 0, change / volatility, 0)

        # Calculate Smoothing Constant (SC)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = np.square(er * (fast_sc - slow_sc) + slow_sc)

        # Fill NaN values
        sc = np.nan_to_num(sc)
        close_array = df['close'].fillna(method='ffill').values

        # Create arrays for vectorized calculation
        kama_array = np.zeros_like(close_array)
        kama_array[period - 1] = close_array[period - 1]

        # Define the recursive function to be used with np.frompyfunc
        def kama_recurrence(prev, idx):
            i = int(idx)
            return prev + sc[i] * (close_array[i] - prev)

        # Generate indices for accumulation (period to end)
        indices = np.arange(period, len(df))

        # Calculate KAMA recursively using numpy's frompyfunc and accumulate
        vec_kama = np.frompyfunc(kama_recurrence, 2, 1)
        kama_values = vec_kama.accumulate(indices, dtype=object, initial=kama_array[period - 1])

        # Assign calculated values to the output array
        kama_array[period:] = kama_values

        # Assign to DataFrame
        df['kama'] = kama_array

        return df

        return df

    def kc(self, df: pd.DataFrame, window: int = 20, atr_window: int = 10, multiplier: float = 2) -> pd.DataFrame:
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

    def kst(self, df: pd.DataFrame, r1: int = 10, r2: int = 15, r3: int = 20, r4: int = 30, s1: int = 10, s2: int = 10, s3: int = 10, s4: int = 15, sp: int = 9) -> pd.DataFrame:
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

    def lr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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

    def macd(self, df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
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

    def mae(self, df: pd.DataFrame, window: int = 20, percent: float = 5) -> pd.DataFrame:
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

    def mfi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
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
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate raw money flow
        money_flow = typical_price * df['volume']

        # Generate money flow signals based on price changes
        price_diff = typical_price.diff()

        # Create positive and negative money flow series
        positive_flow = np.where(price_diff > 0, money_flow, 0)
        negative_flow = np.where(price_diff < 0, money_flow, 0)

        # Convert to Series for rolling operations
        positive_flow_series = pd.Series(positive_flow, index=df.index)
        negative_flow_series = pd.Series(negative_flow, index=df.index)

        # Calculate sums over periods
        positive_sum = positive_flow_series.rolling(window=window).sum()
        negative_sum = negative_flow_series.rolling(window=window).sum()

        # Calculate MFI with handling for division by zero
        df['mfi'] = np.where(negative_sum != 0,
                             100 - (100 / (1 + positive_sum / negative_sum)),
                             100)

        return df

    def mi(self, df: pd.DataFrame, window: int = 25, s1: int = 9, s2: int = 9, p1: int = 9, p2: int = 9) -> pd.DataFrame:
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

    def mp(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def nvi(self, df: pd.DataFrame, r: int = 14) -> pd.DataFrame:
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

        # Calculate volume change and price percent change
        vol_change = df['volume'].diff()
        price_pct_change = df['close'].pct_change()

        # Create multipliers: 1+price_change for volume decrease days, 1 for others
        nvi_mult = np.where(vol_change < 0, 1 + price_pct_change, 1)

        # Replace first value (NaN) with 1
        nvi_mult = np.nan_to_num(nvi_mult, nan=1.0)

        # Calculate NVI using cumulative product
        df['nvi'] = 1000 * pd.Series(nvi_mult).cumprod()

        # Calculate signal line
        df['nvis'] = df['nvi'].rolling(r).mean()

        return df

    def obv(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def pc(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
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

    def pp(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def ppo(self, df: pd.DataFrame, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
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

    def proc(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
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

    def psar(self, df: pd.DataFrame, step: float = 0.02, max_acceleration: float = 0.2) -> pd.DataFrame:
        """
        Parabolic Stop and Reverse (PSAR)

        A trend-following indicator that places dots above or below price bars to indicate
        potential entry and exit points.

        Parameters:
        - df (DataFrame): Input data containing 'high' and 'low' columns.
        - step (float): Acceleration factor step.
        - max_acceleration (float): Maximum acceleration factor.

        Returns:
        - DataFrame with an added 'psar' column.
        """
        df = self._prepare_df(df, required_cols=['high', 'low'])

        # Extract price data as arrays for faster operations
        high = df['high'].values
        low = df['low'].values
        n = len(df)

        # Pre-allocate output arrays (improves performance)
        psar = np.zeros(n)
        ep = np.zeros(n)  # Extreme Point
        af = np.zeros(n)  # Acceleration Factor
        trend = np.zeros(n, dtype=int)  # 1 for uptrend, -1 for downtrend

        # Set initial values
        trend[0] = 1  # Start with uptrend
        psar[0] = np.min(low[:2]) - 0.01  # Start below first low
        ep[0] = high[0]  # First extreme point
        af[0] = step  # Initial acceleration

        # Core PSAR calculation - this loop is unavoidable due to the nature of PSAR
        for i in range(1, n):
            # Calculate PSAR based on previous values
            psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])

            # Apply trend-based adjustments and check for reversals
            if trend[i - 1] > 0:  # Previous uptrend
                # Adjust PSAR to respect prior lows
                psar[i] = min(psar[i], low[max(0, i - 2)], low[i - 1])

                # Check for trend reversal (price moved below PSAR)
                if low[i] < psar[i]:
                    # Reverse trend direction
                    trend[i] = -1
                    psar[i] = ep[i - 1]  # Reset PSAR to prior extreme
                    ep[i] = low[i]  # Set new extreme point
                    af[i] = step  # Reset acceleration factor
                else:
                    # Continue uptrend
                    trend[i] = 1
                    ep[i] = max(ep[i - 1], high[i])  # Update extreme point if needed
                    # Update acceleration factor if new extreme found
                    af[i] = min(af[i - 1] + step, max_acceleration) if ep[i] > ep[i - 1] else af[i - 1]
            else:  # Previous downtrend
                # Adjust PSAR to respect prior highs
                psar[i] = max(psar[i], high[max(0, i - 2)], high[i - 1])

                # Check for trend reversal (price moved above PSAR)
                if high[i] > psar[i]:
                    # Reverse trend direction
                    trend[i] = 1
                    psar[i] = ep[i - 1]  # Reset PSAR to prior extreme
                    ep[i] = high[i]  # Set new extreme point
                    af[i] = step  # Reset acceleration factor
                else:
                    # Continue downtrend
                    trend[i] = -1
                    ep[i] = min(ep[i - 1], low[i])  # Update extreme point if needed
                    # Update acceleration factor if new extreme found
                    af[i] = min(af[i - 1] + step, max_acceleration) if ep[i] < ep[i - 1] else af[i - 1]

        # Assign calculated PSAR to dataframe
        df['psar'] = psar

        return df

    def psar2(self, df: pd.DataFrame, step: float = 0.02, max_acceleration: float = 0.2) -> pd.DataFrame:
        """
        Parabolic SAR (Stop and Reverse)
        """
        df = self._prepare_df(df, required_cols=['high', 'low', 'close'])

        # For PSAR we need to use a more creative approach with auxiliary columns
        # This is an approximation that uses features of the original algorithm
        # but doesn't require explicit loops

        # Calculate initial values using momentum
        price_diff = df['close'].diff()
        initial_trend = np.where(price_diff.iloc[1] > 0, 1, -1)

        # Calculate ATR to serve as a baseline for PSAR distance
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=14).mean()

        # Create adaptive acceleration factor
        volatility_ratio = atr / atr.rolling(window=30).mean()
        adaptive_af = np.minimum(step * volatility_ratio, max_acceleration)
        adaptive_af = adaptive_af.fillna(step)

        # Calculate basic PSAR levels
        high_psar = df['high'].rolling(window=14).max() - (adaptive_af * atr)
        low_psar = df['low'].rolling(window=14).min() + (adaptive_af * atr)

        # Determine trend using price momentum and moving average crossover
        short_ma = df['close'].rolling(window=5).mean()
        long_ma = df['close'].rolling(window=20).mean()
        trend = np.where(short_ma > long_ma, 1, -1)

        # Calculate PSAR based on trend
        df['psar'] = np.where(trend > 0, low_psar, high_psar)

        return df

    def pvo(self, df: pd.DataFrame, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
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

    def roc(self, df: pd.DataFrame, period: int = 14, r: int = 6) -> pd.DataFrame:
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

    def rsi(self, df: pd.DataFrame, w1: int = 14) -> pd.DataFrame:
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

    def sma(self, df: pd.DataFrame, w: int = 20, ws: int = 10) -> pd.DataFrame:
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

    def so(self, df: pd.DataFrame, kw: int = 14, dw: int = 3) -> pd.DataFrame:
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

    def sroc(self, df: pd.DataFrame, period: int = 14, smoothing: int = 3) -> pd.DataFrame:
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

    def srsi(self, df: pd.DataFrame, w1: int = 14, w2: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
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

    def stc(self, df: pd.DataFrame, p1: int = 23, p2: int = 50, p3: int = 10, f: float = 0.5) -> pd.DataFrame:
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

    def sz(self, df: pd.DataFrame, period: int = 14, multiplier: float = 1) -> pd.DataFrame:
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

    def tmf(self, df: pd.DataFrame, period: int = 21) -> pd.DataFrame:
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

    def tmo(self, df: pd.DataFrame, short_period: int = 10, long_period: int = 21) -> pd.DataFrame:
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

    def tp(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def trix(self, df: pd.DataFrame, w: int = 15, sw: int = 7) -> pd.DataFrame:
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

    def tsi(self, df: pd.DataFrame, short_period: int = 13, long_period: int = 25, tsi_p: int = 5, ema_period: int = 7) -> pd.DataFrame:
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

    def tti(self, df: pd.DataFrame, period: int = 21) -> pd.DataFrame:
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

    def tv(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
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

    def ui(self, df: pd.DataFrame, w1: int = 14, w2: int = 14) -> pd.DataFrame:
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

    def uo(self, df: pd.DataFrame, window1: int = 7, window2: int = 14, window3: int = 28) -> pd.DataFrame:
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

        # Calculate previous close
        prev_close = df['close'].shift(1)

        # Calculate true low (min of current low or previous close)
        true_low = np.minimum(df['low'], prev_close)

        # Calculate buying pressure
        buying_pressure = df['close'] - true_low

        # Calculate true range components
        hl = df['high'] - df['low']
        hpc = np.abs(df['high'] - prev_close)
        lpc = np.abs(df['low'] - prev_close)

        # Get max of the three for true range
        true_range = np.maximum(hl, np.maximum(hpc, lpc))

        # Calculate the three averages using weights
        avg1 = buying_pressure.rolling(window=window1).sum() / true_range.rolling(window=window1).sum()
        avg2 = buying_pressure.rolling(window=window2).sum() / true_range.rolling(window=window2).sum()
        avg3 = buying_pressure.rolling(window=window3).sum() / true_range.rolling(window=window3).sum()

        # Calculate weighted Ultimate Oscillator
        df['uo'] = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)

        return df

    def vhf(self, df: pd.DataFrame, period: int = 28) -> pd.DataFrame:
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

    def vi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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

    def vo(self, df: pd.DataFrame, short_period: int = 14, long_period: int = 28) -> pd.DataFrame:
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

    def vpt(self, df: pd.DataFrame, sp: int = 5, lp: int = 20) -> pd.DataFrame:
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

    def vr(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
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

    def vroc(self, df: pd.DataFrame, short_period: int = 12, long_period: int = 26) -> pd.DataFrame:
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

    def vs(self, df: pd.DataFrame, period: int = 14, multiplier: float = 2) -> pd.DataFrame:
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

    def vwap(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
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

    def wad(self, df: pd.DataFrame, p: int = 14) -> pd.DataFrame:
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

    def wma(self, df: pd.DataFrame, p1: int = 20, p2: int = 9) -> pd.DataFrame:
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

    def wr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
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

    def null(self, df: pd.DataFrame) -> pd.DataFrame:
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


    # ── TA-Lib Expansion: New Indicators ───────────────────────────────

    def dema(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Double Exponential Moving Average (DEMA).

        Reduces lag vs standard EMA: ``2 * EMA - EMA(EMA)``.
        Matches TA-Lib DEMA.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 20
            EMA period.

        Adds columns
        ------------
        ``dema`` : Double Exponential Moving Average of close.
        """
        df = self._prepare_df(df, required_cols=["close"])
        ema1 = df["close"].ewm(span=window, min_periods=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, min_periods=window, adjust=False).mean()
        df["dema"] = 2 * ema1 - ema2
        return df

    def tema(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Triple Exponential Moving Average (TEMA).

        Further reduces lag: ``3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))``.
        Matches TA-Lib TEMA.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 20
            EMA period.

        Adds columns
        ------------
        ``tema`` : Triple Exponential Moving Average of close.
        """
        df = self._prepare_df(df, required_cols=["close"])
        ema1 = df["close"].ewm(span=window, min_periods=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, min_periods=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, min_periods=window, adjust=False).mean()
        df["tema"] = 3 * ema1 - 3 * ema2 + ema3
        return df

    def t3(self, df: pd.DataFrame, window: int = 5, vfactor: float = 0.7) -> pd.DataFrame:
        """Tillson T3 Moving Average.

        A 6th-order smoothed EMA cascade with adjustable volume factor.
        Matches TA-Lib T3 with default vfactor=0.7.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 5
            EMA period for each cascade stage.
        vfactor : float, default 0.7
            Volume factor controlling smoothness vs lag (0 to 1).

        Adds columns
        ------------
        ``t3`` : Tillson T3 moving average of close.
        """
        df = self._prepare_df(df, required_cols=["close"])
        c1 = -(vfactor ** 3)
        c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
        c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
        c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2
        e1 = df["close"].ewm(span=window, min_periods=window, adjust=False).mean()
        e2 = e1.ewm(span=window, min_periods=window, adjust=False).mean()
        e3 = e2.ewm(span=window, min_periods=window, adjust=False).mean()
        e4 = e3.ewm(span=window, min_periods=window, adjust=False).mean()
        e5 = e4.ewm(span=window, min_periods=window, adjust=False).mean()
        e6 = e5.ewm(span=window, min_periods=window, adjust=False).mean()
        df["t3"] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        return df

    def trima(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Triangular Moving Average (TRIMA).

        SMA of SMA, giving more weight to the centre of the window.
        Matches TA-Lib TRIMA.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 20
            Period.

        Adds columns
        ------------
        ``trima`` : Triangular Moving Average of close.
        """
        df = self._prepare_df(df, required_cols=["close"])
        half = (window + 1) // 2
        sma1 = df["close"].rolling(window=half, min_periods=half).mean()
        sma2_period = half if window % 2 == 1 else half + 1
        df["trima"] = sma1.rolling(window=sma2_period, min_periods=sma2_period).mean()
        return df

    def mom(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Momentum (MOM): difference between current price and N bars ago.

        Matches TA-Lib MOM: ``close - close.shift(window)``.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 10
            Look-back period.

        Adds columns
        ------------
        ``mom`` : Price momentum (absolute difference).
        """
        df = self._prepare_df(df, required_cols=["close"])
        df["mom"] = df["close"] - df["close"].shift(window)
        return df

    def apo(self, df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
        """Absolute Price Oscillator (APO).

        Absolute difference between fast and slow EMAs.
        Matches TA-Lib APO: ``EMA(fast) - EMA(slow)``.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        fast : int, default 12
            Fast EMA period.
        slow : int, default 26
            Slow EMA period.

        Adds columns
        ------------
        ``apo`` : Absolute Price Oscillator.
        """
        df = self._prepare_df(df, required_cols=["close"])
        ema_fast = df["close"].ewm(span=fast, min_periods=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, min_periods=slow, adjust=False).mean()
        df["apo"] = ema_fast - ema_slow
        return df

    def bop(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance of Power (BOP).

        Measures buyer vs seller strength: ``(close - open) / (high - low)``.
        Matches TA-Lib BOP.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``open``, ``high``, ``low``, ``close``.

        Adds columns
        ------------
        ``bop`` : Balance of Power in [-1, +1].
        """
        df = self._prepare_df(df, required_cols=["open", "high", "low", "close"])
        df["bop"] = self._safe_div(df["close"] - df["open"], df["high"] - df["low"], fill=0.0)
        return df

    def adosc(self, df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.DataFrame:
        """Chaikin A/D Oscillator (ADOSC).

        EMA(fast) - EMA(slow) of the Accumulation/Distribution line.
        Matches TA-Lib ADOSC.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``, ``high``, ``low``, ``volume``.
        fast : int, default 3
            Fast EMA period.
        slow : int, default 10
            Slow EMA period.

        Adds columns
        ------------
        ``adosc`` : Chaikin A/D Oscillator.
        """
        df = self._prepare_df(df, required_cols=["close", "high", "low", "volume"])
        hl_range = df["high"] - df["low"]
        mfm = self._safe_div(
            (df["close"] - df["low"]) - (df["high"] - df["close"]),
            hl_range,
            fill=0.0,
        )
        mfv = mfm * df["volume"]
        ad_line = pd.Series(np.asarray(mfv), index=df.index).cumsum()
        ema_fast = ad_line.ewm(span=fast, min_periods=fast, adjust=False).mean()
        ema_slow = ad_line.ewm(span=slow, min_periods=slow, adjust=False).mean()
        df["adosc"] = ema_fast - ema_slow
        return df

    def natr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Normalized Average True Range (NATR).

        ATR expressed as a percentage of close: ``100 * ATR / close``.
        Matches TA-Lib NATR.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``high``, ``low``, ``close``.
        window : int, default 14
            ATR period.

        Adds columns
        ------------
        ``natr`` : Normalized ATR as percentage.
        """
        df = self._prepare_df(df, required_cols=["high", "low", "close"])
        prev_close = df["close"].shift(1)
        true_range = np.maximum.reduce([
            (df["high"] - df["low"]).to_numpy(dtype=float),
            (df["high"] - prev_close).abs().to_numpy(dtype=float),
            (df["low"] - prev_close).abs().to_numpy(dtype=float),
        ])
        tr = pd.Series(true_range, index=df.index)
        atr_val = self._wilder_rma(tr, window)
        df["natr"] = self._safe_div(atr_val, df["close"]) * 100
        return df

    def trange(self, df: pd.DataFrame) -> pd.DataFrame:
        """True Range (TRANGE): single-bar volatility measure.

        Matches TA-Lib TRANGE: ``max(H-L, |H-Cp|, |L-Cp|)``.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``high``, ``low``, ``close``.

        Adds columns
        ------------
        ``trange`` : True Range per bar.
        """
        df = self._prepare_df(df, required_cols=["high", "low", "close"])
        prev_close = df["close"].shift(1)
        df["trange"] = np.maximum.reduce([
            (df["high"] - df["low"]).to_numpy(dtype=float),
            (df["high"] - prev_close).abs().to_numpy(dtype=float),
            (df["low"] - prev_close).abs().to_numpy(dtype=float),
        ])
        return df

    def sof(self, df: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.DataFrame:
        """Fast Stochastic Oscillator (STOCHF).

        Raw %K (no %K smoothing) with %D = SMA(%K, smooth).
        Matches TA-Lib STOCHF.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``high``, ``low``, ``close``.
        window : int, default 14
            Look-back for highest-high and lowest-low.
        smooth : int, default 3
            SMA period for %D line.

        Adds columns
        ------------
        ``sofk`` : Fast %K (raw).
        ``sofd`` : Fast %D (SMA of %K).
        """
        df = self._prepare_df(df, required_cols=["high", "low", "close"])
        lowest = df["low"].rolling(window=window).min()
        highest = df["high"].rolling(window=window).max()
        df["sofk"] = self._safe_div(df["close"] - lowest, highest - lowest) * 100
        df["sofd"] = df["sofk"].rolling(window=smooth).mean()
        return df

    def aiosc(self, df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
        """Aroon Oscillator: Aroon Up minus Aroon Down.

        Matches TA-Lib AROONOSC.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``high``, ``low``.
        window : int, default 25
            Aroon look-back period.

        Adds columns
        ------------
        ``aiosc`` : Aroon Oscillator in [-100, +100].
        """
        df = self._prepare_df(df, required_cols=["high", "low"])
        up_offset = self._sliding_argmax(df["high"], window)
        down_offset = self._sliding_argmin(df["low"], window)
        periods_since_high = (window - 1) - up_offset
        periods_since_low = (window - 1) - down_offset
        aiu = 100.0 * (window - periods_since_high) / window
        aid = 100.0 * (window - periods_since_low) / window
        df["aiosc"] = aiu - aid
        return df

    def adxr(self, df: pd.DataFrame, window: int = 14, rating_period: int = 14) -> pd.DataFrame:
        """Average Directional Movement Index Rating (ADXR).

        Smoothed ADX: ``(ADX + ADX.shift(rating_period)) / 2``.
        Matches TA-Lib ADXR.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``high``, ``low``, ``close``.
        window : int, default 14
            ADX calculation period.
        rating_period : int, default 14
            Look-back for the rating average.

        Adds columns
        ------------
        ``adxr`` : ADX Rating.
        """
        df = self.dx(df, period=window)
        df["adxr"] = (df["adx"] + df["adx"].shift(rating_period)) / 2
        return df

    def imi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Intraday Momentum Index (IMI).

        Measures buying pressure via up-close candle ranges to total ranges.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``open``, ``close``.
        window : int, default 14
            Rolling window.

        Adds columns
        ------------
        ``imi`` : Intraday Momentum Index in [0, 100].
        """
        df = self._prepare_df(df, required_cols=["open", "close"])
        gains = (df["close"] - df["open"]).clip(lower=0)
        losses = (df["open"] - df["close"]).clip(lower=0)
        sum_gains = gains.rolling(window=window, min_periods=window).sum()
        sum_losses = losses.rolling(window=window, min_periods=window).sum()
        df["imi"] = self._safe_div(sum_gains, sum_gains + sum_losses) * 100
        return df

    def accb(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Acceleration Bands (ACCBANDS).

        Upper and lower bands based on high-low range scaled by a factor.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``high``, ``low``, ``close``.
        window : int, default 20
            SMA period.

        Adds columns
        ------------
        ``accbu`` : Upper acceleration band.
        ``accbl`` : Lower acceleration band.
        ``accbm`` : Middle band (SMA of close).
        """
        df = self._prepare_df(df, required_cols=["high", "low", "close"])
        factor = self._safe_div(df["high"] - df["low"], (df["high"] + df["low"]) / 2, fill=0.0)
        upper = df["high"] * (1 + 2 * factor)
        lower = df["low"] * (1 - 2 * factor)
        df["accbu"] = upper.rolling(window=window, min_periods=window).mean()
        df["accbl"] = lower.rolling(window=window, min_periods=window).mean()
        df["accbm"] = df["close"].rolling(window=window, min_periods=window).mean()
        return df

    def midpt(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """MidPoint over period.

        Matches TA-Lib MIDPOINT: ``(highest(close, n) + lowest(close, n)) / 2``.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 14
            Look-back period.

        Adds columns
        ------------
        ``midpt`` : Midpoint of close over window.
        """
        df = self._prepare_df(df, required_cols=["close"])
        df["midpt"] = (
            df["close"].rolling(window=window).max()
            + df["close"].rolling(window=window).min()
        ) / 2
        return df

    def midpr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Midpoint Price over period.

        Matches TA-Lib MIDPRICE: ``(rolling_max(high) + rolling_min(low)) / 2``.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``high``, ``low``.
        window : int, default 14
            Look-back period.

        Adds columns
        ------------
        ``midpr`` : Midpoint price over window.
        """
        df = self._prepare_df(df, required_cols=["high", "low"])
        df["midpr"] = (
            df["high"].rolling(window=window).max()
            + df["low"].rolling(window=window).min()
        ) / 2
        return df

    def lrs(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Linear Regression Slope.

        Matches TA-Lib LINEARREG_SLOPE.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 14
            Regression period.

        Adds columns
        ------------
        ``lrs`` : Slope of linear regression over window.
        """
        df = self._prepare_df(df, required_cols=["close"])
        close_arr = df["close"].to_numpy(dtype=float)
        n = len(close_arr)
        slopes = np.full(n, np.nan, dtype=float)
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        if n >= window:
            windows = np.lib.stride_tricks.sliding_window_view(close_arr, window)
            y_means = windows.mean(axis=1)
            slopes[window - 1:] = (
                (windows - y_means[:, None]) * (x - x_mean)
            ).sum(axis=1) / x_var
        df["lrs"] = slopes
        return df

    def rocp(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Rate of Change Percentage (fractional).

        Matches TA-Lib ROCP: ``(close - close.shift(n)) / close.shift(n)``.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 10
            Look-back period.

        Adds columns
        ------------
        ``rocp`` : Fractional rate of change.
        """
        df = self._prepare_df(df, required_cols=["close"])
        prev = df["close"].shift(window)
        df["rocp"] = self._safe_div(df["close"] - prev, prev)
        return df

    def stdv(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Rolling Standard Deviation.

        Matches TA-Lib STDDEV.

        Parameters
        ----------
        df : pd.DataFrame
            Requires ``close``.
        window : int, default 20
            Rolling window.

        Adds columns
        ------------
        ``stdv`` : Rolling standard deviation of close.
        """
        df = self._prepare_df(df, required_cols=["close"])
        df["stdv"] = df["close"].rolling(window=window, min_periods=window).std()
        return df

    '''
    ---------------------------- Signals --------------------------------
    '''

    # Momentum Signals

    def adi_m(self, df: pd.DataFrame, r1: int = 7, r2: int = 21) -> pd.DataFrame:
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

    def ai_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def atr_m(self, df: pd.DataFrame, d: int = 2) -> pd.DataFrame:
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

    def awo_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def cc_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def cci_m(self, df: pd.DataFrame, bz: float = 100, sz: float = 100) -> pd.DataFrame:
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

    def dc_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def di_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def dpo_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def ema_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def eom_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def fr_m(self, df: pd.DataFrame, p: int = 14) -> pd.DataFrame:
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

    def ha_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def ic_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def kama_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def kc_m(self, df: pd.DataFrame, w: int = 14, q: float = 1.35) -> pd.DataFrame:
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

    def kst_m(self, df: pd.DataFrame, bz: float = 50, sz: float = 50) -> pd.DataFrame:
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

    def macd_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def mae_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def obv_m(self, df: pd.DataFrame, r1: int = 7, r2: int = 21) -> pd.DataFrame:
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

    def pp_m(self, df: pd.DataFrame, p: int = 14) -> pd.DataFrame:
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

    def ppo_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def proc_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def psar_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def pvo_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def roc_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def rsi_m(self, df: pd.DataFrame, bz: float = 30, sz: float = 70) -> pd.DataFrame:
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

    def sma_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def so_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def srsi_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def stc_m(self, df: pd.DataFrame, bz: float = 25, sz: float = 75) -> pd.DataFrame:
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

    def tmo_m(self, df: pd.DataFrame, bz: float = 20, sz: float = 20) -> pd.DataFrame:
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

    def trix_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def tsi_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def uo_m(self, df: pd.DataFrame, r1: int = 3) -> pd.DataFrame:
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

    def vi_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def vpt_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def vs_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def wma_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def null_m(self, df: pd.DataFrame) -> pd.DataFrame:
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

    # ── TA-Lib Expansion: Momentum Signals ─────────────────────────────

    def dema_m(self, df: pd.DataFrame, fast: int = 10, slow: int = 20) -> pd.DataFrame:
        """DEMA momentum signal from fast/slow DEMA crossovers.

        Adds columns
        ------------
        ``dema_m`` : int8 signal. Signal: -1 (buy) on bullish crossover, +1 (sell) on bearish crossover.
        """
        df = self._prepare_df(df, required_cols=["close"])
        dema_f = df["close"].ewm(span=fast, min_periods=fast, adjust=False).mean()
        ef2 = dema_f.ewm(span=fast, min_periods=fast, adjust=False).mean()
        fast_dema = 2 * dema_f - ef2
        dema_s = df["close"].ewm(span=slow, min_periods=slow, adjust=False).mean()
        es2 = dema_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
        slow_dema = 2 * dema_s - es2
        prev_fast = fast_dema.shift(1)
        prev_slow = slow_dema.shift(1)
        df.loc[(prev_fast <= prev_slow) & (fast_dema > slow_dema), "dema_m"] = -1
        df.loc[(prev_fast >= prev_slow) & (fast_dema < slow_dema), "dema_m"] = 1
        return self._finalize_signal(df, "dema_m")

    def tema_m(self, df: pd.DataFrame, fast: int = 10, slow: int = 20) -> pd.DataFrame:
        """TEMA momentum signal from fast/slow TEMA crossovers.

        Adds columns
        ------------
        ``tema_m`` : int8 signal. Signal: -1 (buy) on bullish crossover, +1 (sell) on bearish crossover.
        """
        df = self._prepare_df(df, required_cols=["close"])
        def _tema(series, w):
            e1 = series.ewm(span=w, min_periods=w, adjust=False).mean()
            e2 = e1.ewm(span=w, min_periods=w, adjust=False).mean()
            e3 = e2.ewm(span=w, min_periods=w, adjust=False).mean()
            return 3 * e1 - 3 * e2 + e3
        fast_tema = _tema(df["close"], fast)
        slow_tema = _tema(df["close"], slow)
        prev_f = fast_tema.shift(1)
        prev_s = slow_tema.shift(1)
        df.loc[(prev_f <= prev_s) & (fast_tema > slow_tema), "tema_m"] = -1
        df.loc[(prev_f >= prev_s) & (fast_tema < slow_tema), "tema_m"] = 1
        return self._finalize_signal(df, "tema_m")

    def t3_m(self, df: pd.DataFrame) -> pd.DataFrame:
        """T3 momentum signal from slope direction change.

        Adds columns
        ------------
        ``t3_m`` : int8 signal. Signal: -1 (buy) when slope turns positive, +1 (sell) when slope turns negative.
        """
        df = self.t3(df)
        slope = df["t3"] - df["t3"].shift(1)
        prev_slope = slope.shift(1)
        df.loc[(prev_slope <= 0) & (slope > 0), "t3_m"] = -1
        df.loc[(prev_slope >= 0) & (slope < 0), "t3_m"] = 1
        return self._finalize_signal(df, "t3_m")

    def trima_m(self, df: pd.DataFrame, fast: int = 10, slow: int = 20) -> pd.DataFrame:
        """TRIMA momentum signal from fast/slow TRIMA crossovers.

        Adds columns
        ------------
        ``trima_m`` : int8 signal. Signal: -1 (buy) on bullish crossover, +1 (sell) on bearish crossover.
        """
        df = self._prepare_df(df, required_cols=["close"])
        half_f = (fast + 1) // 2
        sma1_f = df["close"].rolling(window=half_f).mean()
        fast_trima = sma1_f.rolling(window=half_f).mean()
        half_s = (slow + 1) // 2
        sma1_s = df["close"].rolling(window=half_s).mean()
        slow_trima = sma1_s.rolling(window=half_s).mean()
        prev_f = fast_trima.shift(1)
        prev_s = slow_trima.shift(1)
        df.loc[(prev_f <= prev_s) & (fast_trima > slow_trima), "trima_m"] = -1
        df.loc[(prev_f >= prev_s) & (fast_trima < slow_trima), "trima_m"] = 1
        return self._finalize_signal(df, "trima_m")

    def mom_m(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Momentum zero-cross signal.

        Adds columns
        ------------
        ``mom_m`` : int8 signal. Signal: -1 (buy) when momentum crosses above zero, +1 (sell) when below.
        """
        df = self.mom(df, window=window)
        prev = df["mom"].shift(1)
        df.loc[(prev <= 0) & (df["mom"] > 0), "mom_m"] = -1
        df.loc[(prev >= 0) & (df["mom"] < 0), "mom_m"] = 1
        return self._finalize_signal(df, "mom_m")

    def apo_m(self, df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
        """APO zero-cross signal.

        Adds columns
        ------------
        ``apo_m`` : int8 signal. Signal: -1 (buy) when APO crosses above zero, +1 (sell) when below.
        """
        df = self.apo(df, fast=fast, slow=slow)
        prev = df["apo"].shift(1)
        df.loc[(prev <= 0) & (df["apo"] > 0), "apo_m"] = -1
        df.loc[(prev >= 0) & (df["apo"] < 0), "apo_m"] = 1
        return self._finalize_signal(df, "apo_m")

    def bop_m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance of Power zero-cross signal.

        Adds columns
        ------------
        ``bop_m`` : int8 signal. Signal: -1 (buy) when BOP crosses above zero, +1 (sell) when below.
        """
        df = self.bop(df)
        prev = df["bop"].shift(1)
        df.loc[(prev <= 0) & (df["bop"] > 0), "bop_m"] = -1
        df.loc[(prev >= 0) & (df["bop"] < 0), "bop_m"] = 1
        return self._finalize_signal(df, "bop_m")

    def adosc_m(self, df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.DataFrame:
        """Chaikin A/D Oscillator zero-cross signal.

        Adds columns
        ------------
        ``adosc_m`` : int8 signal. Signal: -1 (buy) when ADOSC crosses above zero, +1 (sell) when below.
        """
        df = self.adosc(df, fast=fast, slow=slow)
        prev = df["adosc"].shift(1)
        df.loc[(prev <= 0) & (df["adosc"] > 0), "adosc_m"] = -1
        df.loc[(prev >= 0) & (df["adosc"] < 0), "adosc_m"] = 1
        return self._finalize_signal(df, "adosc_m")

    def sof_m(self, df: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.DataFrame:
        """Fast Stochastic %K/%D crossover signal.

        Adds columns
        ------------
        ``sof_m`` : int8 signal. Signal: -1 (buy) when %%K crosses above %%D, +1 (sell) when below.
        """
        df = self.sof(df, window=window, smooth=smooth)
        prev_k = df["sofk"].shift(1)
        prev_d = df["sofd"].shift(1)
        df.loc[(prev_k <= prev_d) & (df["sofk"] > df["sofd"]), "sof_m"] = -1
        df.loc[(prev_k >= prev_d) & (df["sofk"] < df["sofd"]), "sof_m"] = 1
        return self._finalize_signal(df, "sof_m")

    def aiosc_m(self, df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
        """Aroon Oscillator zero-cross signal.

        Adds columns
        ------------
        ``aiosc_m`` : int8 signal. Signal: -1 (buy) when oscillator crosses above zero, +1 (sell) when below.
        """
        df = self.aiosc(df, window=window)
        prev = df["aiosc"].shift(1)
        df.loc[(prev <= 0) & (df["aiosc"] > 0), "aiosc_m"] = -1
        df.loc[(prev >= 0) & (df["aiosc"] < 0), "aiosc_m"] = 1
        return self._finalize_signal(df, "aiosc_m")

    def lrs_m(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Linear Regression Slope sign-change signal.

        Adds columns
        ------------
        ``lrs_m`` : int8 signal. Signal: -1 (buy) when slope turns positive, +1 (sell) when negative.
        """
        df = self.lrs(df, window=window)
        prev = df["lrs"].shift(1)
        df.loc[(prev <= 0) & (df["lrs"] > 0), "lrs_m"] = -1
        df.loc[(prev >= 0) & (df["lrs"] < 0), "lrs_m"] = 1
        return self._finalize_signal(df, "lrs_m")

    def rocp_m(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """ROCP zero-cross signal.

        Adds columns
        ------------
        ``rocp_m`` : int8 signal. Signal: -1 (buy) when ROCP crosses above zero, +1 (sell) when below.
        """
        df = self.rocp(df, window=window)
        prev = df["rocp"].shift(1)
        df.loc[(prev <= 0) & (df["rocp"] > 0), "rocp_m"] = -1
        df.loc[(prev >= 0) & (df["rocp"] < 0), "rocp_m"] = 1
        return self._finalize_signal(df, "rocp_m")

    # Zone Signals

    def bb_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def cc_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def cci_z(self, df: pd.DataFrame, bz: float = 100, sz: float = 100) -> pd.DataFrame:
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

    def dc_z(self, df: pd.DataFrame, m: int = 2) -> pd.DataFrame:
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

    def di_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def dpo_z(self, df: pd.DataFrame, qs: float = 0.85, qb: float = -0.85) -> pd.DataFrame:
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

    def dx_z(self, df: pd.DataFrame, q: int = 35) -> pd.DataFrame:
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

    def ema_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def eom_z(self, df: pd.DataFrame, qb: float = -13000, qs: float = 13000) -> pd.DataFrame:
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

    def fi_z(self, df: pd.DataFrame, qb: float = -1500, qs: float = 1500) -> pd.DataFrame:
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

    def fr_z(self, df: pd.DataFrame, p: int = 14) -> pd.DataFrame:
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

    def ic_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def kama_z(self, df: pd.DataFrame, qb: float = 500, qs: float = 3500) -> pd.DataFrame:
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

    def kc_z(self, df: pd.DataFrame, bz: float = 0.5, sz: float = 0.5) -> pd.DataFrame:
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

    def mae_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def mfi_z(self, df: pd.DataFrame, bz: float = 20, sz: float = 80) -> pd.DataFrame:
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

    def pp_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def proc_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def roc_z(self, df: pd.DataFrame, qb: float = -0.2, qs: float = 0.2) -> pd.DataFrame:
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

    def rsi_z(self, df: pd.DataFrame, bz: float = 30, sz: float = 70) -> pd.DataFrame:
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

    def sma_z(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple Moving Average Zone Signal

        Identifies price zones relative to the SMA.
        Signal: 1 (sell zone) when price > SMA,
        -1 (buy zone) when price < SMA.
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

    def so_z(self, df: pd.DataFrame, bz: float = 30, sz: float = 70) -> pd.DataFrame:
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

    def srsi_z(self, df: pd.DataFrame, bz: float = 20, sz: float = 80) -> pd.DataFrame:
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

    def stc_z(self, df: pd.DataFrame, bz: float = 25, sz: float = 75) -> pd.DataFrame:
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

    def sz_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def tmo_z(self, df: pd.DataFrame, bz: float = 20, sz: float = 20) -> pd.DataFrame:
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

    def tsi_z(self, df: pd.DataFrame, qb: float = 6, qs: float = 190) -> pd.DataFrame:
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

    def tv_z(self, df: pd.DataFrame, qb: float = 24, qs: float = 4) -> pd.DataFrame:
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

    def uo_z(self, df: pd.DataFrame, bz: float = 30, sz: float = 70) -> pd.DataFrame:
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

    def vs_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def wad_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def wma_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def wr_z(self, df: pd.DataFrame, bz: float = 80, sz: float = 20) -> pd.DataFrame:
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

    def null_z(self, df: pd.DataFrame) -> pd.DataFrame:
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

    # ── TA-Lib Expansion: Zone Signals ─────────────────────────────────

    def mom_z(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Momentum zone signal (sign of momentum).

        Adds columns
        ------------
        ``mom_z`` : int8 signal. Signal: -1 (buy zone) when momentum > 0, +1 (sell zone) when < 0.
        """
        df = self.mom(df, window=window)
        df.loc[df["mom"] > 0, "mom_z"] = -1
        df.loc[df["mom"] < 0, "mom_z"] = 1
        return self._finalize_signal(df, "mom_z")

    def bop_z(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Balance of Power zone signal (strong buy/sell pressure).

        Adds columns
        ------------
        ``bop_z`` : int8 signal. Signal: -1 (buy zone) above threshold, +1 (sell zone) below -threshold.
        """
        df = self.bop(df)
        df.loc[df["bop"] > threshold, "bop_z"] = -1
        df.loc[df["bop"] < -threshold, "bop_z"] = 1
        return self._finalize_signal(df, "bop_z")

    def sof_z(self, df: pd.DataFrame, window: int = 14, bz: float = 20, sz: float = 80) -> pd.DataFrame:
        """Fast Stochastic zone signal (oversold/overbought).

        Adds columns
        ------------
        ``sof_z`` : int8 signal. Signal: -1 (buy) exiting oversold, +1 (sell) exiting overbought.
        """
        df = self.sof(df, window=window)
        prev = df["sofk"].shift(1)
        df.loc[(prev <= bz) & (df["sofk"] > bz), "sof_z"] = -1
        df.loc[(prev >= sz) & (df["sofk"] < sz), "sof_z"] = 1
        return self._finalize_signal(df, "sof_z")

    def imi_z(self, df: pd.DataFrame, window: int = 14, bz: float = 30, sz: float = 70) -> pd.DataFrame:
        """IMI zone signal (oversold/overbought).

        Adds columns
        ------------
        ``imi_z`` : int8 signal. Signal: -1 (buy) exiting oversold, +1 (sell) exiting overbought.
        """
        df = self.imi(df, window=window)
        prev = df["imi"].shift(1)
        df.loc[(prev <= bz) & (df["imi"] > bz), "imi_z"] = -1
        df.loc[(prev >= sz) & (df["imi"] < sz), "imi_z"] = 1
        return self._finalize_signal(df, "imi_z")

    def accb_z(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Acceleration Bands breakout signal.

        Adds columns
        ------------
        ``accb_z`` : int8 signal. Signal: -1 (buy) on upper band breakout, +1 (sell) on lower band breakout.
        """
        df = self.accb(df, window=window)
        df.loc[df["close"] > df["accbu"], "accb_z"] = -1
        df.loc[df["close"] < df["accbl"], "accb_z"] = 1
        return self._finalize_signal(df, "accb_z")

    # Trend Signals

    def adi_t(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
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

    def ai_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def atr_t(self, df: pd.DataFrame, divisor: int = 3) -> pd.DataFrame:
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

    def awo_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def bb_t(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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
        df = self.dx(df, period)
        df.loc[(df['close'] >= df['bbh']) & (df['dip'] > df['din']), 'bb_t'] = 1
        df.loc[(df['close'] <= df['bbl']) & (df['dip'] < df['din']), 'bb_t'] = -1
        if initial_cols + 1 < len(df.columns):
            df = df.drop(['adx', 'dip', 'din'], axis=1)
        return df

    def cmf_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def cmo_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def dc_t(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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
        df = self.dx(df, period)
        df.loc[(df['close'] > df['dcm']) & (df['dip'] > df['din']), 'dc_t'] = -1
        df.loc[(df['close'] < df['dcm']) & (df['dip'] < df['din']), 'dc_t'] = 1
        if initial_cols + 1 < len(df.columns):
            df = df.drop(['adx', 'dip', 'din'], axis=1)
        return df

    def dma_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def dx_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def eom_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def eri_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def fi_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def ha_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def ic_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def kama_t(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
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

    def kc_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def kst_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def lr_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def macd_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def mfi_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def nvi_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def obv_t(self, df: pd.DataFrame, r1: int = 7, r2: int = 21) -> pd.DataFrame:
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
            df = self.adi(df)
            add_adi = True

        df['obvmacd'] = df['obv'].ewm(span=r2, adjust=False).mean() - df['adi'].ewm(span=r1, adjust=False).mean()
        df['obvmacds'] = df['obvmacd'].rolling(r2).mean()
        df.loc[df['obvmacd'] < df['obvmacds'], 'obv_t'] = 1
        df.loc[df['obvmacd'] > df['obvmacds'], 'obv_t'] = -1

        if add_adi:
            df = df.drop(['adi'], axis=1)
        return df.drop(['obvmacd', 'obvmacds'], axis=1)

    def pc_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def ppo_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def psar_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def pvo_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def roc_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def so_t(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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

    def sroc_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def tmf_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def trix_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def tsi_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def tti_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def uo_t(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
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

    def vi_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def vpt_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def vroc_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def vwap_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def wr_t(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def null_t(self, df: pd.DataFrame) -> pd.DataFrame:
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


    # ── TA-Lib Expansion: Trend Signals ────────────────────────────────

    def dema_t(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """DEMA trend signal: price above/below DEMA.

        Adds columns
        ------------
        ``dema_t`` : int8 signal. Signal: -1 (buy) when price above DEMA, +1 (sell) when below.
        """
        df = self.dema(df, window=window)
        df.loc[df["close"] > df["dema"], "dema_t"] = -1
        df.loc[df["close"] < df["dema"], "dema_t"] = 1
        return self._finalize_signal(df, "dema_t")

    def tema_t(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """TEMA trend signal: price above/below TEMA.

        Adds columns
        ------------
        ``tema_t`` : int8 signal. Signal: -1 (buy) when price above TEMA, +1 (sell) when below.
        """
        df = self.tema(df, window=window)
        df.loc[df["close"] > df["tema"], "tema_t"] = -1
        df.loc[df["close"] < df["tema"], "tema_t"] = 1
        return self._finalize_signal(df, "tema_t")

    def t3_t(self, df: pd.DataFrame) -> pd.DataFrame:
        """T3 trend signal: price above/below T3.

        Adds columns
        ------------
        ``t3_t`` : int8 signal. Signal: -1 (buy) when price above T3, +1 (sell) when below.
        """
        df = self.t3(df)
        df.loc[df["close"] > df["t3"], "t3_t"] = -1
        df.loc[df["close"] < df["t3"], "t3_t"] = 1
        return self._finalize_signal(df, "t3_t")

    def trima_t(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """TRIMA trend signal: price above/below TRIMA.

        Adds columns
        ------------
        ``trima_t`` : int8 signal. Signal: -1 (buy) when price above TRIMA, +1 (sell) when below.
        """
        df = self.trima(df, window=window)
        df.loc[df["close"] > df["trima"], "trima_t"] = -1
        df.loc[df["close"] < df["trima"], "trima_t"] = 1
        return self._finalize_signal(df, "trima_t")

    def midpt_t(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Midpoint trend signal.

        Adds columns
        ------------
        ``midpt_t`` : int8 signal. Signal: -1 (buy) when price above midpoint, +1 (sell) when below.
        """
        df = self.midpt(df, window=window)
        df.loc[df["close"] > df["midpt"], "midpt_t"] = -1
        df.loc[df["close"] < df["midpt"], "midpt_t"] = 1
        return self._finalize_signal(df, "midpt_t")

    def midpr_t(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Midprice trend signal.

        Adds columns
        ------------
        ``midpr_t`` : int8 signal. Signal: -1 (buy) when price above midprice, +1 (sell) when below.
        """
        df = self.midpr(df, window=window)
        df.loc[df["close"] > df["midpr"], "midpr_t"] = -1
        df.loc[df["close"] < df["midpr"], "midpr_t"] = 1
        return self._finalize_signal(df, "midpr_t")

    def adxr_t(self, df: pd.DataFrame, window: int = 14, threshold: float = 25) -> pd.DataFrame:
        """ADXR trend strength signal (trending when above threshold).

        Adds columns
        ------------
        ``adxr_t`` : int8 signal. Signal: -1 (buy/trending) when ADXR above threshold, +1 (sell/ranging) when below.
        """
        df = self.adxr(df, window=window)
        df.loc[df["adxr"] > threshold, "adxr_t"] = -1
        df.loc[df["adxr"] <= threshold, "adxr_t"] = 1
        return self._finalize_signal(df, "adxr_t")

    def atr_v(self, df: pd.DataFrame, threshold: float = 1.35) -> pd.DataFrame:
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

    def bb_v(self, df: pd.DataFrame, threshold: float = 5) -> pd.DataFrame:
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

    def ci1_v(self, df: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
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

    def ci2_v(self, df: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
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

    def ui_v(self, df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
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

    def vhf1_v(self, df: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
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

    def vhf2_v(self, df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
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

    def vo_v(self, df: pd.DataFrame, threshold: float = 30) -> pd.DataFrame:
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

    def null_v(self, df: pd.DataFrame) -> pd.DataFrame:
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

    # ── TA-Lib Expansion: Volatility Signals ───────────────────────────

    def natr_v(self, df: pd.DataFrame, window: int = 14, threshold: float = 1.5) -> pd.DataFrame:
        """NATR volatility expansion flag.

        Adds columns
        ------------
        ``natr_v`` : int8 flag {0, 1}. Flag: 1 when NATR exceeds 1.5x rolling median (high volatility), 0 otherwise.
        """
        df = self.natr(df, window=window)
        median_natr = df["natr"].rolling(window=50, min_periods=10).median()
        df.loc[df["natr"] > median_natr * threshold, "natr_v"] = 1
        return self._finalize_signal(df, "natr_v")

    def accb_v(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Acceleration Bands width volatility flag.

        Adds columns
        ------------
        ``accb_v`` : int8 flag {0, 1}. Flag: 1 when band width exceeds 1.5x rolling median (high volatility), 0 otherwise.
        """
        df = self.accb(df, window=window)
        width = self._safe_div(df["accbu"] - df["accbl"], df["accbm"]) * 100
        median_w = width.rolling(window=50, min_periods=10).median()
        df.loc[width > median_w * 1.5, "accb_v"] = 1
        return self._finalize_signal(df, "accb_v")

    def stdv_v(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Standard deviation volatility expansion flag.

        Adds columns
        ------------
        ``stdv_v`` : int8 flag {0, 1}. Flag: 1 when std dev exceeds 1.5x rolling median (high volatility), 0 otherwise.
        """
        df = self.stdv(df, window=window)
        median_std = df["stdv"].rolling(window=50, min_periods=10).median()
        df.loc[df["stdv"] > median_std * 1.5, "stdv_v"] = 1
        return self._finalize_signal(df, "stdv_v")

