import pandas as pd
import numpy as np
import sys
from src.mlproject.logger import logging
from src.exception import CustomException

output_data_path = r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\un_processed_data\stock_price_with_indicators.csv'

class stock_indicators:
    """
    Technical indicator calculation utilities for stock data.
    """
    
    @staticmethod
    def load_data(data_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(data_path)
            logging.info(f"Data loaded successfully from {data_path}")
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logging.error(f"Error in loading data: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def SMA(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Simple Moving Averages"""
        try:
            logging.info(f"Calculating SMA for column: {col_name}")
            df['SMA_20'] = df.groupby(col_name)['Close'].transform(lambda x: x.rolling(window=20).mean())
            df['SMA_50'] = df.groupby(col_name)['Close'].transform(lambda x: x.rolling(window=50).mean())
            df['SMA_100'] = df.groupby(col_name)['Close'].transform(lambda x: x.rolling(window=100).mean())
            df['SMA_200'] = df.groupby(col_name)['Close'].transform(lambda x: x.rolling(window=200).mean())
            logging.info("SMA calculation completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error in SMA calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def EMA(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Exponential Moving Averages"""
        try:
            logging.info(f"Calculating EMA for column: {col_name}")
            df['EMA_50'] = df.groupby(col_name)['Close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
            df['EMA_100'] = df.groupby(col_name)['Close'].transform(lambda x: x.ewm(span=100, adjust=False).mean())
            df['EMA_200'] = df.groupby(col_name)['Close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())
            logging.info("EMA calculation completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error in EMA calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def Bollinger_bands(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        try:
            logging.info(f"Calculating Bollinger Bands for column: {col_name}")
            std_20 = df.groupby(col_name)['Close'].transform(lambda x: x.rolling(window=20).std())
            df['upper_band_bb'] = df['SMA_20'] + (2 * std_20)
            df['lower_band_bb'] = df['SMA_20'] - (2 * std_20)
            logging.info("Bollinger Bands calculation completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error in Bollinger Bands calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def pivot_points(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Pivot Points"""
        try:
            logging.info(f"Calculating Pivot Points for column: {col_name}")
            def compute_pivots(group):
                group = group.sort_values('Date')
                H, L, C = group['High'], group['Low'], group['Close']
                P = (H + L + C) / 3
                group['Pivot'] = P
                group['S1'] = (2 * P) - H
                group['S2'] = P - (H - L)
                group['S3'] = P - 2 * (H - L)
                group['R1'] = (2 * P) - L
                group['R2'] = P + (H - L)
                group['R3'] = P + 2 * (H - L)
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_pivots)
            logging.info("Pivot Points calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in Pivot Points calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def ichimoku_cloud(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Ichimoku Cloud"""
        try:
            logging.info(f"Calculating Ichimoku Cloud for column: {col_name}")
            def compute_ichimoku(group):
                group = group.sort_values('Date')
                group['Tenkan_Sen'] = (group['High'].rolling(9).max() + group['Low'].rolling(9).min()) / 2
                group['Kijun_Sen'] = (group['High'].rolling(26).max() + group['Low'].rolling(26).min()) / 2
                group['Senkou_Span_A'] = ((group['Tenkan_Sen'] + group['Kijun_Sen']) / 2).shift(26)
                group['Senkou_Span_B'] = ((group['High'].rolling(52).max() + group['Low'].rolling(52).min()) / 2).shift(26)
                group['Chikou_Span'] = group['Close'].shift(-26)
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_ichimoku)
            logging.info("Ichimoku Cloud calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in Ichimoku Cloud calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def MACD(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate MACD"""
        try:
            logging.info(f"Calculating MACD for column: {col_name}")
            def compute_macd(group):
                group = group.sort_values('Date')
                group['EMA_12'] = group['Close'].ewm(span=12, adjust=False).mean()
                group['EMA_26'] = group['Close'].ewm(span=26, adjust=False).mean()
                group['MACD'] = group['EMA_12'] - group['EMA_26']
                group['Signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
                group['MACD_Histogram'] = group['MACD'] - group['Signal']
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_macd)
            logging.info("MACD calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in MACD calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def RSI(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate RSI"""
        try:
            logging.info(f"Calculating RSI for column: {col_name}")
            def compute_rsi(group):
                group = group.sort_values('Date')
                delta = group['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                group['RSI'] = 100 - (100 / (1 + rs))
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_rsi)
            logging.info("RSI calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in RSI calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def ATR(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Average True Range"""
        try:
            logging.info(f"Calculating ATR for column: {col_name}")
            def compute_atr(group):
                group = group.sort_values('Date')
                high_low = group['High'] - group['Low']
                high_close = (group['High'] - group['Close'].shift()).abs()
                low_close = (group['Low'] - group['Close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                group['ATR'] = tr.rolling(window=14).mean()
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_atr)
            logging.info("ATR calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in ATR calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def stochastic_oscillator(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        try:
            logging.info(f"Calculating Stochastic Oscillator for column: {col_name}")
            def compute_stochastic(group):
                group = group.sort_values('Date')
                l14 = group['Low'].rolling(window=14).min()
                h14 = group['High'].rolling(window=14).max()
                k = ((group['Close'] - l14) / (h14 - l14)) * 100
                group['so_K'] = k
                group['so_D'] = k.rolling(window=3).mean()
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_stochastic)
            logging.info("Stochastic Oscillator calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in Stochastic Oscillator calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def SAR(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Parabolic SAR"""
        try:
            logging.info(f"Calculating SAR for column: {col_name}")
            def compute_sar(group):
                group = group.sort_values('Date').reset_index(drop=True)
                sar = []
                trend = []
                ep = group['High'].iloc[0]
                af = 0.02
                sar_val = group['Low'].iloc[0]
                uptrend = True
                
                for i in range(len(group)):
                    if i == 0:
                        sar.append(sar_val)
                        trend.append(uptrend)
                        continue
                    
                    prev_sar = sar[-1]
                    
                    if uptrend:
                        new_sar = prev_sar + af * (ep - prev_sar)
                        new_sar = min(new_sar, group['Low'].iloc[i - 1], group['Low'].iloc[i])
                        if group['Low'].iloc[i] < new_sar:
                            uptrend = False
                            sar_val = ep
                            ep = group['Low'].iloc[i]
                            af = 0.02
                        else:
                            sar_val = new_sar
                            if group['High'].iloc[i] > ep:
                                ep = group['High'].iloc[i]
                                af = min(af + 0.02, 0.2)
                    else:
                        new_sar = prev_sar + af * (ep - prev_sar)
                        new_sar = max(new_sar, group['High'].iloc[i - 1], group['High'].iloc[i])
                        if group['High'].iloc[i] > new_sar:
                            uptrend = True
                            sar_val = ep
                            ep = group['High'].iloc[i]
                            af = 0.02
                        else:
                            sar_val = new_sar
                            if group['Low'].iloc[i] < ep:
                                ep = group['Low'].iloc[i]
                                af = min(af + 0.02, 0.2)
                    
                    sar.append(sar_val)
                    trend.append(uptrend)
                
                group['SAR'] = sar
                group['Trend_Up'] = trend
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_sar)
            logging.info("SAR calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in SAR calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def OBV(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate On-Balance Volume"""
        try:
            logging.info(f"Calculating OBV for column: {col_name}")
            def compute_obv(group):
                group = group.sort_values('Date')
                obv = [0]
                for i in range(1, len(group)):
                    if group['Close'].iloc[i] > group['Close'].iloc[i - 1]:
                        obv.append(obv[-1] + group['Volume'].iloc[i])
                    elif group['Close'].iloc[i] < group['Close'].iloc[i - 1]:
                        obv.append(obv[-1] - group['Volume'].iloc[i])
                    else:
                        obv.append(obv[-1])
                group['OBV'] = obv
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_obv)
            logging.info("OBV calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in OBV calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def fibonacci_retracement(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Fibonacci Retracement"""
        try:
            logging.info(f"Calculating Fibonacci Retracement for column: {col_name}")
            def compute_fibonacci(group):
                group = group.sort_values('Date')
                high = group['High'].max()
                low = group['Low'].min()
                diff = high - low
                levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
                for lvl in levels:
                    group[f'Fib_{int(lvl*100)}'] = high - diff * lvl
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_fibonacci)
            logging.info("Fibonacci Retracement calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in Fibonacci Retracement calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def garman_klass_volatility(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Garman-Klass Volatility"""
        try:
            logging.info(f"Calculating Garman-Klass Volatility for column: {col_name}")
            def compute_gk(group):
                group = group.sort_values('Date')
                group['GK'] = np.sqrt(
                    (0.5 * np.log(group['High'] / group['Low']) ** 2) -
                    ((2 * np.log(2) - 1) * (np.log(group['Close'] / group['Open']) ** 2))
                )
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_gk)
            logging.info("Garman-Klass Volatility calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in Garman-Klass Volatility calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def donchian_channel(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Calculate Donchian Channel"""
        try:
            logging.info(f"Calculating Donchian Channel for column: {col_name}")
            def compute_dc(group):
                group = group.sort_values('Date')
                group['DC_Upper'] = group['High'].rolling(window=20).max()
                group['DC_Lower'] = group['Low'].rolling(window=20).min()
                return group
            result = df.groupby(col_name, group_keys=False).apply(compute_dc)
            logging.info("Donchian Channel calculation completed successfully")
            return result
        except Exception as e:
            logging.error(f"Error in Donchian Channel calculation: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def set_all_indicators(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Apply all technical indicators to the dataframe"""
        try:
            logging.info(f"Applying all technical indicators for column: {col_name}")
            # Ensure all price/volume columns are numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Apply indicators in order (some depend on others)
            df = stock_indicators.SMA(df, col_name)
            df = stock_indicators.EMA(df, col_name)
            df = stock_indicators.Bollinger_bands(df, col_name)
            df = stock_indicators.pivot_points(df, col_name)
            df = stock_indicators.ichimoku_cloud(df, col_name)
            df = stock_indicators.MACD(df, col_name)
            df = stock_indicators.RSI(df, col_name)
            df = stock_indicators.ATR(df, col_name)
            df = stock_indicators.stochastic_oscillator(df, col_name)
            df = stock_indicators.SAR(df, col_name)
            df = stock_indicators.fibonacci_retracement(df, col_name)
            df = stock_indicators.OBV(df, col_name)
            df = stock_indicators.garman_klass_volatility(df, col_name)
            df = stock_indicators.donchian_channel(df, col_name)
            logging.info("All technical indicators applied successfully")
            return df
        except Exception as e:
            logging.error(f"Error applying indicators: {e}")
            raise CustomException(e, sys)


class set_indicators:
    """Helper class for setting indicators on data"""
    
    @staticmethod
    def set_indicators(data, output_path: str = None) -> pd.DataFrame:
        """
        Load data and apply all technical indicators
        Args:
            data: DataFrame or path to the input CSV file
            output_path: Path to save the output CSV file (optional)
        Returns:
            DataFrame with all indicators applied
        """
        try:
            # Accept either a DataFrame or a file path
            if isinstance(data, str):
                logging.info(f"Loading data from {data}")
                df = stock_indicators.load_data(data)
            else:
                df = data.copy()
            # Apply all indicators
            df = stock_indicators.set_all_indicators(df, 'Ticker')
            # Save to output path if provided
            if output_path:
                df.to_csv(output_path, index=False)
                logging.info(f"Data with indicators saved to {output_path}")
            return df
        except Exception as e:
            logging.error(f"Error in setting indicators: {e}")
            raise CustomException(e, sys)
