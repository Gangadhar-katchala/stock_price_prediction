"""
Plotting utilities for the stock portfolio project.
Provides functions and classes to visualize all major technical indicators and price data.
"""
import plotly.graph_objects as go
import sys
import numpy as np
from itertools import groupby
from operator import itemgetter
from src.mlproject.logger import logging
from src.exception import CustomException

class plot_indicators:
    @staticmethod
    def plot_candlestick(df, symbol, col_name='Ticker'):
        """Plot a candlestick chart for the given symbol."""
        try:
            logging.info(f"Plotting candlestick for {symbol}")
            data = df[df[col_name] == symbol]
            fig = go.Figure(data=[go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            fig.update_layout(title=f'Candlestick chart for {symbol}')
            fig.show()
            logging.info("Candlestick plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in candlestick plotting: {e}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_line(df, symbol, col_name='Ticker', indicators=None):
        """Plot a line chart for the given symbol and list of indicator columns."""
        try:
            if indicators is None:
                indicators = []
            logging.info(f"Plotting line chart for {symbol} with indicators: {indicators}")
            data = df[df[col_name] == symbol]
            fig = go.Figure()
            for indicator in indicators:
                if indicator in data.columns:
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[indicator], name=indicator))
                else:
                    logging.warning(f"Indicator {indicator} not found in data columns.")
            fig.update_layout(title=f"Line chart for {symbol} with {', '.join(indicators)}")
            fig.show()
            logging.info("Line chart plotted successfully.")
        except Exception as e:
            logging.error(f"Error in line chart plotting: {e}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_pivots(df, symbol, col_name='Ticker'):
        """Plot pivot points and candlestick chart for the given symbol."""
        try:
            logging.info(f"Plotting pivot points for {symbol}")
            if col_name not in df.columns:
                df = df.reset_index()
            df_symbol = df[df[col_name] == symbol].sort_values('Date')
            if df_symbol.empty:
                logging.warning(f"No data for symbol: {symbol}")
                return
            last_row = df_symbol.iloc[-1]
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_symbol['Date'],
                open=df_symbol['Open'],
                high=df_symbol['High'],
                low=df_symbol['Low'],
                close=df_symbol['Close'],
                name='Candlesticks'
            ))
            for level in ['Pivot', 'S1', 'S2', 'S3', 'R1', 'R2', 'R3']:
                if level in last_row:
                    y_val = last_row[level]
                    fig.add_trace(go.Scatter(
                        x=[df_symbol['Date'].min(), df_symbol['Date'].max()],
                        y=[y_val, y_val],
                        mode='lines',
                        name=level,
                        line=dict(dash='dash')
                    ))
            fig.update_layout(
                title=f'{symbol} - Last Day Pivot Levels',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template='plotly_dark'
            )
            fig.show()
            logging.info("Pivot points plotted successfully.")
        except Exception as e:
            logging.error(f"Error in pivot points plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_ichimoku(df, symbol, col_name='Ticker'):
        """Plot Ichimoku Cloud with conditional Kumo shading for the given symbol."""
        try:
            logging.info(f"Plotting Ichimoku for {symbol}")
            data = df[df[col_name] == symbol].copy()
            fig = go.Figure()
            for name in ['Tenkan_Sen', 'Kijun_Sen', 'Chikou_Span']:
                if name in data.columns:
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[name], name=name))
            if 'Senkou_Span_A' in data.columns and 'Senkou_Span_B' in data.columns:
                a = data['Senkou_Span_A']
                b = data['Senkou_Span_B']
                x = data['Date']
                fig.add_trace(go.Scatter(
                    x=x,
                    y=a,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=x,
                    y=b,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0,200,0,0.2)',
                    line=dict(width=0),
                    showlegend=True,
                    hoverinfo='skip',
                    name='Kumo Bullish'
                ))
                mask = a < b
                indices = np.where(mask)[0]
                for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
                    group = list(map(itemgetter(1), g))
                    if len(group) > 1:
                        fig.add_trace(go.Scatter(
                            x=x.iloc[group],
                            y=a.iloc[group],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        fig.add_trace(go.Scatter(
                            x=x.iloc[group],
                            y=b.iloc[group],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(200,0,0,0.2)',
                            line=dict(width=0),
                            showlegend=True,
                            hoverinfo='skip',
                            name='Kumo Bearish'
                        ))
            fig.update_layout(title=f'Ichimoku for {symbol}')
            fig.show()
            logging.info("Ichimoku plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in Ichimoku plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_rsi(df, symbol, col_name='Ticker'):
        """Plot RSI with overbought/oversold/neutral lines for the given symbol."""
        try:
            logging.info(f"Plotting RSI for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            data = data.dropna(subset=['RSI'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ))
            fig.add_hline(y=70, line=dict(color='red', dash='dash'), annotation_text='Overbought', annotation_position='top left', showlegend=False)
            fig.add_hline(y=50, line=dict(color='gray', dash='dash'), annotation_text='Neutral', annotation_position='top right', showlegend=False)
            fig.add_hline(y=30, line=dict(color='green', dash='dash'), annotation_text='Oversold', annotation_position='bottom left', showlegend=False)
            fig.update_layout(
                title=f'{symbol} RSI Indicator',
                xaxis_title='Date',
                yaxis_title='RSI',
                yaxis=dict(range=[0, 100]),
                template='plotly_dark',
                hovermode='x unified'
            )
            fig.show()
            logging.info("RSI plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in RSI plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def MACD_Indicator(df, stock_col):
        """Calculate MACD, Signal, and MACD Histogram columns for the DataFrame."""
        df.groupby(stock_col).apply(lambda x: x.sort_values('Date', inplace=True))
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        return df

    @staticmethod
    def plot_sar(df, symbol, col_name='Ticker'):
        """Plot Parabolic SAR and candlestick chart for the given symbol."""
        try:
            logging.info(f"Plotting Parabolic SAR for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'
            ))
            if 'SAR' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['SAR'],
                    mode='markers',
                    name='SAR',
                    marker=dict(color='cyan', size=5, symbol='circle')
                ))
            fig.update_layout(
                title=f'{symbol} Parabolic SAR',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                hovermode='x unified'
            )
            fig.show()
            logging.info("Parabolic SAR plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in Parabolic SAR plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_fibonacci(df, symbol, col_name='Ticker'):
        """Plot Fibonacci retracement levels and candlestick chart for the given symbol."""
        try:
            logging.info(f"Plotting Fibonacci Retracement for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            fib_cols = [col for col in data.columns if col.startswith('Fib_')]
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'
            ))
            if not data.empty:
                last = data.iloc[-1]
                for col in fib_cols:
                    fig.add_hline(y=last[col], line_dash='dash', annotation_text=col, annotation_position='right')
            fig.update_layout(
                title=f'{symbol} Fibonacci Retracement',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark'
            )
            fig.show()
            logging.info("Fibonacci Retracement plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in Fibonacci Retracement plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_bollinger_bands(df, symbol, col_name='Ticker', show_close=True):
        """Plot Bollinger Bands and optionally close price for the given symbol."""
        try:
            logging.info(f"Plotting Bollinger Bands for {symbol}")
            data = df[df[col_name] == symbol]
            fig = go.Figure()
            if 'SMA_20' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], name='SMA_20', line=dict(color='black')))
            if 'upper_band_bb' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['upper_band_bb'], name='Upper Band', line=dict(color='blue')))
            if 'lower_band_bb' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['lower_band_bb'], name='Lower Band', line=dict(color='red')))
            fig.update_layout(title=f'Bollinger Bands for {symbol}')
            fig.show()
            logging.info("Bollinger Bands plotted successfully.")
        except Exception as e:
            logging.error(f"Error in Bollinger Bands plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_stochastic_oscillator(df, symbol, col_name='Ticker'):
        """Plot Stochastic Oscillator (%K, %D) with overbought/oversold/neutral lines for the given symbol."""
        try:
            logging.info(f"Plotting Stochastic Oscillator for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            fig = go.Figure()
            if 'so_K' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['so_K'], name='%K', line=dict(color='blue')))
            if 'so_D' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['so_D'], name='%D', line=dict(color='orange')))
            fig.add_hline(y=80, line=dict(color='red', dash='dash'), annotation_text='Overbought', annotation_position='top left', showlegend=False)
            fig.add_hline(y=50, line=dict(color='gray', dash='dash'), annotation_text='Neutral', annotation_position='top right', showlegend=False)
            fig.add_hline(y=20, line=dict(color='green', dash='dash'), annotation_text='Oversold', annotation_position='bottom left', showlegend=False)
            fig.update_layout(
                title=f'{symbol} Stochastic Oscillator',
                xaxis_title='Date',
                yaxis_title='Stochastic Value',
                yaxis=dict(range=[0, 100]),
                template='plotly_dark',
                hovermode='x unified'
            )
            fig.show()
            logging.info("Stochastic Oscillator plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in Stochastic Oscillator plotting: {e}")
            raise CustomException(e, sys)
    
    @staticmethod
    def plot_obv(df, symbol, col_name='Ticker'):
        """Plot On-Balance Volume (OBV) for the given symbol."""
        try:
            logging.info(f"Plotting OBV for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            fig = go.Figure()
            if 'OBV' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['OBV'], name='OBV', line=dict(color='blue')))
            else:
                raise ValueError('OBV column not found in data')
            fig.update_layout(
                title=f'{symbol} On-Balance Volume (OBV)',
                xaxis_title='Date',
                yaxis_title='OBV',
                template='plotly_dark',
                hovermode='x unified'
            )
            fig.show()
            logging.info("OBV plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in OBV plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_garman_klass(df, symbol, col_name='Ticker'):
        """Plot Garman-Klass Volatility for the given symbol."""
        try:
            logging.info(f"Plotting Garman-Klass Volatility for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            fig = go.Figure()
            if 'GK' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['GK'], name='Garman-Klass Volatility', line=dict(color='orange')))
            else:
                raise ValueError('GK column not found in data')
            fig.update_layout(
                title=f'{symbol} Garman-Klass Volatility',
                xaxis_title='Date',
                yaxis_title='GK',
                template='plotly_dark',
                hovermode='x unified'
            )
            fig.show()
            logging.info("Garman-Klass Volatility plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in Garman-Klass Volatility plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_donchian_channel(df, symbol, col_name='Ticker', show_close=True):
        """Plot Donchian Channel (upper/lower bands and optionally close price) for the given symbol."""
        try:
            logging.info(f"Plotting Donchian Channel for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            fig = go.Figure()
            if show_close and 'Close' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close', line=dict(color='black')))
            if 'DC_Upper' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['DC_Upper'], name='Donchian Upper', line=dict(color='blue')))
            if 'DC_Lower' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['DC_Lower'], name='Donchian Lower', line=dict(color='red')))
            fig.update_layout(
                title=f'{symbol} Donchian Channel',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark',
                hovermode='x unified'
            )
            fig.show()
            logging.info("Donchian Channel plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in Donchian Channel plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_macd(df, symbol, col_name='Ticker'):
        """Plot MACD, Signal, and MACD Histogram for the given symbol."""
        try:
            logging.info(f"Plotting MACD for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            fig = go.Figure()
            if 'MACD' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD', line=dict(color='blue')))
            if 'Signal' in data.columns:
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Signal'], name='Signal', line=dict(color='orange')))
            if 'MACD_Histogram' in data.columns:
                colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
                fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_Histogram'], name='MACD Histogram', marker_color=colors, opacity=0.7))
            fig.update_layout(title=f'{symbol} MACD', xaxis_title='Date', yaxis_title='Value', template='plotly_dark', barmode='relative')
            fig.show()
            logging.info("MACD plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in MACD plotting: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def plot_all_indicators(df, symbol, col_name='Ticker', indicators_list=None):
        """Plot selected indicators in a single figure for the given symbol. If indicators_list is None, plot all."""
        try:
            logging.info(f"Plotting selected indicators for {symbol}")
            data = df[df[col_name] == symbol].copy()
            data = data.sort_values('Date')
            fig = go.Figure()
            # Define all possible indicators and their columns
            indicator_map = {
                'close': ['Close'],
                'sma': ['SMA_20', 'SMA_50', 'SMA_100', 'SMA_200'],
                'ema': ['EMA_50', 'EMA_100', 'EMA_200'],
                'bollinger_bands': ['upper_band_bb', 'lower_band_bb'],
                'macd': ['MACD', 'Signal', 'MACD_Histogram'],
                'rsi': ['RSI'],
                'atr': ['ATR'],
                'sar': ['SAR'],
                'obv': ['OBV'],
                'donchian_channel': ['DC_Upper', 'DC_Lower'],
                'garman_klass': ['GK'],
                'stochastic_oscillator': ['so_K', 'so_D'],
                'ichimoku': ['Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span'],
                'fibonacci': [col for col in data.columns if col.startswith('Fib_')],
            }
            # If no list, plot all
            if indicators_list is None:
                indicators_list = list(indicator_map.keys())
            for key in indicators_list:
                if key in indicator_map:
                    for col in indicator_map[key]:
                        if col in data.columns:
                            fig.add_trace(go.Scatter(x=data['Date'], y=data[col], name=col))
            # Fibonacci as horizontal lines (last row)
            if 'fibonacci' in indicators_list and not data.empty:
                last = data.iloc[-1]
                for col in indicator_map['fibonacci']:
                    fig.add_hline(y=last[col], line_dash='dash', annotation_text=col, annotation_position='right')
            fig.update_layout(title=f'Selected Indicators for {symbol}', xaxis_title='Date', template='plotly_dark')
            fig.show()
            logging.info("Selected indicators plot completed successfully.")
        except Exception as e:
            logging.error(f"Error in selected indicators plotting: {e}")
            raise CustomException(e, sys)

class plotting_indicators:
    @staticmethod
    def plot(df, symbol, indicators, all_in_one=False):
        """Plot multiple indicators for a given symbol using plot_indicators methods. If all_in_one is True, plot all in one graph."""
        if all_in_one:
            plot_indicators.plot_all_indicators(df, symbol, indicators_list=indicators)
        else:
            for indicator in indicators:
                if indicator == 'candlestick':
                    plot_indicators.plot_candlestick(df, symbol)
                elif indicator == 'pivots':
                    plot_indicators.plot_pivots(df, symbol)
                elif indicator == 'ichimoku':
                    plot_indicators.plot_ichimoku(df, symbol)
                elif indicator == 'rsi':
                    plot_indicators.plot_rsi(df, symbol)
                elif indicator == 'macd':
                    plot_indicators.plot_macd(df, symbol)
                elif indicator == 'sar':
                    plot_indicators.plot_sar(df, symbol)
                elif indicator == 'fibonacci':
                    plot_indicators.plot_fibonacci(df, symbol)
                elif indicator == 'bollinger_bands':
                    plot_indicators.plot_bollinger_bands(df, symbol)
                elif indicator == 'stochastic_oscillator':
                    plot_indicators.plot_stochastic_oscillator(df, symbol)
                elif indicator == 'obv':
                    plot_indicators.plot_obv(df, symbol)
                elif indicator == 'garman_klass':
                    plot_indicators.plot_garman_klass(df, symbol)
                elif indicator == 'donchian_channel':
                    plot_indicators.plot_donchian_channel(df, symbol)
                elif indicator == 'all_indicators':
                    plot_indicators.plot_all_indicators(df, symbol)
                else:
                    raise ValueError(f"Invalid indicator: {indicator}")


        
    




