import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
import pickle
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import tensorflow as tf

from src.mlproject.logger import logging
from src.exception import CustomException
from src.mlproject.indicators import set_indicators

save_dir=r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\models'
metadata_path=r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\model_metadata.pkl'
output_dir=r'C:\Users\katch\Desktop\projects\stock_portfolio\images'

class DataReader:
    @staticmethod
    def read(file_path):
        try:
            df = pd.read_csv(file_path)
            logging.info('Data loaded to DataFrame successfully')
            return df
        except Exception as e:
            logging.info('Data path not found')
            raise CustomException(e, sys)


class DataProcessing:
    @staticmethod
    def process(df):
        try:
            # Robust date parsing for mixed formats
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            except TypeError:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Range'] = df['High'] - df['Low']
            df['Change'] = df['Close'] - df['Open']
            df['Return_%'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df['daily_return'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change())
            df['Avg_Price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
            df['Volatility'] = (df['High'] - df['Low']) / df['Open'] * 100
            df['log_open'] = df.groupby('Ticker')['Open'].transform(lambda x: np.log(x))
            df['log_close'] = df.groupby('Ticker')['Close'].transform(lambda x: np.log(x))
            df['log_high'] = df.groupby('Ticker')['High'].transform(lambda x: np.log(x))
            df['log_low'] = df.groupby('Ticker')['Low'].transform(lambda x: np.log(x))
            # Apply technical indicators
            df = set_indicators.set_indicators(df)
            # Convert boolean columns to numeric for model compatibility
            for col in df.columns:
                if df[col].dtype == bool:
                    df[col] = df[col].astype(int)
                    logging.info(f"Converted boolean column {col} to numeric")
            # Define the full 62-feature list in the correct order
            feature_list = [
                'Close','High','Low','Open','Volume',
                'SMA_20','SMA_50','SMA_100','SMA_200',
                'EMA_50','EMA_100','EMA_200',
                'upper_band_bb','lower_band_bb',
                'Pivot','S1','S2','S3','R1','R2','R3',
                'Tenkan_Sen','Kijun_Sen','Senkou_Span_A','Senkou_Span_B','Chikou_Span',
                'EMA_12','EMA_26','MACD','Signal','MACD_Histogram',
                'RSI','ATR','so_K','so_D','SAR','Trend_Up',
                'Fib_0','Fib_23','Fib_38','Fib_50','Fib_61','Fib_78','Fib_100',
                'OBV','GK','DC_Upper','DC_Lower',
                'Year','Month','Day','DayOfWeek','Range','Change','Return_%','daily_return',
                'Avg_Price','Volatility','log_open','log_close','log_high','log_low'
            ]
            # Keep only the features in the list (if present)
            features_present = [f for f in feature_list if f in df.columns]
            df = df[['Date','Ticker'] + features_present]
            # Drop rows with NaNs in any essential feature
            before_drop = len(df)
            df = df.dropna(subset=features_present)
            after_drop = len(df)
            dropped = before_drop - after_drop
            if dropped > 0:
                logging.warning(f"Dropped {dropped} rows with NaNs in essential features.")
            # Log NaN counts for each feature
            nan_counts = df[features_present].isna().sum()
            for col, nans in nan_counts.items():
                if nans > 0:
                    logging.warning(f"Column {col} has {nans} NaNs after dropna.")
            # Print summary
            logging.info(f"Processed data shape: {df.shape}")
            logging.info(f"First 3 rows:\n{df.head(3)}")
            return df
        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise CustomException(e, sys)



class DataTransformation:
    @staticmethod
    def create_sequences(scaled_data, lookback, target_indices):
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, :])
            y.append(scaled_data[i, target_indices])
        return np.array(X), np.array(y)

    @staticmethod
    def transform(df, target_col=['Close', 'Open', 'High', 'Low'], lookback=52, final_dict=None):
        try:
            if final_dict is None:
                final_dict = {}
            # Use the same feature list as in DataProcessing
            feature_list = [
                'Close','High','Low','Open','Volume',
                'SMA_20','SMA_50','SMA_100','SMA_200',
                'EMA_50','EMA_100','EMA_200',
                'upper_band_bb','lower_band_bb',
                'Pivot','S1','S2','S3','R1','R2','R3',
                'Tenkan_Sen','Kijun_Sen','Senkou_Span_A','Senkou_Span_B','Chikou_Span',
                'EMA_12','EMA_26','MACD','Signal','MACD_Histogram',
                'RSI','ATR','so_K','so_D','SAR','Trend_Up',
                'Fib_0','Fib_23','Fib_38','Fib_50','Fib_61','Fib_78','Fib_100',
                'OBV','GK','DC_Upper','DC_Lower',
                'Year','Month','Day','DayOfWeek','Range','Change','Return_%','daily_return',
                'Avg_Price','Volatility','log_open','log_close','log_high','log_low'
            ]
            for ticker, group in df.groupby('Ticker'):
                group = group.sort_values('Date')  # Keep time order
                # Ensure minimum data requirements
                if len(group) < lookback + 10:
                    logging.warning(f"Insufficient data for {ticker}: {len(group)} rows")
                    continue
                # Only use the defined features
                num_cols = [f for f in feature_list if f in group.columns]
                # Abort if any NaNs remain
                if group[num_cols].isna().any().any():
                    logging.error(f"NaNs remain in features for {ticker}. Skipping.")
                    continue
                mm = MinMaxScaler()
                scaled_data = mm.fit_transform(group[num_cols])
                target_indices = [num_cols.index(col) for col in target_col if col in num_cols]
                if len(target_indices) != len(target_col):
                    logging.warning(f"Some target columns not found for {ticker}")
                    continue
                X, y = DataTransformation.create_sequences(scaled_data, lookback, target_indices)
                if len(X) == 0:
                    logging.warning(f"No sequences created for {ticker}")
                    continue
                # Split using 80/20 ratio
                split_idx = int(len(X) * 0.8)
                # Ensure we have enough training and test data
                if split_idx < 10 or len(X) - split_idx < 5:
                    logging.warning(f"Insufficient train/test data for {ticker}")
                    continue
                final_dict[ticker] = {
                    'x_train': X[:split_idx],
                    'y_train': y[:split_idx],
                    'x_test': X[split_idx:],
                    'y_test': y[split_idx:],
                    'scaler': mm,
                    'num_cols': num_cols,
                    'target_cols': target_col,
                    'target_indices': target_indices
                }
                logging.info(f"Transformed {ticker}: {len(X)} sequences, {split_idx} train, {len(X)-split_idx} test")
            return final_dict
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)




class ModelBuilding:
    @staticmethod
    def build(final_dict, model_dict=None):
        if model_dict is None:
            model_dict = {}
        np.random.seed(42)
        tf.random.set_seed(42)
        random.seed(42)
        
        for ticker, data in final_dict.items():
            x_train = data['x_train']
            y_train = data['y_train']

            input_shape = x_train.shape[1:]  # (lookback, features)
            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

            model = Sequential()
            model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(output_dim))

            model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mae'])
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(
                x_train, y_train, 
                validation_split=0.1, 
                epochs=15, 
                batch_size=32, 
                callbacks=[early_stop], 
                verbose=0
            )
            model_dict[ticker] = {'model': model, 'history': history}
        
        return model_dict



class ModelEvaluation:
    @staticmethod
    def evaluate(final_dict, model_dict, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        summary, detailed = [], []
        for ticker in model_dict:
            model = model_dict[ticker]['model']
            data = final_dict[ticker]
            mm = data['scaler']
            num_cols = data['num_cols']
            target_cols = data['target_cols']
            target_indices = data['target_indices']
            x_test = data['x_test']
            y_test = data['y_test']
            
            # Check if test data exists
            if len(x_test) == 0:
                logging.warning(f"No test data available for {ticker}")
                continue
            
            pred_scaled = model.predict(x_test, verbose=0)
            # Create dummy arrays for inverse scaling
            dummy_actual = np.zeros((len(y_test), len(num_cols)))
            dummy_pred = np.zeros((len(y_test), len(num_cols)))
            dummy_actual[:, target_indices] = y_test
            dummy_pred[:, target_indices] = pred_scaled
            # Inverse scale to get original values
            original_actual = mm.inverse_transform(dummy_actual)[:, target_indices]
            original_pred = mm.inverse_transform(dummy_pred)[:, target_indices]
            # Calculate metrics for each target column
            for i, col in enumerate(target_cols):
                mae = mean_absolute_error(original_actual[:, i], original_pred[:, i])
                rmse = np.sqrt(mean_squared_error(original_actual[:, i], original_pred[:, i]))
                summary.append({
                    'Ticker': ticker, 
                    'Column': col, 
                    'MAE': mae, 
                    'RMSE': rmse
                })
                # Store detailed predictions
                for j in range(len(original_actual[:, i])):
                    detailed.append({
                        'Ticker': ticker,
                        'Column': col,
                        'Index': j,
                        'Actual': original_actual[j, i],
                        'Predicted': original_pred[j, i]
                    })
        # Save evaluation results
        summary_df = pd.DataFrame(summary)
        detailed_df = pd.DataFrame(detailed)
        summary_path = os.path.join(save_dir, "evaluation_summary.csv")
        detailed_path = os.path.join(save_dir, "detailed_predictions.csv")
        summary_df.to_csv(summary_path, index=False)
        detailed_df.to_csv(detailed_path, index=False)
        logging.info(f"Evaluation results saved to {save_dir}")
        return summary_df, detailed_df


class SaveModel:
    @staticmethod
    def save(model_dict, save_dir, metadata_path):
        os.makedirs(save_dir, exist_ok=True)
        model_info = {}
        for ticker in model_dict:
            # Ensure .NS suffix
            if not ticker.endswith('.NS'):
                ticker_ns = ticker + '.NS'
            else:
                ticker_ns = ticker
            model_path = os.path.join(save_dir, f"{ticker_ns}_lstm_model.keras")
            model_dict[ticker]['model'].save(model_path)
            model_info[ticker_ns] = {'model_path': model_path, 'history': model_dict[ticker]['history'].history}
        with open(metadata_path, "wb") as f:
            pickle.dump(model_info, f)


class PlotModel:
    @staticmethod
    def price_plot(ticker_list, ochl=['Close', 'Open', 'High', 'Low'], 
             summary_path=r"C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\un_processed_data\detailed_predictions.csv"):
        df = pd.read_csv(summary_path)
        for ticker in ticker_list:
            for col in ochl:
                subset = df[(df['Ticker'] == ticker) & (df['Column'] == col)]
                if not subset.empty:
                    plt.figure(figsize=(10, 5))
                    plt.plot(subset['Index'], subset['Actual'], label=f'Actual {col}')
                    plt.plot(subset['Index'], subset['Predicted'], label=f'Predicted {col}', linestyle='--')
                    plt.title(f'{ticker} - Actual vs Predicted {col}')
                    plt.legend()
                    plt.grid(True)
                    filename = f"{ticker}_{col}_actual_vs_predicted.png"
                    plt.show()
    
   
    @staticmethod
    def stats_plot(summary_path=None, output_dir='plots'):
        try:
            if summary_path is None:
                summary_path = r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\models\evaluation_summary.csv'
            
            df = pd.read_csv(summary_path)
            os.makedirs(output_dir, exist_ok=True)

            metrics = ['MAE', 'RMSE']
            column_colors = {
                'Close': '#1f77b4',  # blue
                'High': '#ff7f0e',   # orange
                'Low': '#2ca02c',    # green
                'Open': '#d62728',   # red
            }

            for metric in metrics:
                if metric not in df.columns:
                    logging.warning(f"{metric} not found in dataframe columns.")
                    continue

                pivot = df.pivot(index='Ticker', columns='Column', values=metric).fillna(0)
                pivot[f'avg_{metric.lower()}'] = pivot.mean(axis=1)
                pivot = pivot.sort_values(by=f'avg_{metric.lower()}')
                avg_vals = pivot[f'avg_{metric.lower()}']
                pivot = pivot.drop(columns=f'avg_{metric.lower()}')

                plt.figure(figsize=(14, 6))
                bottom = np.zeros(len(pivot))

                for col in ['Close', 'High', 'Low', 'Open']:
                    if col in pivot.columns:
                        plt.bar(pivot.index, pivot[col], label=col, bottom=bottom, color=column_colors[col])
                        bottom += pivot[col]

                # Add average metric value on top
                for i, val in enumerate(avg_vals):
                    plt.text(i, bottom[i] + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

                plt.title(f'Stacked {metric} per Stock (Open, Close, High, Low)')
                plt.xlabel('Ticker')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.legend(title='Column')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{metric.lower()}_per_stock.png"))
                plt.close()
        
        except Exception as e:
            logging.error(f"Error in plotting stats: {e}")
            print(f"Warning: Could not generate evaluation plots: {e}")

# Example usage:
# StatsPlotter.stats_plot()


class ModelTraining:
    @staticmethod
    def train(data_path: str, save_dir=None, metadata_path=None):
        try:
            # Use default paths if not provided
            if save_dir is None:
                save_dir = r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\models'
            if metadata_path is None:
                metadata_path = r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\model_metadata.pkl'
            
            df = DataReader.read(data_path)
            df = DataProcessing.process(df)
            final_dict = DataTransformation.transform(df)
            model_dict = ModelBuilding.build(final_dict)
            ModelEvaluation.evaluate(final_dict, model_dict, save_dir)
            SaveModel.save(model_dict, save_dir, metadata_path)
            
            # Add models_dir to the returned dictionary for reference
            model_dict['models_dir'] = save_dir
            model_dict['metadata_path'] = metadata_path
            
            return model_dict
        except Exception as e:
            logging.info(f"Error in training model: {e}")
            raise CustomException(e, sys)
