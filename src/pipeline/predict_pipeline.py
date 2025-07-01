import os
# Suppress TensorFlow oneDNN & verbose logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.getcwd())

from src.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.model_building import DataReader, DataProcessing, DataTransformation

# Removed local DataTransformation class. Now using the one from model_building.

class PredictPipeline:
    def __init__(self, models_dir=None, metadata_path=None, data_path=None):
        self.models_dir = models_dir or r"C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\models"
        self.metadata_path = metadata_path or r"C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\models\model_metadata.pkl"
        self.data_path = data_path or r"C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\un_processed_data\stock_price_with_indicators.csv"

        self.models = {}
        self.global_metadata = {}
        self.data_dict = {}
        self.data_tickers = set()

        self.load_and_prepare_data()
        self.load_models_and_metadata()

    def load_and_prepare_data(self):
        try:
            logging.info("Loading and preparing data...")
            df = DataReader.read(self.data_path)
            df = DataProcessing.process(df)
            self.data_dict = DataTransformation.transform(df)
            self.data_tickers = set(self.data_dict.keys())
            logging.info(f"Data prepared for {len(self.data_dict)} tickers")
        except Exception as e:
            raise CustomException(f"Error preparing data: {str(e)}", sys)

    def load_models_and_metadata(self):
        try:
            logging.info("Loading trained LSTM models and global metadata...")
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "rb") as f:
                    self.global_metadata = pickle.load(f)

            # Load training data to get feature information
            df = DataReader.read(self.data_path)
            df = DataProcessing.process(df)
            training_data_dict = DataTransformation.transform(df)

            for filename in os.listdir(self.models_dir):
                if filename.endswith('_lstm_model.keras'):
                    ticker = filename.replace('_lstm_model.keras', '')
                    if ticker.endswith('.NS') and ticker in self.data_tickers:
                        model = load_model(os.path.join(self.models_dir, filename))
                        self.models[ticker] = model
                        # Add feature information to metadata if not present
                        if ticker in training_data_dict:
                            if ticker not in self.global_metadata:
                                self.global_metadata[ticker] = {}
                            self.global_metadata[ticker]['num_cols'] = training_data_dict[ticker]['num_cols']
                            self.global_metadata[ticker]['target_cols'] = training_data_dict[ticker]['target_cols']
                            self.global_metadata[ticker]['target_indices'] = training_data_dict[ticker]['target_indices']
                        logging.info(f"Loaded model for {ticker}")
        except Exception as e:
            raise CustomException(f"Error loading models and metadata: {str(e)}", sys)

    def predict_single_stock(self, ticker: str, prediction_date: str = None) -> Dict:
        try:
            if ticker not in self.models:
                raise ValueError(f"No trained model found for ticker: {ticker}")
            if ticker not in self.data_dict:
                raise ValueError(f"No data found for ticker: {ticker}")
            if ticker not in self.global_metadata:
                raise ValueError(f"No metadata found for ticker: {ticker}")

            model = self.models[ticker]
            data = self.data_dict[ticker]
            expected_input_shape = model.input_shape
            expected_features = expected_input_shape[-1]
            last_sequence = data['x_test'][-1:].reshape(1, data['x_test'].shape[1], data['x_test'].shape[2])
            current_features = last_sequence.shape[-1]
            if current_features != expected_features:
                if current_features < expected_features:
                    padding = np.zeros((1, last_sequence.shape[1], expected_features - current_features))
                    last_sequence = np.concatenate([last_sequence, padding], axis=2)
                else:
                    last_sequence = last_sequence[:, :, :expected_features]
            pred_scaled = model.predict(last_sequence, verbose=0)
            if np.isnan(pred_scaled).any():
                raise ValueError(f"Model output contains NaN for {ticker}")
            mm = data['scaler']
            num_cols = data['num_cols']
            target_cols = data['target_cols']
            target_indices = data['target_indices']
            dummy_pred = np.zeros((1, len(num_cols)))
            dummy_pred[:, target_indices] = pred_scaled[0]
            dummy_pred = np.clip(dummy_pred, 0, 1)
            original_pred = mm.inverse_transform(dummy_pred)[:, target_indices]
            if np.isnan(original_pred).any():
                raise ValueError(f"Inverse transformed values contain NaN for {ticker}")
            raw_preds = {col: float(original_pred[0, i]) for i, col in enumerate(target_cols)}
            open_ = raw_preds.get("Open", 0)
            high = raw_preds.get("High", 0)
            low = raw_preds.get("Low", 0)
            close = raw_preds.get("Close", 0)
            values = [open_, high, low, close]
            high = max(values)
            low = min(values)
            open_ = sorted(values)[1]
            close = sorted(values)[2]
            predictions = {
                'ticker': ticker,
                'prediction_date': prediction_date or (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'Open': round(open_, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'model_confidence': self._calculate_confidence(ticker, pred_scaled[0])
            }
            return predictions
        except Exception as e:
            raise CustomException(f"Error predicting for {ticker}: {str(e)}", sys)

    def predict_multiple_stocks(self, tickers: List[str], prediction_date: str = None) -> Dict:
        try:
            results = {}
            for ticker in tickers:
                if ticker not in self.models:
                    results[ticker] = {'error': f'No trained model for {ticker}'}
                elif ticker not in self.global_metadata:
                    results[ticker] = {'error': f'No metadata for {ticker}'}
                else:
                    try:
                        results[ticker] = self.predict_single_stock(ticker, prediction_date)
                    except Exception as e:
                        results[ticker] = {'error': str(e)}
            return results
        except Exception as e:
            raise CustomException(f"Error in batch prediction: {str(e)}", sys)

    def _calculate_confidence(self, ticker: str, pred_scaled: np.ndarray) -> float:
        try:
            variance = np.var(pred_scaled)
            confidence = round(max(0.1, 1.0 - variance), 3)
            return confidence
        except:
            return 0.5

    def get_available_tickers(self) -> List[str]:
        return [t for t in self.models.keys() if t.endswith('.NS')]

# 🧪 Simple Test Run
if __name__ == "__main__":
    predictor = PredictPipeline()
    tickers = predictor.get_available_tickers()
    predictions = predictor.predict_multiple_stocks(tickers)
    print("\n📊 Prediction Results:\n" + "-" * 30)
    for ticker, pred in predictions.items():
        if 'error' in pred:
            print(f"❌ {ticker}: {pred['error']}")
        else:
            print(f"✅ {ticker}:")
            print(f"   📈 Open: ₹{pred['Open']}")
            print(f"   📊 High: ₹{pred['High']}")
            print(f"   📉 Low: ₹{pred['Low']}")
            print(f"   📋 Close: ₹{pred['Close']}")
            print(f"   🎯 Confidence: {pred['model_confidence'] * 100:.2f}%\n")
