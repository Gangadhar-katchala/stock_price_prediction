import sys
import os

# Ensure project root is in sys.path for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.mlproject.data_collection.data_collection import get_data_simple
from src.mlproject.model_building import ModelTraining, DataProcessing
from src.mlproject.logger import logging
from src.mlproject.indicators import set_indicators

# Data paths
stock_list_path = r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\processed_data\stock_list.xlsx'
price_data_path = r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\processed_data\stock_price_data.csv'
indicators_path = r'C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\un_processed_data\stock_price_with_indicators.csv'


def main():
    print("\n🚀 Starting Stock Portfolio Training Pipeline (All Stocks)", flush=True)
    print("==================================================", flush=True)
    # Collect and process data for all stocks
    data_path = get_data_simple()  # Uses default Excel and CSV paths
    print("✅ Data collection complete.", flush=True)
    ModelTraining.train(data_path)
    print("✅ Model training complete for all stocks.", flush=True)

if __name__ == "__main__":
    main()
