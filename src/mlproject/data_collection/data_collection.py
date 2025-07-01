"""
Data collection utilities for the stock portfolio project.
Provides functions to download and validate stock data from Excel files and Yahoo Finance.
"""
from src.mlproject.logger import logging
from src.exception import CustomException
import pandas as pd
import yfinance as yf
import datetime as dt
import sys
import os

def get_data(excel_file_path: str, output_csv_path: str = None, years_back: int = 5) -> str:
    """
    Function to read stock symbols from Excel file and download stock data to CSV
    
    Args:
        excel_file_path (str): Path to the Excel file containing stock symbols
        output_csv_path (str): Path where the CSV file should be saved (optional)
        years_back (int): Number of years of historical data to download (default: 5)
    
    Returns:
        str: Path to the generated CSV file
    
    Raises:
        CustomException: If there's an error in data collection
    """
    try:
        logging.info("Starting stock data collection process")
        
        # Validate input file exists
        if not os.path.exists(excel_file_path):
            raise CustomException(f"Excel file not found: {excel_file_path}", sys)
        
        # Read the Excel file
        logging.info(f"Reading Excel file from: {excel_file_path}")
        stocks = pd.read_excel(excel_file_path)

        
        # Validate required column exists
        if 'Stock' not in stocks.columns:
            raise CustomException("Excel file must contain a 'Stock' column with stock symbols", sys)
        
        # Remove any empty or NaN values from stock list
        stock_list = stocks['Stock'].dropna().tolist()
        if not stock_list:
            raise CustomException("No valid stock symbols found in the Excel file", sys)
        
        logging.info(f"Found {len(stock_list)} stock symbols in Excel file")
        
        # Add market indices and ETFs
        additional_symbols = ['^NSEI', 'BANKBEES.NS', 'NIFTYBEES.NS', 'GOLDBEES.NS', 'MID150BEES.NS', 'JUNIORBEES.NS']
        stock_list.extend(additional_symbols)
        logging.info(f"Total symbols to download: {len(stock_list)}")
        
        # Set date range
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365 * years_back)
        logging.info(f"Downloading data from {start_date.date()} to {end_date.date()}")
        
        # Download stock data
        logging.info("Downloading stock data from Yahoo Finance...")
        df = yf.download(stock_list, start=start_date, end=end_date, progress=False)
        
        # Check if data was downloaded successfully
        if df.empty:
            raise CustomException("No data was downloaded. Please check your stock symbols and internet connection.", sys)
        
        # Reshape the data
        logging.info("Processing downloaded data...")
        df = df.stack(future_stack=True)
        df = df.reset_index()
        
        # Set output path
        if output_csv_path is None:
            # Save in same directory as Excel file
            output_csv_path = os.path.join(os.path.dirname(excel_file_path), "stock_price_data.csv")
        else:
            # If output_csv_path is provided, use it directly (don't append filename)
            if os.path.isdir(output_csv_path):
                # If it's a directory, append filename
                output_csv_path = os.path.join(output_csv_path, "stock_price_data.csv")
            # If it's already a file path, use as is
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        # Save to CSV
        logging.info(f"Saving data to CSV: {output_csv_path}")
        df.to_csv(output_csv_path, index=False)
        
        logging.info(f"Data collection completed successfully. CSV saved to: {output_csv_path}")
        logging.info(f"Total records: {len(df)}")
        logging.info(f"Unique tickers: {df['Ticker'].nunique()}")
        
        return output_csv_path
        
    except Exception as e:
        logging.error(f"Error in data collection: {str(e)}")
        raise CustomException(e, sys)

def validate_stock_symbols(excel_file_path: str) -> pd.DataFrame:
    """
    Function to validate stock symbols in Excel file
    
    Args:
        excel_file_path (str): Path to the Excel file containing stock symbols
    
    Returns:
        pd.DataFrame: DataFrame with validation results
    """
    try:
        logging.info("Validating stock symbols...")
        
        # Validate input file exists
        if not os.path.exists(excel_file_path):
            raise CustomException(f"Excel file not found: {excel_file_path}", sys)
        
        # Read the Excel file
        stocks = pd.read_excel(excel_file_path)
        
        if 'Stock' not in stocks.columns:
            raise CustomException("Excel file must contain a 'Stock' column with stock symbols", sys)
        
        # Remove empty values before validation
        stocks = stocks.dropna(subset=['Stock'])
        
        # Validate symbols
        def is_valid_symbol(stock):
            try:
                info = yf.Ticker(stock).info
                return 'valid' if 'shortName' in info and info['shortName'] else 'non valid'
            except Exception:
                return 'error'
        
        stocks['is_valid'] = stocks['Stock'].apply(is_valid_symbol)
        
        # Print validation summary
        valid_count = (stocks['is_valid'] == 'valid').sum()
        invalid_count = (stocks['is_valid'] == 'non valid').sum()
        error_count = (stocks['is_valid'] == 'error').sum()
        
        logging.info(f"Validation complete:")
        logging.info(f"Valid symbols: {valid_count}")
        logging.info(f"Invalid symbols: {invalid_count}")
        logging.info(f"Error symbols: {error_count}")
        
        return stocks
        
    except Exception as e:
        logging.error(f"Error in symbol validation: {str(e)}")
        raise CustomException(e, sys)

def get_data_simple() -> str:
    """
    Simplified function that uses hardcoded paths - just call get_data_simple() without parameters
    
    Returns:
        str: Path to the generated CSV file
    """
    # Hardcoded paths - modify these as needed
    excel_file_path = r"C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\processed_data\stock_list.xlsx"
    output_csv_path = r"C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\processed_data\stock_price_data.csv"
    
    return get_data(excel_file_path, output_csv_path)