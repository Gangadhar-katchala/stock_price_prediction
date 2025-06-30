"""
Data processing utilities for the stock portfolio project.
Provides the DataProcessor class for loading, cleaning, processing, and saving financial data.
"""
import pandas as pd
import os
import sys
from mlproject.logger import logging
from exception import CustomException

class DataProcessor:
    """
    Loads, cleans, processes, and saves financial data for the stock portfolio project.
    """
    def __init__(self, data_path=None):
        """
        Initialize DataProcessor with the path to financial data files
        
        Args:
            data_path (str): Path to the directory containing financial data CSV files
                           Default: OneDrive Documents/data path
        """
        if data_path is None:
            self.data_path = r"C:\Users\katch\OneDrive\Documents\data"
        else:
            self.data_path = data_path
        
        self.quarterly_result_data = None
        self.cash_flow_data = None
        self.profit_loss_data = None
        self.balance_sheet_data = None
        self.shareholding_data = None
        
    def load_data(self):
        """
        Load all financial data CSV files from the data path
        """
        try:
            logging.info("Loading financial data files...")
            
            # Load quarterly results data
            quarterly_path = os.path.join(self.data_path, 'quarterly_data.csv')
            if os.path.exists(quarterly_path):
                self.quarterly_result_data = pd.read_csv(quarterly_path)
                logging.info(f"Loaded quarterly_data.csv with {len(self.quarterly_result_data)} rows")
            else:
                raise FileNotFoundError(f"quarterly_data.csv not found at {quarterly_path}")
            
            # Load cash flow data
            cash_flow_path = os.path.join(self.data_path, 'cash_flow_data.csv')
            if os.path.exists(cash_flow_path):
                self.cash_flow_data = pd.read_csv(cash_flow_path)
                logging.info(f"Loaded cash_flow_data.csv with {len(self.cash_flow_data)} rows")
            else:
                raise FileNotFoundError(f"cash_flow_data.csv not found at {cash_flow_path}")
            
            # Load profit loss data
            profit_loss_path = os.path.join(self.data_path, 'profit_loss_data.csv')
            if os.path.exists(profit_loss_path):
                self.profit_loss_data = pd.read_csv(profit_loss_path)
                logging.info(f"Loaded profit_loss_data.csv with {len(self.profit_loss_data)} rows")
            else:
                raise FileNotFoundError(f"profit_loss_data.csv not found at {profit_loss_path}")
            
            # Load balance sheet data
            balance_sheet_path = os.path.join(self.data_path, 'balance_sheet_data.csv')
            if os.path.exists(balance_sheet_path):
                self.balance_sheet_data = pd.read_csv(balance_sheet_path)
                logging.info(f"Loaded balance_sheet_data.csv with {len(self.balance_sheet_data)} rows")
            else:
                raise FileNotFoundError(f"balance_sheet_data.csv not found at {balance_sheet_path}")
            
            # Load shareholding data
            shareholding_path = os.path.join(self.data_path, 'shareholding_data.csv')
            if os.path.exists(shareholding_path):
                self.shareholding_data = pd.read_csv(shareholding_path)
                logging.info(f"Loaded shareholding_data.csv with {len(self.shareholding_data)} rows")
            else:
                raise FileNotFoundError(f"shareholding_data.csv not found at {shareholding_path}")
                
            logging.info("All financial data files loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise CustomException(e, sys)
    
    def clean_column_names(self):
        """
        Clean column names by removing special characters and whitespace
        Same logic as in Model_Training notebook
        """
        try:
            logging.info("Cleaning column names...")
            
            # Clean quarterly_result_data columns
            if self.quarterly_result_data is not None:
                self.quarterly_result_data.columns = self.quarterly_result_data.columns.str.replace('\xa0', '', regex=False)
                self.quarterly_result_data.columns = self.quarterly_result_data.columns.str.replace('+', '', regex=False)
                self.quarterly_result_data.columns = self.quarterly_result_data.columns.str.strip()
            
            # Clean shareholding_data columns
            if self.shareholding_data is not None:
                self.shareholding_data.columns = self.shareholding_data.columns.str.replace('\xa0', '', regex=False)
                self.shareholding_data.columns = self.shareholding_data.columns.str.replace('+', '', regex=False)
                self.shareholding_data.columns = self.shareholding_data.columns.str.strip()
            
            # Clean balance_sheet_data columns
            if self.balance_sheet_data is not None:
                self.balance_sheet_data.columns = self.balance_sheet_data.columns.str.replace('+', '', regex=False)
                self.balance_sheet_data.columns = self.balance_sheet_data.columns.str.strip()
            
            # Clean profit_loss_data columns
            if self.profit_loss_data is not None:
                self.profit_loss_data.columns = self.profit_loss_data.columns.str.replace('+', '', regex=False)
                self.profit_loss_data.columns = self.profit_loss_data.columns.str.strip()
            
            # Clean cash_flow_data columns
            if self.cash_flow_data is not None:
                self.cash_flow_data.columns = self.cash_flow_data.columns.str.replace('+', '', regex=False)
                self.cash_flow_data.columns = self.cash_flow_data.columns.str.strip()
            
            logging.info("Column names cleaned successfully")
            
        except Exception as e:
            logging.error(f"Error cleaning column names: {str(e)}")
            raise CustomException(e, sys)
    
    def process_data(self):
        """
        Process and merge data to create quarterly_data and annual_data
        Same logic as in Model_Training notebook
        """
        try:
            logging.info("Processing and merging data...")
            
            # Convert Period columns to datetime
            if self.quarterly_result_data is not None:
                self.quarterly_result_data['Period'] = pd.to_datetime(self.quarterly_result_data['Period'], errors='coerce')
            
            if self.shareholding_data is not None:
                self.shareholding_data['Period'] = pd.to_datetime(self.shareholding_data['Period'], errors='coerce')
            
            if self.balance_sheet_data is not None:
                self.balance_sheet_data['Period'] = pd.to_datetime(self.balance_sheet_data['Period'], errors='coerce')
            
            if self.profit_loss_data is not None:
                self.profit_loss_data['Period'] = pd.to_datetime(self.profit_loss_data['Period'], errors='coerce')
            
            if self.cash_flow_data is not None:
                self.cash_flow_data['Period'] = pd.to_datetime(self.cash_flow_data['Period'], errors='coerce')
            
            # Handle Sales/Revenue column mapping (same as notebook)
            if self.quarterly_result_data is not None:
                self.quarterly_result_data['Sales'] = self.quarterly_result_data['Sales'].fillna(self.quarterly_result_data['Revenue'])
            
            # Create quarterly_data (same as Cell 32 in notebook)
            quarterly_data = pd.merge(
                self.quarterly_result_data[['Period', 'Stock', 'Sales', 'Net Profit', 'EPS in Rs']],
                self.shareholding_data[['Period', 'Stock', 'Promoters', 'FIIs', 'DIIs', 'Government', 'Public']],
                on=['Period', 'Stock'],
                how='inner'
            )
            
            # Create annual_data (same as Cell 41 in notebook)
            merged1 = pd.merge(
                self.balance_sheet_data[['Period', 'Stock', 'Total Assets', 'Investments', 'Borrowings']],
                self.profit_loss_data[['Period', 'Stock', 'Sales', 'Net Profit', 'EPS in Rs', 'Dividend Payout %']],
                on=['Period', 'Stock'],
                how='inner'
            )
            
            annual_data = pd.merge(
                merged1,
                self.cash_flow_data[['Period', 'Stock', 'Net Cash Flow']],
                on=['Period', 'Stock'],
                how='inner'
            )
            
            logging.info(f"Created quarterly_data with {len(quarterly_data)} rows")
            logging.info(f"Created annual_data with {len(annual_data)} rows")
            
            return quarterly_data, annual_data
            
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise CustomException(e, sys)
    
    def save_data(self, quarterly_data, annual_data, output_path=None):
        """
        Save processed data to CSV files
        
        Args:
            quarterly_data (pd.DataFrame): Processed quarterly data
            annual_data (pd.DataFrame): Processed annual data
            output_path (str): Path to save output files (default: notebooks/Data/processed_data)
        """
        try:
            if output_path is None:
                output_path = r"C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\processed_data"
            
            # Create directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Save quarterly_data
            quarterly_output_path = os.path.join(output_path, 'quarterly_data.csv')
            quarterly_data.to_csv(quarterly_output_path, index=False)
            logging.info(f"Saved quarterly_data.csv to {quarterly_output_path}")
            
            # Save annual_data
            annual_output_path = os.path.join(output_path, 'annual_data.csv')
            annual_data.to_csv(annual_output_path, index=False)
            logging.info(f"Saved annual_data.csv to {annual_output_path}")
            
        except Exception as e:
            logging.error(f"Error saving data: {str(e)}")
            raise CustomException(e, sys)
    
    def run_pipeline(self, output_path=None):
        """
        Run the complete data processing pipeline
        
        Args:
            output_path (str): Path to save output files (default: notebooks/Data/processed_data)
        
        Returns:
            tuple: (quarterly_data, annual_data) DataFrames
        """
        try:
            logging.info("Starting data processing pipeline...")
            
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Clean column names
            self.clean_column_names()
            
            # Step 3: Process and merge data
            quarterly_data, annual_data = self.process_data()
            
            # Step 4: Save data
            self.save_data(quarterly_data, annual_data, output_path)
            
            logging.info("Data processing pipeline completed successfully")
            
            return quarterly_data, annual_data
            
        except Exception as e:
            logging.error(f"Error in data processing pipeline: {str(e)}")
            raise CustomException(e, sys)

def main():
    """
    Main function to run the data processor
    """
    try:
        # Initialize data processor
        processor = DataProcessor()
        
        # Run the complete pipeline
        quarterly_data, annual_data = processor.run_pipeline()
        
        print("Data processing completed successfully!")
        print(f"Quarterly data shape: {quarterly_data.shape}")
        print(f"Annual data shape: {annual_data.shape}")
        
        # Display sample data
        print("\nQuarterly data sample:")
        print(quarterly_data.head())
        
        print("\nAnnual data sample:")
        print(annual_data.head())
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

# Usage example:
# from mlproject.data_processing import DataProcessor
# processor = DataProcessor()
# quarterly_data, annual_data = processor.run_pipeline()
