"""
Financial data scraping utilities for the stock portfolio project.
Provides classes to scrape and organize financial data from Screener.in.
"""
import os
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from mlproject.logger import logging
from exception import CustomException

class FinancialDataScraper:
    """
    Scrapes financial data for a list of stocks from Screener.in and saves as CSV files.
    """
    DATA_PATH = r"C:\Users\katch\Desktop\projects\stock_portfolio\notebooks\Data\un_processed_data"
    SECTION_MAP = {
        'quarters': 'quarters',
        'profit-loss': 'profit-loss',
        'balance-sheet': 'balance-sheet',
        'cash-flow': 'cash-flow',
        'shareholding': 'shareholding'
    }
    USER_AGENTS = [
        {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...'},
        {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...'},
        {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)...'}
    ]

    def __init__(self, stock_list, data_path=None):
        """
        Initialize the scraper with a list of stock symbols and optional data path.
        """
        self.stock_list = [stock.replace(".NS", "") for stock in stock_list]
        self.failed_stocks = []
        self.DATA_PATH = data_path if data_path else self.DATA_PATH
        os.makedirs(self.DATA_PATH, exist_ok=True)

    def _get_soup_with_retry(self, stock, retries=5):
        """
        Attempt to fetch and parse the HTML for a stock's Screener.in page, with retries on failure.
        """
        url = f"https://www.screener.in/company/{stock}/consolidated/"
        for i in range(retries):
            try:
                headers = random.choice(self.USER_AGENTS)
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 429:
                    wait = 2 ** i
                    logging.warning(f"Rate limited for {stock}. Retrying in {wait} sec")
                    time.sleep(wait)
                else:
                    response.raise_for_status()
                    return BeautifulSoup(response.content, 'html.parser')
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed for {stock}: {str(e)}")
                time.sleep(2 ** i)
        self.failed_stocks.append(stock)
        return None

    def _extract_table_data(self, soup, section_id, stock):
        """
        Extracts table data from a given section of the HTML soup for a stock.
        """
        section = soup.find('section', {'id': section_id}) if soup else None
        if not section:
            logging.warning(f"Section '{section_id}' not found for {stock}")
            self.failed_stocks.append(stock)
            return None

        table = section.find('table')
        if not table:
            logging.warning(f"No table in section '{section_id}' for {stock}")
            self.failed_stocks.append(stock)
            return None

        try:
            df = pd.read_html(str(table))[0]
            if section_id == 'balance-sheet':
                df = df.set_index(df.columns[0], drop=True)
                df_transposed = df.T.reset_index()
                df_transposed.rename(columns={'index': 'Period'}, inplace=True)
            else:
                df_transposed = df.set_index(df.columns[0]).T.reset_index()
                df_transposed.rename(columns={'index': 'Period'}, inplace=True)

            df_transposed['Stock'] = stock  # Changed from Ticker to Stock
            return df_transposed
        except Exception as e:
            logging.error(f"Table error for {stock} ({section_id}): {str(e)}")
            self.failed_stocks.append(stock)
            return None

    def _scrape_financial_section(self, section_key, filename):
        """
        Scrapes a specific financial section for all stocks and saves the result as a CSV.
        """
        section_id = self.SECTION_MAP[section_key]
        all_data = []

        for stock in self.stock_list:
            soup = self._get_soup_with_retry(stock)
            df_stock = self._extract_table_data(soup, section_id, stock)
            if df_stock is not None:
                all_data.append(df_stock)
            time.sleep(random.uniform(3, 6))  # Throttle requests

        if not all_data:
            return pd.DataFrame()

        final_df = pd.concat(all_data, ignore_index=True)
        if 'Period' in final_df.columns:
            try:
                final_df['Period'] = pd.to_datetime(final_df['Period'], format='%b %Y', errors='coerce')
                final_df.sort_values(by=['Period', 'Stock'], inplace=True)
            except:
                final_df.sort_values(by=['Stock', 'Period'], inplace=True)

        final_df.reset_index(drop=True, inplace=True)

        csv_path = os.path.join(self.DATA_PATH, f"{filename}.csv")
        final_df.to_csv(csv_path, index=False,encoding='utf-8')
        logging.info(f"Saved: {csv_path}")

        # Save failed tickers
        if self.failed_stocks:
            fail_log = os.path.join(self.DATA_PATH, f"failed_{filename}.txt")
            with open(fail_log, 'w') as f:
                f.write('\n'.join(self.failed_stocks))
            logging.warning(f"Failed stocks saved to {fail_log}")

        return final_df

    def quarterly_data(self):
        """Scrape and return quarterly financial data for all stocks."""
        return self._scrape_financial_section('quarters', 'quarterly_data')

    def profit_loss_data(self):
        """Scrape and return profit/loss data for all stocks."""
        return self._scrape_financial_section('profit-loss', 'profit_loss_data')

    def balance_sheet_data(self):
        """Scrape and return balance sheet data for all stocks."""
        return self._scrape_financial_section('balance-sheet', 'balance_sheet_data')

    def cash_flow_data(self):
        """Scrape and return cash flow data for all stocks."""
        return self._scrape_financial_section('cash-flow', 'cash_flow_data')

    def shareholding_data(self):
        """Scrape and return shareholding data for all stocks."""
        return self._scrape_financial_section('shareholding', 'shareholding_data')

    def get_all_data(self):
        """Scrape and return all available financial data for all stocks as a dictionary."""
        return {
            "quarterly": self.quarterly_data(),
            "profit_loss": self.profit_loss_data(),
            "balance_sheet": self.balance_sheet_data(),
            "cash_flow": self.cash_flow_data(),
            "shareholding": self.shareholding_data(),
        }

class GetFinancialData:
    """
    Interface for retrieving different types of financial data using FinancialDataScraper.
    """
    def __init__(self, stock_list, data_type='all_data', data_path=None):
        """
        Initialize with a stock list, data type, and optional data path.
        """
        self.scraper = FinancialDataScraper(stock_list, data_path=data_path)
        self.data = None

        if data_type == 'financial_data':
            self.data = self.scraper.quarterly_data()
        elif data_type == 'quarterly_data':
            self.data = self.scraper.quarterly_data()
        elif data_type == 'profit_loss_data':
            self.data = self.scraper.profit_loss_data()
        elif data_type == 'balance_sheet_data':
            self.data = self.scraper.balance_sheet_data()
        elif data_type == 'cash_flow_data':
            self.data = self.scraper.cash_flow_data()
        elif data_type == 'shareholding_data':
            self.data = self.scraper.shareholding_data()
        elif data_type == 'all_data':
            self.data = self.scraper.get_all_data()
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

    def get_data(self):
        """Return the requested financial data."""
        return self.data
