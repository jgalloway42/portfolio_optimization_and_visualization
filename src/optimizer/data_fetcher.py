"""
Modern data fetching module using yahooquery as yfinance alternative.
Includes caching, error handling, and efficient data storage.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import time
import logging
from dataclasses import dataclass
import json

# Using yahooquery instead of yfinance for Python 3.12 compatibility
from yahooquery import Ticker

from config import config


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FundData:
    """Data class for storing fund information and price data."""
    ticker: str
    description: str
    data: pd.DataFrame
    last_updated: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'description': self.description,
            'last_updated': self.last_updated.isoformat(),
            'data_shape': self.data.shape
        }


class DataFetcher:
    """
    Modern data fetching class with caching and error handling.
    Uses yahooquery for Python 3.12 compatibility.
    """
    
    def __init__(self):
        self.cache_dir = config.data.cache_dir
        self.processed_dir = config.data.processed_data_dir
        self.funds_data: Dict[str, FundData] = {}
    
    def fetch_single_fund(self, ticker: str, 
                         start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None,
                         period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Fetch data for a single fund using yahooquery.
        
        Args:
            ticker: Fund ticker symbol
            start_date: Start date for data (optional)
            end_date: End date for data (optional)  
            period: Period string if dates not provided (default: "2y")
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            logger.info(f"Fetching data for {ticker}")
            
            # Create ticker object
            ticker_obj = Ticker(ticker)
            
            # Fetch historical data
            if start_date and end_date:
                data = ticker_obj.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
            else:
                data = ticker_obj.history(period=period, interval='1d')
            
            # Check if data is valid
            if data is None or data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
                
            # Handle multi-index if present
            if isinstance(data.index, pd.MultiIndex):
                data = data.droplevel(0)
            
            # Ensure we have the expected columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns.str.lower() for col in required_cols):
                logger.warning(f"Missing required columns for {ticker}")
                return None
            
            # Standardize column names
            data.columns = [col.title() for col in data.columns]
            
            # Clean the data
            data = data.dropna()
            data = data[data['Close'] > 0]  # Remove invalid prices
            
            # Add delay to avoid rate limiting
            time.sleep(config.data.request_delay)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def fetch_fund_info(self, ticker: str) -> str:
        """
        Fetch fund description/name.
        
        Args:
            ticker: Fund ticker symbol
            
        Returns:
            Fund description or fallback name
        """
        try:
            ticker_obj = Ticker(ticker)
            info = ticker_obj.summary_detail
            
            if ticker in info and 'longName' in info[ticker]:
                return info[ticker]['longName']
            else:
                # Fallback to config description
                return config.funds.get_fund_info(ticker)
                
        except Exception as e:
            logger.warning(f"Could not fetch info for {ticker}: {str(e)}")
            return config.funds.get_fund_info(ticker)
    
    def fetch_multiple_funds(self, tickers: List[str], 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           period: str = "2y",
                           use_cache: bool = True) -> Dict[str, FundData]:
        """
        Fetch data for multiple funds with progress tracking.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            period: Period string if dates not provided
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping tickers to FundData objects
        """
        funds_data = {}
        total_funds = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{total_funds})")
            
            # Check cache first
            if use_cache:
                cached_data = self._load_from_cache(ticker)
                if cached_data is not None:
                    funds_data[ticker] = cached_data
                    continue
            
            # Fetch fresh data
            data = self.fetch_single_fund(ticker, start_date, end_date, period)
            
            if data is not None:
                # Get fund description
                description = self.fetch_fund_info(ticker)
                
                # Create FundData object
                fund_data = FundData(
                    ticker=ticker,
                    description=description,
                    data=data,
                    last_updated=datetime.now()
                )
                
                funds_data[ticker] = fund_data
                
                # Cache the data
                self._save_to_cache(fund_data)
            else:
                logger.warning(f"Failed to fetch data for {ticker}")
        
        self.funds_data = funds_data
        return funds_data
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get cache file path for a ticker."""
        return self.cache_dir / f"{ticker}_data.parquet"
    
    def _get_metadata_path(self, ticker: str) -> Path:
        """Get metadata file path for a ticker."""
        return self.cache_dir / f"{ticker}_metadata.json"
    
    def _save_to_cache(self, fund_data: FundData) -> None:
        """Save fund data to cache."""
        try:
            # Save data as parquet for efficiency
            data_path = self._get_cache_path(fund_data.ticker)
            fund_data.data.to_parquet(data_path)
            
            # Save metadata as JSON
            metadata_path = self._get_metadata_path(fund_data.ticker)
            metadata = {
                'ticker': fund_data.ticker,
                'description': fund_data.description,
                'last_updated': fund_data.last_updated.isoformat(),
                'data_shape': fund_data.data.shape
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache data for {fund_data.ticker}: {str(e)}")
    
    def _load_from_cache(self, ticker: str) -> Optional[FundData]:
        """Load fund data from cache if available and recent."""
        try:
            data_path = self._get_cache_path(ticker)
            metadata_path = self._get_metadata_path(ticker)
            
            if not (data_path.exists() and metadata_path.exists()):
                return None
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is recent (less than 1 day old)
            last_updated = datetime.fromisoformat(metadata['last_updated'])
            if datetime.now() - last_updated > timedelta(days=1):
                logger.info(f"Cache for {ticker} is stale, fetching fresh data")
                return None
            
            # Load data
            data = pd.read_parquet(data_path)
            
            return FundData(
                ticker=metadata['ticker'],
                description=metadata['description'],
                data=data,
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {ticker}: {str(e)}")
            return None
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def save_to_processed(self, filename: str = "fund_data.parquet") -> None:
        """Save all fund data to processed directory."""
        if not self.funds_data:
            logger.warning("No data to save")
            return
        
        try:
            # Combine all close prices into a single DataFrame
            close_prices = {}
            for ticker, fund_data in self.funds_data.items():
                close_prices[ticker] = fund_data.data['Close']
            
            combined_df = pd.DataFrame(close_prices)
            
            # Save to processed directory
            save_path = self.processed_dir / filename
            combined_df.to_parquet(save_path)
            
            # Save metadata
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'tickers': list(self.funds_data.keys()),
                'fund_descriptions': {ticker: fund.description 
                                    for ticker, fund in self.funds_data.items()},
                'data_shape': combined_df.shape
            }
            
            metadata_path = self.processed_dir / "fund_data_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Data saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")


def fetch_all_vanguard_funds(use_cache: bool = True, 
                           period: str = "2y") -> Dict[str, FundData]:
    """
    Convenience function to fetch all Vanguard funds.
    
    Args:
        use_cache: Whether to use cached data
        period: Data period to fetch
        
    Returns:
        Dictionary of fund data
    """
    fetcher = DataFetcher()
    tickers = config.funds.get_tickers()
    
    return fetcher.fetch_multiple_funds(
        tickers=tickers,
        period=period,
        use_cache=use_cache
    )