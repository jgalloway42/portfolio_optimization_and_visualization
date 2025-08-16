"""
Configuration settings for the portfolio optimization system.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import json


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization parameters."""
    risk_free_rate: float = 0.02  # 10-year Treasury yield
    max_iterations: int = 1000000
    tolerance: float = 1e-6
    optimization_method: str = 'SLSQP'
    

@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    
    # Data fetching parameters
    default_period: str = "2y"  # Default data period
    max_retries: int = 3
    request_delay: float = 0.1  # Delay between API requests
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.data_dir, self.raw_data_dir, 
                        self.processed_data_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class FundUniverse:
    """Manages the universe of funds available for optimization."""
    
    VANGUARD_FUNDS = {
        # Core Market Funds
        "VBTLX": "Vanguard Total Bond Market Index Fund Admiral Shares",
        "VTABX": "Vanguard Total International Bond Index Fund Admiral Shares",
        "VTIAX": "Vanguard Total International Stock Index Fund Admiral Shares",
        "VTSAX": "Vanguard Total Stock Market Index Fund Admiral Shares",
        "VTAPX": "Vanguard Short-Term Inflation-Protected Securities Index Fund Admiral Shares",
        "VBIAX": "Vanguard Balanced Index Fund Admiral Shares",
        "VFIAX": "Vanguard 500 Index Fund Admiral Shares",
        "VIMAX": "Vanguard Mid-Cap Index Fund Admiral Shares",
        "VSMAX": "Vanguard Small-Cap Index Fund Admiral Shares",
        
        # Sector ETFs
        "VDE": "Vanguard Energy Index Fund ETF Shares",
        "VAW": "Vanguard Materials Index Fund ETF Shares",
        "VIS": "Vanguard Industrials Index Fund ETF Shares",
        "VPU": "Vanguard Utilities Index Fund ETF Shares",
        "VHT": "Vanguard Health Care Index Fund ETF Shares",
        "VFH": "Vanguard Financials Index Fund ETF Shares",
        "VDC": "Vanguard Consumer Staples Index Fund ETF Shares",
        "VCR": "Vanguard Consumer Discretionary Index Fund ETF Shares",
        "VGT": "Vanguard Information Technology Index Fund ETF Shares",
        "VOX": "Vanguard Communication Services Index Fund ETF Shares",
        "VNQ": "Vanguard Real Estate Index Fund ETF Shares",
    }
    
    @classmethod
    def get_tickers(cls) -> List[str]:
        """Get list of all ticker symbols."""
        return list(cls.VANGUARD_FUNDS.keys())
    
    @classmethod
    def get_descriptions(cls) -> Dict[str, str]:
        """Get dictionary mapping tickers to descriptions."""
        return cls.VANGUARD_FUNDS.copy()
    
    @classmethod
    def get_fund_info(cls, ticker: str) -> str:
        """Get description for a specific ticker."""
        return cls.VANGUARD_FUNDS.get(ticker, f"Unknown fund: {ticker}")
    
    @classmethod
    def save_to_json(cls, filepath: Path) -> None:
        """Save fund universe to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(cls.VANGUARD_FUNDS, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: Path) -> Dict[str, str]:
        """Load fund universe from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


class Config:
    """Main configuration class that combines all settings."""
    
    def __init__(self):
        self.optimization = OptimizationConfig()
        self.data = DataConfig()
        self.funds = FundUniverse()
        
        # Analysis parameters
        self.analysis_start_date = datetime.now() - timedelta(weeks=2*52)
        self.analysis_end_date = datetime.now()
        
        # Portfolio parameters
        self.default_investment_amount = 300000.0  # dollars
        
    def update_dates(self, start_date: datetime = None, end_date: datetime = None):
        """Update analysis date range."""
        if start_date:
            self.analysis_start_date = start_date
        if end_date:
            self.analysis_end_date = end_date
    
    def get_analysis_period_years(self) -> float:
        """Calculate analysis period in years."""
        delta = self.analysis_end_date - self.analysis_start_date
        return delta.total_seconds() / (3600 * 24 * 365.25)


# Global configuration instance
config = Config()