# Modern Portfolio Optimization System

A refactored and modernized portfolio optimization system for Python 3.12, featuring clean architecture, efficient data handling, and interactive visualizations.

## 🚀 Key Improvements

### ✅ Python 3.12 Compatibility
- **Replaced yfinance** with `yahooquery` for reliable data fetching
- Updated all dependencies for Python 3.12 support
- Modern async-compatible architecture

### ✅ Clean, Modular Architecture
- **Separation of concerns**: Each module has a single responsibility
- **Configuration management**: Centralized settings in `config.py`
- **Type hints**: Full type annotation support
- **Error handling**: Comprehensive exception handling and logging

### ✅ Efficient Data Storage
- **Caching system**: Automatic data caching with expiration
- **Parquet format**: Fast data storage and retrieval
- **Metadata tracking**: Complete data provenance tracking

### ✅ Modern Visualization
- **Interactive charts**: Plotly-based visualizations
- **Dashboard views**: Comprehensive portfolio analysis
- **Export capabilities**: Save charts as HTML files

## 📁 Project Structure

```
portfolio_optimization/
├── config.py              # Configuration settings and fund universe
├── data_fetcher.py         # Modern data fetching with yahooquery
├── analysis.py             # Financial analysis and technical indicators
├── optimizer.py            # Portfolio optimization engine
├── visualization.py        # Interactive visualizations
├── main.py                 # Main execution pipeline
├── requirements.txt        # Dependencies
├── README.md              # This file
├── Portfolio_Optimization_Modern.ipynb  # Jupyter notebook
├── data/                  # Data directories
│   ├── raw/              # Raw downloaded data
│   ├── processed/        # Processed data files
│   └── cache/            # Cached data with metadata
└── output/               # Generated results and charts
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- pip package manager

### Setup

1. **Clone or download the project files**

2. **Create virtual environment** (recommended):
```bash
python -m venv portfolio_env
source portfolio_env/bin/activate  # On Windows: portfolio_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up Jupyter kernel** (optional):
```bash
python -m ipykernel install --user --name=portfolio_env
```

## 🎯 Quick Start

### Option 1: Python Script
```python
from main import run_quick_optimization

# Run with default settings ($300K investment)
results = run_quick_optimization(investment_amount=300000)
```

### Option 2: Jupyter Notebook
```bash
jupyter notebook Portfolio_Optimization_Modern.ipynb
```

### Option 3: Custom Optimization
```python
from main import run_custom_optimization

# Conservative portfolio
results = run_custom_optimization(
    tickers=['VBTLX', 'VTABX', 'VTSAX', 'VFIAX'],
    investment_amount=250000,
    min_weight=0.10,  # Minimum 10% allocation
    max_weight=0.40   # Maximum 40% allocation
)
```

## 📊 Available Funds

The system optimizes across 20 Vanguard funds:

**Core Market Funds:**
- VBTLX - Total Bond Market Index
- VTABX - Total International Bond Index
- VTIAX - Total International Stock Index
- VTSAX - Total Stock Market Index
- VTAPX - Short-Term Inflation-Protected Securities
- VBIAX - Balanced Index Fund
- VFIAX - 500 Index Fund
- VIMAX - Mid-Cap Index Fund
- VSMAX - Small-Cap Index Fund

**Sector ETFs:**
- VDE - Energy
- VAW - Materials
- VIS - Industrials
- VPU - Utilities
- VHT - Health Care
- VFH - Financials
- VDC - Consumer Staples
- VCR - Consumer Discretionary
- VGT - Information Technology
- VOX - Communication Services
- VNQ - Real Estate

## 🔧 Configuration

### Basic Settings (`config.py`)
```python
# Investment parameters
default_investment_amount = 300000.0

# Optimization settings
risk_free_rate = 0.02  # 2% Treasury rate
max_iterations = 1000000
tolerance = 1e-6

# Data settings
default_period = "2y"  # 2 years of historical data
cache_expiry = 1  # 1 day cache expiry
```

### Custom Fund Universe
```python
from config import FundUniverse

# Add custom funds
custom_funds = {
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust"
}

# Or modify existing universe
tickers = FundUniverse.get_tickers()
```

## 📈 Features

### 1. Data Fetching
- **Reliable API**: Uses yahooquery for stable data access
- **Automatic caching**: Reduces API calls and improves performance
- **Error recovery**: Handles API failures gracefully
- **Progress tracking**: Real-time fetch progress

### 2. Financial Analysis
- **Returns calculation**: Daily, annual, and cumulative returns
- **Risk metrics**: Volatility, Sharpe ratio, maximum drawdown
- **Technical indicators**: Moving averages, Bollinger Bands, RSI
- **Correlation analysis**: Fund relationship analysis

### 3. Portfolio Optimization
- **Sharpe ratio maximization**: Modern portfolio theory implementation
- **Constraints support**: Min/max weight constraints
- **Multiple algorithms**: SLSQP, L-BFGS-B optimization methods
- **Efficient frontier**: Risk-return optimization curve

### 4. Visualization
- **Interactive charts**: Plotly-based dynamic visualizations
- **Portfolio allocation**: Pie charts and bar charts
- **Risk-return scatter**: Fund positioning analysis
- **Correlation heatmap**: Fund relationship visualization
- **Performance comparison**: Historical performance tracking
- **Dashboard view**: Comprehensive portfolio overview

### 5. Results Export
- **JSON format**: Machine-readable optimization results
- **CSV exports**: Fund statistics and correlation matrices
- **HTML charts**: Shareable interactive visualizations
- **Comprehensive logging**: Detailed execution logs

## 🔍 Example Outputs

### Optimization Results
```
🎯 PORTFOLIO OPTIMIZATION RESULTS
====================================
📊 Sharpe Ratio: 1.8654
📈 Annual Return: 32.12%
📉 Annual Volatility: 16.15%
💰 Total Investment: $300,000

🏆 TOP FUND ALLOCATIONS:
  • VTIAX: 30.3% ($90,900)
  • VGT: 21.9% ($65,700)
  • VIS: 21.5% ($64,500)
  • VBTLX: 14.7% ($44,100)
  • VTABX: 3.4% ($10,200)
```

### Risk Analysis
```
📊 DIVERSIFICATION METRICS:
  • Number of funds: 5
  • Max allocation: 30.3%
  • Concentration ratio: 0.247
  
🔗 CORRELATION INSIGHTS:
  • Average correlation: 0.456
  • Highest correlation: VGT-VCR (0.865)
  • Lowest correlation: VBTLX-VDE (-0.272)
```

## 🎛️ Advanced Usage

### Custom Optimization Constraints
```python
from optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
optimizer.prepare_data(funds_data)

# Sector-neutral constraints
result = optimizer.optimize_portfolio(
    investment_amount=500000,
    min_weight=0.05,    # Minimum 5% per fund
    max_weight=0.25     # Maximum 25% per fund
)
```

### Multiple Time Periods
```python
from data_fetcher import DataFetcher
from datetime import datetime, timedelta

fetcher = DataFetcher()

# 5-year analysis
funds_data = fetcher.fetch_multiple_funds(
    tickers=['VTSAX', 'VTIAX', 'VBTLX'],
    start_date=datetime.now() - timedelta(days=5*365),
    end_date=datetime.now()
)
```

### Batch Processing
```python
# Test multiple investment amounts
investment_amounts = [100000, 250000, 500000, 1000000]
results = {}

for amount in investment_amounts:
    results[amount] = run_quick_optimization(amount)
```

## 🐛 Troubleshooting

### Common Issues

**1. yahooquery Data Errors**
```python
# Clear cache and retry
from data_fetcher import DataFetcher
fetcher = DataFetcher()
fetcher.clear_cache()
```

**2. Optimization Convergence Issues**
```python
# Adjust optimization parameters
config.optimization.max_iterations = 2000000
config.optimization.tolerance = 1e-8
```

**3. Memory Issues with Large Datasets**
```python
# Use smaller date ranges
config.analysis_start_date = datetime.now() - timedelta(days=365)
```

**4. Visualization Display Issues**
```python
# For Jupyter notebooks
import plotly.io as pio
pio.renderers.default = "notebook"
```

## 📚 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Optimization algorithms
- **yahooquery**: Financial data API (yfinance replacement)

### Visualization
- **plotly**: Interactive charts
- **matplotlib**: Static plots (backup)

### Utilities
- **python-dateutil**: Date parsing
- **requests**: HTTP requests
- **openpyxl**: Excel file support
- **pyarrow**: Parquet file format

### Development
- **jupyter**: Notebook interface
- **black**: Code formatting
- **pytest**: Testing framework

## 🔄 Migration from Old System

### Key Changes
1. **Replace yfinance imports**: 
   ```python
   # Old
   import yfinance as yf
   
   # New
   from data_fetcher import DataFetcher
   ```

2. **Update data fetching**:
   ```python
   # Old
   data = yf.download(ticker, start=start_date, end=end_date)
   
   # New
   fetcher = DataFetcher()
   data = fetcher.fetch_single_fund(ticker, start_date, end_date)
   ```

3. **Use new configuration**:
   ```python
   # Old
   tickers = pd.read_csv('tickers.csv').columns
   
   # New
   from config import FundUniverse
   tickers = FundUniverse.get_tickers()
   ```

### Data Migration
```python
# Convert old CSV data to new format
from pathlib import Path
import pandas as pd

# Load old data
old_data = pd.read_csv('old_fund_data.csv')

# Save in new format
output_dir = Path('data/processed')
old_data.to_parquet(output_dir / 'historical_data.parquet')
```

## 📄 License & Disclaimer

This project is for educational purposes only. It is not financial advice.

**Important Notes:**
- Past performance does not guarantee future results
- Consider consulting with a financial advisor
- Review fund prospectuses before investing
- Understand all fees and tax implications

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional optimization algorithms
- More asset classes (cryptocurrencies, commodities)
- Advanced risk metrics (VaR, CVaR)
- Real-time data streaming
- Portfolio rebalancing automation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example notebook
3. Check the logs in `portfolio_optimization.log`
4. Ensure all dependencies are properly installed