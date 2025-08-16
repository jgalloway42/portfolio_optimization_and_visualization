"""
Financial analysis functions for portfolio optimization.
Refactored and modernized from original StockAnalysisFunctions2.py
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import logging

from data_fetcher import FundData

logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """
    Modern financial analysis class with clean, efficient methods.
    """
    
    def __init__(self):
        self.returns_data: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
    def calculate_returns(self, funds_data: Dict[str, FundData]) -> pd.DataFrame:
        """
        Calculate daily returns for all funds.
        
        Args:
            funds_data: Dictionary of FundData objects
            
        Returns:
            DataFrame with daily returns for each fund
        """
        returns_dict = {}
        
        for ticker, fund_data in funds_data.items():
            if fund_data.data is not None and not fund_data.data.empty:
                # Calculate daily returns
                prices = fund_data.data['Close']
                daily_returns = prices.pct_change().dropna()
                returns_dict[ticker] = daily_returns
        
        # Combine into single DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Drop rows where any fund has missing data
        returns_df = returns_df.dropna()
        
        self.returns_data = returns_df
        return returns_df
    
    def calculate_annual_statistics(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate annualized return and volatility statistics.
        
        Args:
            returns_df: DataFrame of daily returns
            
        Returns:
            DataFrame with annual statistics
        """
        stats = pd.DataFrame(index=returns_df.columns)
        
        # Annualized returns (assuming 252 trading days)
        stats['annual_return'] = returns_df.mean() * 252
        
        # Annualized volatility
        stats['annual_volatility'] = returns_df.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate from config)
        from config import config
        risk_free_rate = config.optimization.risk_free_rate
        stats['sharpe_ratio'] = (stats['annual_return'] - risk_free_rate) / stats['annual_volatility']
        
        # Additional statistics
        stats['max_drawdown'] = self._calculate_max_drawdown(returns_df)
        stats['skewness'] = returns_df.skew()
        stats['kurtosis'] = returns_df.kurtosis()
        
        return stats
    
    def _calculate_max_drawdown(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate maximum drawdown for each asset."""
        max_drawdowns = {}
        
        for column in returns_df.columns:
            # Calculate cumulative returns
            cum_returns = (1 + returns_df[column]).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max
            
            # Get maximum drawdown (most negative value)
            max_drawdowns[column] = drawdown.min()
        
        return pd.Series(max_drawdowns)
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix of returns.
        
        Args:
            returns_df: DataFrame of daily returns
            
        Returns:
            Correlation matrix
        """
        corr_matrix = returns_df.corr()
        self.correlation_matrix = corr_matrix
        return corr_matrix
    
    def calculate_covariance_matrix(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate covariance matrix for optimization.
        
        Args:
            returns_df: DataFrame of daily returns
            
        Returns:
            Covariance matrix as numpy array
        """
        # Annualize the covariance matrix
        return returns_df.cov().values * 252
    
    def add_technical_indicators(self, fund_data: FundData) -> pd.DataFrame:
        """
        Add technical indicators to price data.
        
        Args:
            fund_data: FundData object
            
        Returns:
            DataFrame with added technical indicators
        """
        df = fund_data.data.copy()
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (2 * rolling_std)
        df['BB_lower'] = df['BB_middle'] - (2 * rolling_std)
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Daily returns and cumulative returns
        df['daily_return'] = df['Close'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """
        Create an interactive correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Fund Correlation Matrix',
            xaxis_title='Funds',
            yaxis_title='Funds',
            width=800,
            height=800
        )
        
        return fig
    
    def create_returns_comparison(self, funds_data: Dict[str, FundData], 
                                tickers: List[str] = None) -> go.Figure:
        """
        Create cumulative returns comparison chart.
        
        Args:
            funds_data: Dictionary of FundData objects
            tickers: List of tickers to plot (if None, plot all)
            
        Returns:
            Plotly figure
        """
        if tickers is None:
            tickers = list(funds_data.keys())
        
        fig = go.Figure()
        
        for ticker in tickers:
            if ticker in funds_data:
                fund_data = funds_data[ticker]
                df_with_indicators = self.add_technical_indicators(fund_data)
                
                fig.add_trace(go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['cumulative_return'],
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_price_chart_with_indicators(self, fund_data: FundData) -> go.Figure:
        """
        Create comprehensive price chart with technical indicators.
        
        Args:
            fund_data: FundData object
            
        Returns:
            Plotly figure with subplots
        """
        df = self.add_technical_indicators(fund_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{fund_data.ticker} Price and Moving Averages', 
                          'Volume', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price and moving averages
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines', name='Close Price',
            line=dict(color='black', width=2)
        ), row=1, col=1)
        
        # Moving averages
        for ma, color in [('SMA_20', 'blue'), ('SMA_50', 'orange'), ('SMA_200', 'red')]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ma],
                mode='lines', name=ma,
                line=dict(color=color, width=1)
            ), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_upper'],
            mode='lines', name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_lower'],
            mode='lines', name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            name='Volume', marker_color='lightblue'
        ), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            mode='lines', name='RSI',
            line=dict(color='purple', width=2)
        ), row=3, col=1)
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Oversold", row=3, col=1)
        
        fig.update_layout(
            title=f'{fund_data.ticker} - {fund_data.description}',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_analysis_summary(self, funds_data: Dict[str, FundData]) -> Dict:
        """
        Generate comprehensive analysis summary.
        
        Args:
            funds_data: Dictionary of FundData objects
            
        Returns:
            Dictionary with analysis summary
        """
        if not funds_data:
            return {}
        
        # Calculate returns and statistics
        returns_df = self.calculate_returns(funds_data)
        stats_df = self.calculate_annual_statistics(returns_df)
        corr_matrix = self.calculate_correlation_matrix(returns_df)
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_funds': len(funds_data),
            'data_period': {
                'start': returns_df.index.min().isoformat(),
                'end': returns_df.index.max().isoformat(),
                'trading_days': len(returns_df)
            },
            'best_performers': {
                'by_return': stats_df.nlargest(5, 'annual_return')['annual_return'].to_dict(),
                'by_sharpe': stats_df.nlargest(5, 'sharpe_ratio')['sharpe_ratio'].to_dict(),
                'by_low_volatility': stats_df.nsmallest(5, 'annual_volatility')['annual_volatility'].to_dict()
            },
            'correlation_insights': {
                'highest_correlation': self._find_highest_correlations(corr_matrix, n=5),
                'lowest_correlation': self._find_lowest_correlations(corr_matrix, n=5),
                'average_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            }
        }
        
        return summary
    
    def _find_highest_correlations(self, corr_matrix: pd.DataFrame, n: int = 5) -> List[Tuple[str, str, float]]:
        """Find pairs with highest correlations."""
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                correlations.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))
        
        # Sort by correlation value and return top n
        correlations.sort(key=lambda x: x[2], reverse=True)
        return correlations[:n]
    
    def _find_lowest_correlations(self, corr_matrix: pd.DataFrame, n: int = 5) -> List[Tuple[str, str, float]]:
        """Find pairs with lowest correlations."""
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                correlations.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_value
                ))
        
        # Sort by correlation value and return bottom n
        correlations.sort(key=lambda x: x[2])
        return correlations[:n]