"""
Modern portfolio optimization module with Sharpe ratio maximization.
Clean, efficient implementation with comprehensive result analysis.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from config import config
from data_fetcher import FundData
from analysis import FinancialAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Data class for storing optimization results."""
    success: bool
    weights: np.ndarray
    sharpe_ratio: float
    annual_return: float
    annual_volatility: float
    fund_allocations: Dict[str, float]
    dollar_allocations: Dict[str, float]
    optimization_message: str
    convergence_info: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'sharpe_ratio': self.sharpe_ratio,
            'annual_return': self.annual_return,
            'annual_volatility': self.annual_volatility,
            'fund_allocations': self.fund_allocations,
            'dollar_allocations': self.dollar_allocations,
            'optimization_message': self.optimization_message,
            'convergence_info': self.convergence_info
        }


class PortfolioOptimizer:
    """
    Modern portfolio optimizer using Sharpe ratio maximization.
    """
    
    def __init__(self):
        self.analyzer = FinancialAnalyzer()
        self.returns_data: Optional[pd.DataFrame] = None
        self.mean_returns: Optional[np.ndarray] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.tickers: Optional[List[str]] = None
        
    def prepare_data(self, funds_data: Dict[str, FundData]) -> bool:
        """
        Prepare data for optimization.
        
        Args:
            funds_data: Dictionary of FundData objects
            
        Returns:
            True if data preparation successful
        """
        try:
            # Calculate returns
            self.returns_data = self.analyzer.calculate_returns(funds_data)
            
            if self.returns_data.empty:
                logger.error("No valid returns data available")
                return False
            
            # Store tickers
            self.tickers = list(self.returns_data.columns)
            
            # Calculate mean annual returns
            self.mean_returns = self.returns_data.mean().values * 252
            
            # Calculate annualized covariance matrix
            self.cov_matrix = self.analyzer.calculate_covariance_matrix(self.returns_data)
            
            logger.info(f"Data prepared for {len(self.tickers)} funds")
            logger.info(f"Data period: {self.returns_data.index.min()} to {self.returns_data.index.max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return False
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        # Portfolio return
        portfolio_return = np.sum(weights * self.mean_returns)
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Sharpe ratio
        if portfolio_volatility > 0:
            sharpe_ratio = (portfolio_return - config.optimization.risk_free_rate) / portfolio_volatility
        else:
            sharpe_ratio = 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def objective_function(self, weights: np.ndarray) -> float:
        """
        Objective function to minimize (negative Sharpe ratio).
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Negative Sharpe ratio
        """
        _, _, sharpe_ratio = self.calculate_portfolio_metrics(weights)
        return -sharpe_ratio  # Minimize negative Sharpe ratio = maximize Sharpe ratio
    
    def optimize_portfolio(self, 
                         investment_amount: float = None,
                         min_weight: float = 0.0,
                         max_weight: float = 1.0) -> OptimizationResult:
        """
        Optimize portfolio to maximize Sharpe ratio.
        
        Args:
            investment_amount: Total investment amount in dollars
            min_weight: Minimum weight for any asset
            max_weight: Maximum weight for any asset
            
        Returns:
            OptimizationResult object
        """
        if self.mean_returns is None or self.cov_matrix is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if investment_amount is None:
            investment_amount = config.default_investment_amount
        
        num_assets = len(self.tickers)
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Constraints: weights sum to 1
        constraints = ({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Bounds: weights between min_weight and max_weight
        bounds = tuple([(min_weight, max_weight) for _ in range(num_assets)])
        
        # Optimization options
        options = {
            'maxiter': config.optimization.max_iterations,
            'ftol': config.optimization.tolerance,
            'disp': True
        }
        
        try:
            logger.info("Starting portfolio optimization...")
            
            # Run optimization
            result = minimize(
                fun=self.objective_function,
                x0=initial_weights,
                method=config.optimization.optimization_method,
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return, portfolio_volatility, sharpe_ratio = \
                    self.calculate_portfolio_metrics(optimal_weights)
                
                # Create allocation dictionaries
                fund_allocations = {
                    ticker: weight for ticker, weight in zip(self.tickers, optimal_weights)
                }
                
                dollar_allocations = {
                    ticker: weight * investment_amount 
                    for ticker, weight in zip(self.tickers, optimal_weights)
                }
                
                # Filter out zero allocations for cleaner display
                fund_allocations = {k: v for k, v in fund_allocations.items() if v > 0.001}
                dollar_allocations = {k: v for k, v in dollar_allocations.items() if v > 100}
                
                logger.info(f"Optimization successful! Sharpe ratio: {sharpe_ratio:.4f}")
                
                return OptimizationResult(
                    success=True,
                    weights=optimal_weights,
                    sharpe_ratio=sharpe_ratio,
                    annual_return=portfolio_return,
                    annual_volatility=portfolio_volatility,
                    fund_allocations=fund_allocations,
                    dollar_allocations=dollar_allocations,
                    optimization_message=result.message,
                    convergence_info={
                        'iterations': result.nit,
                        'function_evaluations': result.nfev,
                        'success': result.success
                    }
                )
            else:
                logger.error(f"Optimization failed: {result.message}")
                
                return OptimizationResult(
                    success=False,
                    weights=np.array([]),
                    sharpe_ratio=0.0,
                    annual_return=0.0,
                    annual_volatility=0.0,
                    fund_allocations={},
                    dollar_allocations={},
                    optimization_message=result.message,
                    convergence_info={'success': False}
                )
                
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            
            return OptimizationResult(
                success=False,
                weights=np.array([]),
                sharpe_ratio=0.0,
                annual_return=0.0,
                annual_volatility=0.0,
                fund_allocations={},
                dollar_allocations={},
                optimization_message=f"Error: {str(e)}",
                convergence_info={'success': False}
            )
    
    def calculate_efficient_frontier(self, 
                                   num_portfolios: int = 100) -> pd.DataFrame:
        """
        Calculate efficient frontier points.
        
        Args:
            num_portfolios: Number of portfolios to calculate
            
        Returns:
            DataFrame with efficient frontier data
        """
        if self.mean_returns is None or self.cov_matrix is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Generate target returns
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                # Minimize volatility for target return
                portfolio = self._minimize_volatility_for_return(target_return)
                
                if portfolio is not None:
                    ret, vol, sharpe = self.calculate_portfolio_metrics(portfolio)
                    efficient_portfolios.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe_ratio': sharpe,
                        'weights': portfolio
                    })
            except:
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def _minimize_volatility_for_return(self, target_return: float) -> Optional[np.ndarray]:
        """Minimize volatility for a target return."""
        num_assets = len(self.tickers)
        
        # Objective: minimize volatility
        def volatility_objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0},
            {'type': 'eq', 'fun': lambda weights: np.sum(weights * self.mean_returns) - target_return}
        ]
        
        # Bounds
        bounds = tuple([(0.0, 1.0) for _ in range(num_assets)])
        
        # Initial guess
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        try:
            result = minimize(
                fun=volatility_objective,
                x0=initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x if result.success else None
        except:
            return None
    
    def create_optimization_summary(self, 
                                  result: OptimizationResult,
                                  funds_data: Dict[str, FundData]) -> Dict:
        """
        Create comprehensive optimization summary.
        
        Args:
            result: OptimizationResult object
            funds_data: Dictionary of FundData objects
            
        Returns:
            Summary dictionary
        """
        if not result.success:
            return {'error': 'Optimization failed', 'message': result.optimization_message}
        
        # Fund descriptions for allocations
        allocation_details = []
        for ticker, allocation in result.fund_allocations.items():
            if ticker in funds_data:
                allocation_details.append({
                    'ticker': ticker,
                    'fund_name': funds_data[ticker].description,
                    'weight_percent': round(allocation * 100, 2),
                    'dollar_amount': round(result.dollar_allocations.get(ticker, 0), 2)
                })
        
        # Sort by allocation percentage
        allocation_details.sort(key=lambda x: x['weight_percent'], reverse=True)
        
        summary = {
            'optimization_date': datetime.now().isoformat(),
            'success': result.success,
            'portfolio_metrics': {
                'sharpe_ratio': round(result.sharpe_ratio, 4),
                'annual_return_percent': round(result.annual_return * 100, 2),
                'annual_volatility_percent': round(result.annual_volatility * 100, 2),
                'risk_free_rate_percent': round(config.optimization.risk_free_rate * 100, 2)
            },
            'total_investment': sum(result.dollar_allocations.values()),
            'fund_allocations': allocation_details,
            'diversification_metrics': {
                'number_of_funds': len(result.fund_allocations),
                'max_allocation_percent': max([d['weight_percent'] for d in allocation_details]),
                'allocation_concentration': self._calculate_concentration_ratio(result.weights)
            },
            'convergence_info': result.convergence_info
        }
        
        return summary
    
    def _calculate_concentration_ratio(self, weights: np.ndarray) -> float:
        """Calculate Herfindahl concentration index."""
        # Filter out zero weights
        non_zero_weights = weights[weights > 0.001]
        return np.sum(non_zero_weights ** 2)