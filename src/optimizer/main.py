"""
Main execution script for the modern portfolio optimization system.
Clean, modular approach with comprehensive error handling and logging.
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from config import config, FundUniverse
from data_fetcher import DataFetcher, fetch_all_vanguard_funds
from analysis import FinancialAnalyzer
from optimizer import PortfolioOptimizer
from visualization import PortfolioVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PortfolioOptimizationPipeline:
    """
    Main pipeline for portfolio optimization workflow.
    """
    
    def __init__(self, investment_amount: float = None):
        self.investment_amount = investment_amount or config.default_investment_amount
        self.fetcher = DataFetcher()
        self.analyzer = FinancialAnalyzer()
        self.optimizer = PortfolioOptimizer()
        self.visualizer = PortfolioVisualizer()
        
        self.funds_data = {}
        self.optimization_result = None
        
    def run_full_optimization(self, 
                            use_cache: bool = True,
                            save_results: bool = True,
                            create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Run the complete portfolio optimization pipeline.
        
        Args:
            use_cache: Whether to use cached data
            save_results: Whether to save results to files
            create_visualizations: Whether to generate charts
            
        Returns:
            Dictionary with complete results
        """
        logger.info("Starting portfolio optimization pipeline...")
        
        try:
            # Step 1: Fetch data
            logger.info("Step 1: Fetching fund data...")
            self.funds_data = self._fetch_data(use_cache)
            
            if not self.funds_data:
                raise ValueError("No fund data available")
            
            # Step 2: Analyze data
            logger.info("Step 2: Analyzing fund data...")
            analysis_summary = self._analyze_data()
            
            # Step 3: Optimize portfolio
            logger.info("Step 3: Optimizing portfolio...")
            self.optimization_result = self._optimize_portfolio()
            
            # Step 4: Create comprehensive summary
            logger.info("Step 4: Creating summary...")
            optimization_summary = self.optimizer.create_optimization_summary(
                self.optimization_result, self.funds_data
            )
            
            # Step 5: Generate visualizations
            visualization_files = {}
            if create_visualizations and self.optimization_result.success:
                logger.info("Step 5: Creating visualizations...")
                visualization_files = self._create_visualizations()
            
            # Step 6: Save results
            if save_results:
                logger.info("Step 6: Saving results...")
                self._save_results(analysis_summary, optimization_summary)
            
            # Compile final results
            results = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'analysis_summary': analysis_summary,
                'optimization_summary': optimization_summary,
                'visualization_files': visualization_files,
                'data_info': {
                    'total_funds': len(self.funds_data),
                    'investment_amount': self.investment_amount,
                    'analysis_period_years': config.get_analysis_period_years()
                }
            }
            
            logger.info("Portfolio optimization completed successfully!")
            self._print_key_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _fetch_data(self, use_cache: bool) -> Dict:
        """Fetch fund data."""
        return fetch_all_vanguard_funds(use_cache=use_cache, period="2y")
    
    def _analyze_data(self) -> Dict:
        """Analyze fund data and generate summary."""
        return self.analyzer.generate_analysis_summary(self.funds_data)
    
    def _optimize_portfolio(self):
        """Run portfolio optimization."""
        # Prepare data for optimization
        if not self.optimizer.prepare_data(self.funds_data):
            raise ValueError("Failed to prepare data for optimization")
        
        # Run optimization
        return self.optimizer.optimize_portfolio(
            investment_amount=self.investment_amount
        )
    
    def _create_visualizations(self) -> Dict[str, str]:
        """Create and save visualizations."""
        if not self.optimization_result or not self.optimization_result.success:
            logger.warning("Cannot create visualizations - optimization failed")
            return {}
        
        return self.visualizer.save_all_charts(
            self.optimization_result,
            self.funds_data,
            self.optimizer,
            output_dir="output"
        )
    
    def _save_results(self, analysis_summary: Dict, optimization_summary: Dict):
        """Save results to files."""
        # Ensure output directory exists
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save analysis summary
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        
        # Save optimization summary
        with open(output_dir / "optimization_summary.json", 'w') as f:
            json.dump(optimization_summary, f, indent=2, default=str)
        
        # Save fund universe for reference
        FundUniverse.save_to_json(output_dir / "fund_universe.json")
        
        # Save raw optimization result
        if self.optimization_result:
            with open(output_dir / "optimization_result.json", 'w') as f:
                json.dump(self.optimization_result.to_dict(), f, indent=2, default=str)
    
    def _print_key_results(self, results: Dict):
        """Print key results to console."""
        if not results['success']:
            print(f"\n❌ Optimization Failed: {results.get('error', 'Unknown error')}")
            return
        
        opt_summary = results['optimization_summary']
        
        print("\n" + "="*60)
        print("🎯 PORTFOLIO OPTIMIZATION RESULTS")
        print("="*60)
        
        if opt_summary.get('success'):
            metrics = opt_summary['portfolio_metrics']
            print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']}")
            print(f"📈 Annual Return: {metrics['annual_return_percent']}%")
            print(f"📉 Annual Volatility: {metrics['annual_volatility_percent']}%")
            print(f"💰 Total Investment: ${opt_summary['total_investment']:,.0f}")
            
            print(f"\n🏆 TOP FUND ALLOCATIONS:")
            for fund in opt_summary['fund_allocations'][:5]:
                print(f"  • {fund['ticker']}: {fund['weight_percent']}% "
                      f"(${fund['dollar_amount']:,.0f})")
            
            diversification = opt_summary['diversification_metrics']
            print(f"\n📊 DIVERSIFICATION:")
            print(f"  • Number of funds: {diversification['number_of_funds']}")
            print(f"  • Max allocation: {diversification['max_allocation_percent']}%")
            
        else:
            print("❌ Optimization failed")
        
        print("="*60)


def run_quick_optimization(investment_amount: float = 300000) -> Dict:
    """
    Quick optimization run with default settings.
    
    Args:
        investment_amount: Total investment amount
        
    Returns:
        Results dictionary
    """
    pipeline = PortfolioOptimizationPipeline(investment_amount)
    return pipeline.run_full_optimization()


def run_custom_optimization(
    tickers: list = None,
    investment_amount: float = 300000,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    use_cache: bool = True
) -> Dict:
    """
    Run optimization with custom parameters.
    
    Args:
        tickers: List of tickers to include (if None, use all Vanguard funds)
        investment_amount: Total investment amount
        min_weight: Minimum weight for any asset
        max_weight: Maximum weight for any asset
        use_cache: Whether to use cached data
        
    Returns:
        Results dictionary
    """
    pipeline = PortfolioOptimizationPipeline(investment_amount)
    
    # If custom tickers provided, update the fund universe
    if tickers:
        logger.info(f"Using custom ticker list: {tickers}")
        # Filter funds data to only include specified tickers
        pipeline.funds_data = {
            ticker: data for ticker, data in 
            fetch_all_vanguard_funds(use_cache=use_cache).items()
            if ticker in tickers
        }
    
    # Run optimization with custom parameters
    if not pipeline.optimizer.prepare_data(pipeline.funds_data):
        raise ValueError("Failed to prepare data for optimization")
    
    pipeline.optimization_result = pipeline.optimizer.optimize_portfolio(
        investment_amount=investment_amount,
        min_weight=min_weight,
        max_weight=max_weight
    )
    
    return pipeline.run_full_optimization(
        use_cache=use_cache,
        save_results=True,
        create_visualizations=True
    )


if __name__ == "__main__":
    # Example usage
    print("🚀 Modern Portfolio Optimization System")
    print("="*40)
    
    # Run with default settings
    results = run_quick_optimization(investment_amount=500000)
    
    # Example of custom optimization
    # results = run_custom_optimization(
    #     tickers=['VTSAX', 'VTIAX', 'VBTLX', 'VGT', 'VHT'],
    #     investment_amount=250000,
    #     min_weight=0.05,  # Minimum 5% allocation
    #     max_weight=0.40   # Maximum 40% allocation
    # )