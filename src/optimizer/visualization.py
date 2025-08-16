"""
Modern visualization module for portfolio optimization results.
Clean, interactive plots using Plotly.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging

from optimizer import OptimizationResult
from data_fetcher import FundData

logger = logging.getLogger(__name__)


class PortfolioVisualizer:
    """
    Modern visualization class for portfolio analysis and optimization results.
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_portfolio_allocation(self, result: OptimizationResult, 
                                funds_data: Dict[str, FundData]) -> go.Figure:
        """
        Create portfolio allocation pie chart.
        
        Args:
            result: OptimizationResult object
            funds_data: Dictionary of FundData objects
            
        Returns:
            Plotly figure
        """
        if not result.success or not result.fund_allocations:
            return self._create_error_figure("No allocation data available")
        
        # Prepare data for pie chart
        tickers = list(result.fund_allocations.keys())
        weights = [result.fund_allocations[ticker] for ticker in tickers]
        labels = [f"{ticker}<br>{funds_data[ticker].description[:30]}..." 
                 if ticker in funds_data else ticker for ticker in tickers]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=self.color_palette[:len(tickers)])
        )])
        
        fig.update_layout(
            title=f'Optimal Portfolio Allocation<br>'
                  f'<sub>Sharpe Ratio: {result.sharpe_ratio:.3f} | '
                  f'Annual Return: {result.annual_return*100:.1f}% | '
                  f'Volatility: {result.annual_volatility*100:.1f}%</sub>',
            showlegend=True,
            height=600,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig
    
    def plot_allocation_bar_chart(self, result: OptimizationResult,
                                funds_data: Dict[str, FundData]) -> go.Figure:
        """
        Create horizontal bar chart of portfolio allocations.
        
        Args:
            result: OptimizationResult object
            funds_data: Dictionary of FundData objects
            
        Returns:
            Plotly figure
        """
        if not result.success or not result.fund_allocations:
            return self._create_error_figure("No allocation data available")
        
        # Sort allocations by weight
        sorted_allocations = sorted(result.fund_allocations.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        tickers = [item[0] for item in sorted_allocations]
        weights = [item[1] * 100 for item in sorted_allocations]  # Convert to percentage
        
        # Create labels with fund names
        labels = []
        for ticker in tickers:
            if ticker in funds_data:
                fund_name = funds_data[ticker].description
                # Truncate long names
                if len(fund_name) > 40:
                    fund_name = fund_name[:37] + "..."
                labels.append(f"{ticker} - {fund_name}")
            else:
                labels.append(ticker)
        
        fig = go.Figure(data=[go.Bar(
            x=weights,
            y=labels,
            orientation='h',
            marker=dict(
                color=weights,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Allocation %")
            ),
            text=[f'{w:.1f}%' for w in weights],
            textposition='auto',
        )])
        
        fig.update_layout(
            title='Portfolio Allocation by Fund',
            xaxis_title='Allocation (%)',
            yaxis_title='Funds',
            height=max(400, len(tickers) * 40),
            showlegend=False
        )
        
        return fig
    
    def plot_risk_return_scatter(self, funds_data: Dict[str, FundData],
                               result: OptimizationResult = None) -> go.Figure:
        """
        Create risk-return scatter plot of individual funds.
        
        Args:
            funds_data: Dictionary of FundData objects
            result: OptimizationResult object (optional, to highlight optimal portfolio)
            
        Returns:
            Plotly figure
        """
        # Calculate statistics for each fund
        from analysis import FinancialAnalyzer
        analyzer = FinancialAnalyzer()
        returns_df = analyzer.calculate_returns(funds_data)
        stats_df = analyzer.calculate_annual_statistics(returns_df)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Individual funds
        fig.add_trace(go.Scatter(
            x=stats_df['annual_volatility'] * 100,
            y=stats_df['annual_return'] * 100,
            mode='markers+text',
            text=stats_df.index,
            textposition='top center',
            marker=dict(
                size=10,
                color=stats_df['sharpe_ratio'],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Individual Funds',
            hovertemplate='<b>%{text}</b><br>' +
                         'Return: %{y:.2f}%<br>' +
                         'Volatility: %{x:.2f}%<br>' +
                         'Sharpe: %{marker.color:.3f}<extra></extra>'
        ))
        
        # Add optimal portfolio point if provided
        if result and result.success:
            fig.add_trace(go.Scatter(
                x=[result.annual_volatility * 100],
                y=[result.annual_return * 100],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name='Optimal Portfolio',
                hovertemplate='<b>Optimal Portfolio</b><br>' +
                             f'Return: {result.annual_return*100:.2f}%<br>' +
                             f'Volatility: {result.annual_volatility*100:.2f}%<br>' +
                             f'Sharpe: {result.sharpe_ratio:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Risk-Return Profile of Funds',
            xaxis_title='Annual Volatility (%)',
            yaxis_title='Annual Return (%)',
            hovermode='closest',
            height=600
        )
        
        return fig
    
    def plot_correlation_heatmap(self, funds_data: Dict[str, FundData]) -> go.Figure:
        """
        Create correlation heatmap of fund returns.
        
        Args:
            funds_data: Dictionary of FundData objects
            
        Returns:
            Plotly figure
        """
        from analysis import FinancialAnalyzer
        analyzer = FinancialAnalyzer()
        returns_df = analyzer.calculate_returns(funds_data)
        corr_matrix = analyzer.calculate_correlation_matrix(returns_df)
        
        return analyzer.create_correlation_heatmap(corr_matrix)
    
    def plot_efficient_frontier(self, optimizer, 
                              result: OptimizationResult = None) -> go.Figure:
        """
        Plot efficient frontier with optimal portfolio highlighted.
        
        Args:
            optimizer: PortfolioOptimizer object with prepared data
            result: OptimizationResult object (optional)
            
        Returns:
            Plotly figure
        """
        try:
            # Calculate efficient frontier
            efficient_df = optimizer.calculate_efficient_frontier(num_portfolios=50)
            
            if efficient_df.empty:
                return self._create_error_figure("Could not calculate efficient frontier")
            
            fig = go.Figure()
            
            # Plot efficient frontier
            fig.add_trace(go.Scatter(
                x=efficient_df['volatility'] * 100,
                y=efficient_df['return'] * 100,
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                hovertemplate='Return: %{y:.2f}%<br>' +
                             'Volatility: %{x:.2f}%<br>' +
                             'Sharpe: %{customdata:.3f}<extra></extra>',
                customdata=efficient_df['sharpe_ratio']
            ))
            
            # Add optimal portfolio if provided
            if result and result.success:
                fig.add_trace(go.Scatter(
                    x=[result.annual_volatility * 100],
                    y=[result.annual_return * 100],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='star',
                        line=dict(color='black', width=2)
                    ),
                    name='Optimal Portfolio',
                    hovertemplate='<b>Optimal Portfolio</b><br>' +
                                 f'Return: {result.annual_return*100:.2f}%<br>' +
                                 f'Volatility: {result.annual_volatility*100:.2f}%<br>' +
                                 f'Sharpe: {result.sharpe_ratio:.3f}<extra></extra>'
                ))
            
            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Annual Volatility (%)',
                yaxis_title='Annual Return (%)',
                hovermode='closest',
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting efficient frontier: {str(e)}")
            return self._create_error_figure("Error calculating efficient frontier")
    
    def plot_performance_comparison(self, funds_data: Dict[str, FundData],
                                  selected_funds: List[str] = None) -> go.Figure:
        """
        Plot cumulative performance comparison of selected funds.
        
        Args:
            funds_data: Dictionary of FundData objects
            selected_funds: List of funds to plot (if None, plot all)
            
        Returns:
            Plotly figure
        """
        if selected_funds is None:
            selected_funds = list(funds_data.keys())
        
        # Limit to 10 funds for readability
        if len(selected_funds) > 10:
            selected_funds = selected_funds[:10]
        
        from analysis import FinancialAnalyzer
        analyzer = FinancialAnalyzer()
        
        fig = go.Figure()
        
        for i, ticker in enumerate(selected_funds):
            if ticker in funds_data:
                fund_data = funds_data[ticker]
                df_with_indicators = analyzer.add_technical_indicators(fund_data)
                
                color = self.color_palette[i % len(self.color_palette)]
                
                fig.add_trace(go.Scatter(
                    x=df_with_indicators.index,
                    y=(df_with_indicators['cumulative_return'] - 1) * 100,
                    mode='lines',
                    name=ticker,
                    line=dict(color=color, width=2),
                    hovertemplate=f'<b>{ticker}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Return: %{y:.2f}%<extra></extra>'
                ))
        
        fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_portfolio_dashboard(self, result: OptimizationResult,
                                 funds_data: Dict[str, FundData],
                                 optimizer) -> go.Figure:
        """
        Create comprehensive portfolio dashboard with multiple subplots.
        
        Args:
            result: OptimizationResult object
            funds_data: Dictionary of FundData objects
            optimizer: PortfolioOptimizer object
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Allocation', 'Risk-Return Profile',
                          'Correlation Heatmap', 'Performance Comparison'),
            specs=[[{"type": "domain"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Portfolio allocation (pie chart data only - simplified for subplot)
        if result.success and result.fund_allocations:
            tickers = list(result.fund_allocations.keys())
            weights = [result.fund_allocations[ticker] for ticker in tickers]
            
            fig.add_trace(go.Pie(
                labels=tickers,
                values=weights,
                name="Allocation",
                hole=0.3
            ), row=1, col=1)
        
        # Risk-Return scatter
        from analysis import FinancialAnalyzer
        analyzer = FinancialAnalyzer()
        returns_df = analyzer.calculate_returns(funds_data)
        stats_df = analyzer.calculate_annual_statistics(returns_df)
        
        fig.add_trace(go.Scatter(
            x=stats_df['annual_volatility'] * 100,
            y=stats_df['annual_return'] * 100,
            mode='markers',
            text=stats_df.index,
            name='Funds',
            marker=dict(size=8)
        ), row=1, col=2)
        
        if result.success:
            fig.add_trace(go.Scatter(
                x=[result.annual_volatility * 100],
                y=[result.annual_return * 100],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Optimal'
            ), row=1, col=2)
        
        # Correlation heatmap
        corr_matrix = analyzer.calculate_correlation_matrix(returns_df)
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            name="Correlation"
        ), row=2, col=1)
        
        # Performance comparison (top 5 allocated funds)
        if result.success:
            top_funds = sorted(result.fund_allocations.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
            
            for ticker, _ in top_funds:
                if ticker in funds_data:
                    fund_data = funds_data[ticker]
                    df_with_indicators = analyzer.add_technical_indicators(fund_data)
                    
                    fig.add_trace(go.Scatter(
                        x=df_with_indicators.index,
                        y=(df_with_indicators['cumulative_return'] - 1) * 100,
                        mode='lines',
                        name=ticker,
                        showlegend=False
                    ), row=2, col=2)
        
        fig.update_layout(
            title='Portfolio Optimization Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_error_figure(self, message: str) -> go.Figure:
        """Create a figure displaying an error message."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        
        return fig
    
    def save_all_charts(self, result: OptimizationResult,
                       funds_data: Dict[str, FundData],
                       optimizer, output_dir: str = "output") -> Dict[str, str]:
        """
        Generate and save all charts to HTML files.
        
        Args:
            result: OptimizationResult object
            funds_data: Dictionary of FundData objects
            optimizer: PortfolioOptimizer object
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Portfolio allocation pie chart
        try:
            fig = self.plot_portfolio_allocation(result, funds_data)
            file_path = output_path / "portfolio_allocation.html"
            fig.write_html(file_path)
            saved_files['allocation_pie'] = str(file_path)
        except Exception as e:
            logger.error(f"Error saving allocation chart: {e}")
        
        # Allocation bar chart
        try:
            fig = self.plot_allocation_bar_chart(result, funds_data)
            file_path = output_path / "allocation_bar_chart.html"
            fig.write_html(file_path)
            saved_files['allocation_bar'] = str(file_path)
        except Exception as e:
            logger.error(f"Error saving bar chart: {e}")
        
        # Risk-return scatter
        try:
            fig = self.plot_risk_return_scatter(funds_data, result)
            file_path = output_path / "risk_return_scatter.html"
            fig.write_html(file_path)
            saved_files['risk_return'] = str(file_path)
        except Exception as e:
            logger.error(f"Error saving risk-return chart: {e}")
        
        # Correlation heatmap
        try:
            fig = self.plot_correlation_heatmap(funds_data)
            file_path = output_path / "correlation_heatmap.html"
            fig.write_html(file_path)
            saved_files['correlation'] = str(file_path)
        except Exception as e:
            logger.error(f"Error saving correlation chart: {e}")
        
        # Efficient frontier
        try:
            fig = self.plot_efficient_frontier(optimizer, result)
            file_path = output_path / "efficient_frontier.html"
            fig.write_html(file_path)
            saved_files['efficient_frontier'] = str(file_path)
        except Exception as e:
            logger.error(f"Error saving efficient frontier: {e}")
        
        # Performance comparison
        try:
            # Use top allocated funds for comparison
            if result.success:
                top_funds = [ticker for ticker, _ in 
                           sorted(result.fund_allocations.items(), 
                                 key=lambda x: x[1], reverse=True)[:8]]
            else:
                top_funds = list(funds_data.keys())[:8]
                
            fig = self.plot_performance_comparison(funds_data, top_funds)
            file_path = output_path / "performance_comparison.html"
            fig.write_html(file_path)
            saved_files['performance'] = str(file_path)
        except Exception as e:
            logger.error(f"Error saving performance chart: {e}")
        
        # Dashboard
        try:
            fig = self.create_portfolio_dashboard(result, funds_data, optimizer)
            file_path = output_path / "portfolio_dashboard.html"
            fig.write_html(file_path)
            saved_files['dashboard'] = str(file_path)
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")
        
        logger.info(f"Saved {len(saved_files)} charts to {output_dir}")
        return saved_files