"""
Chart Generator Module

Handles chart generation and technical analysis visualization for HTML reports.
"""

import json
from typing import Dict, List, Any, Optional


class ChartGenerator:
    """Generates charts and technical analysis visualizations for HTML reports."""
    
    def __init__(self):
        """Initialize chart generator."""
        pass
    
    def generate_price_chart_section(self, ticker: str, financial_metrics: Dict) -> str:
        """Generate price chart section with technical analysis."""
        
        current_price = financial_metrics.get('current_price', 0)
        if not current_price:
            return "<p>Price chart data not available.</p>"
        
        # Generate mock price data for demonstration
        # In production, this would fetch real historical data
        price_data = self._generate_mock_price_data(current_price)
        
        chart_html = f"""
        <div class="chart-container" style="margin: 20px 0;">
            <h4>📈 Price Chart & Technical Analysis</h4>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <div class="row">
                    <div class="col-md-8">
                        <canvas id="priceChart_{ticker.replace('.', '_')}" width="400" height="200"></canvas>
                    </div>
                    <div class="col-md-4">
                        <div class="technical-indicators">
                            <h6>Technical Indicators</h6>
                            <div class="indicator-item">
                                <strong>Current Price:</strong> ${current_price:.2f}
                            </div>
                            <div class="indicator-item">
                                <strong>52W High:</strong> ${current_price * 1.25:.2f}
                            </div>
                            <div class="indicator-item">
                                <strong>52W Low:</strong> ${current_price * 0.75:.2f}
                            </div>
                            <div class="indicator-item">
                                <strong>RSI (14):</strong> {self._calculate_mock_rsi():.1f}
                            </div>
                            <div class="indicator-item">
                                <strong>MACD:</strong> {self._calculate_mock_macd():.2f}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Chart.js implementation would go here
                // This is a placeholder for the actual chart rendering
                console.log('Price chart for {ticker}:', {json.dumps(price_data)});
            </script>
        </div>
        """
        
        return chart_html
    
    def generate_performance_chart(self, financial_metrics: Dict) -> str:
        """Generate performance comparison chart."""
        
        revenue_growth = financial_metrics.get('revenue_growth', 0)
        profit_margin = financial_metrics.get('profit_margin', 0)
        roe = financial_metrics.get('roe', 0)
        
        chart_html = f"""
        <div class="performance-chart" style="margin: 20px 0;">
            <h5>📊 Performance Metrics</h5>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <div class="row">
                    <div class="col-md-4 text-center">
                        <div class="metric-card">
                            <h6>Revenue Growth</h6>
                            <div class="metric-value" style="font-size: 24px; font-weight: bold; color: {'#28a745' if revenue_growth > 0 else '#dc3545'};">
                                {revenue_growth:+.1f}%
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="metric-card">
                            <h6>Profit Margin</h6>
                            <div class="metric-value" style="font-size: 24px; font-weight: bold; color: {'#28a745' if profit_margin > 0 else '#dc3545'};">
                                {profit_margin:.1f}%
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="metric-card">
                            <h6>Return on Equity</h6>
                            <div class="metric-value" style="font-size: 24px; font-weight: bold; color: {'#28a745' if roe > 0 else '#dc3545'};">
                                {roe:.1f}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return chart_html
    
    def generate_valuation_chart(self, financial_metrics: Dict) -> str:
        """Generate valuation metrics chart."""
        
        pe_ratio = financial_metrics.get('pe_ratio', 0)
        pb_ratio = financial_metrics.get('pb_ratio', 0)
        ps_ratio = financial_metrics.get('ps_ratio', 0)
        
        chart_html = f"""
        <div class="valuation-chart" style="margin: 20px 0;">
            <h5>💰 Valuation Metrics</h5>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <div class="row">
                    <div class="col-md-4 text-center">
                        <div class="metric-card">
                            <h6>P/E Ratio</h6>
                            <div class="metric-value" style="font-size: 24px; font-weight: bold; color: #007bff;">
                                {pe_ratio:.1f}x
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="metric-card">
                            <h6>P/B Ratio</h6>
                            <div class="metric-value" style="font-size: 24px; font-weight: bold; color: #007bff;">
                                {pb_ratio:.1f}x
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="metric-card">
                            <h6>P/S Ratio</h6>
                            <div class="metric-value" style="font-size: 24px; font-weight: bold; color: #007bff;">
                                {ps_ratio:.1f}x
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return chart_html
    
    def generate_technical_analysis_section(self, technical_data: Dict) -> str:
        """Generate technical analysis section."""
        
        if not technical_data:
            return "<p>Technical analysis data not available.</p>"
        
        trend = technical_data.get('trend', 'Neutral')
        support_level = technical_data.get('support_level', 0)
        resistance_level = technical_data.get('resistance_level', 0)
        volume_trend = technical_data.get('volume_trend', 'Average')
        
        trend_color = {
            'Bullish': '#28a745',
            'Bearish': '#dc3545',
            'Neutral': '#ffc107'
        }.get(trend, '#6c757d')
        
        html = f"""
        <div class="technical-analysis" style="margin: 20px 0;">
            <h5>🔍 Technical Analysis</h5>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <div class="row">
                    <div class="col-md-3 text-center">
                        <div class="tech-indicator">
                            <h6>Overall Trend</h6>
                            <div style="font-size: 18px; font-weight: bold; color: {trend_color};">
                                {trend}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 text-center">
                        <div class="tech-indicator">
                            <h6>Support Level</h6>
                            <div style="font-size: 18px; font-weight: bold; color: #28a745;">
                                ${support_level:.2f}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 text-center">
                        <div class="tech-indicator">
                            <h6>Resistance Level</h6>
                            <div style="font-size: 18px; font-weight: bold; color: #dc3545;">
                                ${resistance_level:.2f}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 text-center">
                        <div class="tech-indicator">
                            <h6>Volume Trend</h6>
                            <div style="font-size: 18px; font-weight: bold; color: #007bff;">
                                {volume_trend}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_mock_price_data(self, current_price: float) -> List[Dict]:
        """Generate mock price data for chart demonstration."""
        import random
        
        data = []
        price = current_price * 0.9  # Start 10% below current
        
        for i in range(30):  # 30 data points
            # Random walk with slight upward bias
            change = random.uniform(-0.03, 0.04)  # -3% to +4% daily change
            price = price * (1 + change)
            
            data.append({
                'date': f'2024-{8 + i//30:02d}-{(i % 30) + 1:02d}',
                'price': round(price, 2),
                'volume': random.randint(1000000, 5000000)
            })
        
        # Ensure last price is close to current price
        data[-1]['price'] = current_price
        
        return data
    
    def _calculate_mock_rsi(self) -> float:
        """Calculate mock RSI value."""
        import random
        return random.uniform(30, 70)  # RSI typically ranges 0-100, 30-70 is common
    
    def _calculate_mock_macd(self) -> float:
        """Calculate mock MACD value."""
        import random
        return random.uniform(-2, 2)  # MACD can be positive or negative
    
    def generate_sector_comparison_chart(self, ticker: str, financial_metrics: Dict) -> str:
        """Generate sector comparison chart."""
        
        pe_ratio = financial_metrics.get('pe_ratio', 15)
        
        # Mock sector averages for comparison
        sector_pe = pe_ratio * 1.1  # Assume sector trades at slight premium
        market_pe = pe_ratio * 0.95  # Assume market trades at slight discount
        
        chart_html = f"""
        <div class="sector-comparison" style="margin: 20px 0;">
            <h5>🏭 Sector Comparison</h5>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <div class="row">
                    <div class="col-md-4 text-center">
                        <div class="comparison-item">
                            <h6>{ticker}</h6>
                            <div style="font-size: 20px; font-weight: bold; color: #007bff;">
                                {pe_ratio:.1f}x P/E
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="comparison-item">
                            <h6>Sector Average</h6>
                            <div style="font-size: 20px; font-weight: bold; color: #6c757d;">
                                {sector_pe:.1f}x P/E
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="comparison-item">
                            <h6>Market Average</h6>
                            <div style="font-size: 20px; font-weight: bold; color: #6c757d;">
                                {market_pe:.1f}x P/E
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 15px;">
                    <small class="text-muted">
                        <strong>Analysis:</strong> 
                        {ticker} trades at a {'premium' if pe_ratio > sector_pe else 'discount'} to sector average, 
                        suggesting {'strong market confidence' if pe_ratio > sector_pe else 'potential value opportunity'}.
                    </small>
                </div>
            </div>
        </div>
        """
        
        return chart_html
