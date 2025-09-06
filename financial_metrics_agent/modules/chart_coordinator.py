"""
Chart Coordinator

Coordinates chart generation and integration with the existing ChartEnhancementAgent.
Handles chart data preparation and script generation.
"""

import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional
from .report_data_processor import ProcessedFinancialData

logger = logging.getLogger(__name__)

class ChartCoordinator:
    """Coordinates chart generation and integration."""
    
    def __init__(self, chart_enhancement_agent=None):
        """
        Initialize chart coordinator.
        
        Args:
            chart_enhancement_agent: Optional ChartEnhancementAgent instance
        """
        self.chart_enhancement_agent = chart_enhancement_agent
        self.logger = logging.getLogger(__name__)
        self._enhanced_chart_scripts = []
    
    def generate_enhanced_price_technical_section(self, historical_data: Dict[str, Any],
                                                technical_data: Dict[str, Any],
                                                data: ProcessedFinancialData) -> str:
        """
        Generate enhanced combined price and technical analysis section.
        
        Args:
            historical_data: Historical price data
            technical_data: Technical analysis data
            data: Processed financial data
            
        Returns:
            HTML string for the enhanced price and technical analysis section
        """
        try:
            # Check if chart enhancement agent is available
            if not self.chart_enhancement_agent:
                # Fallback to separate sections
                price_section = self._generate_price_chart_section(historical_data, data)
                technical_section = self._generate_technical_analysis_section(technical_data, data)
                return price_section + technical_section
            
            # Use chart enhancement agent
            enhancement_result = self._run_chart_enhancement(historical_data, technical_data, data.ticker)
            
            if enhancement_result.get('success'):
                # Generate enhanced chart scripts
                chart_scripts = ""
                if enhancement_result.get('chart_config'):
                    chart_scripts = self.chart_enhancement_agent.generate_enhanced_chart_scripts(
                        enhancement_result['chart_config']
                    )
                
                # Store chart scripts for later inclusion
                self._enhanced_chart_scripts.append(chart_scripts)
                
                return enhancement_result['enhanced_html']
            else:
                # Fallback on enhancement failure
                self.logger.warning(f"Chart enhancement failed: {enhancement_result.get('error', 'Unknown error')}")
                return self._generate_fallback_combined_section(historical_data, technical_data, data)
        
        except Exception as e:
            self.logger.error(f"Error in enhanced price technical section generation: {e}")
            # Fallback to separate sections
            return self._generate_fallback_combined_section(historical_data, technical_data, data)
    
    def generate_chart_scripts(self, historical_data: Dict[str, Any], data: ProcessedFinancialData) -> str:
        """
        Generate chart JavaScript code.
        
        Args:
            historical_data: Historical price data
            data: Processed financial data
            
        Returns:
            JavaScript code for charts
        """
        try:
            # Include any enhanced chart scripts
            enhanced_scripts = "\n".join(self._enhanced_chart_scripts)
            
            # Generate basic chart script if no enhanced scripts available
            if not enhanced_scripts and historical_data:
                basic_script = self._generate_basic_chart_script(historical_data, data)
                return basic_script
            
            return enhanced_scripts
            
        except Exception as e:
            self.logger.error(f"Error generating chart scripts: {e}")
            return ""
    
    def _run_chart_enhancement(self, historical_data: Dict[str, Any], technical_data: Dict[str, Any],
                             ticker: str) -> Dict[str, Any]:
        """Run chart enhancement synchronously."""
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, run in a thread
                result = None
                exception = None
                
                def run_in_thread():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(
                            self.chart_enhancement_agent.enhance_price_chart_section(
                                historical_data, technical_data, ticker
                            )
                        )
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join(timeout=30)  # 30 second timeout
                
                if exception:
                    raise exception
                if result is None:
                    raise TimeoutError("Chart enhancement timed out")
                
                return result
            
            except RuntimeError:
                # No running loop, we can use asyncio.run
                return asyncio.run(
                    self.chart_enhancement_agent.enhance_price_chart_section(
                        historical_data, technical_data, ticker
                    )
                )
        
        except Exception as e:
            self.logger.error(f"Error running chart enhancement: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_price_chart_section(self, historical_data: Dict[str, Any], data: ProcessedFinancialData) -> str:
        """Generate basic price chart section."""
        if not historical_data:
            return f"""
            <div class="section">
                <h2>📈 Price Chart</h2>
                <div class="alert alert-warning">
                    <p>Price chart data not available for {data.ticker}</p>
                </div>
            </div>"""
        
        return f"""
        <div class="section">
            <h2>📈 Price Chart - {data.ticker}</h2>
            <div class="chart-container">
                <canvas id="priceChart_{data.ticker.replace('.', '_')}"></canvas>
            </div>
            <div class="chart-info">
                <p><strong>Current Price:</strong> ${data.current_price:.2f}</p>
                <p><strong>Market Cap:</strong> ${data.market_cap/1e9:.1f}B</p>
            </div>
        </div>"""
    
    def _generate_technical_analysis_section(self, technical_data: Dict[str, Any], data: ProcessedFinancialData) -> str:
        """Generate basic technical analysis section."""
        if not technical_data:
            return f"""
            <div class="section">
                <h2>📊 Technical Analysis</h2>
                <div class="alert alert-warning">
                    <p>Technical analysis data not available for {data.ticker}</p>
                </div>
            </div>"""
        
        return f"""
        <div class="section">
            <h2>📊 Technical Analysis - {data.ticker}</h2>
            <div class="technical-metrics">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Beta</div>
                        <div class="metric-value">{data.beta:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Volatility Profile</div>
                        <div class="metric-value">{'High' if data.beta > 1.5 else 'Moderate' if data.beta > 0.8 else 'Low'}</div>
                    </div>
                </div>
            </div>
        </div>"""
    
    def _generate_fallback_combined_section(self, historical_data: Dict[str, Any], technical_data: Dict[str, Any],
                                          data: ProcessedFinancialData) -> str:
        """Generate fallback combined section when enhancement fails."""
        return f"""
        <div class="section">
            <h2>📈 Price & Technical Analysis</h2>
            <div class="alert alert-warning">
                <h5>⚠️ Enhanced Chart Unavailable</h5>
                <p>Enhanced chart functionality is temporarily unavailable. Individual sections are displayed below.</p>
            </div>
        </div>
        {self._generate_price_chart_section(historical_data, data)}
        {self._generate_technical_analysis_section(technical_data, data)}"""
    
    def _generate_basic_chart_script(self, historical_data: Dict[str, Any], data: ProcessedFinancialData) -> str:
        """Generate basic chart JavaScript code."""
        chart_id = f"priceChart_{data.ticker.replace('.', '_')}"
        
        # Extract price data if available
        prices = historical_data.get('prices', [])
        dates = historical_data.get('dates', [])
        
        if not prices or not dates:
            return ""
        
        # Prepare data for Chart.js
        chart_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            if i < len(dates) and i < len(prices):
                chart_data.append(f"{{x: '{date}', y: {price}}}")
        
        data_points = ",".join(chart_data[:100])  # Limit to 100 points for performance
        
        return f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const ctx = document.getElementById('{chart_id}');
            if (ctx) {{
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        datasets: [{{
                            label: '{data.ticker} Price',
                            data: [{data_points}],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            x: {{
                                type: 'time',
                                time: {{
                                    unit: 'day'
                                }},
                                title: {{
                                    display: true,
                                    text: 'Date'
                                }}
                            }},
                            y: {{
                                title: {{
                                    display: true,
                                    text: 'Price ($)'
                                }}
                            }}
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: '{data.ticker} - Price Chart'
                            }},
                            legend: {{
                                display: true,
                                position: 'top'
                            }}
                        }}
                    }}
                }});
            }}
        }});
        </script>"""
    
    def generate_multi_ticker_charts(self, tickers_data: Dict[str, Any]) -> str:
        """
        Generate charts for multi-ticker comparison.
        
        Args:
            tickers_data: Dictionary of ticker data
            
        Returns:
            HTML string for multi-ticker charts
        """
        try:
            charts_html = """
            <div class="section">
                <h2>📊 Price Comparison Charts</h2>
                <div class="chart-container">
                    <canvas id="multiTickerComparisonChart"></canvas>
                </div>
            </div>"""
            
            return charts_html
            
        except Exception as e:
            self.logger.error(f"Error generating multi-ticker charts: {e}")
            return ""
    
    def generate_multi_ticker_chart_scripts(self, tickers_data: Dict[str, Any]) -> str:
        """
        Generate chart scripts for multi-ticker comparison.
        
        Args:
            tickers_data: Dictionary of ticker data
            
        Returns:
            JavaScript code for multi-ticker charts
        """
        try:
            # Prepare data for comparison chart
            datasets = []
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            
            for i, (ticker, ticker_data) in enumerate(tickers_data.items()):
                financial_metrics = ticker_data.get('financial_metrics', {})
                current_price = financial_metrics.get('current_price', 0)
                market_cap = financial_metrics.get('market_cap', 0)
                
                if current_price > 0:
                    color = colors[i % len(colors)]
                    datasets.append(f"""{{
                        label: '{ticker}',
                        data: [{{x: '{ticker}', y: {current_price}}}],
                        backgroundColor: '{color}',
                        borderColor: '{color}',
                        borderWidth: 2
                    }}""")
            
            datasets_str = ",".join(datasets)
            
            return f"""
            <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const ctx = document.getElementById('multiTickerComparisonChart');
                if (ctx) {{
                    new Chart(ctx, {{
                        type: 'bar',
                        data: {{
                            datasets: [{datasets_str}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: 'Ticker'
                                    }}
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'Current Price ($)'
                                    }}
                                }}
                            }},
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: 'Multi-Ticker Price Comparison'
                                }},
                                legend: {{
                                    display: true,
                                    position: 'top'
                                }}
                            }}
                        }}
                    }});
                }}
            }});
            </script>"""
            
        except Exception as e:
            self.logger.error(f"Error generating multi-ticker chart scripts: {e}")
            return ""
    
    def reset_chart_scripts(self):
        """Reset stored chart scripts."""
        self._enhanced_chart_scripts = []
