"""
Template Renderer

Handles HTML template generation and styling for financial reports.
Provides responsive design and professional formatting.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .report_data_processor import ProcessedFinancialData, CompanyProfile
from .financial_analyzer import InvestmentRecommendation
from .content_generator import BullsBears
from .citation_manager import CitationManager

logger = logging.getLogger(__name__)

class TemplateRenderer:
    """Renders HTML templates for financial reports."""
    
    def __init__(self, citation_manager: CitationManager):
        """
        Initialize template renderer.
        
        Args:
            citation_manager: Citation manager instance
        """
        self.citation_manager = citation_manager
        self.logger = logging.getLogger(__name__)
    
    def render_single_ticker_report(self, data: ProcessedFinancialData, profile: CompanyProfile,
                                  recommendation: InvestmentRecommendation, bulls_bears: BullsBears,
                                  executive_summary: str, financial_highlights: str,
                                  chart_scripts: str = "", title: str = "Financial Analysis Report") -> str:
        """
        Render complete single ticker HTML report.
        
        Args:
            data: Processed financial data
            profile: Company profile
            recommendation: Investment recommendation
            bulls_bears: Bulls and bears analysis
            executive_summary: Executive summary HTML
            financial_highlights: Financial highlights HTML
            chart_scripts: Chart JavaScript code
            title: Report title
            
        Returns:
            Complete HTML report
        """
        try:
            # Generate report sections
            header_html = self._render_report_header(title, data, profile)
            executive_summary_section = self._render_executive_summary_section(executive_summary)
            investment_recommendation_section = self._render_investment_recommendation_section(
                recommendation, bulls_bears, data, profile
            )
            financial_highlights_section = self._render_financial_highlights_section(financial_highlights)
            citations_section = self.citation_manager.generate_numbered_references_section()
            footer_html = self._render_report_footer()
            
            # Combine all sections
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {data.ticker}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial"></script>
    {self._get_css_styles()}
</head>
<body>
    <div class="container">
        {header_html}
        
        <div class="report-content">
            {executive_summary_section}
            {investment_recommendation_section}
            {financial_highlights_section}
            {citations_section}
        </div>
        
        {footer_html}
    </div>
    
    {chart_scripts}
</body>
</html>"""
            
            self.logger.info(f"✅ Rendered single ticker report for {data.ticker}")
            return html_content
            
        except Exception as e:
            self.logger.error(f"❌ Error rendering single ticker report: {e}")
            raise
    
    def render_multi_ticker_report(self, tickers_data: Dict[str, Any], title: str = "Multi-Ticker Analysis") -> str:
        """
        Render multi-ticker comparison report.
        
        Args:
            tickers_data: Dictionary of ticker data
            title: Report title
            
        Returns:
            Complete HTML report
        """
        try:
            # Generate multi-ticker sections
            header_html = self._render_multi_ticker_header(title, tickers_data)
            comparison_section = self._render_ticker_comparison_section(tickers_data)
            citations_section = self.citation_manager.generate_numbered_references_section()
            footer_html = self._render_report_footer()
            
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Multi-Ticker Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial"></script>
    {self._get_css_styles()}
</head>
<body>
    <div class="container">
        {header_html}
        
        <div class="report-content">
            {comparison_section}
            {citations_section}
        </div>
        
        {footer_html}
    </div>
</body>
</html>"""
            
            self.logger.info(f"✅ Rendered multi-ticker report for {len(tickers_data)} tickers")
            return html_content
            
        except Exception as e:
            self.logger.error(f"❌ Error rendering multi-ticker report: {e}")
            raise
    
    def _render_report_header(self, title: str, data: ProcessedFinancialData, profile: CompanyProfile) -> str:
        """Render report header section."""
        return f"""
        <header class="report-header">
            <h1>{title}</h1>
            <div class="ticker-info">
                <span class="ticker">{data.ticker}</span>
                <span class="company-name">{data.company_name}</span>
            </div>
            <div class="company-details">
                <span class="sector">{data.sector}</span>
                <span class="size-category">{profile.size_category.replace('_', ' ').title()}</span>
                <span class="business-model">{profile.business_model.title()}</span>
            </div>
            <div class="report-meta">
                <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span>Market Cap: ${data.market_cap/1e9:.1f}B</span>
                <span>Current Price: ${data.current_price:.2f}</span>
            </div>
        </header>"""
    
    def _render_multi_ticker_header(self, title: str, tickers_data: Dict[str, Any]) -> str:
        """Render multi-ticker report header."""
        return f"""
        <header class="report-header">
            <h1>{title}</h1>
            <div class="ticker-info">
                <span class="ticker-count">{len(tickers_data)} Tickers Analyzed</span>
            </div>
            <div class="report-meta">
                <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span>Comparative Analysis Report</span>
            </div>
        </header>"""
    
    def _render_executive_summary_section(self, executive_summary: str) -> str:
        """Render executive summary section."""
        return f"""
        <div class="section">
            <h2>📋 Executive Summary</h2>
            <div class="alert alert-info">
                {executive_summary}
            </div>
        </div>"""
    
    def _render_investment_recommendation_section(self, recommendation: InvestmentRecommendation,
                                                bulls_bears: BullsBears, data: ProcessedFinancialData,
                                                profile: CompanyProfile) -> str:
        """Render investment recommendation section."""
        # Determine recommendation styling
        if recommendation.rating == 'BUY':
            rec_class = 'success'
            rec_color = '#27ae60'
        elif recommendation.rating == 'SELL':
            rec_class = 'danger'
            rec_color = '#e74c3c'
        else:  # HOLD
            rec_class = 'warning'
            rec_color = '#f39c12'
        
        # Confidence indicator
        confidence_bars = '█' * recommendation.confidence_score + '░' * (10 - recommendation.confidence_score)
        
        # Render bulls and bears analysis
        bulls_bears_html = self._render_bulls_bears_analysis(bulls_bears)
        
        return f"""
        <div class="section">
            <h2>🎯 Investment Recommendation</h2>
            
            <!-- Main Recommendation -->
            <div class="alert alert-{rec_class}" style="text-align: center; margin: 20px 0;">
                <h1 style="font-size: 3em; margin: 10px 0; color: {rec_color};">
                    {recommendation.rating} {recommendation.emoji}
                </h1>
                <h3 style="margin: 10px 0;">Confidence Score: {recommendation.confidence_score}/10</h3>
                <div style="font-family: monospace; font-size: 1.2em; letter-spacing: 2px;">
                    {confidence_bars}
                </div>
                <p style="font-size: 1.1em; margin: 15px 0;"><strong>{recommendation.key_rationale}</strong></p>
            </div>
            
            <!-- Detailed Reasoning -->
            <div class="alert alert-light" style="margin: 20px 0;">
                <h4 style="margin-bottom: 15px;">📋 Decision Rationale</h4>
                <div style="line-height: 1.6;">
                    {recommendation.detailed_reasoning}
                </div>
            </div>
            
            <!-- Price Analysis -->
            {self._render_price_analysis_section(recommendation.price_analysis)}
            
            <!-- Bulls and Bears Analysis -->
            {bulls_bears_html}
        </div>"""
    
    def _render_financial_highlights_section(self, financial_highlights: str) -> str:
        """Render financial highlights section."""
        return f"""
        <div class="section">
            <h2>📊 Financial Highlights</h2>
            <div class="financial-highlights-content">
                {financial_highlights}
            </div>
        </div>"""
    
    def _render_bulls_bears_analysis(self, bulls_bears: BullsBears) -> str:
        """Render bulls and bears analysis section."""
        bulls_html = ""
        for bull in bulls_bears.bulls_analysis:
            bulls_html += f"""
            <div class="analysis-point bull-point">
                <h5>{bull['title']}</h5>
                <p>{bull['content']}</p>
                <div class="quantitative-support">
                    <strong>Quantitative Support:</strong> {bull['quantitative_support']}
                </div>
            </div>"""
        
        bears_html = ""
        for bear in bulls_bears.bears_analysis:
            bears_html += f"""
            <div class="analysis-point bear-point">
                <h5>{bear['title']}</h5>
                <p>{bear['content']}</p>
                <div class="quantitative-support">
                    <strong>Quantitative Support:</strong> {bear['quantitative_support']}
                </div>
            </div>"""
        
        return f"""
        <div class="bulls-bears-analysis">
            <h4>📈 Bulls Say vs 📉 Bears Say Analysis</h4>
            
            <div class="analysis-grid">
                <div class="bulls-section">
                    <h5 style="color: #27ae60;">🐂 Bulls Say</h5>
                    {bulls_html}
                    <div class="summary">
                        <strong>Summary:</strong> {bulls_bears.bulls_summary}
                    </div>
                </div>
                
                <div class="bears-section">
                    <h5 style="color: #e74c3c;">🐻 Bears Say</h5>
                    {bears_html}
                    <div class="summary">
                        <strong>Summary:</strong> {bulls_bears.bears_summary}
                    </div>
                </div>
            </div>
        </div>"""
    
    def _render_price_analysis_section(self, price_analysis: Dict[str, Any]) -> str:
        """Render price analysis section."""
        current_price = price_analysis.get('current_price', 0)
        target_mean = price_analysis.get('target_mean')
        upside_potential = price_analysis.get('upside_potential', 0)
        
        if not current_price:
            return ""
        
        return f"""
        <div class="price-analysis">
            <h4>💰 Price Analysis</h4>
            <div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                <div class="metric-card">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${current_price:.2f}</div>
                </div>
                {f'''<div class="metric-card">
                    <div class="metric-label">Analyst Target</div>
                    <div class="metric-value">${target_mean:.2f}</div>
                </div>''' if target_mean else ''}
                {f'''<div class="metric-card">
                    <div class="metric-label">Upside Potential</div>
                    <div class="metric-value" style="color: {'#27ae60' if upside_potential > 0 else '#e74c3c'};">
                        {upside_potential:+.1f}%
                    </div>
                </div>''' if upside_potential != 0 else ''}
            </div>
        </div>"""
    
    def _render_ticker_comparison_section(self, tickers_data: Dict[str, Any]) -> str:
        """Render ticker comparison section."""
        comparison_html = """
        <div class="section">
            <h2>📊 Ticker Comparison</h2>
            <div class="comparison-table-container">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Company</th>
                            <th>Recommendation</th>
                            <th>Current Price</th>
                            <th>Market Cap</th>
                            <th>P/E Ratio</th>
                            <th>Dividend Yield</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        for ticker, ticker_data in tickers_data.items():
            basic_info = ticker_data.get('basic_info', {})
            financial_metrics = ticker_data.get('financial_metrics', {})
            investment_decision = ticker_data.get('investment_decision', {})
            
            company_name = basic_info.get('long_name', ticker)
            recommendation = investment_decision.get('recommendation', 'N/A')
            current_price = financial_metrics.get('current_price', 0)
            market_cap = financial_metrics.get('market_cap', 0)
            pe_ratio = financial_metrics.get('pe_ratio', 0)
            dividend_yield = financial_metrics.get('dividend_yield', 0)
            
            rec_color = '#27ae60' if recommendation == 'BUY' else '#e74c3c' if recommendation == 'SELL' else '#f39c12'
            
            comparison_html += f"""
                        <tr>
                            <td><strong>{ticker}</strong></td>
                            <td>{company_name}</td>
                            <td style="color: {rec_color}; font-weight: bold;">{recommendation}</td>
                            <td>${current_price:.2f}</td>
                            <td>${market_cap/1e9:.1f}B</td>
                            <td>{pe_ratio:.1f}x</td>
                            <td>{dividend_yield*100:.1f}%</td>
                        </tr>"""
        
        comparison_html += """
                    </tbody>
                </table>
            </div>
        </div>"""
        
        return comparison_html
    
    def _render_report_footer(self) -> str:
        """Render report footer."""
        return f"""
        <footer class="report-footer">
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            AgentInvest Financial Analysis System | 
            Data sources: Yahoo Finance, Annual Reports, Market Data Providers</p>
            <p style="font-size: 0.8em; color: #666;">
                Disclaimer: This report is for informational purposes only and does not constitute investment advice. 
                Past performance does not guarantee future results. Please consult with a qualified financial advisor 
                before making investment decisions.
            </p>
        </footer>"""
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report."""
        return """
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; background-color: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .report-header { text-align: center; padding: 30px 0; border-bottom: 3px solid #2c3e50; margin-bottom: 30px; }
        .report-header h1 { color: #2c3e50; font-size: 2.5em; margin-bottom: 10px; }
        .ticker-info { margin: 15px 0; }
        .ticker { background-color: #3498db; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold; font-size: 1.2em; margin-right: 10px; }
        .company-name { color: #7f8c8d; font-size: 1.1em; }
        .company-details { margin: 10px 0; }
        .company-details span { background-color: #ecf0f1; padding: 4px 8px; border-radius: 3px; margin: 0 5px; font-size: 0.9em; }
        .report-meta { color: #95a5a6; font-size: 0.9em; margin-top: 10px; }
        .report-meta span { margin: 0 15px; }
        .section { margin: 30px 0; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .section h2 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        .metric-label { font-weight: bold; color: #7f8c8d; font-size: 0.9em; }
        .metric-value { font-size: 1.3em; color: #2c3e50; margin-top: 5px; }
        .chart-container { position: relative; height: 400px; margin: 20px 0; }
        .comparison-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .comparison-table th, .comparison-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .comparison-table th { background-color: #f2f2f2; font-weight: bold; color: #2c3e50; }
        .comparison-table tr:hover { background-color: #f5f5f5; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .report-footer { text-align: center; padding: 20px 0; border-top: 1px solid #ecf0f1; margin-top: 30px; color: #7f8c8d; font-size: 0.9em; }
        .alert { padding: 15px; margin: 20px 0; border-radius: 5px; }
        .alert-info { background-color: #d1ecf1; border-left: 4px solid #bee5eb; color: #0c5460; }
        .alert-warning { background-color: #fff3cd; border-left: 4px solid #ffeaa7; color: #856404; }
        .alert-success { background-color: #d4edda; border-left: 4px solid #c3e6cb; color: #155724; }
        .alert-danger { background-color: #f8d7da; border-left: 4px solid #f5c6cb; color: #721c24; }
        .alert-light { background-color: #fefefe; border-left: 4px solid #e9ecef; color: #6c757d; }
        .executive-summary-content { line-height: 1.6; }
        .executive-summary-content h4 { color: #2c3e50; margin: 15px 0 10px 0; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }
        .executive-summary-content .investment-thesis p { font-size: 1.1em; margin-bottom: 15px; }
        .executive-summary-content .key-insights ul { margin: 10px 0; padding-left: 20px; }
        .executive-summary-content .key-insights li { margin: 8px 0; }
        .executive-summary-content .balance-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 10px; }
        .executive-summary-content .opportunities, .executive-summary-content .risks { padding: 10px; border-radius: 5px; }
        .executive-summary-content .opportunities { background-color: #d4edda; border-left: 3px solid #28a745; }
        .executive-summary-content .risks { background-color: #f8d7da; border-left: 3px solid #dc3545; }
        .executive-summary-content .opportunities ul, .executive-summary-content .risks ul { margin: 5px 0; padding-left: 15px; }
        .bulls-bears-analysis { margin: 20px 0; }
        .analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .bulls-section, .bears-section { padding: 15px; border-radius: 5px; }
        .bulls-section { background-color: #d4edda; border-left: 4px solid #28a745; }
        .bears-section { background-color: #f8d7da; border-left: 4px solid #dc3545; }
        .analysis-point { margin: 15px 0; padding: 10px; background-color: rgba(255,255,255,0.7); border-radius: 3px; }
        .analysis-point h5 { margin-bottom: 8px; }
        .quantitative-support { font-size: 0.9em; color: #666; margin-top: 8px; }
        .summary { margin-top: 15px; padding: 10px; background-color: rgba(255,255,255,0.9); border-radius: 3px; font-style: italic; }
        .financial-highlights-content h4 { color: #2c3e50; margin: 20px 0 10px 0; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }
        .price-analysis { margin: 20px 0; }
        @media (max-width: 768px) { 
            .container { padding: 10px; } 
            .metrics-grid { grid-template-columns: 1fr; } 
            .report-header h1 { font-size: 2em; } 
            .executive-summary-content .balance-grid { grid-template-columns: 1fr; }
            .analysis-grid { grid-template-columns: 1fr; }
        }
    </style>"""
