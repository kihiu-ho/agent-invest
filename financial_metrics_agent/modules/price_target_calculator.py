"""
Price Target Calculator Module

Handles price target calculations and valuation analysis for investment recommendations.
"""

from typing import Dict, Any


class PriceTargetCalculator:
    """Calculates price targets based on financial metrics and recommendation types."""
    
    def __init__(self):
        """Initialize price target calculator."""
        pass
    
    def generate_enhanced_price_target(self, financial_metrics: Dict, annual_report_data: Dict, 
                                     ticker: str, recommendation: str = None) -> str:
        """Generate enhanced price target with specific methodology and timeframe, aligned with recommendation."""

        current_price = financial_metrics.get('current_price', 0)
        pe_ratio = financial_metrics.get('pe_ratio', 15)
        dividend_yield = financial_metrics.get('dividend_yield', 0)

        if not current_price:
            return "Price target analysis pending comprehensive valuation model completion"

        # Enhanced price target calculation with annual report factors
        annual_strength = self._assess_annual_report_strength(annual_report_data)

        # Recommendation-aligned price target calculation
        if recommendation == 'BUY':
            # BUY: 15-30% upside potential
            base_multiplier = 1.15 + (annual_strength * 0.15)  # 15-30% upside
        elif recommendation == 'SELL':
            # SELL: 5-15% downside potential
            base_multiplier = 0.95 - (annual_strength * 0.10)  # 5-15% downside
        else:  # HOLD
            # HOLD: -5% to +12% range (conservative)
            base_multiplier = 1.02 + (annual_strength * 0.10)  # 2-12% upside for quality companies
            # Cap at 12% for HOLD recommendations
            base_multiplier = min(base_multiplier, 1.12)

        # Apply dividend yield and strategic adjustments (smaller for HOLD)
        dividend_component = dividend_yield * 0.05 if recommendation == 'HOLD' else dividend_yield * 0.10
        strategic_premium = annual_strength * 0.03 if recommendation == 'HOLD' else annual_strength * 0.08

        # Calculate conservative target price for HOLD
        target_price = current_price * base_multiplier * (1 + dividend_component + strategic_premium)

        # Calculate upside/downside
        price_change = ((target_price - current_price) / current_price) * 100
        direction = "upside" if price_change > 0 else "downside"

        currency = "HK$" if current_price > 50 else "$"  # Simple heuristic for HK stocks

        return f"{currency}{target_price:.2f} ({price_change:+.1f}% {direction} potential over 18-month horizon)"
    
    def generate_price_target_analysis(self, current_price: float, financial_metrics: Dict, 
                                     annual_report_data: Dict) -> str:
        """Generate MTR-style price target analysis."""

        if not current_price:
            return "Price target analysis pending additional data"

        # Simple price target calculation based on P/E and growth
        pe_ratio = financial_metrics.get('pe_ratio', 15)
        revenue_growth = financial_metrics.get('revenue_growth', 0)

        # Annual report strength assessment
        annual_strength = self._assess_annual_report_strength(annual_report_data)
        adjustment_factor = 1.0 + (annual_strength * 0.15)  # Up to 15% adjustment

        # Conservative target calculation
        target_pe = pe_ratio * (1 + max(revenue_growth, 0.02))  # Minimum 2% growth assumption
        target_price = current_price * (target_pe / pe_ratio) * adjustment_factor

        upside_potential = ((target_price - current_price) / current_price) * 100

        currency = "HK$" if current_price > 50 else "$"  # Simple heuristic for HK stocks

        return f"{currency}{target_price:.2f} ({upside_potential:+.1f}% upside/downside potential)"
    
    def _assess_annual_report_strength(self, annual_report_data: Dict) -> float:
        """Assess the strength of annual report data for price target calculations."""
        if not annual_report_data:
            return 0.5  # Default moderate strength
        
        strength_score = 0.0
        max_score = 0.0
        
        # Check for key data categories
        categories = ['business_model', 'financial_highlights', 'strategic_positioning', 
                     'risk_management', 'esg_framework']
        
        for category in categories:
            max_score += 1.0
            if category in annual_report_data:
                category_data = annual_report_data[category]
                if isinstance(category_data, dict) and category_data:
                    # Score based on data completeness
                    if len(category_data) >= 3:  # Has multiple data points
                        strength_score += 1.0
                    elif len(category_data) >= 1:  # Has some data
                        strength_score += 0.5
        
        # Normalize to 0-1 range
        if max_score > 0:
            return min(1.0, strength_score / max_score)
        else:
            return 0.5
    
    def generate_price_targets_subsection(self, price_targets: Dict) -> str:
        """Generate price targets subsection."""
        current_price = price_targets.get('current_price')
        target_mean = price_targets.get('target_mean')
        upside_potential = price_targets.get('upside_potential')

        if not current_price:
            return "<p>Price target data not available.</p>"

        upside_color = "text-success" if upside_potential and upside_potential > 0 else "text-danger"
        upside_text = f"{upside_potential:+.1f}%" if upside_potential else "N/A"

        return f"""
        <h3>🎯 Price Targets</h3>
        <div class="alert alert-light">
            <div class="row">
                <div class="col-md-4">
                    <strong>Current Price:</strong> ${current_price:.2f}
                </div>
                <div class="col-md-4">
                    <strong>Target Mean:</strong> ${target_mean:.2f}
                </div>
                <div class="col-md-4">
                    <strong>Upside Potential:</strong> <span class="{upside_color}">{upside_text}</span>
                </div>
            </div>
        </div>"""
