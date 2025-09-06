"""
Report Data Processor

Processes and normalizes financial data from various sources for report generation.
Extracts company characteristics and calculates derived metrics without hardcoded logic.
"""

import logging
from typing import Dict, List, Any, Optional, NamedTuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProcessedFinancialData:
    """Processed financial data structure."""
    ticker: str
    company_name: str
    sector: str
    current_price: float
    market_cap: float
    pe_ratio: float
    dividend_yield: float
    revenue_growth: float
    earnings_growth: float
    profit_margin: float
    debt_to_equity: float
    roe: float
    beta: float
    analyst_count: int
    recommendation_score: float
    price_target: Optional[float]
    raw_data: Dict[str, Any]

@dataclass
class CompanyProfile:
    """Company characteristics derived from financial data."""
    size_category: str  # "large_cap", "mid_cap", "small_cap"
    growth_profile: str  # "high_growth", "moderate_growth", "stable", "declining"
    dividend_profile: str  # "high_yield", "moderate_yield", "low_yield", "no_dividend"
    profitability_profile: str  # "highly_profitable", "profitable", "break_even", "unprofitable"
    risk_profile: str  # "low_risk", "moderate_risk", "high_risk"
    valuation_profile: str  # "undervalued", "fairly_valued", "overvalued"
    business_model: str  # "technology", "financial", "industrial", "consumer", "healthcare", "other"
    geographic_scope: str  # "global", "regional", "domestic"

class ReportDataProcessor:
    """Processes financial data and extracts company characteristics."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
    
    def process_financial_data(self, raw_data: Dict[str, Any]) -> ProcessedFinancialData:
        """
        Process raw financial data into a standardized format.
        
        Args:
            raw_data: Raw financial data from various sources
            
        Returns:
            ProcessedFinancialData object with normalized data
        """
        try:
            # Extract basic information
            ticker = raw_data.get("ticker", "Unknown")
            basic_info = raw_data.get("basic_info", {})
            
            # Extract financial metrics from multiple possible locations
            financial_metrics = self._extract_financial_metrics(raw_data)
            
            # Create processed data object
            processed_data = ProcessedFinancialData(
                ticker=ticker,
                company_name=basic_info.get('long_name', basic_info.get('short_name', ticker)),
                sector=basic_info.get('sector', 'Unknown'),
                current_price=self._safe_float(financial_metrics.get('current_price', 0)),
                market_cap=self._safe_float(financial_metrics.get('market_cap', 0)),
                pe_ratio=self._safe_float(financial_metrics.get('pe_ratio', 0)),
                dividend_yield=self._safe_float(financial_metrics.get('dividend_yield', 0)),
                revenue_growth=self._safe_float(financial_metrics.get('revenue_growth', 0)),
                earnings_growth=self._safe_float(financial_metrics.get('earnings_growth', 0)),
                profit_margin=self._safe_float(financial_metrics.get('profit_margin', 0)),
                debt_to_equity=self._safe_float(financial_metrics.get('debt_to_equity', 0)),
                roe=self._safe_float(financial_metrics.get('roe', 0)),
                beta=self._safe_float(financial_metrics.get('beta', 1.0)),
                analyst_count=int(financial_metrics.get('analyst_count', 0)),
                recommendation_score=self._safe_float(financial_metrics.get('recommendation_score', 3.0)),
                price_target=self._safe_float(financial_metrics.get('price_target')),
                raw_data=raw_data
            )
            
            self.logger.info(f"✅ Processed financial data for {ticker}: {processed_data.company_name}")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Error processing financial data: {e}")
            raise
    
    def extract_company_characteristics(self, data: ProcessedFinancialData) -> CompanyProfile:
        """
        Extract company characteristics based on financial metrics.
        
        Args:
            data: Processed financial data
            
        Returns:
            CompanyProfile with derived characteristics
        """
        try:
            profile = CompanyProfile(
                size_category=self._determine_size_category(data.market_cap),
                growth_profile=self._determine_growth_profile(data.revenue_growth, data.earnings_growth),
                dividend_profile=self._determine_dividend_profile(data.dividend_yield),
                profitability_profile=self._determine_profitability_profile(data.profit_margin, data.roe),
                risk_profile=self._determine_risk_profile(data.beta, data.debt_to_equity),
                valuation_profile=self._determine_valuation_profile(data.pe_ratio, data.revenue_growth),
                business_model=self._determine_business_model(data.sector, data.company_name),
                geographic_scope=self._determine_geographic_scope(data.company_name, data.sector)
            )
            
            self.logger.info(f"✅ Extracted company profile for {data.ticker}: {profile.business_model} {profile.size_category}")
            return profile
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting company characteristics: {e}")
            raise
    
    def calculate_derived_metrics(self, data: ProcessedFinancialData) -> Dict[str, Any]:
        """
        Calculate additional derived metrics for analysis.
        
        Args:
            data: Processed financial data
            
        Returns:
            Dictionary of derived metrics
        """
        try:
            derived_metrics = {
                'upside_potential': self._calculate_upside_potential(data.current_price, data.price_target),
                'growth_quality_score': self._calculate_growth_quality_score(data),
                'financial_strength_score': self._calculate_financial_strength_score(data),
                'valuation_attractiveness': self._calculate_valuation_attractiveness(data),
                'dividend_sustainability': self._calculate_dividend_sustainability(data),
                'analyst_confidence': self._calculate_analyst_confidence(data),
                'risk_adjusted_return': self._calculate_risk_adjusted_return(data)
            }
            
            self.logger.info(f"✅ Calculated derived metrics for {data.ticker}")
            return derived_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating derived metrics: {e}")
            return {}
    
    def _extract_financial_metrics(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial metrics from various possible locations in raw data."""
        financial_metrics = {}
        
        # Try multiple possible locations for financial data
        possible_locations = [
            raw_data.get('market_data', {}).get('financial_metrics', {}),
            raw_data.get('historical_data', {}).get('financial_metrics', {}),
            raw_data.get('financial_metrics', {}),
            raw_data.get('investment_decision', {}).get('financial_metrics', {})
        ]
        
        # Merge data from all locations, with later sources taking precedence
        for location in possible_locations:
            if isinstance(location, dict):
                financial_metrics.update(location)
        
        return financial_metrics
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _determine_size_category(self, market_cap: float) -> str:
        """Determine company size category based on market cap."""
        if market_cap >= 200e9:  # $200B+
            return "mega_cap"
        elif market_cap >= 10e9:  # $10B+
            return "large_cap"
        elif market_cap >= 2e9:  # $2B+
            return "mid_cap"
        elif market_cap >= 300e6:  # $300M+
            return "small_cap"
        else:
            return "micro_cap"
    
    def _determine_growth_profile(self, revenue_growth: float, earnings_growth: float) -> str:
        """Determine growth profile based on growth rates."""
        avg_growth = (revenue_growth + earnings_growth) / 2
        
        if avg_growth >= 0.20:  # 20%+
            return "high_growth"
        elif avg_growth >= 0.10:  # 10%+
            return "moderate_growth"
        elif avg_growth >= 0.0:  # Positive
            return "stable"
        else:
            return "declining"
    
    def _determine_dividend_profile(self, dividend_yield: float) -> str:
        """Determine dividend profile based on yield."""
        if dividend_yield >= 0.05:  # 5%+
            return "high_yield"
        elif dividend_yield >= 0.02:  # 2%+
            return "moderate_yield"
        elif dividend_yield > 0:
            return "low_yield"
        else:
            return "no_dividend"
    
    def _determine_profitability_profile(self, profit_margin: float, roe: float) -> str:
        """Determine profitability profile."""
        if profit_margin >= 0.20 and roe >= 0.15:  # 20% margin, 15% ROE
            return "highly_profitable"
        elif profit_margin >= 0.10 and roe >= 0.10:  # 10% margin, 10% ROE
            return "profitable"
        elif profit_margin >= 0.0 and roe >= 0.0:
            return "break_even"
        else:
            return "unprofitable"
    
    def _determine_risk_profile(self, beta: float, debt_to_equity: float) -> str:
        """Determine risk profile based on beta and leverage."""
        if beta <= 0.8 and debt_to_equity <= 0.3:
            return "low_risk"
        elif beta <= 1.2 and debt_to_equity <= 0.6:
            return "moderate_risk"
        else:
            return "high_risk"
    
    def _determine_valuation_profile(self, pe_ratio: float, revenue_growth: float) -> str:
        """Determine valuation profile based on P/E and growth."""
        if pe_ratio <= 0:
            return "not_applicable"
        
        # PEG-like calculation
        peg_ratio = pe_ratio / (revenue_growth * 100) if revenue_growth > 0 else float('inf')
        
        if peg_ratio <= 1.0:
            return "undervalued"
        elif peg_ratio <= 1.5:
            return "fairly_valued"
        else:
            return "overvalued"
    
    def _determine_business_model(self, sector: str, company_name: str) -> str:
        """Determine business model based on sector and company name."""
        sector_lower = sector.lower()
        name_lower = company_name.lower()
        
        if any(term in sector_lower for term in ['technology', 'software', 'internet']):
            return "technology"
        elif any(term in sector_lower for term in ['financial', 'bank', 'insurance']):
            return "financial"
        elif any(term in sector_lower for term in ['healthcare', 'pharmaceutical', 'biotech']):
            return "healthcare"
        elif any(term in sector_lower for term in ['consumer', 'retail', 'food']):
            return "consumer"
        elif any(term in sector_lower for term in ['industrial', 'manufacturing', 'materials']):
            return "industrial"
        elif any(term in sector_lower for term in ['energy', 'oil', 'gas', 'utilities']):
            return "energy"
        else:
            return "other"
    
    def _determine_geographic_scope(self, company_name: str, sector: str) -> str:
        """Determine geographic scope based on company characteristics."""
        name_lower = company_name.lower()
        
        # Look for global indicators
        global_indicators = ['global', 'international', 'worldwide', 'holdings']
        if any(indicator in name_lower for indicator in global_indicators):
            return "global"
        
        # Financial services often have broader scope
        if 'financial' in sector.lower():
            return "global"
        
        # Default to regional for most companies
        return "regional"
    
    def _calculate_upside_potential(self, current_price: float, price_target: Optional[float]) -> float:
        """Calculate upside potential percentage."""
        if not price_target or current_price <= 0:
            return 0.0
        return ((price_target - current_price) / current_price) * 100
    
    def _calculate_growth_quality_score(self, data: ProcessedFinancialData) -> float:
        """Calculate growth quality score (0-10)."""
        score = 5.0  # Base score
        
        # Revenue growth component
        if data.revenue_growth > 0.15:
            score += 2.0
        elif data.revenue_growth > 0.05:
            score += 1.0
        
        # Earnings growth component
        if data.earnings_growth > 0.15:
            score += 2.0
        elif data.earnings_growth > 0.05:
            score += 1.0
        
        # Profitability component
        if data.profit_margin > 0.15:
            score += 1.0
        
        return min(10.0, max(0.0, score))
    
    def _calculate_financial_strength_score(self, data: ProcessedFinancialData) -> float:
        """Calculate financial strength score (0-10)."""
        score = 5.0  # Base score
        
        # ROE component
        if data.roe > 0.15:
            score += 2.0
        elif data.roe > 0.10:
            score += 1.0
        
        # Debt management
        if data.debt_to_equity < 0.3:
            score += 2.0
        elif data.debt_to_equity < 0.6:
            score += 1.0
        
        # Profitability
        if data.profit_margin > 0.15:
            score += 1.0
        
        return min(10.0, max(0.0, score))
    
    def _calculate_valuation_attractiveness(self, data: ProcessedFinancialData) -> float:
        """Calculate valuation attractiveness score (0-10)."""
        if data.pe_ratio <= 0:
            return 5.0  # Neutral for companies without P/E
        
        score = 5.0  # Base score
        
        # P/E evaluation
        if data.pe_ratio < 15:
            score += 2.0
        elif data.pe_ratio < 25:
            score += 1.0
        elif data.pe_ratio > 40:
            score -= 2.0
        
        # Growth adjustment
        if data.revenue_growth > 0.10:
            score += 1.0
        
        # Dividend yield bonus
        if data.dividend_yield > 0.03:
            score += 1.0
        
        return min(10.0, max(0.0, score))
    
    def _calculate_dividend_sustainability(self, data: ProcessedFinancialData) -> float:
        """Calculate dividend sustainability score (0-10)."""
        if data.dividend_yield <= 0:
            return 0.0  # No dividend
        
        score = 5.0  # Base score
        
        # Yield evaluation
        if 0.02 <= data.dividend_yield <= 0.06:  # Sweet spot
            score += 2.0
        elif data.dividend_yield > 0.08:  # Too high might be unsustainable
            score -= 1.0
        
        # Profitability support
        if data.profit_margin > 0.10:
            score += 2.0
        
        # Financial strength
        if data.debt_to_equity < 0.5:
            score += 1.0
        
        return min(10.0, max(0.0, score))
    
    def _calculate_analyst_confidence(self, data: ProcessedFinancialData) -> float:
        """Calculate analyst confidence score (0-10)."""
        score = 5.0  # Base score
        
        # Analyst coverage
        if data.analyst_count >= 10:
            score += 2.0
        elif data.analyst_count >= 5:
            score += 1.0
        
        # Recommendation score (1=Strong Buy, 5=Strong Sell)
        if data.recommendation_score <= 2.0:
            score += 2.0
        elif data.recommendation_score <= 2.5:
            score += 1.0
        elif data.recommendation_score >= 4.0:
            score -= 2.0
        
        return min(10.0, max(0.0, score))
    
    def _calculate_risk_adjusted_return(self, data: ProcessedFinancialData) -> float:
        """Calculate risk-adjusted return score (0-10)."""
        score = 5.0  # Base score
        
        # Return potential
        upside = self._calculate_upside_potential(data.current_price, data.price_target)
        if upside > 20:
            score += 2.0
        elif upside > 10:
            score += 1.0
        elif upside < -10:
            score -= 2.0
        
        # Risk adjustment
        if data.beta < 0.8:
            score += 1.0
        elif data.beta > 1.5:
            score -= 1.0
        
        # Dividend cushion
        if data.dividend_yield > 0.03:
            score += 1.0
        
        return min(10.0, max(0.0, score))
