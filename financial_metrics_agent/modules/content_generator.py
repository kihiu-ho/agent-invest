"""
Content Generator

Generates investment content based on financial data patterns and company characteristics.
Eliminates hardcoded ticker-specific logic in favor of data-driven content generation.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .report_data_processor import ProcessedFinancialData, CompanyProfile
from .financial_analyzer import InvestmentAnalysis, InvestmentRecommendation, RiskAssessment
from .citation_manager import CitationManager

logger = logging.getLogger(__name__)

@dataclass
class BullsBears:
    """Bulls and bears analysis structure."""
    bulls_analysis: List[Dict[str, Any]]
    bears_analysis: List[Dict[str, Any]]
    bulls_summary: str
    bears_summary: str

@dataclass
class ContentTheme:
    """Content theme structure."""
    theme: str
    title: str
    content: str
    citations: List[str]
    quantitative_support: str
    point_number: int

class ContentGenerator:
    """Generates investment content based on financial data patterns."""
    
    def __init__(self, citation_manager: CitationManager):
        """
        Initialize content generator.
        
        Args:
            citation_manager: Citation manager instance
        """
        self.citation_manager = citation_manager
        self.logger = logging.getLogger(__name__)
    
    def generate_bulls_bears_analysis(self, data: ProcessedFinancialData, profile: CompanyProfile,
                                    analysis: InvestmentAnalysis, derived_metrics: Dict[str, Any]) -> BullsBears:
        """
        Generate bulls and bears analysis based on financial data.
        
        Args:
            data: Processed financial data
            profile: Company profile
            analysis: Investment analysis
            derived_metrics: Derived metrics
            
        Returns:
            BullsBears analysis structure
        """
        try:
            # Generate bull points
            bulls_analysis = self._generate_bull_points(data, profile, analysis, derived_metrics)
            
            # Generate bear points
            bears_analysis = self._generate_bear_points(data, profile, analysis, derived_metrics)
            
            # Generate summaries
            bulls_summary = self._generate_bulls_summary(bulls_analysis, data, profile)
            bears_summary = self._generate_bears_summary(bears_analysis, data, profile)
            
            bulls_bears = BullsBears(
                bulls_analysis=bulls_analysis,
                bears_analysis=bears_analysis,
                bulls_summary=bulls_summary,
                bears_summary=bears_summary
            )
            
            self.logger.info(f"✅ Generated bulls/bears analysis for {data.ticker}: {len(bulls_analysis)} bulls, {len(bears_analysis)} bears")
            return bulls_bears
            
        except Exception as e:
            self.logger.error(f"❌ Error generating bulls/bears analysis: {e}")
            raise
    
    def generate_executive_summary(self, data: ProcessedFinancialData, profile: CompanyProfile,
                                 recommendation: InvestmentRecommendation) -> str:
        """
        Generate executive summary based on company profile and recommendation.
        
        Args:
            data: Processed financial data
            profile: Company profile
            recommendation: Investment recommendation
            
        Returns:
            Executive summary HTML
        """
        try:
            # Generate investment thesis
            thesis = self._generate_investment_thesis_content(data, profile, recommendation)
            
            # Generate key insights
            key_insights = self._generate_key_insights(data, profile, recommendation)
            
            # Generate opportunities and risks
            opportunities = self._generate_opportunities_content(data, profile)
            risks = self._generate_risks_content(data, profile)
            
            executive_summary = f"""
            <div class="executive-summary-content">
                <h4>Investment Thesis</h4>
                {thesis}
                
                <h4>Key Investment Insights</h4>
                {key_insights}
                
                <div class="balance-grid">
                    <div class="opportunities">
                        <h5>🚀 Key Opportunities</h5>
                        {opportunities}
                    </div>
                    <div class="risks">
                        <h5>⚠️ Key Risks</h5>
                        {risks}
                    </div>
                </div>
            </div>"""
            
            self.logger.info(f"✅ Generated executive summary for {data.ticker}")
            return executive_summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating executive summary: {e}")
            raise
    
    def generate_financial_highlights(self, data: ProcessedFinancialData, profile: CompanyProfile,
                                    derived_metrics: Dict[str, Any]) -> str:
        """
        Generate financial highlights section.
        
        Args:
            data: Processed financial data
            profile: Company profile
            derived_metrics: Derived metrics
            
        Returns:
            Financial highlights HTML
        """
        try:
            # Generate performance narrative
            performance_narrative = self._generate_performance_narrative(data, profile)
            
            # Generate valuation narrative
            valuation_narrative = self._generate_valuation_narrative(data, profile, derived_metrics)
            
            # Generate dividend narrative if applicable
            dividend_narrative = self._generate_dividend_narrative(data, profile) if profile.dividend_profile != "no_dividend" else ""
            
            highlights = f"""
            <div class="financial-highlights">
                <h4>📊 Financial Performance</h4>
                {performance_narrative}
                
                <h4>💰 Valuation Analysis</h4>
                {valuation_narrative}
                
                {f'<h4>💵 Dividend Analysis</h4>{dividend_narrative}' if dividend_narrative else ''}
            </div>"""
            
            self.logger.info(f"✅ Generated financial highlights for {data.ticker}")
            return highlights
            
        except Exception as e:
            self.logger.error(f"❌ Error generating financial highlights: {e}")
            raise
    
    def _generate_bull_points(self, data: ProcessedFinancialData, profile: CompanyProfile,
                            analysis: InvestmentAnalysis, derived_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate bull points based on company strengths."""
        bull_points = []
        point_number = 1
        
        # Growth strength
        if profile.growth_profile in ["high_growth", "moderate_growth"]:
            bull_points.append(self._create_growth_bull_point(data, profile, point_number))
            point_number += 1
        
        # Profitability strength
        if profile.profitability_profile in ["highly_profitable", "profitable"]:
            bull_points.append(self._create_profitability_bull_point(data, profile, point_number))
            point_number += 1
        
        # Scale advantage
        if profile.size_category in ["mega_cap", "large_cap"]:
            bull_points.append(self._create_scale_bull_point(data, profile, point_number))
            point_number += 1
        
        # Dividend strength
        if profile.dividend_profile in ["high_yield", "moderate_yield"]:
            bull_points.append(self._create_dividend_bull_point(data, profile, point_number))
            point_number += 1
        
        # Valuation opportunity
        if profile.valuation_profile == "undervalued":
            bull_points.append(self._create_valuation_bull_point(data, profile, point_number))
            point_number += 1
        
        return bull_points[:4]  # Limit to top 4 bull points
    
    def _generate_bear_points(self, data: ProcessedFinancialData, profile: CompanyProfile,
                            analysis: InvestmentAnalysis, derived_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate bear points based on company risks."""
        bear_points = []
        point_number = 1
        
        # Growth concerns
        if profile.growth_profile == "declining":
            bear_points.append(self._create_growth_bear_point(data, profile, point_number))
            point_number += 1
        
        # Profitability concerns
        if profile.profitability_profile == "unprofitable":
            bear_points.append(self._create_profitability_bear_point(data, profile, point_number))
            point_number += 1
        
        # Valuation concerns
        if profile.valuation_profile == "overvalued":
            bear_points.append(self._create_valuation_bear_point(data, profile, point_number))
            point_number += 1
        
        # Financial risk concerns
        if data.debt_to_equity > 0.6:
            bear_points.append(self._create_financial_risk_bear_point(data, profile, point_number))
            point_number += 1
        
        # Market risk concerns
        if data.beta > 1.5:
            bear_points.append(self._create_market_risk_bear_point(data, profile, point_number))
            point_number += 1
        
        return bear_points[:4]  # Limit to top 4 bear points
    
    def _create_growth_bull_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create growth-focused bull point."""
        growth_descriptor = "robust" if data.revenue_growth > 0.15 else "solid"
        
        # Create dynamic citation
        citation = self.citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Revenue Growth Excellence',
            'title': f"💰 Strong Revenue Growth and Business Expansion",
            'content': (
                f"{data.company_name}'s {growth_descriptor} revenue growth of {data.revenue_growth*100:.1f}% "
                f"demonstrates strong business momentum and market demand for its {profile.business_model} "
                f"offerings. Combined with earnings growth of {data.earnings_growth*100:+.1f}%, the company "
                f"shows effective operational execution and scalable business model characteristics that "
                f"support long-term value creation for growth-oriented investors."
            ),
            'citations': [citation],
            'quantitative_support': f"Revenue growth: {data.revenue_growth*100:+.1f}%, Earnings growth: {data.earnings_growth*100:+.1f}%",
            'point_number': point_number
        }
    
    def _create_profitability_bull_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create profitability-focused bull point."""
        profitability_descriptor = "exceptional" if data.profit_margin > 0.20 else "strong"
        
        citation = self.citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Operational Excellence',
            'title': f"📈 {profitability_descriptor.title()} Profitability and Operational Efficiency",
            'content': (
                f"{data.company_name} demonstrates {profitability_descriptor} operational efficiency with "
                f"profit margins of {data.profit_margin*100:.1f}% and return on equity of {data.roe*100:.1f}%. "
                f"This {profile.profitability_profile.replace('_', ' ')} profile indicates effective cost management, "
                f"pricing power, and operational leverage that supports sustainable competitive advantages "
                f"in the {profile.business_model} sector."
            ),
            'citations': [citation],
            'quantitative_support': f"Profit margin: {data.profit_margin*100:.1f}%, ROE: {data.roe*100:.1f}%",
            'point_number': point_number
        }
    
    def _create_scale_bull_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create scale advantage bull point."""
        scale_descriptor = "exceptional" if profile.size_category == "mega_cap" else "significant"
        
        citation = self.citation_manager.create_dynamic_citation(
            'market_data', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Market Leadership and Scale',
            'title': f"🏢 {scale_descriptor.title()} Market Position and Scale Advantages",
            'content': (
                f"As a {profile.size_category.replace('_', ' ')} company with ${data.market_cap/1e9:.1f}B "
                f"market capitalization, {data.company_name} benefits from {scale_descriptor} scale advantages "
                f"including operational leverage, market access, and institutional investor appeal. "
                f"This {profile.geographic_scope} presence provides diversification benefits and "
                f"competitive positioning that supports long-term market leadership in {profile.business_model} services."
            ),
            'citations': [citation],
            'quantitative_support': f"Market cap: ${data.market_cap/1e9:.1f}B, {profile.size_category.replace('_', ' ')} status",
            'point_number': point_number
        }
    
    def _create_dividend_bull_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create dividend-focused bull point."""
        yield_descriptor = "attractive" if data.dividend_yield > 0.04 else "solid"
        
        citation = self.citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Income Generation',
            'title': f"💵 {yield_descriptor.title()} Dividend Income and Shareholder Returns",
            'content': (
                f"{data.company_name} offers {yield_descriptor} dividend income with a yield of "
                f"{data.dividend_yield*100:.1f}%, providing compelling income generation for yield-focused investors. "
                f"The {profile.dividend_profile.replace('_', ' ')} profile, supported by {profile.profitability_profile.replace('_', ' ')} "
                f"operations, indicates sustainable dividend policy and management commitment to shareholder returns "
                f"while maintaining financial flexibility for growth investments."
            ),
            'citations': [citation],
            'quantitative_support': f"Dividend yield: {data.dividend_yield*100:.1f}%, {profile.dividend_profile.replace('_', ' ')} profile",
            'point_number': point_number
        }
    
    def _create_valuation_bull_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create valuation opportunity bull point."""
        citation = self.citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Valuation Opportunity',
            'title': f"📊 Attractive Valuation Entry Opportunity",
            'content': (
                f"Trading at {data.pe_ratio:.1f}x P/E ratio, {data.company_name} appears {profile.valuation_profile.replace('_', ' ')} "
                f"relative to its {profile.growth_profile.replace('_', ' ')} profile and {profile.business_model} sector fundamentals. "
                f"The current valuation provides compelling risk-adjusted return potential for value-oriented investors "
                f"seeking exposure to {profile.profitability_profile.replace('_', ' ')} companies with sustainable competitive advantages."
            ),
            'citations': [citation],
            'quantitative_support': f"P/E ratio: {data.pe_ratio:.1f}x, {profile.valuation_profile.replace('_', ' ')} assessment",
            'point_number': point_number
        }
    
    def _create_growth_bear_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create growth concern bear point."""
        citation = self.citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Growth Challenges',
            'title': f"📉 Revenue Decline and Business Momentum Concerns",
            'content': (
                f"{data.company_name} faces significant business momentum challenges with revenue declining "
                f"{abs(data.revenue_growth)*100:.1f}% and earnings growth of {data.earnings_growth*100:+.1f}%. "
                f"This {profile.growth_profile} trend indicates potential market share loss, competitive pressures, "
                f"or operational challenges that require management attention and strategic repositioning to restore "
                f"sustainable growth trajectory in the {profile.business_model} sector."
            ),
            'citations': [citation],
            'quantitative_support': f"Revenue growth: {data.revenue_growth*100:+.1f}%, Earnings growth: {data.earnings_growth*100:+.1f}%",
            'point_number': point_number
        }
    
    def _create_profitability_bear_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create profitability concern bear point."""
        citation = self.citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Profitability Challenges',
            'title': f"⚠️ Operational Efficiency and Profitability Concerns",
            'content': (
                f"Operational challenges are evident with {data.company_name} reporting negative profit margins "
                f"of {data.profit_margin*100:.1f}% and return on equity of {data.roe*100:.1f}%. This {profile.profitability_profile} "
                f"profile indicates cost structure issues, pricing pressures, or operational inefficiencies that "
                f"require significant management intervention to restore sustainable profitability and shareholder value creation."
            ),
            'citations': [citation],
            'quantitative_support': f"Profit margin: {data.profit_margin*100:.1f}%, ROE: {data.roe*100:.1f}%",
            'point_number': point_number
        }
    
    def _create_valuation_bear_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create valuation concern bear point."""
        citation = self.citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Valuation Concerns',
            'title': f"📈 Elevated Valuation and Price Risk",
            'content': (
                f"Current valuation appears stretched with {data.company_name} trading at {data.pe_ratio:.1f}x P/E ratio, "
                f"which seems {profile.valuation_profile.replace('_', ' ')} relative to {profile.growth_profile.replace('_', ' ')} "
                f"fundamentals and {profile.business_model} sector comparables. This elevated valuation creates "
                f"limited upside potential and increased downside risk if the company fails to meet high growth expectations "
                f"or faces operational headwinds."
            ),
            'citations': [citation],
            'quantitative_support': f"P/E ratio: {data.pe_ratio:.1f}x, {profile.valuation_profile.replace('_', ' ')} assessment",
            'point_number': point_number
        }
    
    def _create_financial_risk_bear_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create financial risk bear point."""
        citation = self.citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Financial Risk',
            'title': f"⚖️ Elevated Financial Leverage and Credit Risk",
            'content': (
                f"Financial risk is elevated with {data.company_name} maintaining debt-to-equity ratio of "
                f"{data.debt_to_equity:.1f}x, indicating {profile.risk_profile.replace('_', ' ')} leverage profile. "
                f"This financial structure creates refinancing risk, interest rate sensitivity, and potential "
                f"covenant concerns that could constrain operational flexibility and limit strategic options "
                f"during economic downturns or industry challenges."
            ),
            'citations': [citation],
            'quantitative_support': f"Debt-to-equity: {data.debt_to_equity:.1f}x, {profile.risk_profile.replace('_', ' ')} profile",
            'point_number': point_number
        }
    
    def _create_market_risk_bear_point(self, data: ProcessedFinancialData, profile: CompanyProfile, point_number: int) -> Dict[str, Any]:
        """Create market risk bear point."""
        citation = self.citation_manager.create_dynamic_citation(
            'market_data', ticker=data.ticker, company_name=data.company_name
        )
        
        return {
            'theme': 'Market Volatility Risk',
            'title': f"🌪️ High Market Volatility and Beta Risk",
            'content': (
                f"{data.company_name}'s high beta of {data.beta:.1f} indicates elevated market volatility exposure "
                f"and {profile.risk_profile.replace('_', ' ')} characteristics. This sensitivity to market movements "
                f"creates amplified downside risk during market corrections and increased portfolio volatility "
                f"that may not be suitable for risk-averse investors seeking stable returns in the {profile.business_model} sector."
            ),
            'citations': [citation],
            'quantitative_support': f"Beta: {data.beta:.1f}, {profile.risk_profile.replace('_', ' ')} volatility profile",
            'point_number': point_number
        }
    
    def _generate_bulls_summary(self, bulls_analysis: List[Dict[str, Any]], data: ProcessedFinancialData, 
                              profile: CompanyProfile) -> str:
        """Generate bulls summary."""
        if not bulls_analysis:
            return f"{data.company_name} shows limited positive investment factors based on current financial metrics."
        
        key_themes = [bull['theme'] for bull in bulls_analysis]
        return (f"Bulls highlight {data.company_name}'s strengths in {', '.join(key_themes[:2]).lower()} "
               f"with {profile.size_category.replace('_', ' ')} scale advantages and {profile.profitability_profile.replace('_', ' ')} "
               f"operations supporting long-term value creation potential.")
    
    def _generate_bears_summary(self, bears_analysis: List[Dict[str, Any]], data: ProcessedFinancialData, 
                              profile: CompanyProfile) -> str:
        """Generate bears summary."""
        if not bears_analysis:
            return f"{data.company_name} shows limited significant risk factors based on current financial metrics."
        
        key_concerns = [bear['theme'] for bear in bears_analysis]
        return (f"Bears cite concerns about {', '.join(key_concerns[:2]).lower()} "
               f"with {profile.risk_profile.replace('_', ' ')} profile and {profile.valuation_profile.replace('_', ' ')} "
               f"metrics requiring careful risk assessment for potential investors.")
    
    def _generate_investment_thesis_content(self, data: ProcessedFinancialData, profile: CompanyProfile,
                                          recommendation: InvestmentRecommendation) -> str:
        """Generate investment thesis content."""
        return f"""
        <p>{data.company_name} represents a {profile.valuation_profile.replace('_', ' ')} investment opportunity 
        in the {profile.business_model} sector with {profile.growth_profile.replace('_', ' ')} characteristics 
        and {profile.size_category.replace('_', ' ')} market position.</p>
        
        <p>The company's {profile.profitability_profile.replace('_', ' ')} operations, combined with 
        {profile.risk_profile.replace('_', ' ')} risk profile, support a <strong>{recommendation.rating}</strong> 
        recommendation with {recommendation.confidence_score}/10 confidence based on comprehensive financial analysis.</p>
        """
    
    def _generate_key_insights(self, data: ProcessedFinancialData, profile: CompanyProfile,
                             recommendation: InvestmentRecommendation) -> str:
        """Generate key insights list."""
        insights = []
        
        # Financial performance insight
        if profile.profitability_profile in ["highly_profitable", "profitable"]:
            insights.append(f"Strong profitability with {data.profit_margin*100:.1f}% margins demonstrates operational efficiency")
        
        # Growth insight
        if profile.growth_profile in ["high_growth", "moderate_growth"]:
            insights.append(f"Revenue growth of {data.revenue_growth*100:+.1f}% indicates business momentum")
        
        # Dividend insight
        if profile.dividend_profile != "no_dividend":
            insights.append(f"Dividend yield of {data.dividend_yield*100:.1f}% provides income generation")
        
        # Valuation insight
        insights.append(f"Current P/E ratio of {data.pe_ratio:.1f}x appears {profile.valuation_profile.replace('_', ' ')}")
        
        # Risk insight
        insights.append(f"Overall risk profile assessed as {profile.risk_profile.replace('_', ' ')}")
        
        return "<ul>" + "".join([f"<li>{insight}</li>" for insight in insights[:5]]) + "</ul>"
    
    def _generate_opportunities_content(self, data: ProcessedFinancialData, profile: CompanyProfile) -> str:
        """Generate opportunities content."""
        opportunities = []
        
        if profile.growth_profile in ["high_growth", "moderate_growth"]:
            opportunities.append("Business expansion and market share growth potential")
        
        if profile.valuation_profile == "undervalued":
            opportunities.append("Valuation re-rating opportunity as fundamentals improve")
        
        if profile.size_category in ["mega_cap", "large_cap"]:
            opportunities.append("Scale advantages and operational leverage benefits")
        
        if profile.dividend_profile in ["moderate_yield", "high_yield"]:
            opportunities.append("Sustainable dividend income with potential for increases")
        
        return "<ul>" + "".join([f"<li>{opp}</li>" for opp in opportunities[:4]]) + "</ul>"
    
    def _generate_risks_content(self, data: ProcessedFinancialData, profile: CompanyProfile) -> str:
        """Generate risks content."""
        risks = []
        
        if profile.growth_profile == "declining":
            risks.append("Business momentum challenges and market share risks")
        
        if profile.valuation_profile == "overvalued":
            risks.append("Valuation compression risk if growth expectations not met")
        
        if data.debt_to_equity > 0.6:
            risks.append("Financial leverage and refinancing risks")
        
        if data.beta > 1.5:
            risks.append("High market volatility and systematic risk exposure")
        
        if profile.profitability_profile == "unprofitable":
            risks.append("Operational challenges and profitability restoration needs")
        
        return "<ul>" + "".join([f"<li>{risk}</li>" for risk in risks[:4]]) + "</ul>"
    
    def _generate_performance_narrative(self, data: ProcessedFinancialData, profile: CompanyProfile) -> str:
        """Generate performance narrative."""
        performance_descriptor = self._get_performance_descriptor(data.revenue_growth)
        
        return f"""
        <p>{data.company_name} demonstrates {performance_descriptor} financial performance with revenue growth of 
        {data.revenue_growth*100:+.1f}% and profit margins of {data.profit_margin*100:.1f}%. The company's 
        {profile.profitability_profile.replace('_', ' ')} profile reflects {self._get_efficiency_descriptor(data.profit_margin)} 
        operational execution in the {profile.business_model} sector.</p>
        """
    
    def _generate_valuation_narrative(self, data: ProcessedFinancialData, profile: CompanyProfile, 
                                    derived_metrics: Dict[str, Any]) -> str:
        """Generate valuation narrative."""
        valuation_attractiveness = derived_metrics.get('valuation_attractiveness', 5.0)
        attractiveness_descriptor = self._get_attractiveness_descriptor(valuation_attractiveness)
        
        return f"""
        <p>Current valuation metrics indicate {attractiveness_descriptor} investment opportunity with P/E ratio of 
        {data.pe_ratio:.1f}x appearing {profile.valuation_profile.replace('_', ' ')} relative to growth prospects. 
        The valuation assessment considers {profile.growth_profile.replace('_', ' ')} characteristics and 
        {profile.business_model} sector comparables.</p>
        """
    
    def _generate_dividend_narrative(self, data: ProcessedFinancialData, profile: CompanyProfile) -> str:
        """Generate dividend narrative."""
        sustainability_descriptor = self._get_sustainability_descriptor(data.dividend_yield, data.profit_margin)
        
        return f"""
        <p>Dividend policy offers {data.dividend_yield*100:.1f}% yield with {sustainability_descriptor} sustainability 
        profile based on {profile.profitability_profile.replace('_', ' ')} operations and conservative payout approach. 
        The {profile.dividend_profile.replace('_', ' ')} strategy supports income-focused investment objectives.</p>
        """
    
    def _get_performance_descriptor(self, growth_rate: float) -> str:
        """Get performance descriptor based on growth rate."""
        if growth_rate > 0.15:
            return "exceptional"
        elif growth_rate > 0.05:
            return "solid"
        elif growth_rate > 0:
            return "modest"
        else:
            return "challenging"
    
    def _get_efficiency_descriptor(self, profit_margin: float) -> str:
        """Get efficiency descriptor based on profit margin."""
        if profit_margin > 0.20:
            return "highly efficient"
        elif profit_margin > 0.10:
            return "efficient"
        elif profit_margin > 0:
            return "adequate"
        else:
            return "challenged"
    
    def _get_attractiveness_descriptor(self, attractiveness_score: float) -> str:
        """Get attractiveness descriptor based on score."""
        if attractiveness_score >= 7.0:
            return "an attractive"
        elif attractiveness_score >= 5.0:
            return "a reasonable"
        else:
            return "a challenging"
    
    def _get_sustainability_descriptor(self, dividend_yield: float, profit_margin: float) -> str:
        """Get sustainability descriptor based on yield and profitability."""
        if dividend_yield <= 0.06 and profit_margin > 0.10:
            return "strong"
        elif dividend_yield <= 0.08 and profit_margin > 0.05:
            return "adequate"
        else:
            return "uncertain"
