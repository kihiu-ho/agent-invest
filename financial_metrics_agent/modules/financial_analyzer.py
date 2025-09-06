"""
Financial Analyzer

Performs comprehensive financial analysis and generates investment insights
based on processed financial data without hardcoded ticker-specific logic.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .report_data_processor import ProcessedFinancialData, CompanyProfile

logger = logging.getLogger(__name__)

@dataclass
class InvestmentAnalysis:
    """Investment analysis results."""
    recommendation: str  # "BUY", "HOLD", "SELL"
    confidence_score: int  # 1-10
    key_strengths: List[str]
    key_risks: List[str]
    price_target: Optional[float]
    upside_potential: float
    investment_thesis: str
    risk_assessment: str
    valuation_summary: str

@dataclass
class InvestmentRecommendation:
    """Final investment recommendation."""
    rating: str
    emoji: str
    confidence_score: int
    key_rationale: str
    detailed_reasoning: str
    supporting_factors: List[str]
    risk_factors: List[str]
    price_analysis: Dict[str, Any]

@dataclass
class RiskAssessment:
    """Risk assessment results."""
    overall_risk_level: str  # "Low", "Moderate", "High"
    operational_risks: List[str]
    market_risks: List[str]
    financial_risks: List[str]
    regulatory_risks: List[str]
    risk_mitigation_factors: List[str]

class FinancialAnalyzer:
    """Analyzes financial data and generates investment insights."""
    
    def __init__(self):
        """Initialize the financial analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_investment_potential(self, data: ProcessedFinancialData, profile: CompanyProfile, 
                                   derived_metrics: Dict[str, Any]) -> InvestmentAnalysis:
        """
        Analyze investment potential based on financial data and company profile.
        
        Args:
            data: Processed financial data
            profile: Company profile with characteristics
            derived_metrics: Calculated derived metrics
            
        Returns:
            InvestmentAnalysis with comprehensive analysis
        """
        try:
            # Generate investment recommendation
            recommendation = self._determine_investment_recommendation(data, profile, derived_metrics)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(data, profile, derived_metrics)
            
            # Identify key strengths and risks
            key_strengths = self._identify_key_strengths(data, profile, derived_metrics)
            key_risks = self._identify_key_risks(data, profile, derived_metrics)
            
            # Generate investment thesis
            investment_thesis = self._generate_investment_thesis(data, profile, derived_metrics)
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(data, profile)
            
            # Generate valuation summary
            valuation_summary = self._generate_valuation_summary(data, profile, derived_metrics)
            
            analysis = InvestmentAnalysis(
                recommendation=recommendation,
                confidence_score=confidence_score,
                key_strengths=key_strengths,
                key_risks=key_risks,
                price_target=data.price_target,
                upside_potential=derived_metrics.get('upside_potential', 0.0),
                investment_thesis=investment_thesis,
                risk_assessment=risk_assessment,
                valuation_summary=valuation_summary
            )
            
            self.logger.info(f"✅ Generated investment analysis for {data.ticker}: {recommendation} ({confidence_score}/10)")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing investment potential: {e}")
            raise
    
    def generate_recommendation(self, analysis: InvestmentAnalysis, data: ProcessedFinancialData, 
                              profile: CompanyProfile) -> InvestmentRecommendation:
        """
        Generate final investment recommendation.
        
        Args:
            analysis: Investment analysis results
            data: Processed financial data
            profile: Company profile
            
        Returns:
            InvestmentRecommendation with final recommendation
        """
        try:
            # Determine emoji based on recommendation
            emoji_map = {
                "BUY": "🚀",
                "HOLD": "⚖️", 
                "SELL": "⚠️"
            }
            
            # Generate key rationale
            key_rationale = self._generate_key_rationale(analysis, data, profile)
            
            # Generate detailed reasoning
            detailed_reasoning = self._generate_detailed_reasoning(analysis, data, profile)
            
            # Generate price analysis
            price_analysis = self._generate_price_analysis(data, analysis)
            
            recommendation = InvestmentRecommendation(
                rating=analysis.recommendation,
                emoji=emoji_map.get(analysis.recommendation, "📊"),
                confidence_score=analysis.confidence_score,
                key_rationale=key_rationale,
                detailed_reasoning=detailed_reasoning,
                supporting_factors=analysis.key_strengths,
                risk_factors=analysis.key_risks,
                price_analysis=price_analysis
            )
            
            self.logger.info(f"✅ Generated final recommendation for {data.ticker}")
            return recommendation
            
        except Exception as e:
            self.logger.error(f"❌ Error generating recommendation: {e}")
            raise
    
    def assess_risks_and_opportunities(self, data: ProcessedFinancialData, 
                                     profile: CompanyProfile) -> RiskAssessment:
        """
        Assess risks and opportunities based on company profile.
        
        Args:
            data: Processed financial data
            profile: Company profile
            
        Returns:
            RiskAssessment with comprehensive risk analysis
        """
        try:
            # Determine overall risk level
            overall_risk_level = self._determine_overall_risk_level(data, profile)
            
            # Identify specific risk categories
            operational_risks = self._identify_operational_risks(data, profile)
            market_risks = self._identify_market_risks(data, profile)
            financial_risks = self._identify_financial_risks(data, profile)
            regulatory_risks = self._identify_regulatory_risks(data, profile)
            
            # Identify risk mitigation factors
            risk_mitigation_factors = self._identify_risk_mitigation_factors(data, profile)
            
            risk_assessment = RiskAssessment(
                overall_risk_level=overall_risk_level,
                operational_risks=operational_risks,
                market_risks=market_risks,
                financial_risks=financial_risks,
                regulatory_risks=regulatory_risks,
                risk_mitigation_factors=risk_mitigation_factors
            )
            
            self.logger.info(f"✅ Generated risk assessment for {data.ticker}: {overall_risk_level} risk")
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"❌ Error assessing risks: {e}")
            raise
    
    def _determine_investment_recommendation(self, data: ProcessedFinancialData, profile: CompanyProfile, 
                                           derived_metrics: Dict[str, Any]) -> str:
        """Determine investment recommendation based on multiple factors."""
        score = 0
        
        # Growth factors
        if profile.growth_profile == "high_growth":
            score += 2
        elif profile.growth_profile == "moderate_growth":
            score += 1
        elif profile.growth_profile == "declining":
            score -= 2
        
        # Valuation factors
        if profile.valuation_profile == "undervalued":
            score += 2
        elif profile.valuation_profile == "overvalued":
            score -= 2
        
        # Profitability factors
        if profile.profitability_profile == "highly_profitable":
            score += 2
        elif profile.profitability_profile == "profitable":
            score += 1
        elif profile.profitability_profile == "unprofitable":
            score -= 2
        
        # Risk factors
        if profile.risk_profile == "low_risk":
            score += 1
        elif profile.risk_profile == "high_risk":
            score -= 1
        
        # Analyst confidence
        analyst_confidence = derived_metrics.get('analyst_confidence', 5.0)
        if analyst_confidence >= 7.0:
            score += 1
        elif analyst_confidence <= 3.0:
            score -= 1
        
        # Determine recommendation
        if score >= 3:
            return "BUY"
        elif score <= -2:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_confidence_score(self, data: ProcessedFinancialData, profile: CompanyProfile, 
                                  derived_metrics: Dict[str, Any]) -> int:
        """Calculate confidence score (1-10) based on data quality and consistency."""
        base_score = 5
        
        # Data quality factors
        if data.analyst_count >= 5:
            base_score += 1
        if data.current_price > 0 and data.market_cap > 0:
            base_score += 1
        
        # Consistency factors
        growth_quality = derived_metrics.get('growth_quality_score', 5.0)
        financial_strength = derived_metrics.get('financial_strength_score', 5.0)
        
        if growth_quality >= 7.0 and financial_strength >= 7.0:
            base_score += 2
        elif growth_quality >= 6.0 and financial_strength >= 6.0:
            base_score += 1
        
        return min(10, max(1, base_score))
    
    def _identify_key_strengths(self, data: ProcessedFinancialData, profile: CompanyProfile, 
                              derived_metrics: Dict[str, Any]) -> List[str]:
        """Identify key investment strengths."""
        strengths = []
        
        # Growth strengths
        if profile.growth_profile == "high_growth":
            strengths.append(f"Strong revenue growth of {data.revenue_growth*100:.1f}% demonstrates robust business expansion")
        
        # Profitability strengths
        if profile.profitability_profile in ["highly_profitable", "profitable"]:
            strengths.append(f"Solid profit margins of {data.profit_margin*100:.1f}% indicate efficient operations")
        
        # Dividend strengths
        if profile.dividend_profile in ["high_yield", "moderate_yield"]:
            strengths.append(f"Attractive dividend yield of {data.dividend_yield*100:.1f}% provides income generation")
        
        # Scale strengths
        if profile.size_category in ["mega_cap", "large_cap"]:
            strengths.append(f"Large market capitalization of ${data.market_cap/1e9:.1f}B provides stability and liquidity")
        
        # Valuation strengths
        if profile.valuation_profile == "undervalued":
            strengths.append(f"Attractive valuation with P/E ratio of {data.pe_ratio:.1f}x offers potential upside")
        
        # Financial strength
        financial_strength = derived_metrics.get('financial_strength_score', 5.0)
        if financial_strength >= 7.0:
            strengths.append("Strong balance sheet with healthy financial metrics supports long-term stability")
        
        return strengths[:5]  # Limit to top 5 strengths
    
    def _identify_key_risks(self, data: ProcessedFinancialData, profile: CompanyProfile, 
                          derived_metrics: Dict[str, Any]) -> List[str]:
        """Identify key investment risks."""
        risks = []
        
        # Growth risks
        if profile.growth_profile == "declining":
            risks.append(f"Declining revenue growth of {data.revenue_growth*100:.1f}% indicates business challenges")
        
        # Valuation risks
        if profile.valuation_profile == "overvalued":
            risks.append(f"High P/E ratio of {data.pe_ratio:.1f}x suggests potential valuation concerns")
        
        # Financial risks
        if data.debt_to_equity > 0.6:
            risks.append(f"High debt-to-equity ratio of {data.debt_to_equity:.1f}x increases financial risk")
        
        # Market risks
        if data.beta > 1.5:
            risks.append(f"High beta of {data.beta:.1f} indicates increased market volatility exposure")
        
        # Profitability risks
        if profile.profitability_profile == "unprofitable":
            risks.append("Negative profit margins indicate operational challenges requiring attention")
        
        # Dividend risks
        if profile.dividend_profile == "high_yield" and data.dividend_yield > 0.08:
            risks.append("Very high dividend yield may indicate sustainability concerns")
        
        return risks[:5]  # Limit to top 5 risks
    
    def _generate_investment_thesis(self, data: ProcessedFinancialData, profile: CompanyProfile, 
                                  derived_metrics: Dict[str, Any]) -> str:
        """Generate comprehensive investment thesis."""
        company_name = data.company_name
        
        # Build thesis based on company profile
        thesis_parts = []
        
        # Opening statement
        thesis_parts.append(f"{company_name} presents a {profile.valuation_profile} investment opportunity "
                          f"in the {profile.business_model} sector with {profile.growth_profile} characteristics.")
        
        # Key value proposition
        if profile.size_category in ["mega_cap", "large_cap"]:
            thesis_parts.append(f"As a {profile.size_category} company with ${data.market_cap/1e9:.1f}B market "
                              f"capitalization, it offers institutional-grade scale and market presence.")
        
        # Growth narrative
        if profile.growth_profile == "high_growth":
            thesis_parts.append(f"Strong revenue growth of {data.revenue_growth*100:.1f}% combined with "
                              f"earnings expansion demonstrates robust business momentum.")
        elif profile.dividend_profile in ["high_yield", "moderate_yield"]:
            thesis_parts.append(f"Attractive dividend yield of {data.dividend_yield*100:.1f}% provides "
                              f"compelling income generation for yield-focused investors.")
        
        # Risk-return profile
        risk_return_score = derived_metrics.get('risk_adjusted_return', 5.0)
        if risk_return_score >= 7.0:
            thesis_parts.append("The risk-adjusted return profile appears favorable for long-term investors.")
        elif risk_return_score <= 3.0:
            thesis_parts.append("The risk-return profile requires careful consideration given current metrics.")
        
        return " ".join(thesis_parts)
    
    def _generate_risk_assessment(self, data: ProcessedFinancialData, profile: CompanyProfile) -> str:
        """Generate risk assessment summary."""
        risk_parts = []
        
        # Overall risk level
        risk_level = profile.risk_profile.replace('_', ' ').title()
        risk_parts.append(f"Overall risk profile is assessed as {risk_level} based on financial metrics.")
        
        # Specific risk factors
        if data.beta > 1.2:
            risk_parts.append(f"Market risk is elevated with beta of {data.beta:.1f}, indicating higher volatility.")
        
        if data.debt_to_equity > 0.5:
            risk_parts.append(f"Financial leverage of {data.debt_to_equity:.1f}x debt-to-equity requires monitoring.")
        
        if profile.growth_profile == "declining":
            risk_parts.append("Business momentum risks are present given declining growth trends.")
        
        return " ".join(risk_parts)
    
    def _generate_valuation_summary(self, data: ProcessedFinancialData, profile: CompanyProfile, 
                                  derived_metrics: Dict[str, Any]) -> str:
        """Generate valuation summary."""
        valuation_parts = []
        
        # Current valuation
        if data.pe_ratio > 0:
            valuation_parts.append(f"Currently trading at {data.pe_ratio:.1f}x P/E ratio, "
                                 f"which appears {profile.valuation_profile.replace('_', ' ')} "
                                 f"relative to growth prospects.")
        
        # Price target analysis
        if data.price_target and data.current_price > 0:
            upside = ((data.price_target - data.current_price) / data.current_price) * 100
            valuation_parts.append(f"Analyst price target of ${data.price_target:.2f} implies "
                                 f"{upside:+.1f}% potential return from current levels.")
        
        # Valuation attractiveness
        val_attractiveness = derived_metrics.get('valuation_attractiveness', 5.0)
        if val_attractiveness >= 7.0:
            valuation_parts.append("Valuation metrics suggest attractive entry opportunity.")
        elif val_attractiveness <= 3.0:
            valuation_parts.append("Current valuation appears stretched relative to fundamentals.")
        
        return " ".join(valuation_parts)
    
    def _generate_key_rationale(self, analysis: InvestmentAnalysis, data: ProcessedFinancialData, 
                              profile: CompanyProfile) -> str:
        """Generate key rationale for the recommendation."""
        if analysis.recommendation == "BUY":
            return f"Strong fundamentals with {profile.growth_profile.replace('_', ' ')} profile and attractive valuation metrics"
        elif analysis.recommendation == "SELL":
            return f"Concerning fundamentals with {profile.risk_profile.replace('_', ' ')} risk profile warrant caution"
        else:
            return f"Balanced risk-return profile with {profile.valuation_profile.replace('_', ' ')} valuation suggests holding"
    
    def _generate_detailed_reasoning(self, analysis: InvestmentAnalysis, data: ProcessedFinancialData, 
                                   profile: CompanyProfile) -> str:
        """Generate detailed reasoning for the recommendation."""
        reasoning_parts = []
        
        # Investment thesis
        reasoning_parts.append(f"<p><strong>Investment Thesis:</strong> {analysis.investment_thesis}</p>")
        
        # Key metrics
        reasoning_parts.append(f"<p><strong>Key Metrics:</strong> Current price ${data.current_price:.2f}, "
                             f"Market cap ${data.market_cap/1e9:.1f}B, P/E {data.pe_ratio:.1f}x, "
                             f"Revenue growth {data.revenue_growth*100:+.1f}%</p>")
        
        # Risk assessment
        reasoning_parts.append(f"<p><strong>Risk Assessment:</strong> {analysis.risk_assessment}</p>")
        
        # Valuation summary
        reasoning_parts.append(f"<p><strong>Valuation:</strong> {analysis.valuation_summary}</p>")
        
        return "".join(reasoning_parts)
    
    def _generate_price_analysis(self, data: ProcessedFinancialData, analysis: InvestmentAnalysis) -> Dict[str, Any]:
        """Generate price analysis data."""
        return {
            'current_price': data.current_price,
            'target_mean': data.price_target,
            'upside_potential': analysis.upside_potential
        }
    
    def _determine_overall_risk_level(self, data: ProcessedFinancialData, profile: CompanyProfile) -> str:
        """Determine overall risk level."""
        return profile.risk_profile.replace('_', ' ').title()
    
    def _identify_operational_risks(self, data: ProcessedFinancialData, profile: CompanyProfile) -> List[str]:
        """Identify operational risks."""
        risks = []
        
        if profile.profitability_profile == "unprofitable":
            risks.append("Operational efficiency challenges with negative profit margins")
        
        if profile.growth_profile == "declining":
            risks.append("Business momentum risks with declining revenue trends")
        
        if data.roe < 0.05:
            risks.append("Low return on equity indicates operational effectiveness concerns")
        
        return risks
    
    def _identify_market_risks(self, data: ProcessedFinancialData, profile: CompanyProfile) -> List[str]:
        """Identify market risks."""
        risks = []
        
        if data.beta > 1.5:
            risks.append("High market volatility exposure with elevated beta")
        
        if profile.size_category in ["small_cap", "micro_cap"]:
            risks.append("Liquidity and market access risks for smaller companies")
        
        return risks
    
    def _identify_financial_risks(self, data: ProcessedFinancialData, profile: CompanyProfile) -> List[str]:
        """Identify financial risks."""
        risks = []
        
        if data.debt_to_equity > 0.6:
            risks.append("Elevated financial leverage increases credit and refinancing risks")
        
        if profile.dividend_profile == "high_yield" and data.dividend_yield > 0.08:
            risks.append("Dividend sustainability concerns with very high yield")
        
        return risks
    
    def _identify_regulatory_risks(self, data: ProcessedFinancialData, profile: CompanyProfile) -> List[str]:
        """Identify regulatory risks."""
        risks = []
        
        if profile.business_model == "financial":
            risks.append("Regulatory capital and compliance requirements for financial institutions")
        
        if profile.business_model == "technology":
            risks.append("Data privacy and technology regulation compliance requirements")
        
        return risks
    
    def _identify_risk_mitigation_factors(self, data: ProcessedFinancialData, profile: CompanyProfile) -> List[str]:
        """Identify risk mitigation factors."""
        factors = []
        
        if profile.size_category in ["mega_cap", "large_cap"]:
            factors.append("Large scale provides operational resilience and market stability")
        
        if profile.profitability_profile in ["highly_profitable", "profitable"]:
            factors.append("Strong profitability provides financial flexibility")
        
        if data.debt_to_equity < 0.3:
            factors.append("Conservative debt levels provide financial stability")
        
        if profile.dividend_profile in ["moderate_yield", "low_yield"]:
            factors.append("Sustainable dividend policy supports long-term returns")
        
        return factors
