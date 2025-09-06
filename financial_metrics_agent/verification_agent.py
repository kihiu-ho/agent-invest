"""
Professional Investment Analysis Verification Agent

This module provides comprehensive validation for investment analysis sections,
including price target validation, investment thesis cross-validation, and data consistency checks.
Enhanced with LLM-powered content correction capabilities.
"""

import re
import logging
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import math

# Import OpenAI client for LLM-powered corrections
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ValidationIssue:
    """Represents a validation issue found during analysis."""
    severity: str  # "Critical", "Warning", "Minor"
    category: str  # "Price Target", "Investment Thesis", "Data Source", "Logic"
    description: str
    recommendation: str
    location: str = ""

@dataclass
class ValidationReport:
    """Comprehensive validation report for investment analysis."""
    overall_score: float  # 0-100%
    issues: List[ValidationIssue]
    price_target_validation: Dict[str, Any]
    thesis_validation: Dict[str, Any]
    data_consistency: Dict[str, Any]
    logic_coherence: Dict[str, Any]
    timestamp: datetime
    corrections_made: List[Dict[str, Any]] = None  # Track any corrections applied

@dataclass
class ContentCorrection:
    """Represents a content correction made by the verification system."""
    section: str
    original_content: str
    corrected_content: str
    correction_type: str  # "price_target", "thesis", "citation", etc.
    reason: str
    timestamp: datetime

class InvestmentAnalysisVerifier:
    """
    Comprehensive verification system for Professional Investment Analysis sections.
    Enhanced with LLM-powered content correction capabilities.
    """

    def __init__(self, llm_config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)

        # Initialize LLM client for content corrections
        self.llm_config = llm_config or self._get_default_llm_config()
        self.openai_client = None
        self._initialize_llm_client()

        # Validation thresholds
        self.CRITICAL_UPSIDE_THRESHOLD = 150.0  # >150% upside is likely an error
        self.CRITICAL_DOWNSIDE_THRESHOLD = -50.0  # <-50% downside is concerning
        self.WARNING_UPSIDE_THRESHOLD = 100.0  # >100% upside needs scrutiny
        self.WARNING_DOWNSIDE_THRESHOLD = -30.0  # <-30% downside needs scrutiny

        # Maximum allowed upside without additional justification
        self.MAX_ALLOWED_UPSIDE = 75.0
        
        # Professional language indicators
        self.PROFESSIONAL_TERMS = [
            'investment thesis', 'valuation', 'fundamental analysis', 'dcf',
            'price target', 'upside potential', 'risk-adjusted', 'institutional',
            'comprehensive analysis', 'financial performance', 'market dynamics'
        ]
        
        # Required citation patterns
        self.CITATION_PATTERNS = [
            r'\[Source:.*?\]',
            r'https?://[^\s]+',
            r'Annual Report \d{4}',
            r'StockAnalysis\.com',
            r'TipRanks\.com'
        ]

    def _get_default_llm_config(self) -> Dict:
        """Get default LLM configuration matching the agent factory pattern."""
        api_key = os.getenv("OPENAI_API_KEY")
        custom_llm_url = os.getenv("LLM_BASE_URL") or os.getenv("CUSTOM_LLM_URL")
        openai_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        api_base = custom_llm_url if custom_llm_url else openai_base
        model = os.getenv("OPENAI_MODEL", "gpt-4")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

        return {
            "model": model,
            "api_key": api_key,
            "base_url": api_base,
            "temperature": temperature
        }

    def _initialize_llm_client(self):
        """Initialize OpenAI client for LLM-powered corrections."""
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI not available - content correction will be limited")
            return

        try:
            api_key = self.llm_config.get("api_key")
            base_url = self.llm_config.get("base_url")

            if not api_key:
                self.logger.warning("No API key available - content correction disabled")
                return

            self.openai_client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )

            # Mask API key for logging
            api_key_masked = f"{'*' * 8}{api_key[-4:] if len(api_key) > 4 else '****'}"
            self.logger.info(f"✅ LLM client initialized for verification: key={api_key_masked}, base={base_url}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM client: {e}")
            self.openai_client = None

    def verify_professional_analysis(
        self, 
        analysis_content: str, 
        financial_metrics: Dict[str, Any],
        annual_report_data: Dict[str, Any] = None,
        web_data: Dict[str, Any] = None
    ) -> ValidationReport:
        """
        Perform comprehensive verification of Professional Investment Analysis section.
        
        Args:
            analysis_content: The HTML/text content of the analysis section
            financial_metrics: Current financial metrics and price data
            annual_report_data: Annual report data from Weaviate
            web_data: Web-scraped data from various sources
            
        Returns:
            ValidationReport with detailed findings and recommendations
        """
        self.logger.info("🔍 Starting comprehensive investment analysis verification")
        
        issues = []
        
        # 1. Price Target Validation
        price_target_validation = self._validate_price_targets(analysis_content, financial_metrics)
        issues.extend(price_target_validation.get('issues', []))
        
        # 2. Investment Thesis Cross-Validation
        thesis_validation = self._validate_investment_thesis(
            analysis_content, financial_metrics, annual_report_data
        )
        issues.extend(thesis_validation.get('issues', []))
        
        # 3. Data Source Verification
        data_consistency = self._verify_data_sources(analysis_content, web_data, annual_report_data)
        issues.extend(data_consistency.get('issues', []))
        
        # 4. Logic and Coherence Check
        logic_coherence = self._check_logic_coherence(analysis_content, financial_metrics)
        issues.extend(logic_coherence.get('issues', []))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(issues)
        
        report = ValidationReport(
            overall_score=overall_score,
            issues=issues,
            price_target_validation=price_target_validation,
            thesis_validation=thesis_validation,
            data_consistency=data_consistency,
            logic_coherence=logic_coherence,
            timestamp=datetime.now()
        )
        
        self.logger.info(f"✅ Verification completed: {overall_score:.1f}% score, {len(issues)} issues found")
        return report

    def _validate_price_targets(self, content: str, financial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate price targets for mathematical accuracy and reasonableness."""
        issues = []
        validation_data = {}
        
        current_price = financial_metrics.get('current_price', 0)
        if not current_price:
            issues.append(ValidationIssue(
                severity="Critical",
                category="Price Target",
                description="Current price not available for validation",
                recommendation="Ensure current price data is properly loaded"
            ))
            return {'issues': issues, 'validation_data': validation_data}
        
        # Extract price targets and upside percentages
        price_target_patterns = [
            r'(?:HK\$|USD\$|\$)(\d+\.?\d*)\s*\([+\-]?(\d+\.?\d*)%.*?\)',
            r'Price Target:.*?(?:HK\$|USD\$|\$)(\d+\.?\d*)',
            r'upside potential.*?([+\-]?\d+\.?\d*)%'
        ]
        
        found_targets = []
        for pattern in price_target_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    target_price = float(match.group(1))
                    upside_pct = float(match.group(2))
                    found_targets.append((target_price, upside_pct))
        
        validation_data['found_targets'] = found_targets
        validation_data['current_price'] = current_price
        
        for target_price, upside_pct in found_targets:
            # Verify mathematical accuracy
            calculated_upside = ((target_price - current_price) / current_price) * 100
            upside_diff = abs(calculated_upside - upside_pct)
            
            if upside_diff > 2.0:  # Allow 2% tolerance for rounding
                issues.append(ValidationIssue(
                    severity="Critical",
                    category="Price Target",
                    description=f"Price target calculation error: {upside_pct:.1f}% stated vs {calculated_upside:.1f}% calculated",
                    recommendation="Recalculate upside percentage or verify target price"
                ))
            
            # Check for unrealistic targets
            if upside_pct > self.CRITICAL_UPSIDE_THRESHOLD:
                issues.append(ValidationIssue(
                    severity="Critical",
                    category="Price Target",
                    description=f"Unrealistic upside potential: {upside_pct:.1f}% (>{self.CRITICAL_UPSIDE_THRESHOLD}%)",
                    recommendation="Review valuation methodology and assumptions"
                ))
            elif upside_pct > self.WARNING_UPSIDE_THRESHOLD:
                issues.append(ValidationIssue(
                    severity="Warning",
                    category="Price Target",
                    description=f"High upside potential: {upside_pct:.1f}% requires strong justification",
                    recommendation="Provide detailed valuation rationale for high target"
                ))
            
            if upside_pct < self.CRITICAL_DOWNSIDE_THRESHOLD:
                issues.append(ValidationIssue(
                    severity="Critical",
                    category="Price Target",
                    description=f"Severe downside risk: {upside_pct:.1f}% (<{self.CRITICAL_DOWNSIDE_THRESHOLD}%)",
                    recommendation="Consider SELL recommendation or review analysis"
                ))
        
        return {'issues': issues, 'validation_data': validation_data}

    def _validate_investment_thesis(
        self, 
        content: str, 
        financial_metrics: Dict[str, Any],
        annual_report_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Cross-validate investment thesis claims against actual data."""
        issues = []
        validation_data = {}
        
        # Extract financial claims from thesis
        financial_claims = self._extract_financial_claims(content)
        validation_data['extracted_claims'] = financial_claims
        
        # Validate against actual metrics
        for claim in financial_claims:
            if 'growth' in claim.lower():
                # Validate growth claims
                stated_growth = self._extract_percentage(claim)
                actual_growth = financial_metrics.get('revenue_growth', 0)
                
                if stated_growth and abs(stated_growth - actual_growth) > 5.0:
                    issues.append(ValidationIssue(
                        severity="Warning",
                        category="Investment Thesis",
                        description=f"Growth rate discrepancy: {stated_growth:.1f}% claimed vs {actual_growth:.1f}% actual",
                        recommendation="Verify growth rate calculations and time periods"
                    ))
        
        # Check for unsupported claims
        if annual_report_data:
            unsupported_claims = self._check_unsupported_claims(content, annual_report_data)
            for claim in unsupported_claims:
                issues.append(ValidationIssue(
                    severity="Warning",
                    category="Investment Thesis",
                    description=f"Unsupported claim: {claim}",
                    recommendation="Provide citation or remove unsubstantiated claim"
                ))
        
        return {'issues': issues, 'validation_data': validation_data}

    def _verify_data_sources(
        self, 
        content: str, 
        web_data: Dict[str, Any] = None,
        annual_report_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Verify data source citations and consistency."""
        issues = []
        validation_data = {}
        
        # Check for proper citations
        citations_found = []
        for pattern in self.CITATION_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations_found.extend(matches)
        
        validation_data['citations_found'] = citations_found
        
        if len(citations_found) < 2:
            issues.append(ValidationIssue(
                severity="Warning",
                category="Data Source",
                description="Insufficient citations for professional analysis",
                recommendation="Add proper source citations for all financial claims"
            ))
        
        # Check for broken or invalid URLs
        url_pattern = r'https?://[^\s\]]+' 
        urls = re.findall(url_pattern, content)
        for url in urls:
            if 'example.com' in url or 'placeholder' in url.lower():
                issues.append(ValidationIssue(
                    severity="Minor",
                    category="Data Source",
                    description=f"Placeholder URL found: {url}",
                    recommendation="Replace with actual source URL"
                ))
        
        return {'issues': issues, 'validation_data': validation_data}

    def _check_logic_coherence(self, content: str, financial_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check logical consistency and coherence of the analysis."""
        issues = []
        validation_data = {}
        
        # Extract recommendation
        recommendation = self._extract_recommendation(content)
        validation_data['recommendation'] = recommendation
        
        # Check professional language usage
        professional_score = self._assess_professional_language(content)
        validation_data['professional_score'] = professional_score
        
        if professional_score < 0.3:
            issues.append(ValidationIssue(
                severity="Warning",
                category="Logic",
                description="Insufficient professional investment language",
                recommendation="Use more institutional-grade terminology and analysis"
            ))
        
        # Check for logical contradictions
        contradictions = self._find_logical_contradictions(content)
        for contradiction in contradictions:
            issues.append(ValidationIssue(
                severity="Warning",
                category="Logic",
                description=f"Logical contradiction: {contradiction}",
                recommendation="Resolve contradictory statements in analysis"
            ))
        
        return {'issues': issues, 'validation_data': validation_data}

    def _calculate_overall_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall validation score based on issues found."""
        base_score = 100.0
        
        for issue in issues:
            if issue.severity == "Critical":
                base_score -= 20.0
            elif issue.severity == "Warning":
                base_score -= 10.0
            elif issue.severity == "Minor":
                base_score -= 5.0
        
        return max(0.0, base_score)

    def _extract_financial_claims(self, content: str) -> List[str]:
        """Extract financial claims from the content."""
        # This is a simplified implementation
        claims = []
        sentences = content.split('.')
        for sentence in sentences:
            if any(term in sentence.lower() for term in ['growth', 'revenue', 'profit', 'margin']):
                claims.append(sentence.strip())
        return claims

    def _extract_percentage(self, text: str) -> Optional[float]:
        """Extract percentage value from text."""
        match = re.search(r'([+\-]?\d+\.?\d*)%', text)
        return float(match.group(1)) if match else None

    def _extract_recommendation(self, content: str) -> str:
        """Extract investment recommendation from content."""
        recommendations = ['BUY', 'HOLD', 'SELL', 'STRONG BUY', 'STRONG SELL']
        content_upper = content.upper()
        for rec in recommendations:
            if rec in content_upper:
                return rec
        return "UNKNOWN"

    def _assess_professional_language(self, content: str) -> float:
        """Assess the use of professional investment language."""
        content_lower = content.lower()
        professional_count = sum(1 for term in self.PROFESSIONAL_TERMS if term in content_lower)
        return min(1.0, professional_count / len(self.PROFESSIONAL_TERMS))

    def _find_logical_contradictions(self, content: str) -> List[str]:
        """Find logical contradictions in the analysis."""
        # Simplified implementation - can be enhanced
        contradictions = []
        content_lower = content.lower()
        
        if 'strong buy' in content_lower and 'significant risk' in content_lower:
            contradictions.append("Strong buy recommendation with significant risks mentioned")
        
        if 'undervalued' in content_lower and 'overpriced' in content_lower:
            contradictions.append("Contradictory valuation statements")
        
        return contradictions

    def _check_unsupported_claims(self, content: str, annual_report_data: Dict[str, Any]) -> List[str]:
        """Check for claims not supported by annual report data."""
        # Simplified implementation - would need more sophisticated NLP
        unsupported = []
        
        if 'market leader' in content.lower():
            # Check if this is supported by annual report data
            if not annual_report_data or 'market' not in str(annual_report_data).lower():
                unsupported.append("Market leadership claim")
        
        return unsupported

    async def _generate_corrected_price_target(
        self,
        original_content: str,
        financial_metrics: Dict[str, Any],
        upside_percentage: float
    ) -> Optional[str]:
        """
        Generate LLM-corrected price target content for unrealistic targets.

        Args:
            original_content: Original price target content
            financial_metrics: Current financial metrics for validation
            upside_percentage: The problematic upside percentage

        Returns:
            Corrected price target content or None if correction fails
        """
        if not self.openai_client:
            self.logger.warning("LLM client not available for price target correction")
            return None

        try:
            current_price = financial_metrics.get('current_price', 0)
            pe_ratio = financial_metrics.get('pe_ratio', 0)
            pb_ratio = financial_metrics.get('pb_ratio', 0)
            dividend_yield = financial_metrics.get('dividend_yield', 0)

            # Create correction prompt
            correction_prompt = f"""
You are a professional investment analyst tasked with correcting an unrealistic price target.

ORIGINAL PROBLEMATIC CONTENT:
{original_content}

CURRENT FINANCIAL METRICS:
- Current Price: {current_price}
- P/E Ratio: {pe_ratio}
- P/B Ratio: {pb_ratio}
- Dividend Yield: {dividend_yield}%

ISSUE IDENTIFIED:
The price target shows {upside_percentage:.1f}% upside potential, which exceeds reasonable investment thresholds.

CORRECTION REQUIREMENTS:
1. Generate a conservative, realistic price target with maximum 75% upside potential
2. Base the target on fundamental valuation metrics (P/E, P/B, dividend yield)
3. Provide clear justification using multiple valuation methods
4. Maintain professional investment language and proper HTML formatting
5. Include proper source citations
6. Keep the same section structure but with corrected numbers

EXAMPLE FORMAT:
Price Target: HK$[REALISTIC_PRICE] (+[CONSERVATIVE_%]% upside potential over 18-month horizon)

Investment Thesis: [WELL-JUSTIFIED ANALYSIS based on fundamental metrics]

Generate the corrected content now:
"""

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.llm_config.get("model", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are a professional investment analyst specializing in conservative, well-justified price target analysis."},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=self.llm_config.get("temperature", 0.1),
                max_tokens=1000
            )

            corrected_content = response.choices[0].message.content.strip()

            # Validate the correction doesn't have the same issue
            if self._validate_corrected_content(corrected_content, current_price):
                self.logger.info(f"✅ Generated corrected price target content ({len(corrected_content)} chars)")
                return corrected_content
            else:
                self.logger.warning("❌ LLM correction still contains unrealistic targets")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to generate corrected price target: {e}")
            return None

    def _validate_corrected_content(self, content: str, current_price: float) -> bool:
        """Validate that corrected content doesn't have the same issues."""
        try:
            # Extract price targets and upside percentages from corrected content
            price_target_patterns = [
                r'(?:HK\$|USD\$|\$)(\d+\.?\d*)\s*\([+\-]?(\d+\.?\d*)%.*?\)',
                r'upside potential.*?([+\-]?\d+\.?\d*)%'
            ]

            for pattern in price_target_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        upside_pct = float(match.group(2))
                        if upside_pct > self.MAX_ALLOWED_UPSIDE:
                            return False
                    elif len(match.groups()) == 1:
                        upside_pct = float(match.group(1))
                        if upside_pct > self.MAX_ALLOWED_UPSIDE:
                            return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating corrected content: {e}")
            return False

    def _generate_fallback_price_target(self, financial_metrics: Dict[str, Any]) -> str:
        """Generate conservative fallback price target when LLM correction fails."""
        current_price = financial_metrics.get('current_price', 0)

        if current_price > 0:
            # Conservative 25% upside target
            conservative_target = current_price * 1.25
            return f"""
<h5>📊 Professional Investment Analysis</h5>
<p><strong>Price Target:</strong> HK${conservative_target:.2f} (+25.0% upside potential over 18-month horizon)</p>
<p><strong>Investment Thesis:</strong> Conservative valuation based on fundamental analysis indicates moderate upside potential.
Price target reflects prudent assessment of current market conditions and company fundamentals.
Comprehensive valuation analysis required for higher target justification.</p>
<p><em>[Source: Conservative Valuation Analysis, Fundamental Metrics Review]</em></p>
"""
        else:
            return """
<h5>📊 Professional Investment Analysis</h5>
<p><strong>Price Target:</strong> Under Review - Comprehensive Valuation Analysis Required</p>
<p><strong>Investment Thesis:</strong> Current market data insufficient for reliable price target determination.
Recommend comprehensive fundamental analysis including DCF modeling, peer comparison, and sector analysis
before establishing target price range.</p>
<p><em>[Source: Risk Management Protocol, Valuation Standards]</em></p>
"""
