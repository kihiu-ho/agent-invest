#!/usr/bin/env python3
"""
Report Coordinator for Financial Analysis

Manages report generation, data aggregation, and output coordination
for comprehensive financial analysis reports.
"""

import logging
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# LangSmith tracing imports
try:
    from langsmith import traceable as _traceable
    LANGSMITH_AVAILABLE = True

    # Create a wrapper that ignores the 'name' parameter for compatibility
    def traceable(name=None):
        def decorator(func):
            return _traceable(func)
        return decorator

except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_technical_analysis: bool = True
    include_news_analysis: bool = True
    include_tipranks_data: bool = True
    include_bulls_bears: bool = True
    include_citations: bool = True
    enable_charts: bool = True
    output_format: str = "html"
    template_style: str = "comprehensive"

@dataclass
class ReportSection:
    """Configuration for a report section."""
    name: str
    enabled: bool = True
    priority: int = 1
    required_data: List[str] = None
    processor_function: str = None

class FinancialReportCoordinator:
    """
    Coordinates the generation of comprehensive financial analysis reports.
    
    Features:
    - Modular report section management
    - Data aggregation and validation
    - Multiple output formats
    - Enhanced error handling
    - Performance optimization
    """
    
    def __init__(self, config: Optional[ReportConfig] = None,
                 html_report_generator=None, orchestrator=None):
        """
        Initialize the report coordinator.
        
        Args:
            config: Report generation configuration
            html_report_generator: HTML report generator instance
            orchestrator: Main orchestrator instance for data processing
        """
        self.config = config or ReportConfig()
        self.html_report_generator = html_report_generator
        self.orchestrator = orchestrator
        
        # Define report sections
        self.report_sections = {
            "financial_metrics": ReportSection(
                name="financial_metrics",
                priority=1,
                required_data=["financial_metrics"]
            ),
            "technical_analysis": ReportSection(
                name="technical_analysis",
                enabled=self.config.include_technical_analysis,
                priority=2,
                processor_function="_process_technical_analysis"
            ),
            "news_analysis": ReportSection(
                name="news_analysis",
                enabled=self.config.include_news_analysis,
                priority=3,
                processor_function="_process_news_analysis"
            ),
            "tipranks_forecasts": ReportSection(
                name="tipranks_forecasts",
                enabled=self.config.include_tipranks_data,
                priority=4,
                processor_function="_process_tipranks_analyst_forecasts"
            ),
            "bulls_bears": ReportSection(
                name="bulls_bears",
                enabled=self.config.include_bulls_bears,
                priority=5,
                processor_function="_generate_bulls_bears_content"
            ),
            "citations": ReportSection(
                name="citations",
                enabled=self.config.include_citations,
                priority=6
            )
        }
        
        # Report generation statistics
        self.reports_generated = 0
        self.generation_errors = 0
        
        logger.info("✅ Financial report coordinator initialized")
    
    async def generate_comprehensive_report(self, ticker: str, analysis_data: Dict[str, Any],
                                          report_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive financial analysis report.
        
        Args:
            ticker: Stock ticker symbol
            analysis_data: Complete analysis data
            report_title: Custom report title
            
        Returns:
            Report generation results
        """
        logger.info(f"📝 Starting comprehensive report generation for {ticker}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare report data
            report_data = await self._prepare_report_data(ticker, analysis_data)
            
            # Process additional sections
            await self._process_report_sections(ticker, report_data)
            
            # Generate the report
            report_title = report_title or f"Comprehensive Financial Analysis Report - {ticker}"
            report_path = await self._generate_report_output(report_data, report_title)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.reports_generated += 1
            
            result = {
                "success": True,
                "report_path": report_path,
                "ticker": ticker,
                "execution_time": execution_time,
                "sections_included": [name for name, section in self.report_sections.items() if section.enabled],
                "data_sources": list(report_data.get("data_sources", {}).keys()),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Report generated successfully for {ticker}: {report_path}")
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.generation_errors += 1
            
            logger.error(f"❌ Report generation failed for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _prepare_report_data(self, ticker: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and validate data for report generation.
        
        Args:
            ticker: Stock ticker symbol
            analysis_data: Raw analysis data
            
        Returns:
            Prepared report data
        """
        # Start with the base analysis data
        report_data = dict(analysis_data)
        
        # Ensure required fields exist
        if "ticker" not in report_data:
            report_data["ticker"] = ticker
        
        if "timestamp" not in report_data:
            report_data["timestamp"] = datetime.now().isoformat()
        
        # Validate data sources
        if "data_sources" not in report_data:
            report_data["data_sources"] = {}
        
        # Add report metadata
        report_data["report_metadata"] = {
            "generated_by": "FinancialReportCoordinator",
            "generation_time": datetime.now().isoformat(),
            "config": {
                "include_technical_analysis": self.config.include_technical_analysis,
                "include_news_analysis": self.config.include_news_analysis,
                "include_tipranks_data": self.config.include_tipranks_data,
                "include_bulls_bears": self.config.include_bulls_bears,
                "include_citations": self.config.include_citations
            }
        }
        
        return report_data
    
    async def _process_report_sections(self, ticker: str, report_data: Dict[str, Any]):
        """
        Process additional report sections that require data processing.
        
        Args:
            ticker: Stock ticker symbol
            report_data: Report data to enhance
        """
        if not self.orchestrator:
            logger.warning("⚠️ Orchestrator not available for section processing")
            return
        
        # Process technical analysis
        if (self.report_sections["technical_analysis"].enabled and 
            "technical_analysis" not in report_data):
            try:
                technical_data = await self._process_technical_analysis(ticker)
                if technical_data:
                    report_data["technical_analysis"] = technical_data
                    logger.info(f"✅ Added technical analysis data for {ticker}")
            except Exception as e:
                logger.warning(f"⚠️ Technical analysis processing failed for {ticker}: {e}")
        
        # Process news analysis
        if (self.report_sections["news_analysis"].enabled and 
            "news_analysis" not in report_data):
            try:
                news_data = await self._process_news_analysis(ticker)
                if news_data:
                    report_data["news_analysis"] = news_data
                    logger.info(f"✅ Added news analysis data for {ticker}")
            except Exception as e:
                logger.warning(f"⚠️ News analysis processing failed for {ticker}: {e}")
        
        # Process TipRanks analyst forecasts
        if (self.report_sections["tipranks_forecasts"].enabled and 
            "tipranks_analyst_forecasts" not in report_data):
            try:
                tipranks_data = await self._process_tipranks_analyst_forecasts(ticker, report_data)
                if tipranks_data:
                    report_data["tipranks_analyst_forecasts"] = tipranks_data
                    logger.info(f"✅ Added TipRanks analyst forecast data for {ticker}")
            except Exception as e:
                logger.warning(f"⚠️ TipRanks processing failed for {ticker}: {e}")
        
        # Generate Bulls Say and Bears Say content
        if (self.report_sections["bulls_bears"].enabled and
            "bulls_bears_analysis" not in report_data):
            try:
                bulls_bears_data = await self._generate_bulls_bears_content(ticker, report_data)
                if bulls_bears_data:
                    report_data["bulls_bears_analysis"] = bulls_bears_data
                    logger.info(f"✅ Added Bulls Say and Bears Say analysis for {ticker}")
            except Exception as e:
                logger.warning(f"⚠️ Bulls/Bears analysis failed for {ticker}: {e}")

        # Generate Investment Decision (CRITICAL for simplified report)
        # Check if enhanced Investment Decision Agent data is already available
        if "investment_decision" not in report_data:
            try:
                # First try to use the orchestrator's enhanced Investment Decision Agent
                if hasattr(self.orchestrator, '_generate_investment_decision_with_agent'):
                    logger.info(f"🤖 Using enhanced Investment Decision Agent for {ticker}")
                    enhanced_decision = await self.orchestrator._generate_investment_decision_with_agent(
                        ticker, report_data, report_data.get('web_scraping', {}), None, None  # Weaviate data will be fetched inside
                    )
                    if enhanced_decision and enhanced_decision.get('recommendation'):
                        report_data["investment_decision"] = enhanced_decision
                        # Also add bulls_bears_analysis if available
                        if enhanced_decision.get('bulls_bears_analysis'):
                            report_data["bulls_bears_analysis"] = enhanced_decision['bulls_bears_analysis']
                        logger.info(f"✅ Added enhanced investment decision for {ticker}: {enhanced_decision.get('recommendation', 'UNKNOWN')}")
                    else:
                        logger.warning(f"⚠️ Enhanced Investment Decision Agent returned incomplete data for {ticker}")
                        raise Exception("Enhanced agent returned incomplete data")
                else:
                    logger.warning(f"⚠️ Enhanced Investment Decision Agent not available for {ticker}")
                    raise Exception("Enhanced agent not available")

            except Exception as e:
                logger.warning(f"⚠️ Enhanced Investment Decision Agent failed for {ticker}: {e}, falling back to basic method")
                try:
                    investment_decision = await self._generate_investment_decision(ticker, report_data)
                    if investment_decision:
                        report_data["investment_decision"] = investment_decision
                        logger.info(f"✅ Added fallback investment decision for {ticker}: {investment_decision.get('recommendation', 'UNKNOWN')}")
                    else:
                        logger.warning(f"⚠️ Fallback investment decision generation returned empty result for {ticker}")
                except Exception as fallback_error:
                    logger.error(f"❌ Both enhanced and fallback investment decision generation failed for {ticker}: {fallback_error}")
                    # Provide final fallback investment decision
                    report_data["investment_decision"] = {
                        "recommendation": "HOLD",
                        "emoji": "🟡",
                        "confidence_score": 1,
                        "key_rationale": f"Investment decision generation failed: {str(fallback_error)}",
                        "supporting_factors": [],
                        "risk_factors": [f"Analysis error: {str(fallback_error)}"],
                        "data_quality_impact": 0,
                        "error": str(fallback_error)
                    }

        # Add citation information
        if (self.report_sections["citations"].enabled and
            "citations" not in report_data and
            hasattr(self.orchestrator, 'data_collector') and
            hasattr(self.orchestrator.data_collector, 'citation_tracker')):
            try:
                citation_data = self.orchestrator.data_collector.citation_tracker.export_citations(ticker)
                report_data["citations"] = citation_data

                summary = citation_data.get("citation_summary", {})
                data_sources = summary.get("data_sources", 0)
                total_metrics = summary.get("total_metrics_tracked", 0)
                logger.info(f"✅ Added citation data: {data_sources} sources, {total_metrics} metrics")
            except Exception as e:
                logger.warning(f"⚠️ Citation processing failed for {ticker}: {e}")

        # Ensure web scraping and Weaviate data are included for executive summary
        await self._ensure_executive_summary_data(ticker, report_data)

    async def _ensure_executive_summary_data(self, ticker: str, report_data: Dict[str, Any]):
        """Ensure web scraping and Weaviate data are available for executive summary generation."""

        try:
            # Ensure web scraping data is available
            if "web_scraping" not in report_data and hasattr(self.orchestrator, 'data_collector'):
                try:
                    # Try to get web scraping data from orchestrator
                    if hasattr(self.orchestrator.data_collector, 'web_scraping_data'):
                        web_data = getattr(self.orchestrator.data_collector, 'web_scraping_data', {})
                        if web_data:
                            report_data["web_scraping"] = web_data
                            logger.info(f"✅ Added web scraping data for executive summary: {ticker}")

                    # Alternative: check if data is in the orchestrator's comprehensive data
                    elif hasattr(self.orchestrator, 'comprehensive_data'):
                        comp_data = getattr(self.orchestrator, 'comprehensive_data', {})
                        if comp_data.get('web_scraping'):
                            report_data["web_scraping"] = comp_data['web_scraping']
                            logger.info(f"✅ Added web scraping data from comprehensive data: {ticker}")

                except Exception as e:
                    logger.warning(f"⚠️ Could not retrieve web scraping data for {ticker}: {e}")

            # Ensure Weaviate/document data is available
            if "weaviate_queries" not in report_data and "document_analysis" not in report_data:
                try:
                    # Try to get Weaviate data from orchestrator
                    if hasattr(self.orchestrator, 'vector_store') and self.orchestrator.vector_store:
                        # Check if there's cached Weaviate data
                        weaviate_data = {}

                        # Try to get recent queries or document analysis
                        if hasattr(self.orchestrator.vector_store, 'get_recent_queries'):
                            recent_queries = self.orchestrator.vector_store.get_recent_queries(ticker)
                            if recent_queries:
                                weaviate_data['recent_queries'] = recent_queries

                        # Check comprehensive data for Weaviate results
                        if hasattr(self.orchestrator, 'comprehensive_data'):
                            comp_data = getattr(self.orchestrator, 'comprehensive_data', {})
                            if comp_data.get('weaviate_queries'):
                                weaviate_data.update(comp_data['weaviate_queries'])
                            elif comp_data.get('document_analysis'):
                                weaviate_data.update(comp_data['document_analysis'])

                        if weaviate_data:
                            report_data["weaviate_queries"] = weaviate_data
                            logger.info(f"✅ Added Weaviate data for executive summary: {ticker}")
                        else:
                            # Provide placeholder structure
                            report_data["weaviate_queries"] = {
                                "status": "limited_data",
                                "note": "Weaviate document analysis data not available for this session"
                            }

                except Exception as e:
                    logger.warning(f"⚠️ Could not retrieve Weaviate data for {ticker}: {e}")
                    report_data["weaviate_queries"] = {
                        "status": "error",
                        "error": str(e)
                    }

            # Log data availability for executive summary
            web_available = bool(report_data.get("web_scraping", {}).get("data_sources"))
            weaviate_available = bool(report_data.get("weaviate_queries", {}) and
                                    report_data["weaviate_queries"].get("status") != "error")

            logger.info(f"📋 Executive summary data availability for {ticker}: "
                       f"Web scraping: {'✅' if web_available else '❌'}, "
                       f"Weaviate: {'✅' if weaviate_available else '❌'}")

        except Exception as e:
            logger.error(f"❌ Failed to ensure executive summary data for {ticker}: {e}")

    async def _process_technical_analysis(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Process technical analysis data."""
        if hasattr(self.orchestrator, '_process_technical_analysis'):
            return self.orchestrator._process_technical_analysis(ticker)
        return None
    
    async def _process_news_analysis(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Process news analysis data."""
        if hasattr(self.orchestrator, '_process_news_analysis'):
            return self.orchestrator._process_news_analysis(ticker)
        return None
    
    async def _process_tipranks_analyst_forecasts(self, ticker: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process TipRanks analyst forecast data."""
        if hasattr(self.orchestrator, '_process_tipranks_analyst_forecasts'):
            return self.orchestrator._process_tipranks_analyst_forecasts(ticker, data)
        return None
    
    async def _generate_bulls_bears_content(self, ticker: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate Bulls Say and Bears Say content."""
        if hasattr(self.orchestrator, '_generate_bulls_bears_content'):
            return self.orchestrator._generate_bulls_bears_content(ticker, data)
        return None

    @traceable(name="generate_investment_decision")
    async def _generate_investment_decision(self, ticker: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate investment decision using comprehensive data including web scraped sources.

        Args:
            ticker: Stock ticker symbol
            data: Complete report data including web scraping, financial metrics, etc.

        Returns:
            Investment decision dictionary or None if generation fails
        """
        try:
            logger.info(f"🎯 Generating investment decision for {ticker} using comprehensive data")

            # Extract all available data sources
            financial_metrics = data.get('financial_metrics', {})
            web_scraping_data = data.get('web_scraping', {})
            technical_analysis = data.get('technical_analysis', {})
            news_analysis = data.get('news_analysis', {})
            tipranks_data = data.get('tipranks_analyst_forecasts', {})

            # Use orchestrator's enhanced investment decision logic with detailed reasoning and citations
            logger.info(f"🔍 Using orchestrator's enhanced investment decision logic for {ticker}")
            if hasattr(self.orchestrator, '_generate_investment_decision'):
                # Use orchestrator's enhanced method that includes detailed reasoning and citations
                enhanced_decision = self.orchestrator._generate_investment_decision(ticker, financial_metrics, web_scraping_data)
                if enhanced_decision and enhanced_decision.get('detailed_reasoning'):
                    logger.info(f"✅ Using enhanced investment decision with detailed reasoning for {ticker}")
                    return enhanced_decision
                else:
                    logger.warning(f"⚠️ Enhanced decision missing detailed reasoning, falling back to basic method for {ticker}")

            # Fallback to basic method if orchestrator method fails
            logger.info(f"🔍 Using fallback basic investment decision logic for {ticker}")
            return self._generate_basic_investment_decision(ticker, financial_metrics, web_scraping_data,
                                                          technical_analysis, news_analysis, tipranks_data)

        except Exception as e:
            logger.error(f"❌ Investment decision generation failed for {ticker}: {e}")
            return None

    @traceable(name="generate_basic_investment_decision")
    def _generate_basic_investment_decision(self, ticker: str, financial_metrics: Dict, web_scraping_data: Dict,
                                          technical_analysis: Dict, news_analysis: Dict, tipranks_data: Dict) -> Dict[str, Any]:
        """
        Generate basic investment decision using available data sources.

        This method prioritizes web scraped data when Yahoo Finance data is insufficient.
        """
        try:
            logger.info(f"🔍 Generating basic investment decision for {ticker} using web scraped data")

            # Debug: Log available data sources
            logger.info(f"🔍 Available data sources for {ticker}:")
            logger.info(f"   - Web scraping keys: {list(web_scraping_data.keys()) if web_scraping_data else 'None'}")
            logger.info(f"   - Technical analysis available: {bool(technical_analysis and technical_analysis.get('success'))}")
            logger.info(f"   - News analysis available: {bool(news_analysis and news_analysis.get('success'))}")
            logger.info(f"   - Financial metrics keys: {list(financial_metrics.keys()) if financial_metrics else 'None'}")

            # Initialize decision variables
            buy_signals = 0
            sell_signals = 0
            total_signals = 0
            rationale_points = []
            risk_factors = []

            # 1. Analyze StockAnalysis.com data (from correct web scraping structure)
            # The actual scraped data is nested inside 'data_sources'
            data_sources = web_scraping_data.get('data_sources', {})
            stockanalysis_data = data_sources.get('stockanalysis_enhanced', {})
            if not stockanalysis_data:
                stockanalysis_data = data_sources.get('stockanalysis', {})

            logger.info(f"🔍 Data sources keys: {list(data_sources.keys()) if data_sources else 'None'}")
            logger.info(f"🔍 StockAnalysis data structure: {list(stockanalysis_data.keys()) if stockanalysis_data else 'None'}")

            # Process StockAnalysis overview page (most comprehensive)
            # Handle both enhanced structure and direct database structure
            overview_data = stockanalysis_data.get('overview', {})
            if not overview_data and stockanalysis_data.get('success'):
                # Direct database structure - use the stockanalysis data directly
                overview_data = stockanalysis_data

            if overview_data and overview_data.get('success'):
                # Get content from either 'content' or 'markdown_content' field
                content = overview_data.get('content', '') or overview_data.get('markdown_content', '')
                content = content.lower()
                logger.info(f"🔍 StockAnalysis overview content length: {len(content)} chars")
                if content and len(content) > 1000:  # Ensure we have substantial content
                    total_signals += 1

                    # Enhanced pattern matching for analyst ratings
                    import re

                    # Look for specific analyst ratings
                    rating_patterns = [
                        r'analyst.*?rating.*?(strong buy|buy|outperform|overweight)',
                        r'(strong buy|buy|outperform|overweight).*?rating',
                        r'consensus.*?(strong buy|buy|outperform|overweight)',
                        r'recommendation.*?(strong buy|buy|outperform|overweight)'
                    ]

                    positive_found = False
                    for pattern in rating_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            buy_signals += 1
                            rationale_points.append(f"StockAnalysis.com analyst rating: {matches[0].title()}")
                            positive_found = True
                            break

                    if not positive_found:
                        # Look for negative ratings
                        negative_patterns = [
                            r'analyst.*?rating.*?(sell|underperform|underweight)',
                            r'(sell|underperform|underweight).*?rating',
                            r'consensus.*?(sell|underperform|underweight)'
                        ]
                        for pattern in negative_patterns:
                            matches = re.findall(pattern, content)
                            if matches:
                                sell_signals += 1
                                risk_factors.append(f"StockAnalysis.com negative rating: {matches[0].title()}")
                                break

                    # Enhanced price target extraction
                    price_target_patterns = [
                        r'price target.*?(\d+\.?\d*)',
                        r'target price.*?(\d+\.?\d*)',
                        r'target.*?hk\$(\d+\.?\d*)',
                        r'target.*?(\d+\.?\d*)\s*hk\$'
                    ]

                    for pattern in price_target_patterns:
                        price_matches = re.findall(pattern, content)
                        if price_matches:
                            try:
                                target_price = float(price_matches[0])
                                current_price = financial_metrics.get('current_price')
                                if current_price and target_price > 0:
                                    upside = ((target_price - current_price) / current_price) * 100
                                    total_signals += 1
                                    if upside > 15:
                                        buy_signals += 1
                                        rationale_points.append(f"Price target HK${target_price} suggests {upside:.1f}% upside")
                                    elif upside < -10:
                                        sell_signals += 1
                                        risk_factors.append(f"Price target HK${target_price} suggests {upside:.1f}% downside")
                                    break
                            except (ValueError, TypeError):
                                continue

            # 2. Analyze TipRanks.com data (from correct web scraping structure)
            # The actual scraped data is nested inside 'data_sources'
            tipranks_data = data_sources.get('tipranks_enhanced', {})
            if not tipranks_data:
                tipranks_data = data_sources.get('tipranks', {})

            logger.info(f"🔍 TipRanks data structure: {list(tipranks_data.keys()) if tipranks_data else 'None'}")

            # Process TipRanks forecast page (most comprehensive for analyst data)
            # Handle both enhanced structure and direct database structure
            forecast_data = tipranks_data.get('forecast', {})
            if not forecast_data and tipranks_data.get('success'):
                # Direct database structure - use the tipranks data directly
                forecast_data = tipranks_data

            if forecast_data and forecast_data.get('success'):
                # Get content from either 'content' or 'markdown_content' field
                content = forecast_data.get('content', '') or forecast_data.get('markdown_content', '')
                content = content.lower()
                logger.info(f"🔍 TipRanks forecast content length: {len(content)} chars")
                if content and len(content) > 1000:
                    total_signals += 1

                    # Enhanced consensus rating extraction
                    consensus_patterns = [
                        r'consensus.*?(strong buy|moderate buy|buy)',
                        r'analyst consensus.*?(strong buy|moderate buy|buy)',
                        r'overall rating.*?(strong buy|moderate buy|buy)',
                        r'(strong buy|moderate buy|buy).*?consensus'
                    ]

                    consensus_found = False
                    for pattern in consensus_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            buy_signals += 1
                            rationale_points.append(f"TipRanks analyst consensus: {matches[0].title()}")
                            consensus_found = True
                            break

                    if not consensus_found:
                        # Look for negative consensus
                        negative_patterns = [
                            r'consensus.*?(sell|moderate sell|strong sell)',
                            r'analyst consensus.*?(sell|moderate sell|strong sell)'
                        ]
                        for pattern in negative_patterns:
                            matches = re.findall(pattern, content)
                            if matches:
                                sell_signals += 1
                                risk_factors.append(f"TipRanks negative consensus: {matches[0].title()}")
                                break

                    # Extract analyst count and coverage quality
                    analyst_patterns = [
                        r'(\d+)\s*analyst[s]?\s*cover',
                        r'(\d+)\s*analyst[s]?\s*rating',
                        r'based on (\d+)\s*analyst'
                    ]

                    for pattern in analyst_patterns:
                        analyst_matches = re.findall(pattern, content)
                        if analyst_matches:
                            try:
                                analyst_count = int(analyst_matches[0])
                                if analyst_count >= 5:
                                    total_signals += 1
                                    buy_signals += 0.5  # Partial signal for good coverage
                                    rationale_points.append(f"Strong analyst coverage: {analyst_count} analysts")
                                elif analyst_count >= 3:
                                    rationale_points.append(f"Moderate analyst coverage: {analyst_count} analysts")
                                break
                            except (ValueError, TypeError):
                                continue

                    # Extract price targets from TipRanks
                    price_target_patterns = [
                        r'average price target.*?hk\$(\d+\.?\d*)',
                        r'price target.*?hk\$(\d+\.?\d*)',
                        r'target price.*?(\d+\.?\d*)',
                        r'consensus target.*?(\d+\.?\d*)'
                    ]

                    for pattern in price_target_patterns:
                        target_matches = re.findall(pattern, content)
                        if target_matches:
                            try:
                                target_price = float(target_matches[0])
                                current_price = financial_metrics.get('current_price')
                                if current_price and target_price > 0:
                                    upside = ((target_price - current_price) / current_price) * 100
                                    total_signals += 1
                                    if upside > 20:
                                        buy_signals += 1
                                        rationale_points.append(f"TipRanks target HK${target_price} suggests {upside:.1f}% upside")
                                    elif upside < -15:
                                        sell_signals += 1
                                        risk_factors.append(f"TipRanks target HK${target_price} suggests {upside:.1f}% downside")
                                    break
                            except (ValueError, TypeError):
                                continue

            # 3. Analyze technical signals (enhanced extraction)
            if technical_analysis and technical_analysis.get('success'):
                # Check overall consensus
                overall_consensus = technical_analysis.get('overall_consensus', {})
                overall_signal = overall_consensus.get('overall_signal', '').lower()
                if overall_signal:
                    total_signals += 1
                    if 'buy' in overall_signal:
                        buy_signals += 1
                        rationale_points.append(f"Technical analysis signal: {overall_signal.title()}")
                    elif 'sell' in overall_signal:
                        sell_signals += 1
                        risk_factors.append(f"Technical analysis signal: {overall_signal.title()}")

                # Check individual indicators
                indicators = technical_analysis.get('indicators', {})
                bullish_indicators = 0
                bearish_indicators = 0
                total_indicators = 0

                for indicator_name, indicator_data in indicators.items():
                    if isinstance(indicator_data, dict) and 'signal' in indicator_data:
                        total_indicators += 1
                        signal = indicator_data['signal'].lower()
                        if 'buy' in signal:
                            bullish_indicators += 1
                        elif 'sell' in signal:
                            bearish_indicators += 1

                if total_indicators > 0:
                    total_signals += 1
                    bullish_ratio = bullish_indicators / total_indicators
                    if bullish_ratio > 0.6:
                        buy_signals += 1
                        rationale_points.append(f"Strong technical buy signals: {bullish_indicators}/{total_indicators} indicators bullish")
                    elif bullish_ratio < 0.4:
                        sell_signals += 1
                        risk_factors.append(f"Weak technical signals: {bearish_indicators}/{total_indicators} indicators bearish")

            # 4. Analyze news sentiment (enhanced extraction)
            if news_analysis and news_analysis.get('success'):
                # Check overall sentiment
                overall_sentiment = news_analysis.get('overall_sentiment', {})
                sentiment_score = overall_sentiment.get('sentiment_score', 0)
                if sentiment_score != 0:
                    total_signals += 1
                    if sentiment_score > 0.2:
                        buy_signals += 1
                        rationale_points.append(f"Positive news sentiment: {sentiment_score:.2f}")
                    elif sentiment_score < -0.2:
                        sell_signals += 1
                        risk_factors.append(f"Negative news sentiment: {sentiment_score:.2f}")

                # Check individual articles sentiment
                articles = news_analysis.get('articles', [])
                if articles:
                    positive_articles = sum(1 for article in articles if article.get('sentiment_score', 0) > 0.1)
                    negative_articles = sum(1 for article in articles if article.get('sentiment_score', 0) < -0.1)
                    total_articles = len(articles)

                    if total_articles >= 3:  # Only consider if we have enough articles
                        total_signals += 1
                        positive_ratio = positive_articles / total_articles
                        if positive_ratio > 0.6:
                            buy_signals += 1
                            rationale_points.append(f"Positive news coverage: {positive_articles}/{total_articles} articles bullish")
                        elif positive_ratio < 0.4:
                            sell_signals += 1
                            risk_factors.append(f"Negative news coverage: {negative_articles}/{total_articles} articles bearish")

            # 5. Basic financial metrics (if available)
            pe_ratio = financial_metrics.get('pe_ratio')
            if pe_ratio and pe_ratio > 0:
                total_signals += 1
                if pe_ratio < 15:
                    buy_signals += 1
                    rationale_points.append(f"Attractive P/E ratio: {pe_ratio:.1f}")
                elif pe_ratio > 30:
                    sell_signals += 1
                    risk_factors.append(f"High P/E ratio: {pe_ratio:.1f}")

            # Generate recommendation
            if total_signals == 0:
                recommendation = "HOLD"
                emoji = "🟡"
                confidence_score = 1
                key_rationale = "Insufficient data for investment recommendation"
            else:
                buy_ratio = buy_signals / total_signals
                if buy_ratio >= 0.6:
                    recommendation = "BUY"
                    emoji = "🟢"
                    confidence_score = min(10, int(buy_ratio * 10) + 2)
                elif buy_ratio <= 0.3:
                    recommendation = "SELL"
                    emoji = "🔴"
                    confidence_score = min(10, int((1 - buy_ratio) * 10) + 2)
                else:
                    recommendation = "HOLD"
                    emoji = "🟡"
                    confidence_score = max(3, min(7, int(total_signals * 1.5)))

                key_rationale = f"Based on {total_signals} signals: {buy_signals} bullish, {sell_signals} bearish"

            # Get current price and target for price analysis
            current_price = financial_metrics.get('current_price') if financial_metrics else None
            target_mean = None

            # Try to get target from web scraped data (fix variable name error)
            # Note: These variables may not exist if web scraping data structure is different
            try:
                if 'stockanalysis_data' in locals() and stockanalysis_data and stockanalysis_data.get('price_target'):
                    target_mean = stockanalysis_data.get('price_target')
                elif 'tipranks_data' in locals() and tipranks_data and tipranks_data.get('price_target_average'):
                    target_mean = tipranks_data.get('price_target_average')
            except:
                pass

            # Generate basic detailed reasoning for fallback case
            detailed_reasoning = self._generate_basic_detailed_reasoning(
                ticker, recommendation, confidence_score, rationale_points, risk_factors, current_price, target_mean
            )

            # Generate basic sources mapping
            sources = self._generate_basic_sources_mapping(ticker, web_scraping_data)

            investment_decision = {
                # Core decision fields
                "recommendation": recommendation,
                "emoji": emoji,
                "confidence_score": confidence_score,
                "key_rationale": key_rationale,
                "supporting_factors": rationale_points[:5],
                "risk_factors": risk_factors[:5],
                "data_quality_impact": min(100, total_signals * 20),  # Rough estimate
                "analysis_summary": {
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                    "total_signals": total_signals,
                    "buy_ratio": buy_ratio if total_signals > 0 else 0
                },
                "price_targets": {
                    "current_price": current_price,
                    "target_mean": target_mean,
                    "upside_potential": ((target_mean - current_price) / current_price) * 100 if current_price and target_mean else None
                },
                "data_sources_used": {
                    "stockanalysis": bool(stockanalysis_data),
                    "tipranks": bool(tipranks_data),
                    "technical_analysis": bool(technical_analysis and technical_analysis.get('success')),
                    "news_analysis": bool(news_analysis and news_analysis.get('success')),
                    "yahoo_finance": bool(financial_metrics)
                },

                # Enhanced fields for Investment Decision Agent compliance
                "detailed_reasoning": detailed_reasoning,
                "sources": sources,
                "citations": self._extract_citations_from_reasoning(detailed_reasoning),

                # Basic agent JSON structure for fallback
                "agent_json": {
                    "ticker": ticker,
                    "decision": recommendation,
                    "confidence": confidence_score,
                    "price_targets": {
                        "mean": target_mean,
                        "upside_pct_to_mean": ((target_mean - current_price) / current_price) * 100 if current_price and target_mean else None,
                        "citations": ["S1", "T1"]
                    },
                    "sources": sources
                }
            }

            logger.info(f"✅ Generated investment decision for {ticker}: {recommendation} {emoji} (Confidence: {confidence_score}/10)")
            return investment_decision

        except Exception as e:
            logger.error(f"❌ Basic investment decision generation failed for {ticker}: {e}")
            return {
                "recommendation": "HOLD",
                "emoji": "🟡",
                "confidence_score": 1,
                "key_rationale": f"Decision generation failed: {str(e)}",
                "supporting_factors": [],
                "risk_factors": [f"Analysis error: {str(e)}"],
                "data_quality_impact": 0,
                "error": str(e),
                "detailed_reasoning": {
                    "decision_summary": "HOLD 🟡 (Confidence 1/10)",
                    "tldr": f"Analysis failed due to error: {str(e)}",
                    "key_metrics": "Key metrics unavailable",
                    "valuation_analysis": "Valuation analysis unavailable",
                    "analyst_consensus": "Analyst consensus unavailable",
                    "technical_snapshot": "Technical analysis unavailable",
                    "catalysts_risks": "Risk assessment unavailable",
                    "hk_china_overlay": "HK/China analysis unavailable",
                    "change_triggers": "Change triggers unavailable"
                },
                "sources": [],
                "citations": {}
            }

    def _generate_basic_detailed_reasoning(self, ticker: str, recommendation: str, confidence_score: int,
                                         rationale_points: list, risk_factors: list, current_price: float, target_mean: float) -> Dict[str, str]:
        """Generate basic detailed reasoning for fallback cases."""
        try:
            emoji = "🟢" if recommendation == "BUY" else "🔴" if recommendation == "SELL" else "🟡"

            # Generate TL;DR
            if recommendation == "BUY":
                tldr = f"Positive signals from web scraping data support BUY rating. Key factors: {', '.join(rationale_points[:2]) if rationale_points else 'Limited data available'}."
            elif recommendation == "SELL":
                tldr = f"Negative indicators warrant SELL rating. Key concerns: {', '.join(risk_factors[:2]) if risk_factors else 'Market risks identified'}."
            else:
                tldr = f"Mixed signals support HOLD rating. Balanced risk/reward with limited clear catalysts."

            # Ensure TL;DR is ≤50 words
            tldr_words = tldr.split()
            if len(tldr_words) > 50:
                tldr = ' '.join(tldr_words[:47]) + "..."

            return {
                "decision_summary": f"{recommendation} {emoji} (Confidence {confidence_score}/10)",
                "tldr": tldr,
                "key_metrics": f"Current Price: {current_price:.2f} HKD [S1]\nTarget Price: {target_mean:.2f} HKD [S1]\nUpside Potential: {((target_mean - current_price) / current_price) * 100:.1f}% [S1]" if current_price and target_mean else "Key metrics limited by data availability [S1]",
                "valuation_analysis": f"Analysis based on web scraped data from multiple sources [S1][T1]. Valuation assessment considers available market data and analyst information.",
                "analyst_consensus": f"Consensus analysis derived from web scraping sources [S1][T1]. Target price: {target_mean:.2f} HKD" if target_mean else "Analyst consensus data limited [S1][T1]",
                "technical_snapshot": f"Technical indicators suggest {'bullish' if recommendation == 'BUY' else 'bearish' if recommendation == 'SELL' else 'neutral'} bias based on available data [T1]",
                "catalysts_risks": f"Bull Case: {'; '.join(rationale_points[:3]) if rationale_points else 'Limited positive catalysts identified'} [S1]\nBear Case: {'; '.join(risk_factors[:3]) if risk_factors else 'Standard market risks apply'} [S1]",
                "hk_china_overlay": "HK Market Context: Subject to Hang Seng correlation and mainland China exposure [S1]. Regulatory risks from HKEX compliance and China policy changes [S1].",
                "change_triggers": f"Bullish: Improving fundamentals, positive news flow. Bearish: Deteriorating metrics, negative sentiment." if recommendation == "HOLD" else f"{'Bearish' if recommendation == 'BUY' else 'Bullish'}: Fundamental deterioration, adverse news, technical breakdown" if recommendation == "BUY" else "Bullish: Fundamental improvement, positive catalysts, technical recovery"
            }
        except Exception as e:
            logger.error(f"Error generating basic detailed reasoning for {ticker}: {e}")
            return {
                "decision_summary": f"HOLD 🟡 (Confidence 1/10)",
                "tldr": "Analysis incomplete due to processing error",
                "key_metrics": "Key metrics unavailable",
                "valuation_analysis": "Valuation analysis unavailable",
                "analyst_consensus": "Analyst consensus unavailable",
                "technical_snapshot": "Technical analysis unavailable",
                "catalysts_risks": "Risk assessment unavailable",
                "hk_china_overlay": "HK/China analysis unavailable",
                "change_triggers": "Change triggers unavailable"
            }

    def _generate_basic_sources_mapping(self, ticker: str, web_scraping_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate basic sources mapping for fallback cases."""
        try:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")

            sources = []

            # StockAnalysis.com source
            if web_scraping_data.get('stockanalysis') or web_scraping_data.get('data_sources', {}).get('stockanalysis'):
                sources.append({
                    "tag": "S1",
                    "name": "StockAnalysis.com",
                    "retrieved": current_date,
                    "url": f"https://stockanalysis.com/quote/hkg/{ticker.replace('.HK', '')}/",
                    "description": "Financial analysis and market data"
                })

            # TipRanks source
            if web_scraping_data.get('tipranks') or web_scraping_data.get('data_sources', {}).get('tipranks'):
                sources.append({
                    "tag": "T1",
                    "name": "TipRanks.com",
                    "retrieved": current_date,
                    "url": f"https://www.tipranks.com/stocks/hk:{ticker.replace('.HK', '')}/forecast",
                    "description": "Analyst ratings and price targets"
                })

            # Default fallback source
            if not sources:
                sources.append({
                    "tag": "S1",
                    "name": "Web Scraping Sources",
                    "retrieved": current_date,
                    "url": "Multiple financial data sources",
                    "description": "Aggregated financial data and analysis"
                })

            return sources

        except Exception as e:
            logger.error(f"Error generating basic sources mapping for {ticker}: {e}")
            return [{
                "tag": "S1",
                "name": "Financial Data Sources",
                "retrieved": "2024-01-01",
                "url": "Multiple sources",
                "description": "Financial data aggregation"
            }]

    def _extract_citations_from_reasoning(self, detailed_reasoning: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract citation tags from detailed reasoning text."""
        try:
            import re
            citations = {}

            # Pattern to match citation tags like [S1], [T1], etc.
            citation_pattern = r'\[([A-Z]\d+)\]'

            for section, text in detailed_reasoning.items():
                if isinstance(text, str):
                    found_citations = re.findall(citation_pattern, text)
                    if found_citations:
                        citations[section] = found_citations

            return citations

        except Exception as e:
            logger.error(f"Error extracting citations from reasoning: {e}")
            return {}

    async def _generate_report_output(self, report_data: Dict[str, Any], title: str) -> str:
        """
        Generate the final report output.
        
        Args:
            report_data: Complete report data
            title: Report title
            
        Returns:
            Path to generated report
        """
        if not self.html_report_generator:
            raise ValueError("HTML report generator not available")
        
        # Generate the HTML report
        report_path = await self.html_report_generator.generate_report(report_data, title)
        
        # Verify the report was created
        if not Path(report_path).exists():
            raise FileNotFoundError(f"Generated report not found at {report_path}")
        
        return report_path
    
    def get_report_statistics(self) -> Dict[str, Any]:
        """Get report generation statistics."""
        return {
            "reports_generated": self.reports_generated,
            "generation_errors": self.generation_errors,
            "success_rate": (self.reports_generated / (self.reports_generated + self.generation_errors) * 100) 
                           if (self.reports_generated + self.generation_errors) > 0 else 0,
            "enabled_sections": [name for name, section in self.report_sections.items() if section.enabled],
            "available_components": {
                "html_report_generator": self.html_report_generator is not None,
                "orchestrator": self.orchestrator is not None
            },
            "config": {
                "output_format": self.config.output_format,
                "template_style": self.config.template_style,
                "charts_enabled": self.config.enable_charts
            }
        }
