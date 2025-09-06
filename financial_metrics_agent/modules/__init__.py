"""
AgentInvest HTML Report Generator Modules

Modular components for the HTML report generation system.
Refactored to eliminate hardcoded ticker-specific logic.
"""

from .report_data_processor import ReportDataProcessor, ProcessedFinancialData, CompanyProfile
from .financial_analyzer import FinancialAnalyzer, InvestmentAnalysis, InvestmentRecommendation
from .content_generator import ContentGenerator, BullsBears
from .template_renderer import TemplateRenderer
from .chart_coordinator import ChartCoordinator
from .citation_manager import CitationManager

__all__ = [
    'ReportDataProcessor',
    'ProcessedFinancialData',
    'CompanyProfile',
    'FinancialAnalyzer',
    'InvestmentAnalysis',
    'InvestmentRecommendation',
    'ContentGenerator',
    'BullsBears',
    'TemplateRenderer',
    'ChartCoordinator',
    'CitationManager'
]
