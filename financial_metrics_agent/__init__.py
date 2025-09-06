"""
Financial Metrics Agent System

A comprehensive AutoGen-based system for financial data analysis and reporting.
Provides automated collection of market data, analysis through specialized agents,
and generation of professional HTML reports.

Components:
- orchestrator.py: Main coordination and workflow management
- market_data_collector.py: Yahoo Finance API integration
- html_report_generator.py: Professional HTML report creation
- agent_factory.py: AutoGen agent initialization and configuration

Author: AgentInvest System
Date: 2025-09-01
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AgentInvest System"

# Import main components for easy access
from .orchestrator import FinancialMetricsOrchestrator
from .market_data_collector import MarketDataCollector
from .html_report_generator import HTMLReportGenerator
from .agent_factory import FinancialAgentFactory

__all__ = [
    "FinancialMetricsOrchestrator",
    "MarketDataCollector", 
    "HTMLReportGenerator",
    "FinancialAgentFactory"
]
