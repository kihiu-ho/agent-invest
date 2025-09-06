#!/usr/bin/env python3
"""
Test script for the refactored modular HTML report generator architecture.
Tests the new data-driven approach without hardcoded ticker-specific logic.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_financial_data(ticker: str = "0700.HK") -> dict:
    """Create test financial data for testing the modular architecture."""
    return {
        "ticker": ticker,
        "basic_info": {
            "long_name": "Tencent Holdings Limited" if ticker == "0700.HK" else f"Test Company {ticker}",
            "sector": "Technology" if ticker == "0700.HK" else "Technology",
            "industry": "Internet Content & Information" if ticker == "0700.HK" else "Software",
            "country": "Hong Kong" if ticker.endswith(".HK") else "United States",
            "currency": "HKD" if ticker.endswith(".HK") else "USD"
        },
        "financial_metrics": {
            "current_price": 320.50 if ticker == "0700.HK" else 150.75,
            "market_cap": 3050000000000 if ticker == "0700.HK" else 50000000000,  # 3.05T HKD vs 50B USD
            "pe_ratio": 18.5 if ticker == "0700.HK" else 25.2,
            "dividend_yield": 0.0025 if ticker == "0700.HK" else 0.035,  # 0.25% vs 3.5%
            "revenue_growth": 0.08 if ticker == "0700.HK" else 0.15,  # 8% vs 15%
            "earnings_growth": 0.12 if ticker == "0700.HK" else -0.05,  # 12% vs -5%
            "profit_margin": 0.22 if ticker == "0700.HK" else 0.08,  # 22% vs 8%
            "return_on_equity": 0.15 if ticker == "0700.HK" else 0.12,  # 15% vs 12%
            "debt_to_equity": 0.25 if ticker == "0700.HK" else 0.75,  # 0.25 vs 0.75
            "beta": 1.2 if ticker == "0700.HK" else 1.8,  # 1.2 vs 1.8
            "52_week_high": 450.0 if ticker == "0700.HK" else 200.0,
            "52_week_low": 280.0 if ticker == "0700.HK" else 120.0
        },
        "historical_data": {
            "prices": [300, 310, 315, 320, 325, 320],
            "dates": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"],
            "summary": {
                "period_return": 6.67,  # (320-300)/300 * 100
                "volatility": 15.2
            }
        },
        "technical_analysis": {
            "rsi": 65.5,
            "macd": 2.1,
            "moving_averages": {
                "sma_20": 315.0,
                "sma_50": 310.0,
                "sma_200": 305.0
            }
        },
        "web_insights": {
            "tipranks": {
                "analyst_rating": "Buy" if ticker == "0700.HK" else "Hold",
                "price_target": 380.0 if ticker == "0700.HK" else 160.0,
                "analyst_count": 25 if ticker == "0700.HK" else 12
            },
            "stockanalysis": {
                "consensus_rating": "Strong Buy" if ticker == "0700.HK" else "Hold"
            }
        },
        "weaviate_insights": {
            "documents": [
                {
                    "content": f"Annual report shows {ticker.replace('.HK', '')} has strong market position in technology sector with diversified revenue streams.",
                    "document_title": f"{ticker.replace('.HK', '')} Annual Report 2024",
                    "section_title": "Business Overview",
                    "ticker": ticker
                }
            ]
        }
    }

async def test_modular_architecture():
    """Test the modular architecture with different ticker scenarios."""
    logger.info("🧪 Starting modular architecture test")
    
    try:
        # Import the refactored HTML report generator
        from html_report_generator import HTMLReportGenerator
        
        # Initialize the generator
        generator = HTMLReportGenerator()
        logger.info("✅ HTMLReportGenerator initialized successfully")
        
        # Test scenarios with different company profiles
        test_scenarios = [
            {
                "name": "High Growth Tech (Tencent-like)",
                "ticker": "0700.HK",
                "description": "Large cap, high growth, profitable tech company"
            },
            {
                "name": "Small Cap Growth",
                "ticker": "GROW",
                "description": "Small cap, high growth, unprofitable company"
            },
            {
                "name": "Value Dividend Stock",
                "ticker": "VALUE.HK",
                "description": "Large cap, stable, high dividend yield company"
            }
        ]
        
        for scenario in test_scenarios:
            logger.info(f"🔍 Testing scenario: {scenario['name']} ({scenario['ticker']})")
            
            # Create test data for this scenario
            test_data = create_test_financial_data(scenario['ticker'])
            
            # Adjust data based on scenario
            if scenario['ticker'] == "GROW":
                # Small cap, high growth, unprofitable
                test_data['financial_metrics'].update({
                    'market_cap': 2000000000,  # 2B USD
                    'revenue_growth': 0.45,  # 45% growth
                    'profit_margin': -0.05,  # -5% margin (unprofitable)
                    'dividend_yield': 0.0,  # No dividend
                    'pe_ratio': None,  # No P/E for unprofitable company
                    'debt_to_equity': 0.15  # Low debt
                })
            elif scenario['ticker'] == "VALUE.HK":
                # Large cap, stable, high dividend
                test_data['financial_metrics'].update({
                    'market_cap': 800000000000,  # 800B HKD
                    'revenue_growth': 0.03,  # 3% growth (stable)
                    'profit_margin': 0.18,  # 18% margin
                    'dividend_yield': 0.065,  # 6.5% dividend yield
                    'pe_ratio': 12.5,  # Lower P/E (value)
                    'debt_to_equity': 0.45  # Moderate debt
                })
            
            try:
                # Generate report using modular architecture
                report_path = await generator.generate_report(
                    test_data, 
                    f"Modular Architecture Test - {scenario['name']}"
                )
                
                logger.info(f"✅ Successfully generated report for {scenario['ticker']}: {report_path}")
                
                # Verify the report file exists and has content
                if os.path.exists(report_path):
                    file_size = os.path.getsize(report_path)
                    logger.info(f"📄 Report file size: {file_size:,} bytes")
                    
                    # Read a sample of the content to verify it's not hardcoded
                    with open(report_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for data-driven content (should not contain hardcoded strings)
                    if "0700.HK" in content and scenario['ticker'] != "0700.HK":
                        logger.warning(f"⚠️ Found hardcoded 0700.HK reference in {scenario['ticker']} report")
                    elif "Tencent" in content and scenario['ticker'] != "0700.HK":
                        logger.warning(f"⚠️ Found hardcoded Tencent reference in {scenario['ticker']} report")
                    else:
                        logger.info(f"✅ Report appears to be data-driven for {scenario['ticker']}")
                        
                else:
                    logger.error(f"❌ Report file not found: {report_path}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to generate report for {scenario['ticker']}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("🎉 Modular architecture test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_citation_system():
    """Test the dynamic citation system."""
    logger.info("🧪 Testing dynamic citation system")
    
    try:
        from modules.citation_manager import CitationManager
        
        # Initialize citation manager
        citation_manager = CitationManager()
        
        # Test dynamic citation creation
        citation1 = citation_manager.create_dynamic_citation(
            'yahoo_finance', ticker='0700.HK', company_name='Tencent Holdings'
        )
        citation2 = citation_manager.create_dynamic_citation(
            'annual_report', ticker='0700.HK', company_name='Tencent Holdings', 
            document_name='Annual Report 2024'
        )
        citation3 = citation_manager.create_dynamic_citation(
            'web_source', url='https://example.com', source_name='Financial News'
        )
        
        logger.info(f"✅ Citation 1: {citation1}")
        logger.info(f"✅ Citation 2: {citation2}")
        logger.info(f"✅ Citation 3: {citation3}")
        
        # Test numbered references
        references_html = citation_manager.generate_numbered_references_section()
        logger.info(f"✅ Generated references section: {len(references_html)} characters")
        
        logger.info("🎉 Citation system test completed")
        
    except Exception as e:
        logger.error(f"❌ Citation system test failed: {e}")
        import traceback
        traceback.print_exc()

def test_data_processor():
    """Test the data processor module."""
    logger.info("🧪 Testing data processor module")
    
    try:
        from modules.report_data_processor import ReportDataProcessor
        
        # Initialize data processor
        processor = ReportDataProcessor()
        
        # Test with different company profiles
        test_data = create_test_financial_data("0700.HK")
        
        # Process financial data
        processed_data = processor.process_financial_data(test_data)
        logger.info(f"✅ Processed data for {processed_data.ticker}")
        
        # Extract company characteristics
        company_profile = processor.extract_company_characteristics(processed_data)
        logger.info(f"✅ Company profile: {company_profile.size_category}, {company_profile.growth_profile}, {company_profile.profitability_profile}")
        
        # Calculate derived metrics
        derived_metrics = processor.calculate_derived_metrics(processed_data)
        logger.info(f"✅ Derived metrics calculated: {len(derived_metrics)} metrics")
        
        logger.info("🎉 Data processor test completed")
        
    except Exception as e:
        logger.error(f"❌ Data processor test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting modular architecture tests...")
    
    # Test individual components
    test_citation_system()
    test_data_processor()
    
    # Test full integration
    asyncio.run(test_modular_architecture())
    
    print("✅ All tests completed!")
