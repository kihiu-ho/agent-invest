#!/usr/bin/env python3
"""
Test script for Executive Summary Agent integration
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent))

from agent_factory import FinancialAgentFactory
from html_report_generator import HTMLReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_executive_summary_agent():
    """Test the executive summary agent creation and basic functionality."""
    
    logger.info("🧪 Testing Executive Summary Agent creation...")
    
    try:
        # Create agent factory
        agent_factory = FinancialAgentFactory()
        
        # Test agent creation
        executive_agent = agent_factory.create_executive_summary_agent()
        
        if executive_agent:
            logger.info("✅ Executive Summary Agent created successfully")
            logger.info(f"   Agent name: {getattr(executive_agent, 'name', 'Unknown')}")
            logger.info(f"   Agent description: {getattr(executive_agent, 'description', 'No description')}")
            return True
        else:
            logger.error("❌ Failed to create Executive Summary Agent")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating Executive Summary Agent: {e}")
        return False

def test_html_report_generator():
    """Test the HTML report generator with executive summary section."""
    
    logger.info("🧪 Testing HTML Report Generator with Executive Summary...")
    
    try:
        # Create HTML report generator
        html_generator = HTMLReportGenerator()
        
        # Create sample data
        sample_data = {
            "ticker": "0941.HK",
            "basic_info": {
                "long_name": "China Mobile Limited",
                "sector": "Telecommunications",
                "current_price": 45.50,
                "market_cap": 930000000000
            },
            "web_scraping": {
                "data_sources": {
                    "stockanalysis": {
                        "success": True,
                        "financial_data": {"revenue": 850000000000}
                    },
                    "tipranks": {
                        "success": True,
                        "analyst_data": {"target_price": 50.00}
                    }
                },
                "summary": {
                    "total_sources": 2,
                    "success_rate": 100
                }
            },
            "weaviate_queries": {
                "status": "success",
                "document_insights": ["Strong market position", "Regulatory compliance"]
            },
            "investment_decision": {
                "recommendation": "BUY",
                "confidence_score": 8,
                "key_rationale": "Strong fundamentals and growth prospects"
            },
            "technical_analysis": {
                "overall_consensus": "Bullish",
                "current_price": 45.50
            },
            "historical_data": {
                "current_price": 45.50
            }
        }
        
        # Test executive summary generation
        executive_summary_html = html_generator._generate_executive_summary_section(sample_data, "0941.HK")
        
        if executive_summary_html and len(executive_summary_html) > 100:
            logger.info("✅ Executive Summary section generated successfully")
            logger.info(f"   Generated HTML length: {len(executive_summary_html)} characters")
            
            # Check for key elements
            if "Executive Summary" in executive_summary_html:
                logger.info("   ✅ Contains Executive Summary header")
            if "executive-summary-content" in executive_summary_html:
                logger.info("   ✅ Contains executive summary content div")
                
            return True
        else:
            logger.error("❌ Executive Summary generation failed or returned empty content")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing HTML Report Generator: {e}")
        return False

def test_full_report_generation():
    """Test full report generation with executive summary."""
    
    logger.info("🧪 Testing Full Report Generation with Executive Summary...")
    
    try:
        # Create HTML report generator
        html_generator = HTMLReportGenerator()
        
        # Create comprehensive sample data
        sample_data = {
            "ticker": "0941.HK",
            "basic_info": {
                "long_name": "China Mobile Limited",
                "sector": "Telecommunications",
                "current_price": 45.50,
                "market_cap": 930000000000
            },
            "web_scraping": {
                "data_sources": {
                    "stockanalysis_enhanced": {
                        "success": True,
                        "financial_data": {"revenue": 850000000000, "profit_margin": 15.2}
                    },
                    "tipranks_enhanced": {
                        "success": True,
                        "analyst_data": {"target_price": 50.00, "analyst_count": 12}
                    }
                }
            },
            "weaviate_queries": {
                "status": "success",
                "recent_queries": ["annual report analysis", "competitive positioning"],
                "document_insights": ["Market leader in China", "Strong 5G infrastructure"]
            },
            "investment_decision": {
                "recommendation": "BUY",
                "confidence_score": 8,
                "key_rationale": "Strong fundamentals with 5G growth potential",
                "detailed_reasoning": "Comprehensive analysis shows strong market position"
            },
            "technical_analysis": {
                "overall_consensus": "Bullish",
                "current_price": 45.50,
                "success": True
            },
            "historical_data": {
                "current_price": 45.50,
                "prices": [44.0, 44.5, 45.0, 45.5]
            },
            "bulls_bears_analysis": {
                "bulls_say": [
                    {"content": "Strong 5G infrastructure deployment", "category": "Growth"}
                ],
                "bears_say": [
                    {"content": "Regulatory pressures in China", "category": "Risk"}
                ]
            }
        }
        
        # Generate full report
        report_path = html_generator.generate_report(sample_data, "Test Executive Summary Report")
        
        if report_path and Path(report_path).exists():
            logger.info(f"✅ Full report generated successfully: {report_path}")
            
            # Read and check content
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "Executive Summary" in content:
                logger.info("   ✅ Executive Summary section included in report")
            if "Investment Recommendation" in content:
                logger.info("   ✅ Investment Recommendation section included")
            if "executive-summary-content" in content:
                logger.info("   ✅ Executive summary styling included")
                
            logger.info(f"   Report size: {len(content)} characters")
            return True
        else:
            logger.error("❌ Full report generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing full report generation: {e}")
        return False

def main():
    """Run all tests."""
    
    logger.info("🚀 Starting Executive Summary Integration Tests...")
    
    tests = [
        ("Executive Summary Agent Creation", test_executive_summary_agent),
        ("HTML Report Generator", test_html_report_generator),
        ("Full Report Generation", test_full_report_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Executive Summary integration is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
