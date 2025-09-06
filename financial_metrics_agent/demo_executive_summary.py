#!/usr/bin/env python3
"""
Demo script showing the Executive Summary functionality
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent))

from agent_factory import FinancialAgentFactory
from html_report_generator import HTMLReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_executive_summary():
    """Demonstrate the executive summary functionality."""
    
    logger.info("🚀 Executive Summary Demo")
    logger.info("=" * 50)
    
    # Create HTML report generator
    html_generator = HTMLReportGenerator()
    
    # Create comprehensive sample data that mimics real report data
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
                    "financial_data": {
                        "revenue": 850000000000,
                        "profit_margin": 15.2,
                        "pe_ratio": 12.5,
                        "dividend_yield": 4.2
                    },
                    "key_metrics": {
                        "market_cap": "930B HKD",
                        "enterprise_value": "950B HKD"
                    }
                },
                "tipranks_enhanced": {
                    "success": True,
                    "analyst_data": {
                        "target_price": 50.00,
                        "analyst_count": 12,
                        "strong_buy": 4,
                        "buy": 6,
                        "hold": 2,
                        "sell": 0
                    },
                    "forecast_data": {
                        "eps_estimate": 3.2,
                        "revenue_growth": 5.5
                    }
                }
            },
            "summary": {
                "total_sources": 2,
                "success_rate": 100,
                "data_quality": "high"
            }
        },
        "weaviate_queries": {
            "status": "success",
            "recent_queries": [
                "China Mobile annual report 2023 analysis",
                "5G infrastructure competitive positioning",
                "Telecommunications sector regulatory environment"
            ],
            "document_insights": [
                "Market leader in China with 950M+ subscribers",
                "Strong 5G infrastructure deployment nationwide",
                "Regulatory compliance with Chinese telecommunications policies",
                "Diversification into digital services and cloud computing"
            ],
            "key_findings": {
                "market_position": "Dominant market leader",
                "growth_drivers": ["5G adoption", "Digital transformation", "IoT services"],
                "risk_factors": ["Regulatory changes", "Competition", "Economic slowdown"]
            }
        },
        "investment_decision": {
            "recommendation": "BUY",
            "confidence_score": 8,
            "key_rationale": "Strong fundamentals with 5G growth potential and attractive valuation",
            "detailed_reasoning": "Comprehensive analysis shows strong market position with sustainable competitive advantages"
        },
        "technical_analysis": {
            "overall_consensus": "Bullish",
            "current_price": 45.50,
            "success": True,
            "key_indicators": {
                "rsi": 58.2,
                "macd": "bullish_crossover",
                "moving_averages": "above_50_day"
            }
        },
        "bulls_bears_analysis": {
            "bulls_say": [
                {
                    "content": "Strong 5G infrastructure deployment with nationwide coverage providing competitive moat [S1]",
                    "category": "Growth Prospects"
                },
                {
                    "content": "Attractive dividend yield of 4.2% with stable cash flow generation [S1]",
                    "category": "Income Generation"
                }
            ],
            "bears_say": [
                {
                    "content": "Regulatory pressures in China telecommunications sector may impact pricing power [T1]",
                    "category": "Regulatory Risk"
                },
                {
                    "content": "Intense competition from China Unicom and China Telecom pressuring market share [T1]",
                    "category": "Competition"
                }
            ]
        }
    }
    
    logger.info("📊 Sample Data Prepared:")
    logger.info(f"   Ticker: {sample_data['ticker']}")
    logger.info(f"   Company: {sample_data['basic_info']['long_name']}")
    logger.info(f"   Web Scraping Sources: {len(sample_data['web_scraping']['data_sources'])}")
    logger.info(f"   Weaviate Insights: {len(sample_data['weaviate_queries']['document_insights'])}")
    logger.info(f"   Investment Decision: {sample_data['investment_decision']['recommendation']}")
    
    # Generate executive summary section
    logger.info("\n🎯 Generating Executive Summary...")
    
    try:
        executive_summary_html = html_generator._generate_executive_summary_section(sample_data, "0941.HK")
        
        if executive_summary_html:
            logger.info("✅ Executive Summary Generated Successfully!")
            logger.info(f"   HTML Length: {len(executive_summary_html)} characters")
            
            # Save the executive summary to a file for inspection
            output_file = Path("demo_executive_summary.html")
            
            # Create a complete HTML document for viewing
            complete_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Summary Demo - China Mobile (0941.HK)</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f5f5f5; margin: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .section {{ margin: 30px 0; padding: 20px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-bottom: 20px; }}
        .alert {{ padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .alert-info {{ background-color: #d1ecf1; border-left: 4px solid #bee5eb; color: #0c5460; }}
        .executive-summary-content {{ line-height: 1.6; }}
        .executive-summary-content h4 {{ color: #2c3e50; margin: 15px 0 10px 0; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }}
        .executive-summary-content .investment-thesis p {{ font-size: 1.1em; margin-bottom: 15px; }}
        .executive-summary-content .key-insights ul {{ margin: 10px 0; padding-left: 20px; }}
        .executive-summary-content .key-insights li {{ margin: 8px 0; }}
        .executive-summary-content .balance-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 10px; }}
        .executive-summary-content .opportunities, .executive-summary-content .risks {{ padding: 10px; border-radius: 5px; }}
        .executive-summary-content .opportunities {{ background-color: #d4edda; border-left: 3px solid #28a745; }}
        .executive-summary-content .risks {{ background-color: #f8d7da; border-left: 3px solid #dc3545; }}
        .executive-summary-content .opportunities ul, .executive-summary-content .risks ul {{ margin: 5px 0; padding-left: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Executive Summary Demo</h1>
        <p><strong>Company:</strong> China Mobile Limited (0941.HK)</p>
        <p><strong>Generated:</strong> {sample_data['basic_info']['current_price']} HKD | Market Cap: {sample_data['basic_info']['market_cap']:,} HKD</p>
        
        {executive_summary_html}
        
        <div class="section">
            <h2>📋 Demo Information</h2>
            <p>This executive summary was generated using the new AutoGen-powered Executive Summary Agent that integrates:</p>
            <ul>
                <li><strong>Web Scraped Data:</strong> TipRanks and StockAnalysis.com financial metrics</li>
                <li><strong>Weaviate Vector Database:</strong> Annual reports and HKEX document analysis</li>
                <li><strong>Market Data:</strong> Real-time pricing and technical analysis</li>
                <li><strong>Investment Analysis:</strong> Bulls/Bears perspectives and recommendation logic</li>
            </ul>
            <p>The executive summary appears at the beginning of all financial reports generated by the system.</p>
        </div>
    </div>
</body>
</html>
            """
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(complete_html)
            
            logger.info(f"📄 Complete demo saved to: {output_file.absolute()}")
            logger.info("   You can open this file in a web browser to see the executive summary")
            
            # Show a preview of the content
            logger.info("\n📋 Executive Summary Preview:")
            logger.info("-" * 50)
            
            # Extract text content for preview (simple approach)
            import re
            text_content = re.sub(r'<[^>]+>', '', executive_summary_html)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # Show first 300 characters
            preview = text_content[:300] + "..." if len(text_content) > 300 else text_content
            logger.info(preview)
            
            return True
        else:
            logger.error("❌ Failed to generate executive summary")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error generating executive summary: {e}")
        return False

def main():
    """Run the demo."""
    
    logger.info("🎯 Executive Summary Agent Demo")
    logger.info("This demo shows the new executive summary functionality")
    logger.info("that integrates web scraping, Weaviate, and market data.")
    logger.info("")
    
    success = demo_executive_summary()
    
    if success:
        logger.info("\n🎉 Demo completed successfully!")
        logger.info("The executive summary agent is working correctly and ready for use.")
    else:
        logger.error("\n❌ Demo failed. Please check the implementation.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
