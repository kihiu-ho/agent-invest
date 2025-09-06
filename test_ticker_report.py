#!/usr/bin/env python3
"""
Simple test script to generate HTML report for a specific ticker using the orchestrator.
Usage: python test_ticker_report.py [ticker]
Default ticker: 0007.hk
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add financial_metrics_agent to path
sys.path.insert(0, str(Path(__file__).parent / "financial_metrics_agent"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ticker_report(ticker: str = "0007.hk"):
    """Test report generation for a specific ticker."""
    
    logger.info(f"🚀 Starting report generation test for ticker: {ticker}")
    logger.info("=" * 60)
    
    try:
        # Import the orchestrator
        from orchestrator import FinancialMetricsOrchestrator
        
        # Initialize orchestrator
        logger.info("📊 Initializing FinancialMetricsOrchestrator...")
        orchestrator = FinancialMetricsOrchestrator(
            reports_dir="./reports",
            max_workers=3
        )
        
        logger.info("✅ Orchestrator initialized successfully")
        
        # Generate report
        logger.info(f"📈 Analyzing ticker: {ticker}")
        start_time = datetime.now()
        
        result = await orchestrator.analyze_single_ticker(
            ticker=ticker,
            time_period="1Y",
            use_agents=True,
            generate_report=True,
            enable_pdf_processing=True,
            enable_weaviate_queries=True,
            enable_real_time_data=True
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results
        logger.info("=" * 60)
        logger.info("📋 REPORT GENERATION RESULTS")
        logger.info("=" * 60)
        
        if result.get("success", False):
            logger.info("✅ Report generation SUCCESSFUL!")
            logger.info(f"   Ticker: {result.get('ticker', ticker)}")
            logger.info(f"   Processing time: {processing_time:.2f} seconds")
            
            # Check for report file
            report_file = result.get("report_path") or result.get("report_file")
            if report_file:
                report_path = Path(report_file)
                if report_path.exists():
                    logger.info(f"   Report file: {report_path.absolute()}")
                    logger.info(f"   File size: {report_path.stat().st_size / 1024:.1f} KB")
                    
                    # Show first few lines of the HTML file
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                logger.info("   ✅ HTML report contains content")
                                # Check for key elements
                                if "<html" in content.lower():
                                    logger.info("   ✅ Valid HTML structure detected")
                                if ticker.upper() in content.upper():
                                    logger.info("   ✅ Ticker found in report content")
                                if "chart" in content.lower():
                                    logger.info("   ✅ Charts detected in report")
                            else:
                                logger.warning("   ⚠️ HTML report is empty")
                    except Exception as e:
                        logger.error(f"   ❌ Error reading report file: {e}")
                else:
                    logger.error(f"   ❌ Report file not found: {report_path}")
            else:
                logger.warning("   ⚠️ No report file path in results")
            
            # Show additional metrics
            if result.get("data"):
                data = result["data"]
                logger.info(f"   Data sources collected: {len(data.get('data_sources', {}))}")
                
                if data.get('basic_info'):
                    basic_info = data['basic_info']
                    company_name = basic_info.get('long_name', 'N/A')
                    logger.info(f"   Company: {company_name}")
                    
                if data.get('financial_metrics'):
                    metrics = data['financial_metrics']
                    current_price = metrics.get('current_price', 'N/A')
                    logger.info(f"   Current price: {current_price}")
            
            # Show analysis results
            if result.get("analysis"):
                analysis = result["analysis"]
                logger.info("   Analysis components:")
                for key in analysis.keys():
                    logger.info(f"     - {key}")
            
            return True
            
        else:
            logger.error("❌ Report generation FAILED!")
            error_msg = result.get("error", "Unknown error")
            logger.error(f"   Error: {error_msg}")
            
            # Show partial results if available
            if result.get("data"):
                logger.info("   Partial data was collected:")
                data = result["data"]
                for key, value in data.items():
                    if isinstance(value, dict):
                        logger.info(f"     - {key}: {len(value)} items")
                    else:
                        logger.info(f"     - {key}: {type(value).__name__}")
            
            return False
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("   Make sure you're running from the correct directory")
        logger.error("   and that financial_metrics_agent is available")
        return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.exception("Full traceback:")
        return False

def main():
    """Main function."""
    
    # Get ticker from command line argument or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "0007.hk"
    
    logger.info("🎯 Ticker Report Generation Test")
    logger.info(f"Target ticker: {ticker}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info("")
    
    # Run the test
    success = asyncio.run(test_ticker_report(ticker))
    
    if success:
        logger.info("\n🎉 Test completed successfully!")
        logger.info("Check the reports directory for the generated HTML file.")
        return 0
    else:
        logger.error("\n❌ Test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
