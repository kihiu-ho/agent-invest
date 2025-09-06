#!/usr/bin/env python3
"""
Test Enhanced Streamlined PDF Processor

This script tests the enhanced StreamlinedPDFProcessor with improved LlamaMarkdownReader
functionality for processing the first 20 pages of PDF files.

Features:
- Enhanced LlamaMarkdownReader integration
- Improved error handling and validation
- Test mode for limited page processing
- Comprehensive analysis and reporting
- File output with metadata

Usage:
    python test_enhanced_streamlined_processor.py [pdf_file_path]

Author: Generated for AgentInvest2 project
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced StreamlinedPDFProcessor
try:
    from streamlined_pdf_processor import StreamlinedPDFProcessor
    logger.info("✅ StreamlinedPDFProcessor imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import StreamlinedPDFProcessor: {e}")
    sys.exit(1)


async def test_enhanced_processor(pdf_path: str, max_pages: int = 20):
    """
    Test the enhanced StreamlinedPDFProcessor with LlamaMarkdownReader.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to process
        
    Returns:
        Dict with test results
    """
    logger.info("🚀 Starting Enhanced StreamlinedPDFProcessor test...")
    logger.info(f"📁 Target file: {pdf_path}")
    logger.info(f"📋 Target pages: First {max_pages} pages")
    
    # Initialize the processor
    processor = StreamlinedPDFProcessor(
        download_dir="downloads/test_enhanced",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    try:
        # Test the enhanced extraction
        result = await processor.test_extract_pages(pdf_path, "TEST", max_pages)
        
        if result.get("success"):
            # Display comprehensive results
            print("\n" + "="*70)
            print("ENHANCED STREAMLINED PDF PROCESSOR TEST RESULTS")
            print("="*70)
            print(f"Source PDF: {pdf_path}")
            print(f"Extraction Method: {result.get('extraction_method', 'Unknown')}")
            print(f"Pages Extracted: {result.get('total_pages', 0)}")
            print(f"Total Pages in PDF: {result.get('total_pages_in_pdf', 'Unknown')}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
            print(f"Quality Score: {result.get('quality_score', 0):.2f}")
            print(f"Total Characters: {len(result.get('raw_text', '')):,}")
            print("="*70)
            
            # Show performance metrics if available
            if "performance_metrics" in result:
                metrics = result["performance_metrics"]
                print(f"\nPERFORMANCE METRICS:")
                print(f"Characters per second: {metrics.get('chars_per_second', 0):,.0f}")
                print(f"Documents per second: {metrics.get('docs_per_second', 0):.2f}")
                print(f"Average chars per page: {metrics.get('avg_chars_per_page', 0):,.0f}")
            
            # Show page information
            page_texts = result.get('page_texts', [])
            if page_texts:
                print(f"\nPAGE BREAKDOWN:")
                print("-" * 50)
                for page_info in page_texts[:10]:  # Show first 10 pages
                    page_num = page_info.get('page_number', 0)
                    char_count = page_info.get('char_count', 0)
                    has_content = page_info.get('has_content', False)
                    print(f"Page {page_num}: {char_count:,} chars, Content: {has_content}")
                
                if len(page_texts) > 10:
                    print(f"... and {len(page_texts) - 10} more pages")
            
            # Show content preview
            raw_text = result.get('raw_text', '')
            if raw_text:
                print(f"\nCONTENT PREVIEW (first 500 chars):")
                print("-" * 50)
                preview = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
                print(preview)
            
            # Show saved file information if available
            if "saved_file_info" in result:
                file_info = result["saved_file_info"]
                if file_info.get("success"):
                    print(f"\n📁 SAVED FILES:")
                    print(f"Content file: {file_info.get('file_path', 'Unknown')}")
                    print(f"File size: {file_info.get('file_size_kb', 0):.1f} KB")
            
            # Show LlamaIndex documents information if available
            if "llama_documents" in result:
                llama_docs = result["llama_documents"]
                print(f"\n🦙 LLAMA DOCUMENTS:")
                print(f"Total LlamaIndex documents: {len(llama_docs)}")
                
                # Show metadata from first document
                if llama_docs and hasattr(llama_docs[0], 'metadata'):
                    metadata = llama_docs[0].metadata
                    print(f"Sample metadata keys: {list(metadata.keys())[:5]}")
            
            return {
                "success": True,
                "test_results": result,
                "summary": {
                    "pages_extracted": result.get('total_pages', 0),
                    "processing_time": result.get('processing_time', 0),
                    "quality_score": result.get('quality_score', 0),
                    "total_characters": len(result.get('raw_text', '')),
                    "extraction_method": result.get('extraction_method', 'Unknown')
                }
            }
        
        else:
            print("\n" + "="*70)
            print("ENHANCED STREAMLINED PDF PROCESSOR TEST FAILED")
            print("="*70)
            print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
            print("="*70)
            
            return {
                "success": False,
                "error": result.get('error', 'Unknown error'),
                "processing_time": result.get('processing_time', 0)
            }
    
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def main():
    """Main function to run the enhanced processor test."""
    
    # Get PDF file path from command line or use default
    if len(sys.argv) > 1:
        pdf_file_path = sys.argv[1]
    else:
        # Look for common PDF files
        common_paths = [
            "../downloads/250730-hsbc-holdings-plc-interim-report-2025.pdf",
            "downloads/250730-hsbc-holdings-plc-interim-report-2025.pdf",
            "250730-hsbc-holdings-plc-interim-report-2025.pdf",
            "test.pdf",
            "sample.pdf"
        ]
        
        pdf_file_path = None
        for path in common_paths:
            if os.path.exists(path):
                pdf_file_path = path
                break
        
        if not pdf_file_path:
            print("❌ No PDF file specified and no common PDF files found.")
            print("Usage: python test_enhanced_streamlined_processor.py [pdf_file_path]")
            print("Or place one of these files in an accessible location:")
            for path in common_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    # Validate file exists
    if not os.path.exists(pdf_file_path):
        print(f"❌ PDF file not found: {pdf_file_path}")
        sys.exit(1)
    
    # Run the test with 20 pages
    print(f"📄 Testing with 20-page limit for optimal performance...")
    test_result = await test_enhanced_processor(pdf_file_path, max_pages=20)
    
    if test_result.get("success"):
        print(f"\n🎉 Enhanced StreamlinedPDFProcessor test completed successfully!")
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_enhanced_processor_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                # Remove non-serializable objects for JSON
                clean_results = test_result.copy()
                if "test_results" in clean_results and "llama_documents" in clean_results["test_results"]:
                    del clean_results["test_results"]["llama_documents"]
                
                json.dump(clean_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"📁 Test results saved to: {results_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save test results: {e}")
    
    else:
        print(f"\n❌ Enhanced StreamlinedPDFProcessor test failed!")
        print("Troubleshooting suggestions:")
        print("1. Verify the PDF file exists and is readable")
        print("2. Check if pymupdf4llm is properly installed")
        print("3. Ensure LlamaMarkdownReader is available")
        print("4. Try with a different PDF file")


if __name__ == "__main__":
    asyncio.run(main())
