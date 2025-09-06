#!/usr/bin/env python3
"""
Test LlamaMarkdownReader PDF Processor for AgentInvest2

This script tests the LlamaMarkdownReader functionality for processing
the first 20 pages of PDF files using the Ollama index reference approach.

Features:
- LlamaMarkdownReader for optimized document processing
- First 20 pages extraction for testing
- Comprehensive analysis and output
- File saving with metadata
- Error handling and fallbacks

Requirements:
    pip install pymupdf4llm

Usage:
    python test_llama_pdf_processor.py [pdf_file_path]

Author: Generated for AgentInvest2 project
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required libraries with error handling
try:
    import pymupdf4llm
    logger.info("✅ pymupdf4llm imported successfully")
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    logger.error("❌ pymupdf4llm not found. Please install it using: pip install pymupdf4llm")
    print("\nInstallation instructions:")
    print("pip install pymupdf4llm")
    sys.exit(1)

try:
    from pymupdf4llm import LlamaMarkdownReader
    LLAMA_READER_AVAILABLE = True
    logger.info("✅ LlamaMarkdownReader imported successfully")
except (ImportError, AttributeError):
    LLAMA_READER_AVAILABLE = False
    logger.error("❌ LlamaMarkdownReader not available in this version of pymupdf4llm")


class TestLlamaPDFProcessor:
    """
    Test class for LlamaMarkdownReader PDF processing.
    
    This class provides methods to:
    - Extract PDF pages using LlamaMarkdownReader
    - Process List[LlamaIndexDocument] results
    - Handle page-specific metadata and content
    - Generate formatted output files
    - Test with first 20 pages
    """
    
    def __init__(self, max_pages: int = 20):
        """Initialize the test PDF processor."""
        self.logger = logger
        self.md_reader = None
        self.max_pages = max_pages
        
    def initialize_reader(self) -> bool:
        """
        Initialize the LlamaMarkdownReader.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not LLAMA_READER_AVAILABLE:
            self.logger.error("❌ LlamaMarkdownReader not available")
            return False
            
        try:
            self.md_reader = LlamaMarkdownReader()
            self.logger.info("✅ LlamaMarkdownReader initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LlamaMarkdownReader: {e}")
            return False
    
    def extract_pdf_pages(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract PDF pages using LlamaMarkdownReader.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict containing extraction results
        """
        start_time = datetime.now()
        
        try:
            # Validate file exists
            if not os.path.exists(pdf_path):
                return {
                    "success": False,
                    "error": f"File not found: {pdf_path}",
                    "processing_time": 0
                }
            
            # Validate file is readable
            if not os.access(pdf_path, os.R_OK):
                return {
                    "success": False,
                    "error": f"File not readable: {pdf_path}",
                    "processing_time": 0
                }
            
            self.logger.info(f"📄 Processing PDF file: {pdf_path}")
            self.logger.info(f"📋 Target pages: First {self.max_pages} pages")
            
            # Initialize reader if not already done
            if not self.md_reader:
                if not self.initialize_reader():
                    return {
                        "success": False,
                        "error": "Failed to initialize LlamaMarkdownReader",
                        "processing_time": (datetime.now() - start_time).total_seconds()
                    }
            
            # Extract data using LlamaMarkdownReader
            self.logger.info("🔄 Extracting PDF content using LlamaMarkdownReader...")
            
            # Load data from PDF - this returns List[LlamaIndexDocument]
            data = self.md_reader.load_data(pdf_path)
            
            if not data:
                return {
                    "success": False,
                    "error": "No documents extracted from PDF",
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Limit to first max_pages
            original_count = len(data)
            if len(data) > self.max_pages:
                self.logger.info(f"📋 Limiting extraction to first {self.max_pages} pages (PDF has {original_count} pages)")
                data = data[:self.max_pages]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ Extraction completed successfully!")
            self.logger.info(f"   📄 Pages extracted: {len(data)} (from {original_count} total)")
            self.logger.info(f"   ⏱️ Processing time: {processing_time:.2f} seconds")
            
            return {
                "success": True,
                "data": data,
                "total_pages_extracted": len(data),
                "total_pages_in_pdf": original_count,
                "processing_time": processing_time,
                "pdf_path": pdf_path
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Unexpected error during PDF extraction: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def analyze_extracted_data(self, data: List) -> Dict[str, Any]:
        """
        Analyze the extracted LlamaIndexDocument data.
        
        Args:
            data (List): List of LlamaIndexDocument objects
            
        Returns:
            Dict: Analysis results including metadata and content statistics
        """
        analysis = {
            "total_documents": len(data),
            "pages_info": [],
            "total_content_length": 0,
            "metadata_summary": {},
            "content_preview": {}
        }
        
        for i, doc in enumerate(data):
            try:
                # Extract page information
                page_info = {
                    "page_index": i,
                    "page_number": i + 1,
                    "content_length": len(doc.text) if hasattr(doc, 'text') else 0,
                    "has_metadata": hasattr(doc, 'metadata') and doc.metadata is not None,
                    "metadata_keys": list(doc.metadata.keys()) if hasattr(doc, 'metadata') and doc.metadata else []
                }
                
                # Add content preview for first few pages
                if i < 3 and hasattr(doc, 'text'):
                    analysis["content_preview"][f"page_{i+1}"] = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                
                # Accumulate total content length
                if hasattr(doc, 'text'):
                    analysis["total_content_length"] += len(doc.text)
                
                analysis["pages_info"].append(page_info)
                
                # Analyze metadata
                if hasattr(doc, 'metadata') and doc.metadata:
                    for key in doc.metadata.keys():
                        if key not in analysis["metadata_summary"]:
                            analysis["metadata_summary"][key] = 0
                        analysis["metadata_summary"][key] += 1
                        
            except Exception as e:
                self.logger.warning(f"⚠️ Error analyzing document {i}: {e}")
                continue
        
        return analysis
    
    def save_extracted_content(self, data: List, pdf_path: str, output_dir: str = "downloads/test_output") -> Dict[str, Any]:
        """
        Save the extracted content to files.
        
        Args:
            data (List): List of LlamaIndexDocument objects
            pdf_path (str): Original PDF file path
            output_dir (str): Output directory for saved files
            
        Returns:
            Dict: Save operation results
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            pdf_name = Path(pdf_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_file = output_path / f"{pdf_name}_llama_extracted_{timestamp}.md"
            metadata_file = output_path / f"{pdf_name}_llama_metadata_{timestamp}.json"
            
            self.logger.info(f"💾 Saving extracted content to: {content_file}")
            self.logger.info(f"💾 Saving metadata to: {metadata_file}")
            
            # Save markdown content
            with open(content_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*80 + "\n")
                f.write(f"PDF LLAMA MARKDOWN EXTRACTION - FIRST {self.max_pages} PAGES\n")
                f.write("="*80 + "\n")
                f.write(f"Source PDF: {pdf_path}\n")
                f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Pages Extracted: {len(data)}\n")
                f.write(f"Extraction Method: pymupdf4llm LlamaMarkdownReader\n")
                f.write(f"Generated by: Test LlamaMarkdown Processor v1.0\n")
                f.write("="*80 + "\n\n")
                
                # Write content for each page
                for i, doc in enumerate(data):
                    f.write(f"## PAGE {i+1}\n")
                    f.write("-" * 40 + "\n")
                    
                    # Write metadata if available
                    if hasattr(doc, 'metadata') and doc.metadata:
                        f.write("**Metadata:**\n")
                        for key, value in doc.metadata.items():
                            f.write(f"- {key}: {value}\n")
                        f.write("\n")
                    
                    # Write content
                    f.write("**Content:**\n")
                    if hasattr(doc, 'text'):
                        f.write(doc.text)
                    else:
                        f.write("No text content available")
                    
                    f.write("\n\n" + "="*80 + "\n\n")
            
            # Save metadata as JSON
            metadata_collection = []
            for i, doc in enumerate(data):
                page_metadata = {
                    "page_number": i + 1,
                    "content_length": len(doc.text) if hasattr(doc, 'text') else 0,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') and doc.metadata else {},
                    "has_content": hasattr(doc, 'text') and bool(doc.text),
                    "document_id": getattr(doc, 'id_', f"doc_{i+1}")
                }
                metadata_collection.append(page_metadata)
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "extraction_info": {
                        "source_pdf": pdf_path,
                        "extraction_date": datetime.now().isoformat(),
                        "total_pages": len(data),
                        "max_pages_limit": self.max_pages,
                        "extraction_method": "pymupdf4llm LlamaMarkdownReader"
                    },
                    "pages_metadata": metadata_collection
                }, f, indent=2, ensure_ascii=False)
            
            # Calculate file sizes
            content_size_kb = content_file.stat().st_size / 1024
            metadata_size_kb = metadata_file.stat().st_size / 1024
            
            self.logger.info(f"✅ Content saved successfully!")
            self.logger.info(f"   📁 Content file: {content_file} ({content_size_kb:.2f} KB)")
            self.logger.info(f"   📁 Metadata file: {metadata_file} ({metadata_size_kb:.2f} KB)")
            
            return {
                "success": True,
                "content_file": str(content_file),
                "metadata_file": str(metadata_file),
                "content_size_kb": content_size_kb,
                "metadata_size_kb": metadata_size_kb
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save extracted content: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """Main function to test LlamaMarkdownReader PDF extraction."""
    
    # Get PDF file path from command line or use default
    if len(sys.argv) > 1:
        pdf_file_path = sys.argv[1]
    else:
        # Look for common PDF files in the current directory
        common_names = [
            "250730-hsbc-holdings-plc-interim-report-2025.pdf",
            "test.pdf",
            "sample.pdf"
        ]
        
        pdf_file_path = None
        for name in common_names:
            if os.path.exists(name):
                pdf_file_path = name
                break
        
        if not pdf_file_path:
            print("❌ No PDF file specified and no common PDF files found.")
            print("Usage: python test_llama_pdf_processor.py [pdf_file_path]")
            print("Or place one of these files in the current directory:")
            for name in common_names:
                print(f"  - {name}")
            sys.exit(1)
    
    # Initialize the test processor
    processor = TestLlamaPDFProcessor(max_pages=20)
    
    logger.info("🚀 Starting LlamaMarkdownReader PDF extraction test...")
    logger.info(f"📁 Target file: {pdf_file_path}")
    logger.info(f"📋 Target pages: First {processor.max_pages} pages")
    
    # Extract PDF content
    result = processor.extract_pdf_pages(pdf_file_path)
    
    if result["success"]:
        data = result["data"]
        
        # Display extraction summary
        print("\n" + "="*60)
        print("LLAMA MARKDOWN EXTRACTION TEST SUMMARY")
        print("="*60)
        print(f"Source PDF: {result['pdf_path']}")
        print(f"Pages Extracted: {result['total_pages_extracted']}")
        print(f"Total Pages in PDF: {result['total_pages_in_pdf']}")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print("="*60)
        
        # Analyze the extracted data
        logger.info("📊 Analyzing extracted data...")
        analysis = processor.analyze_extracted_data(data)
        
        print(f"\nDATA ANALYSIS:")
        print(f"Total Documents: {analysis['total_documents']}")
        print(f"Total Content Length: {analysis['total_content_length']:,} characters")
        print(f"Metadata Keys Found: {list(analysis['metadata_summary'].keys())}")
        
        # Show content preview
        if analysis["content_preview"]:
            print(f"\nCONTENT PREVIEW:")
            for page_key, preview in analysis["content_preview"].items():
                print(f"\n{page_key.upper()}:")
                print("-" * 40)
                print(preview)
        
        # Save the content
        save_result = processor.save_extracted_content(data, pdf_file_path)
        if save_result["success"]:
            print(f"\n✅ Content saved successfully!")
            print(f"📁 Content file: {save_result['content_file']}")
            print(f"📁 Metadata file: {save_result['metadata_file']}")
        else:
            print(f"\n❌ Failed to save content: {save_result['error']}")
        
        # Show detailed page information
        print(f"\nDETAILED PAGE INFORMATION:")
        print("-" * 60)
        for page_info in analysis["pages_info"]:
            print(f"Page {page_info['page_number']}: {page_info['content_length']:,} chars, "
                  f"Metadata: {page_info['has_metadata']}")
        
        print(f"\n🎉 Test completed successfully!")
        
    else:
        print("\n" + "="*60)
        print("LLAMA MARKDOWN EXTRACTION TEST FAILED")
        print("="*60)
        print(f"Error: {result['error']}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
        print("="*60)
        
        # Provide troubleshooting suggestions
        print("\nTroubleshooting suggestions:")
        print("1. Verify the PDF file exists and is readable")
        print("2. Check if the PDF file is corrupted")
        print("3. Ensure you have sufficient permissions to read the file")
        print("4. Install required dependencies: pip install pymupdf4llm")
        print("5. Try with a different PDF file to test the script")


if __name__ == "__main__":
    main()
