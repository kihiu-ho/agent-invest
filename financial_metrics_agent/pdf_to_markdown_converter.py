#!/usr/bin/env python3
"""
PDF to Markdown Converter using PyMuPDF RAG Module

This script converts PDF files to Markdown format using PyMuPDF's RAG (Retrieval-Augmented Generation) 
module. It preserves document structure, headings, tables, and formatting while generating clean 
Markdown output.

Requirements:
    pip install pymupdf
    pip install pymupdf4llm

Usage:
    python pdf_to_markdown_converter.py

Author: Generated for AgentInvest2 project
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required libraries with error handling
try:
    import pymupdf
    logger.info("✅ PyMuPDF imported successfully")
except ImportError:
    logger.error("❌ PyMuPDF not found. Please install it using: pip install pymupdf")
    sys.exit(1)

try:
    from pymupdf4llm import to_markdown
    logger.info("✅ pymupdf4llm imported successfully")
except ImportError:
    logger.error("❌ pymupdf4llm not found. Please install it using: pip install pymupdf4llm")
    sys.exit(1)


class PDFToMarkdownConverter:
    """
    A class to convert PDF files to Markdown format using PyMuPDF's RAG module.
    
    This class provides methods to:
    - Convert PDF files to clean Markdown format
    - Preserve document structure and formatting
    - Handle specific page ranges
    - Generate timestamped output files
    - Provide detailed conversion metrics
    """
    
    def __init__(self):
        """Initialize the PDF to Markdown converter."""
        self.logger = logger
        
    def convert_pdf_to_markdown(self, pdf_path: str, pages: Optional[List[int]] = None, 
                               output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert a PDF file to Markdown format.
        
        Args:
            pdf_path (str): Path to the PDF file
            pages (List[int], optional): List of page numbers to convert (0-based indexing)
            output_path (str, optional): Custom output file path
            
        Returns:
            Dict containing conversion results with the following keys:
            - success (bool): Whether conversion was successful
            - markdown_text (str): Generated Markdown content
            - output_file (str): Path to the saved Markdown file
            - total_pages (int): Number of pages in the PDF
            - converted_pages (int): Number of pages converted
            - processing_time (float): Time taken for conversion in seconds
            - file_size_kb (float): Size of generated Markdown file in KB
            - error (str): Error message if conversion failed
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
            
            self.logger.info(f"📄 Opening PDF file: {pdf_path}")
            
            # Open the PDF document
            doc = pymupdf.open(pdf_path)
            total_pages = len(doc)
            
            self.logger.info(f"📖 PDF contains {total_pages} pages")
            
            # Determine which pages to convert
            if pages is None:
                # Default: convert pages 1-10 (0-based indexing: 0-9)
                page_list = list(range(min(10, total_pages)))
                self.logger.info(f"📋 Converting default pages 1-{min(10, total_pages)} (0-based: 0-{min(10, total_pages)-1})")
            else:
                # Validate page numbers
                page_list = [p for p in pages if 0 <= p < total_pages]
                if len(page_list) != len(pages):
                    invalid_pages = [p for p in pages if p < 0 or p >= total_pages]
                    self.logger.warning(f"⚠️ Invalid page numbers removed: {invalid_pages}")
                self.logger.info(f"📋 Converting specified pages: {page_list}")
            
            if not page_list:
                doc.close()
                return {
                    "success": False,
                    "error": "No valid pages to convert",
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Convert PDF to Markdown
            self.logger.info(f"🔄 Converting {len(page_list)} pages to Markdown...")
            
            # Use pymupdf4llm to convert to markdown
            markdown_text = to_markdown(doc, pages=page_list)
            
            # Close the document
            doc.close()
            
            # Generate output filename if not provided
            if output_path is None:
                pdf_name = Path(pdf_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{pdf_name}_markdown_{timestamp}.md"
            
            # Save the Markdown content
            self.logger.info(f"💾 Saving Markdown to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                # Write metadata header
                metadata_header = self._generate_metadata_header(
                    pdf_path, page_list, total_pages, start_time
                )
                output_file.write(metadata_header)
                output_file.write("\n" + "="*80 + "\n")
                output_file.write("CONVERTED MARKDOWN CONTENT\n")
                output_file.write("="*80 + "\n\n")
                
                # Write the converted Markdown
                output_file.write(markdown_text)
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            file_size_kb = os.path.getsize(output_path) / 1024
            
            # Log conversion summary
            self.logger.info(f"✅ Conversion completed successfully!")
            self.logger.info(f"   📄 Total pages in PDF: {total_pages}")
            self.logger.info(f"   📝 Pages converted: {len(page_list)}")
            self.logger.info(f"   📁 Output file: {output_path}")
            self.logger.info(f"   📊 File size: {file_size_kb:.2f} KB")
            self.logger.info(f"   ⏱️ Processing time: {processing_time:.2f} seconds")
            
            return {
                "success": True,
                "markdown_text": markdown_text,
                "output_file": output_path,
                "total_pages": total_pages,
                "converted_pages": len(page_list),
                "processing_time": processing_time,
                "file_size_kb": file_size_kb,
                "pdf_path": pdf_path
            }
            
        except pymupdf.FileDataError as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Corrupted or invalid PDF file: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
            
        except pymupdf.FileNotFoundError as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"PDF file not found: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Unexpected error during PDF conversion: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def _generate_metadata_header(self, pdf_path: str, page_list: List[int], 
                                 total_pages: int, start_time: datetime) -> str:
        """
        Generate a metadata header for the Markdown file.
        
        Args:
            pdf_path (str): Path to the source PDF
            page_list (List[int]): List of converted pages
            total_pages (int): Total pages in PDF
            start_time (datetime): Conversion start time
            
        Returns:
            str: Formatted metadata header
        """
        header = "="*80 + "\n"
        header += "PDF TO MARKDOWN CONVERSION METADATA\n"
        header += "="*80 + "\n"
        header += f"Source PDF: {pdf_path}\n"
        header += f"Conversion Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Total Pages in PDF: {total_pages}\n"
        header += f"Pages Converted: {len(page_list)}\n"
        header += f"Page Range: {page_list[0]+1}-{page_list[-1]+1} (1-based)\n" if page_list else "Page Range: None\n"
        header += f"Conversion Tool: PyMuPDF RAG Module (pymupdf4llm)\n"
        header += f"Generated by: PDF to Markdown Converter v1.0\n"
        
        return header
    
    def convert_with_custom_pages(self, pdf_path: str, page_ranges: str, 
                                 output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert PDF with custom page ranges (e.g., "1-10,15,20-25").
        
        Args:
            pdf_path (str): Path to the PDF file
            page_ranges (str): Page ranges in format "1-10,15,20-25" (1-based)
            output_path (str, optional): Custom output file path
            
        Returns:
            Dict: Conversion results
        """
        try:
            # Parse page ranges
            page_list = self._parse_page_ranges(page_ranges)
            
            # Convert to 0-based indexing
            page_list_zero_based = [p - 1 for p in page_list if p > 0]
            
            self.logger.info(f"📋 Parsed page ranges '{page_ranges}' to: {page_list} (1-based)")
            self.logger.info(f"📋 Converted to 0-based indexing: {page_list_zero_based}")
            
            return self.convert_pdf_to_markdown(pdf_path, page_list_zero_based, output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing page ranges '{page_ranges}': {e}")
            return {
                "success": False,
                "error": f"Invalid page range format: {page_ranges}",
                "processing_time": 0
            }
    
    def _parse_page_ranges(self, page_ranges: str) -> List[int]:
        """
        Parse page ranges string into list of page numbers.
        
        Args:
            page_ranges (str): Page ranges like "1-10,15,20-25"
            
        Returns:
            List[int]: List of page numbers (1-based)
        """
        pages = []
        
        for part in page_ranges.split(','):
            part = part.strip()
            
            if '-' in part:
                # Handle range like "1-10"
                start, end = part.split('-', 1)
                start, end = int(start.strip()), int(end.strip())
                pages.extend(range(start, end + 1))
            else:
                # Handle single page like "15"
                pages.append(int(part))
        
        return sorted(list(set(pages)))  # Remove duplicates and sort


def main():
    """Main function to demonstrate PDF to Markdown conversion."""
    
    # Initialize the converter
    converter = PDFToMarkdownConverter()
    
    # Define the target PDF file path
    pdf_file_path = "downloads/250730-hsbc-holdings-plc-interim-report-2025-1-10.pdf"
    
    logger.info("🚀 Starting PDF to Markdown conversion...")
    logger.info(f"📁 Target file: {pdf_file_path}")
    
    # Check if file exists, if not provide alternative options
    if not os.path.exists(pdf_file_path):
        logger.warning(f"⚠️ Target file not found: {pdf_file_path}")
        
        # Look for alternative PDF files in downloads directory
        downloads_dir = Path("downloads")
        if downloads_dir.exists():
            pdf_files = list(downloads_dir.rglob("*.pdf"))
            if pdf_files:
                logger.info(f"📁 Found {len(pdf_files)} PDF files in downloads directory:")
                for i, pdf_file in enumerate(pdf_files[:5], 1):  # Show first 5
                    logger.info(f"   {i}. {pdf_file}")
                
                # Use the first available PDF for demonstration
                pdf_file_path = str(pdf_files[0])
                logger.info(f"🔄 Using alternative file for demonstration: {pdf_file_path}")
            else:
                logger.error("❌ No PDF files found in downloads directory")
                print("\nTo use this script:")
                print("1. Place the target PDF file in the downloads directory")
                print("2. Or modify the pdf_file_path variable in the script")
                return
        else:
            logger.error("❌ Downloads directory not found")
            return
    
    # Convert PDF to Markdown (pages 1-10, which is 0-9 in 0-based indexing)
    result = converter.convert_pdf_to_markdown(pdf_file_path)
    
    if result["success"]:
        # Display conversion summary
        print("\n" + "="*60)
        print("PDF TO MARKDOWN CONVERSION SUMMARY")
        print("="*60)
        print(f"Source PDF: {result['pdf_path']}")
        print(f"Output File: {result['output_file']}")
        print(f"Total Pages: {result['total_pages']}")
        print(f"Converted Pages: {result['converted_pages']}")
        print(f"File Size: {result['file_size_kb']:.2f} KB")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print("="*60)
        
        # Show first 500 characters as preview
        preview_text = result["markdown_text"][:500]
        print("\nMARKDOWN PREVIEW (first 500 characters):")
        print("-" * 40)
        print(preview_text)
        if len(result["markdown_text"]) > 500:
            print("... (truncated)")
        print("-" * 40)
        
        print(f"\n✅ Conversion completed! Markdown saved to: {result['output_file']}")
        
    else:
        print("\n" + "="*60)
        print("PDF TO MARKDOWN CONVERSION FAILED")
        print("="*60)
        print(f"Error: {result['error']}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
        print("="*60)
        
        # Provide troubleshooting suggestions
        print("\nTroubleshooting suggestions:")
        print("1. Verify the PDF file exists and is readable")
        print("2. Check if the PDF file is corrupted")
        print("3. Ensure you have sufficient permissions to read the file")
        print("4. Install required dependencies: pip install pymupdf pymupdf4llm")
        print("5. Try with a different PDF file to test the script")


if __name__ == "__main__":
    main()
