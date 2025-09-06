#!/usr/bin/env python3
"""
Extraction Method Constants for Financial Metrics Agent

This module defines standardized constants for extraction methods used across
the web scraping and document processing infrastructure to ensure consistency
and eliminate hardcoded string literals.
"""

from typing import List, Set
from enum import Enum


class WebScrapingMethods:
    """Constants for web scraping extraction methods."""
    
    # Primary web scraping methods
    LLM_FILTERED = "llm_filtered"
    BASIC_MARKDOWN = "basic_markdown"
    
    # Fallback and error states
    UNKNOWN = "unknown"
    
    @classmethod
    def get_preferred_method(cls, crawl4ai_available: bool) -> str:
        """
        Get the preferred extraction method based on availability.
        
        Args:
            crawl4ai_available: Whether Crawl4AI is available
            
        Returns:
            Preferred extraction method string
        """
        return cls.LLM_FILTERED if crawl4ai_available else cls.BASIC_MARKDOWN
    
    @classmethod
    def get_valid_methods(cls) -> Set[str]:
        """Get set of all valid web scraping extraction methods."""
        return {cls.LLM_FILTERED, cls.BASIC_MARKDOWN, cls.UNKNOWN}
    
    @classmethod
    def is_valid_method(cls, method: str) -> bool:
        """Check if an extraction method is valid."""
        return method in cls.get_valid_methods()


class PDFExtractionMethods:
    """Constants for PDF extraction methods."""
    
    # PDF processing methods
    PYMUPDF = "pymupdf"
    PDFMINER = "pdfminer"  # Legacy, may be removed
    LANGEXTRACT = "langextract"  # Legacy, may be removed
    OCR = "ocr"  # Legacy, may be removed
    
    # Fallback and error states
    UNKNOWN = "unknown"
    
    @classmethod
    def get_valid_methods(cls) -> Set[str]:
        """Get set of all valid PDF extraction methods."""
        return {cls.PYMUPDF, cls.PDFMINER, cls.LANGEXTRACT, cls.OCR, cls.UNKNOWN}
    
    @classmethod
    def is_valid_method(cls, method: str) -> bool:
        """Check if a PDF extraction method is valid."""
        return method in cls.get_valid_methods()


class ExtractionMethodValidator:
    """Validator for extraction methods across different processing types."""
    
    @staticmethod
    def validate_web_scraping_method(method: str) -> bool:
        """
        Validate a web scraping extraction method.
        
        Args:
            method: Extraction method to validate
            
        Returns:
            True if valid, False otherwise
        """
        return WebScrapingMethods.is_valid_method(method)
    
    @staticmethod
    def validate_pdf_method(method: str) -> bool:
        """
        Validate a PDF extraction method.
        
        Args:
            method: Extraction method to validate
            
        Returns:
            True if valid, False otherwise
        """
        return PDFExtractionMethods.is_valid_method(method)
    
    @staticmethod
    def get_fallback_method(processing_type: str, **kwargs) -> str:
        """
        Get appropriate fallback method for a processing type.
        
        Args:
            processing_type: Type of processing ('web_scraping' or 'pdf')
            **kwargs: Additional parameters (e.g., crawl4ai_available)
            
        Returns:
            Appropriate fallback method
        """
        if processing_type == "web_scraping":
            crawl4ai_available = kwargs.get('crawl4ai_available', False)
            return WebScrapingMethods.get_preferred_method(crawl4ai_available)
        elif processing_type == "pdf":
            return PDFExtractionMethods.PYMUPDF
        else:
            return WebScrapingMethods.UNKNOWN


# Legacy compatibility - can be removed in future versions
class ExtractionMethods:
    """
    Legacy extraction methods class for backward compatibility.
    
    DEPRECATED: Use WebScrapingMethods or PDFExtractionMethods instead.
    """
    
    # Web scraping methods
    LLM_FILTERED = WebScrapingMethods.LLM_FILTERED
    BASIC_MARKDOWN = WebScrapingMethods.BASIC_MARKDOWN
    
    # PDF methods
    PYMUPDF = PDFExtractionMethods.PYMUPDF
    
    # Common
    UNKNOWN = WebScrapingMethods.UNKNOWN
    
    @classmethod
    def get_web_scraping_default(cls, crawl4ai_available: bool) -> str:
        """
        DEPRECATED: Use WebScrapingMethods.get_preferred_method() instead.
        """
        return WebScrapingMethods.get_preferred_method(crawl4ai_available)


# Export commonly used constants for easy importing
__all__ = [
    'WebScrapingMethods',
    'PDFExtractionMethods', 
    'ExtractionMethodValidator',
    'ExtractionMethods'  # Legacy
]
