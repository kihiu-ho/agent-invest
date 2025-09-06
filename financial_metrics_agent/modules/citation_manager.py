"""
Citation Manager Module

Handles citation tracking, numbering, and source attribution for HTML reports.
Provides dynamic citation generation to eliminate hardcoded citations.
"""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime


class CitationManager:
    """Manages citations and source references for HTML reports with dynamic generation."""

    def __init__(self):
        """Initialize citation manager."""
        self.citation_counter = 0
        self.citation_map = {}
        self.logger = logging.getLogger(__name__)
    
    def reset_citation_counter(self):
        """Reset citation counter for new report generation."""
        self.citation_counter = 0
        self.citation_map = {}
    
    def get_citation_number(self, citation_key: str) -> int:
        """Get or create a citation number for the given citation key."""
        if citation_key not in self.citation_map:
            self.citation_counter += 1
            self.citation_map[citation_key] = self.citation_counter
        return self.citation_map[citation_key]
    
    def format_inline_citation(self, citation_key: str) -> str:
        """Format an inline citation with number."""
        citation_num = self.get_citation_number(citation_key)
        return f"[{citation_num}]"

    def create_dynamic_citation(self, source_type: str, ticker: str = None,
                              company_name: str = None, timestamp: datetime = None,
                              url: str = None, document_name: str = None,
                              page_number: int = None, section: str = None) -> str:
        """
        Create a dynamic citation based on actual data sources.

        Args:
            source_type: Type of source ('yahoo_finance', 'annual_report', 'web_scraping', etc.)
            ticker: Stock ticker symbol
            company_name: Company name
            timestamp: When data was retrieved
            url: Source URL if applicable
            document_name: Document filename if applicable
            page_number: Page number if applicable
            section: Document section if applicable

        Returns:
            Formatted citation string
        """
        if timestamp is None:
            timestamp = datetime.now()

        timestamp_str = timestamp.strftime('%Y-%m-%d')

        if source_type == 'yahoo_finance':
            return f"Yahoo Finance API - Real-time Financial Data, Retrieved: {timestamp_str}"

        elif source_type == 'annual_report':
            citation_parts = []
            if company_name:
                citation_parts.append(f"{company_name} Annual Report")
            if document_name:
                citation_parts.append(f"Document: {document_name}")
            if page_number:
                citation_parts.append(f"Page {page_number}")
            if section:
                citation_parts.append(f"Section: {section}")

            return ", ".join(citation_parts)

        elif source_type == 'web_scraping':
            if url:
                domain = self._extract_domain(url)
                if 'stockanalysis.com' in domain:
                    return f"StockAnalysis.com - Financial Data: {url}"
                elif 'tipranks.com' in domain:
                    return f"TipRanks.com - Investment Analysis: {url}"
                else:
                    return f"Financial Data from {domain}: {url}"
            else:
                return f"Web-scraped Financial Data, Retrieved: {timestamp_str}"

        elif source_type == 'weaviate_vector':
            citation_parts = ["Vector Database - Annual Report Analysis"]
            if company_name:
                citation_parts.append(f"Company: {company_name}")
            citation_parts.append(f"Retrieved: {timestamp_str}")
            return ", ".join(citation_parts)

        elif source_type == 'market_data':
            return f"Market Data Provider - Real-time Quotes, Retrieved: {timestamp_str}"

        else:
            # Generic fallback
            return f"Financial Data Source, Retrieved: {timestamp_str}"

    def create_company_annual_report_citation(self, company_name: str, ticker: str,
                                            document_name: str = None, page_number: int = None,
                                            section: str = None) -> str:
        """
        Create a dynamic annual report citation for any company.

        Args:
            company_name: Company name
            ticker: Stock ticker
            document_name: Document filename if available
            page_number: Page number if available
            section: Section name if available

        Returns:
            Dynamic annual report citation
        """
        # Clean ticker for filename generation
        clean_ticker = ticker.replace('.HK', '').replace('.', '_')

        # Generate document name if not provided
        if not document_name:
            current_year = datetime.now().year
            document_name = f"{company_name.replace(' ', '_')}_Annual_Report_{current_year}.pdf"

        citation_parts = [f"{company_name} Annual Report"]

        if page_number:
            citation_parts.append(f"Page {page_number}")

        if section:
            citation_parts.append(section)

        citation_parts.append(document_name)

        return ", ".join(citation_parts)

    def create_web_source_citation(self, url: str, data_type: str = None) -> str:
        """
        Create a dynamic web source citation.

        Args:
            url: Source URL
            data_type: Type of data (e.g., 'financials', 'forecasts', 'technical_analysis')

        Returns:
            Dynamic web source citation
        """
        domain = self._extract_domain(url)

        # Create descriptive citation based on URL and data type
        if 'stockanalysis.com' in domain:
            if data_type == 'financials' or '/financials/' in url:
                return f"StockAnalysis.com - Financial Statements: {url}"
            elif data_type == 'statistics' or '/statistics/' in url:
                return f"StockAnalysis.com - Key Statistics: {url}"
            elif data_type == 'dividend' or '/dividend/' in url:
                return f"StockAnalysis.com - Dividend Information: {url}"
            else:
                return f"StockAnalysis.com - Market Data: {url}"

        elif 'tipranks.com' in domain:
            if data_type == 'forecasts' or '/forecast' in url:
                return f"TipRanks.com - Analyst Forecasts: {url}"
            elif data_type == 'technical_analysis' or '/technical-analysis' in url:
                return f"TipRanks.com - Technical Analysis: {url}"
            elif data_type == 'earnings' or '/earnings' in url:
                return f"TipRanks.com - Earnings Data: {url}"
            else:
                return f"TipRanks.com - Investment Research: {url}"

        else:
            return f"Financial Data from {domain}: {url}"

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return domain_match.group(1) if domain_match else url

    def convert_source_citations_to_numbered(self, text: str) -> str:
        """Convert various citation formats to numbered citations [1], [2], etc."""
        if not text:
            return text

        # Pattern to match [Source: URL] or [Source: description]
        source_pattern = r'\[Source:\s*([^\]]+)\]'

        # Pattern to match Investment Decision Agent citations like [S1: URL] or [T1: URL]
        agent_citation_pattern = r'\[([ST]\d+):\s*([^\]]+)\]'

        def replace_source_citation(match):
            source_info = match.group(1).strip()
            citation_num = self.get_citation_number(source_info)
            return f"[{citation_num}]"

        def replace_agent_citation(match):
            source_tag = match.group(1).strip()  # S1, T1, etc.
            source_url = match.group(2).strip()  # URL

            # Create a descriptive source name based on the URL
            source_description = self._create_source_description(source_tag, source_url)
            citation_num = self.get_citation_number(source_description)
            return f"[{citation_num}]"

        # Apply replacements
        text = re.sub(source_pattern, replace_source_citation, text)
        text = re.sub(agent_citation_pattern, replace_agent_citation, text)

        return text

    def convert_legacy_citations_to_dynamic(self, text: str, ticker: str = None,
                                          company_name: str = None) -> str:
        """
        Convert legacy hardcoded citations to dynamic ones.

        Args:
            text: Text containing legacy citations
            ticker: Stock ticker for context
            company_name: Company name for context

        Returns:
            Text with dynamic citations
        """
        if not text:
            return text

        # Remove hardcoded timestamps
        text = re.sub(r',?\s*Timestamp:\s*\d{4}-\d{2}-\d{2}', '', text)

        # Replace hardcoded company names with dynamic ones if provided
        if company_name:
            text = re.sub(r'Tencent Holdings', company_name, text, flags=re.IGNORECASE)
            text = re.sub(r'HSBC', company_name, text, flags=re.IGNORECASE)

        # Replace hardcoded document names with dynamic ones
        if ticker and company_name:
            current_year = datetime.now().year
            dynamic_doc_name = f"{company_name.replace(' ', '_')}_Annual_Report_{current_year}.pdf"
            text = re.sub(r'Tencent_Holdings_Annual_Report_\d{4}\.pdf', dynamic_doc_name, text)
            text = re.sub(r'HSBC_Annual_Report_\d{4}\.pdf', dynamic_doc_name, text)

        # Add current timestamp if not present
        current_timestamp = datetime.now().strftime('%Y-%m-%d')
        if 'Retrieved:' not in text and 'Timestamp:' not in text and 'API' in text:
            text += f", Retrieved: {current_timestamp}"

        return text

    def _create_source_description(self, source_tag: str, source_url: str) -> str:
        """Create a descriptive source name based on URL and tag."""
        if 'stockanalysis.com' in source_url:
            if source_tag.startswith('S'):
                return f"StockAnalysis.com Financial Data, URL: {source_url}"
            else:
                return f"StockAnalysis.com Technical Analysis, URL: {source_url}"
        elif 'tipranks.com' in source_url:
            if source_tag.startswith('T'):
                return f"TipRanks.com Analyst Forecasts, URL: {source_url}"
            else:
                return f"TipRanks.com Investment Analysis, URL: {source_url}"
        elif 'yahoo' in source_url.lower():
            return f"Yahoo Finance API, Real-time Data, Timestamp: {source_url.split('/')[-1] if '/' in source_url else 'Current'}"
        else:
            return f"Financial Data Source ({source_tag}), URL: {source_url}"
    
    def generate_numbered_references_section(self) -> str:
        """Generate the numbered references section for the report."""
        if not self.citation_map:
            return ""
        
        references_html = """
        <div class="section">
            <h2>📚 References</h2>
            <div class="references-list">
        """
        
        # Sort citations by number
        sorted_citations = sorted(self.citation_map.items(), key=lambda x: x[1])
        
        for source_info, citation_num in sorted_citations:
            # Parse source info to extract components
            source_type, url, page_info, filename = self._parse_source_info(source_info)
            
            references_html += f"""
                <div class="reference-item">
                    <span class="reference-number">[{citation_num}]</span>
                    <span class="reference-content">
                        <strong>{source_type}</strong>
                        {f', {page_info}' if page_info else ''}
                        {f', {filename}' if filename else ''}
                        {f'<br><a href="{url}" target="_blank">{url}</a>' if url and url.startswith('http') else ''}
                    </span>
                </div>
            """
        
        references_html += """
            </div>
        </div>
        """
        
        return references_html
    
    def _parse_source_info(self, source_info: str) -> tuple:
        """Parse source information to extract components."""
        # Default values
        source_type = "Financial Data Source"
        url = ""
        page_info = ""
        filename = ""
        
        # Extract URL if present
        url_match = re.search(r'URL:\s*(https?://[^\s,]+)', source_info)
        if url_match:
            url = url_match.group(1)
        
        # Extract page information
        page_match = re.search(r'Page\s+(\d+)', source_info)
        if page_match:
            page_info = f"Page {page_match.group(1)}"
        
        # Extract filename
        filename_match = re.search(r'([^/,]+\.pdf)', source_info)
        if filename_match:
            filename = filename_match.group(1)
        
        # Determine source type
        if 'StockAnalysis.com' in source_info:
            source_type = "StockAnalysis.com"
        elif 'TipRanks.com' in source_info:
            source_type = "TipRanks.com"
        elif 'Yahoo Finance' in source_info:
            source_type = "Yahoo Finance API"
        elif 'Annual Report' in source_info:
            source_type = "Annual Report"
        elif 'Technical Analysis' in source_info:
            source_type = "Technical Analysis"
        
        return source_type, url, page_info, filename
    
    def convert_to_numbered_citations(self, content: str) -> str:
        """Convert citation markers to numbered format for HTML display."""
        if not content:
            return content
        
        # Convert [Source: ...] patterns to numbered citations
        content = self.convert_source_citations_to_numbered(content)
        
        return content
