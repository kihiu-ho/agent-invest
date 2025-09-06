#!/usr/bin/env python3
"""
Citation Tracking System for Financial Analysis
Implements comprehensive grounding and citation mechanisms using RAG methodology.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
import json

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Structured citation with source attribution."""
    source_url: str
    source_type: str  # 'yahoo_finance', 'stockanalysis', 'tipranks', 'estimated'
    section: str
    passage: str
    retrieved_at: str
    confidence: float = 1.0
    metric_name: Optional[str] = None
    raw_value: Optional[Any] = None
    citation_type: str = 'claim'  # 'claim' for analytical statements, 'data' for raw metrics

    def to_inline_citation(self) -> str:
        """Generate inline citation format."""
        return f"[Source: {self.source_url} | Section: {self.section} | Retrieved: {self.retrieved_at}]"

    def to_detailed_citation(self) -> str:
        """Generate detailed citation with passage."""
        return f"[Source: {self.source_url} | Section: {self.section} | Passage: \"{self.passage}\" | Retrieved: {self.retrieved_at}]"

@dataclass
class DataSource:
    """Track data sources without requiring individual citations."""
    source_url: str
    source_type: str
    description: str
    retrieved_at: str
    confidence: float = 1.0
    metrics_count: int = 0

@dataclass
class CitationCollection:
    """Collection of citations and data sources for a specific analysis."""
    ticker: str
    citations: List[Citation] = field(default_factory=list)  # Only analytical claims
    data_sources: List[DataSource] = field(default_factory=list)  # Raw data sources
    source_urls: Dict[str, str] = field(default_factory=dict)
    retrieval_log: List[Dict[str, Any]] = field(default_factory=list)

    def add_citation(self, citation: Citation) -> str:
        """Add citation for analytical claims and return unique ID."""
        citation_id = f"cite_{len(self.citations) + 1}"
        self.citations.append(citation)

        # Log retrieval operation
        self.retrieval_log.append({
            "citation_id": citation_id,
            "source_url": citation.source_url,
            "source_type": citation.source_type,
            "retrieved_at": citation.retrieved_at,
            "metric_name": citation.metric_name,
            "citation_type": citation.citation_type
        })

        return citation_id

    def add_data_source(self, data_source: DataSource) -> str:
        """Add data source tracking without individual citations."""
        # Check if source already exists
        for existing_source in self.data_sources:
            if existing_source.source_url == data_source.source_url:
                existing_source.metrics_count += data_source.metrics_count
                return f"source_{self.data_sources.index(existing_source) + 1}"

        # Add new source
        source_id = f"source_{len(self.data_sources) + 1}"
        self.data_sources.append(data_source)
        return source_id
    
    def get_citations_by_source(self, source_type: str) -> List[Citation]:
        """Get all citations from a specific source type."""
        return [c for c in self.citations if c.source_type == source_type]
    
    def get_citation_summary(self) -> Dict[str, Any]:
        """Generate citation summary statistics."""
        # Count analytical citations only
        claim_source_counts = {}
        for citation in self.citations:
            if citation.citation_type == 'claim':
                claim_source_counts[citation.source_type] = claim_source_counts.get(citation.source_type, 0) + 1

        # Count data sources
        data_source_counts = {}
        total_metrics = 0
        for data_source in self.data_sources:
            data_source_counts[data_source.source_type] = data_source_counts.get(data_source.source_type, 0) + 1
            total_metrics += data_source.metrics_count

        return {
            "analytical_citations": len([c for c in self.citations if c.citation_type == 'claim']),
            "data_sources": len(self.data_sources),
            "total_metrics_tracked": total_metrics,
            "claim_source_breakdown": claim_source_counts,
            "data_source_breakdown": data_source_counts,
            "unique_claim_sources": len(set(c.source_url for c in self.citations if c.citation_type == 'claim')),
            "estimated_claims": len([c for c in self.citations if c.source_type == 'estimated' and c.citation_type == 'claim'])
        }

class CitationTracker:
    """
    Comprehensive citation tracking system implementing RAG methodology.
    """
    
    def __init__(self):
        self.collections: Dict[str, CitationCollection] = {}
        self.current_ticker: Optional[str] = None
        self.session_start: str = datetime.now().isoformat()
        
    def start_analysis(self, ticker: str) -> None:
        """Start citation tracking for a new ticker analysis."""
        self.current_ticker = ticker
        if ticker not in self.collections:
            self.collections[ticker] = CitationCollection(ticker=ticker)
        logger.info(f"📋 Started citation tracking for {ticker}")
    
    def track_yahoo_finance_data(self, ticker: str, metrics_count: int, api_endpoint: str) -> str:
        """Track Yahoo Finance API data source without individual citations."""
        data_source = DataSource(
            source_url=f"Yahoo Finance API: {api_endpoint}",
            source_type="yahoo_finance",
            description=f"Financial metrics from Yahoo Finance API",
            retrieved_at=datetime.now().isoformat(),
            confidence=1.0,
            metrics_count=metrics_count
        )

        return self.collections[ticker].add_data_source(data_source)

    def track_analytical_claim(self, ticker: str, claim: str, source_url: str,
                              source_type: str, section: str, confidence: float = 0.9) -> str:
        """Track analytical claims that require citations."""
        citation = Citation(
            source_url=source_url,
            source_type=source_type,
            section=section,
            passage=claim,
            retrieved_at=datetime.now().isoformat(),
            confidence=confidence,
            citation_type='claim'
        )

        return self.collections[ticker].add_citation(citation)

    def track_web_scraped_data_source(self, ticker: str, source_url: str, source_type: str,
                                     description: str, metrics_count: int = 1) -> str:
        """Track web-scraped data source without individual citations."""
        data_source = DataSource(
            source_url=source_url,
            source_type=source_type,
            description=description,
            retrieved_at=datetime.now().isoformat(),
            confidence=0.9,  # Slightly lower confidence for web-scraped data
            metrics_count=metrics_count
        )

        return self.collections[ticker].add_data_source(data_source)
    
    def track_estimated_data_source(self, ticker: str, methodology: str,
                                   base_sources: List[str], confidence: float,
                                   metrics_count: int = 1) -> str:
        """Track AI-estimated data source with transparent attribution."""
        data_source = DataSource(
            source_url=f"AI Estimation: {methodology}",
            source_type="estimated",
            description=f"Estimated metrics using {methodology} based on {', '.join(base_sources)}",
            retrieved_at=datetime.now().isoformat(),
            confidence=confidence,
            metrics_count=metrics_count
        )

        return self.collections[ticker].add_data_source(data_source)
    
    def validate_citation_coverage(self, ticker: str, content: str) -> Dict[str, Any]:
        """Validate that content has adequate citation coverage."""
        if ticker not in self.collections:
            return {"valid": False, "error": "No citations tracked for ticker"}
        
        collection = self.collections[ticker]
        
        # Count paragraphs and citations in content
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        citation_count = content.count('[Source:')
        
        # Calculate citation density
        citation_density = citation_count / max(1, len(paragraphs))
        
        # Validation rules
        min_citation_density = 0.5  # At least one citation per 2 paragraphs
        has_adequate_coverage = citation_density >= min_citation_density
        
        return {
            "valid": has_adequate_coverage,
            "citation_density": citation_density,
            "total_citations": citation_count,
            "total_paragraphs": len(paragraphs),
            "min_required_density": min_citation_density,
            "recommendations": self._generate_citation_recommendations(collection, citation_density)
        }
    
    def generate_sources_section(self, ticker: str) -> str:
        """Generate comprehensive sources and references section."""
        if ticker not in self.collections:
            return "No sources tracked for this analysis."
        
        collection = self.collections[ticker]
        summary = collection.get_citation_summary()
        
        sources_html = f"""
        <div class="section">
            <h2>📚 Sources and References</h2>
            
            <!-- Citation Summary -->
            <div class="alert alert-info">
                <h4>📊 Citation Quality Metrics</h4>
                <div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                    <div class="metric-card">
                        <div class="metric-label">Total Citations</div>
                        <div class="metric-value">{summary['total_citations']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Unique Sources</div>
                        <div class="metric-value">{summary['unique_sources']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Estimated Data Points</div>
                        <div class="metric-value">{summary['estimated_data_points']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Citation Density</div>
                        <div class="metric-value">{summary['citation_density']:.2f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Source Breakdown -->
            <h3>🔗 Data Sources by Type</h3>
        """
        
        # Group citations by source type
        for source_type, count in summary['source_breakdown'].items():
            citations = collection.get_citations_by_source(source_type)
            sources_html += f"""
            <h4>{source_type.replace('_', ' ').title()} ({count} citations)</h4>
            <ul>"""
            
            for citation in citations[:10]:  # Limit to first 10 per source type
                sources_html += f"""
                <li>
                    <strong>{citation.metric_name or 'General Data'}:</strong> 
                    <a href="{citation.source_url}" target="_blank">{citation.source_url}</a>
                    <br><small>Section: {citation.section} | Retrieved: {citation.retrieved_at}</small>
                </li>"""
            
            if len(citations) > 10:
                sources_html += f"<li><em>... and {len(citations) - 10} more citations</em></li>"
            
            sources_html += "</ul>"
        
        sources_html += """
            <!-- Data Quality Disclaimer -->
            <div class="alert alert-warning">
                <h4>⚠️ Data Quality and Citation Disclaimer</h4>
                <p><strong>Source Reliability:</strong> Yahoo Finance API data has highest reliability (confidence: 100%), 
                web-scraped data has high reliability (confidence: 90%), and estimated data includes confidence scores.</p>
                <p><strong>Recency:</strong> All data sources include retrieval timestamps. Market data may change rapidly.</p>
                <p><strong>Verification:</strong> Users should verify critical data points by accessing original sources directly.</p>
                <p><strong>Regulatory Compliance:</strong> This analysis includes comprehensive source attribution for audit and compliance purposes.</p>
            </div>
        </div>"""
        
        return sources_html
    
    def _generate_citation_recommendations(self, collection: CitationCollection, 
                                         citation_density: float) -> List[str]:
        """Generate recommendations for improving citation coverage."""
        recommendations = []
        
        if citation_density < 0.3:
            recommendations.append("Increase citation density - add more source attributions")
        
        if len(collection.get_citations_by_source('estimated')) > len(collection.citations) * 0.5:
            recommendations.append("High proportion of estimated data - seek more direct sources")
        
        if len(collection.get_citations_by_source('yahoo_finance')) == 0:
            recommendations.append("Missing Yahoo Finance data - add primary financial metrics")
        
        return recommendations
    
    def get_citation_for_metric(self, ticker: str, metric_name: str) -> Optional[Citation]:
        """Get citation for a specific metric."""
        if ticker not in self.collections:
            return None
        
        for citation in self.collections[ticker].citations:
            if citation.metric_name == metric_name:
                return citation
        
        return None
    
    def export_citations(self, ticker: str) -> Dict[str, Any]:
        """Export citations and data sources for a ticker in structured format."""
        if ticker not in self.collections:
            return {}

        collection = self.collections[ticker]

        return {
            "ticker": ticker,
            "analysis_session": self.session_start,
            "citation_summary": collection.get_citation_summary(),
            "analytical_citations": [
                {
                    "source_url": c.source_url,
                    "source_type": c.source_type,
                    "section": c.section,
                    "passage": c.passage,
                    "retrieved_at": c.retrieved_at,
                    "confidence": c.confidence,
                    "citation_type": c.citation_type
                }
                for c in collection.citations if c.citation_type == 'claim'
            ],
            "data_sources": [
                {
                    "source_url": ds.source_url,
                    "source_type": ds.source_type,
                    "description": ds.description,
                    "retrieved_at": ds.retrieved_at,
                    "confidence": ds.confidence,
                    "metrics_count": ds.metrics_count
                }
                for ds in collection.data_sources
            ],
            "retrieval_log": collection.retrieval_log
        }
