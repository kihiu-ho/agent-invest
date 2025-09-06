"""
HKEX Document Agent for AutoGen Financial Metrics System

This module provides a specialized AutoGen agent that integrates HKEX annual report
data from Weaviate vector database with fallback document processing capabilities.
"""

import asyncio
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Create mock classes for when AutoGen is not available
    class AssistantAgent:
        def __init__(self, *args, **kwargs):
            pass

    class RoundRobinGroupChat:
        def __init__(self, *args, **kwargs):
            pass

    class OpenAIChatCompletionClient:
        def __init__(self, *args, **kwargs):
            pass

from dotenv import load_dotenv

try:
    from weaviate_client import WeaviateClient, WeaviateConnectionError, WeaviateQueryError
    WEAVIATE_CLIENT_AVAILABLE = True
except ImportError:
    WEAVIATE_CLIENT_AVAILABLE = False
    # Create mock classes for when Weaviate client is not available
    class WeaviateClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def check_document_availability(self, ticker):
            return {"available": False, "error": "Weaviate client not available"}

        async def search_documents(self, *args, **kwargs):
            return []

    class WeaviateConnectionError(Exception):
        pass

    class WeaviateQueryError(Exception):
        pass

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class HKEXDocumentAgent:
    """
    Specialized AutoGen agent for HKEX annual report integration.
    
    Features:
    - Semantic search of HKEX annual reports in Weaviate vector database
    - Extraction of specific document sections (executive summary, pros/cons, etc.)
    - Fallback document processing when data is not available
    - Integration with existing AutoGen multi-agent framework
    - Structured data formatting for financial analysis reports
    """
    
    def __init__(self):
        """Initialize HKEX Document Agent with Weaviate integration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Weaviate client
        if WEAVIATE_CLIENT_AVAILABLE:
            try:
                self.weaviate_client = WeaviateClient()
                self.logger.info("✅ HKEX Document Agent initialized with Weaviate integration")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Weaviate client: {e}")
                self.weaviate_client = None
        else:
            self.logger.warning("⚠️ Weaviate client not available - using mock implementation")
            self.weaviate_client = WeaviateClient()  # Mock client
        
        # Initialize AutoGen agents
        self._initialize_autogen_agents()
        
        # Document section types to extract
        self.document_sections = [
            "executive_summary",
            "financial_highlights", 
            "pros_and_cons",
            "risk_factors",
            "business_overview",
            "market_analysis",
            "competitive_advantages",
            "future_outlook"
        ]
    
    def _initialize_autogen_agents(self):
        """Initialize AutoGen agents for document processing."""
        if not AUTOGEN_AVAILABLE:
            self.logger.warning("⚠️ AutoGen not available - using basic document processing")
            self.document_analyzer = None
            self.content_formatter = None
            return

        try:
            # Document Analyzer Agent
            self.document_analyzer = AssistantAgent(
                name="HKEX_Document_Analyzer",
                model_client=OpenAIChatCompletionClient(
                    model=os.getenv("OPENAI_MODEL", "gpt-4"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    api_key=os.getenv("OPENAI_API_KEY")
                ),
                description="Analyzes HKEX annual report content and extracts key insights",
                system_message="""You are a specialized financial document analyst focused on Hong Kong Exchange (HKEX) annual reports.
                Your role is to:
                1. Extract and summarize key financial information from annual reports
                2. Identify pros and cons of investment opportunities
                3. Highlight risk factors and competitive advantages
                4. Provide structured analysis suitable for investment decision-making

                Always provide clear, concise, and actionable insights with proper source attribution."""
            )

            # Content Formatter Agent
            self.content_formatter = AssistantAgent(
                name="HKEX_Content_Formatter",
                model_client=OpenAIChatCompletionClient(
                    model=os.getenv("OPENAI_MODEL", "gpt-4"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    api_key=os.getenv("OPENAI_API_KEY")
                ),
                description="Formats HKEX document content for integration into financial reports",
                system_message="""You are a content formatting specialist for financial reports.
                Your role is to:
                1. Format extracted document content into structured, readable sections
                2. Ensure proper citation and source attribution
                3. Maintain consistency with existing report formatting
                4. Create HTML-compatible content for web reports

                Always maintain professional formatting and clear section organization."""
            )

            self.logger.info("✅ AutoGen agents initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AutoGen agents: {e}")
            self.document_analyzer = None
            self.content_formatter = None
    
    async def analyze_ticker_documents(
        self, 
        ticker: str, 
        analysis_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze HKEX annual report documents for a given ticker.
        
        Args:
            ticker: Hong Kong stock ticker (e.g., '0005.HK')
            analysis_focus: Optional focus area for analysis
            
        Returns:
            Dictionary containing document analysis results
        """
        self.logger.info(f"📊 Starting HKEX document analysis for {ticker}")
        
        try:
            # Step 1: Check document availability in Weaviate
            availability = await self._check_document_availability(ticker)
            
            if availability.get("available"):
                # Step 2: Retrieve documents from Weaviate
                documents = await self._retrieve_documents(ticker, analysis_focus)
                
                if documents:
                    # Step 3: Process documents with AutoGen agents
                    analysis_result = await self._process_documents_with_agents(
                        ticker, documents, analysis_focus
                    )
                    
                    return {
                        "success": True,
                        "ticker": ticker,
                        "source": "weaviate_vector_database",
                        "documents_found": len(documents),
                        "analysis": analysis_result,
                        "availability": availability
                    }
                else:
                    self.logger.warning(f"⚠️ No relevant documents found for {ticker}")
                    return await self._handle_no_documents(ticker, analysis_focus)
            else:
                # Step 4: Fallback to document processing
                self.logger.info(f"📥 No documents in vector database for {ticker}, using fallback")
                return await self._handle_fallback_processing(ticker, analysis_focus)
                
        except Exception as e:
            self.logger.error(f"❌ Error analyzing documents for {ticker}: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "error": str(e),
                "fallback_available": True
            }
    
    async def _check_document_availability(self, ticker: str) -> Dict[str, Any]:
        """Check if documents are available in Weaviate for the ticker."""
        if not self.weaviate_client:
            return {"available": False, "error": "Weaviate client not available"}
        
        try:
            async with self.weaviate_client as client:
                return await client.check_document_availability(ticker)
        except Exception as e:
            self.logger.error(f"❌ Error checking document availability: {e}")
            return {"available": False, "error": str(e)}
    
    async def _retrieve_documents(
        self, 
        ticker: str, 
        analysis_focus: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from Weaviate."""
        if not self.weaviate_client:
            return []
        
        try:
            async with self.weaviate_client as client:
                all_documents = []
                
                # Search for each document section type
                for section_type in self.document_sections:
                    query = analysis_focus if analysis_focus else section_type
                    
                    documents = await client.search_documents(
                        ticker=ticker,
                        query=query,
                        content_types=[section_type],
                        limit=3
                    )
                    
                    all_documents.extend(documents)
                
                # Remove duplicates and sort by similarity score (handle None values from fallback)
                unique_documents = {}
                for doc in all_documents:
                    doc_id = f"{doc.get('content_type')}_{doc.get('section_title', '')}"
                    current_score = doc.get('similarity_score') or 0.0
                    existing_score = unique_documents.get(doc_id, {}).get('similarity_score') or 0.0
                    if doc_id not in unique_documents or current_score > existing_score:
                        unique_documents[doc_id] = doc

                result = list(unique_documents.values())
                result.sort(key=lambda x: x.get('similarity_score') or 0.0, reverse=True)
                
                self.logger.info(f"📄 Retrieved {len(result)} unique documents for {ticker}")
                return result
                
        except Exception as e:
            self.logger.error(f"❌ Error retrieving documents: {e}")
            return []
    
    async def _process_documents_with_agents(
        self, 
        ticker: str, 
        documents: List[Dict[str, Any]], 
        analysis_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process documents using AutoGen agents."""
        if not self.document_analyzer or not self.content_formatter:
            self.logger.warning("⚠️ AutoGen agents not available, using basic processing")
            return self._basic_document_processing(documents)
        
        try:
            # Prepare document content for analysis
            document_content = self._prepare_document_content(documents)
            
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(ticker, document_content, analysis_focus)
            
            # Process with document analyzer (simplified for now)
            analysis_result = {
                "executive_summary": self._extract_section_content(documents, "executive_summary"),
                "financial_highlights": self._extract_section_content(documents, "financial_highlights"),
                "pros_and_cons": self._extract_section_content(documents, "pros_and_cons"),
                "risk_factors": self._extract_section_content(documents, "risk_factors"),
                "business_overview": self._extract_section_content(documents, "business_overview"),
                "source_documents": len(documents),
                "confidence_score": self._calculate_confidence_score(documents),
                "last_updated": max([doc.get('last_updated', '') for doc in documents]) if documents else None
            }
            
            self.logger.info(f"✅ Processed {len(documents)} documents for {ticker}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Error processing documents with agents: {e}")
            return self._basic_document_processing(documents)
    
    def _extract_section_content(self, documents: List[Dict[str, Any]], section_type: str) -> str:
        """Extract content for a specific section type."""
        section_docs = [doc for doc in documents if doc.get('content_type') == section_type]
        
        if not section_docs:
            return f"No {section_type.replace('_', ' ')} information available in annual reports."
        
        # Combine content from multiple documents
        combined_content = []
        for doc in section_docs[:3]:  # Limit to top 3 most relevant
            content = doc.get('content', '').strip()
            if content:
                source_info = f"Source: {doc.get('document_title', 'HKEX Annual Report')}"
                combined_content.append(f"{content}\n\n{source_info}")
        
        return "\n\n---\n\n".join(combined_content) if combined_content else f"Limited {section_type.replace('_', ' ')} information available."
    
    def _calculate_confidence_score(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score based on document quality."""
        if not documents:
            return 0.0
        
        scores = [doc.get('similarity_score') or 0.0 for doc in documents]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _prepare_document_content(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare document content for agent analysis."""
        content_parts = []
        
        for doc in documents[:10]:  # Limit to top 10 documents
            section = f"Section: {doc.get('section_title', 'Unknown')}\n"
            section += f"Type: {doc.get('content_type', 'Unknown')}\n"
            section += f"Content: {doc.get('content', '')[:1000]}...\n"  # Limit content length
            section += f"Confidence: {(doc.get('similarity_score') or 0.0):.2f}\n"
            content_parts.append(section)
        
        return "\n---\n".join(content_parts)
    
    def _create_analysis_prompt(self, ticker: str, content: str, focus: Optional[str] = None) -> str:
        """Create analysis prompt for AutoGen agents."""
        base_prompt = f"""
        Analyze the following HKEX annual report content for {ticker}:
        
        {content}
        
        Please provide:
        1. Executive Summary (key points in 2-3 sentences)
        2. Financial Highlights (key metrics and performance indicators)
        3. Investment Pros and Cons (balanced analysis)
        4. Risk Factors (main risks to consider)
        5. Business Overview (core business and market position)
        """
        
        if focus:
            base_prompt += f"\n\nSpecial focus on: {focus}"
        
        return base_prompt
    
    def _basic_document_processing(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Basic document processing when AutoGen agents are not available."""
        return {
            "executive_summary": self._extract_section_content(documents, "executive_summary"),
            "financial_highlights": self._extract_section_content(documents, "financial_highlights"),
            "pros_and_cons": self._extract_section_content(documents, "pros_and_cons"),
            "risk_factors": self._extract_section_content(documents, "risk_factors"),
            "business_overview": self._extract_section_content(documents, "business_overview"),
            "source_documents": len(documents),
            "confidence_score": self._calculate_confidence_score(documents),
            "processing_method": "basic_extraction"
        }
    
    async def _handle_no_documents(self, ticker: str, analysis_focus: Optional[str] = None) -> Dict[str, Any]:
        """Handle case when no documents are found in vector database."""
        return {
            "success": False,
            "ticker": ticker,
            "message": "No documents found in vector database",
            "fallback_recommended": True,
            "analysis": {
                "executive_summary": "No annual report data available in vector database.",
                "financial_highlights": "Financial highlights not available - consider manual document retrieval.",
                "pros_and_cons": "Investment analysis not available from annual reports.",
                "risk_factors": "Risk assessment not available from annual reports.",
                "business_overview": "Business overview not available from annual reports."
            }
        }
    
    async def _handle_fallback_processing(self, ticker: str, analysis_focus: Optional[str] = None) -> Dict[str, Any]:
        """Handle fallback document processing using HKEX downloader."""
        self.logger.info(f"🔄 Initiating enhanced fallback document processing for {ticker}")

        try:
            # Prefer root-level Crawl4AI HKEX downloader if available
            downloader = None
            try:
                import importlib.util, sys
                root_path = str(Path(__file__).resolve().parents[1])
                module_path = os.path.join(root_path, 'crawl4ai_hkex_downloader.py')
                # Ensure repository root is on sys.path so crawl4ai_hkex_downloader dependencies (e.g., search_agents) can be imported
                if root_path not in sys.path:
                    sys.path.insert(0, root_path)
                if os.path.exists(module_path):
                    # Cache module and agent instance to avoid duplicate initialization
                    if 'crawl4ai_hkex_downloader' in sys.modules:
                        crawl4ai_mod = sys.modules['crawl4ai_hkex_downloader']
                    else:
                        spec = importlib.util.spec_from_file_location('crawl4ai_hkex_downloader', module_path)
                        crawl4ai_mod = importlib.util.module_from_spec(spec)
                        sys.modules['crawl4ai_hkex_downloader'] = crawl4ai_mod
                        spec.loader.exec_module(crawl4ai_mod)
                    if hasattr(crawl4ai_mod, 'HKEXDocumentDownloadAgent'):
                        self.logger.info("Using Crawl4AI HKEXDocumentDownloadAgent for fallback processing")
                        # Cache a singleton instance on the class to prevent duplicate initializations
                        if not hasattr(self.__class__, '_crawl4ai_agent'):
                            self.__class__._crawl4ai_agent = crawl4ai_mod.HKEXDocumentDownloadAgent()
                        agent = self.__class__._crawl4ai_agent
                        # Wrap agent usage with a simple interface; define as method to avoid binding errors
                        class Shim:
                            async def process_and_store_document(self, ticker: str):
                                stock_code = ticker.replace('.HK', '')
                                result = await agent.download_annual_reports(stock_code, max_reports=1, force_refresh=False)
                                return {
                                    "success": result.get("success", False),
                                    "downloads": result.get("downloads", []),
                                    "source": "crawl4ai_agent"
                                }
                        downloader = Shim()
            except Exception as e:
                self.logger.warning(f"Crawl4AI HKEX downloader import failed, falling back to local downloader: {e}")

            if downloader is None:
                from hkex_downloader import HKEXDownloader
                downloader = HKEXDownloader()

            # Download and process the document
            self.logger.info(f"📥 Downloading HKEX annual report for {ticker}")

            # If using Crawl4AI agent shim, perform download then extract using local extractor
            if hasattr(downloader, 'process_and_store_document') and downloader.__class__.__name__ == 'Shim':
                # Use Crawl4AI agent to download latest report
                crawl_result = await downloader.process_and_store_document(ticker)
                if crawl_result.get("success") and crawl_result.get("downloads"):
                    # Find first successful download
                    first = next((d for d in crawl_result["downloads"] if d.get("success")), None)
                    if first and first.get("filepath"):
                        file_path = first["filepath"]
                        # Use local extractor for content and sections
                        from hkex_downloader import HKEXDownloader as LocalExtractor
                        extractor = LocalExtractor()
                        content_extraction = await extractor._extract_document_content(file_path, ticker)
                        if content_extraction.get("success"):
                            sections = content_extraction.get("content", {})
                            storage_result = None
                            if self.weaviate_client:
                                try:
                                    async with self.weaviate_client as client:
                                        storage_result = await client.store_document_sections(ticker, sections)
                                        if storage_result.get("success"):
                                            self.logger.info(f"✅ Stored {storage_result.get('sections_stored', 0)} sections in Weaviate")
                                except Exception as e:
                                    self.logger.warning(f"⚠️ Failed to store in Weaviate: {e}")
                            analysis_result = await self._process_documents_with_agents(
                                ticker,
                                [{"content": v.get("content", v) if isinstance(v, dict) else v, "content_type": k} for k, v in sections.items()],
                                analysis_focus
                            )
                            return {
                                "success": True,
                                "ticker": ticker,
                                "source": "crawl4ai_hkex_downloader",
                                "download_result": crawl_result,
                                "content_extraction": content_extraction,
                                "storage_result": storage_result,
                                "analysis": analysis_result,
                                "documents_processed": len(sections),
                                "processing_method": "enhanced_fallback_crawl4ai"
                            }
                # If Crawl4AI path fails, fall back to local
                download_result = await HKEXDownloader().process_and_store_document(ticker)
            else:
                download_result = await downloader.process_and_store_document(ticker)

            if download_result.get("success"):
                # Extract the processed content
                content_extraction = download_result.get("content_extraction", {})

                if content_extraction.get("success"):
                    sections = content_extraction.get("content", {})

                    # Store in Weaviate if available
                    storage_result = None
                    if self.weaviate_client:
                        try:
                            async with self.weaviate_client as client:
                                storage_result = await client.store_document_sections(ticker, sections)
                                if storage_result.get("success"):
                                    self.logger.info(f"✅ Stored {storage_result.get('sections_stored', 0)} sections in Weaviate")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to store in Weaviate: {e}")

                    # Process with AutoGen agents if available
                    analysis_result = await self._process_documents_with_agents(
                        ticker,
                        [{"content": content, "content_type": section_type} for section_type, content in sections.items()],
                        analysis_focus
                    )

                    return {
                        "success": True,
                        "ticker": ticker,
                        "source": "hkex_downloader_processing",
                        "download_result": download_result,
                        "storage_result": storage_result,
                        "analysis": analysis_result,
                        "documents_processed": len(sections),
                        "processing_method": "enhanced_fallback"
                    }
                else:
                    return {
                        "success": False,
                        "ticker": ticker,
                        "source": "hkex_downloader_processing",
                        "error": "Document content extraction failed",
                        "download_result": download_result
                    }
            else:
                return {
                    "success": False,
                    "ticker": ticker,
                    "source": "hkex_downloader_processing",
                    "error": f"Document download failed: {download_result.get('error', 'Unknown error')}",
                    "download_result": download_result
                }

        except ImportError:
            self.logger.error("❌ HKEX Downloader not available")
            return {
                "success": False,
                "ticker": ticker,
                "source": "fallback_processing",
                "error": "HKEX Downloader not available",
                "message": "Enhanced document processing requires HKEX downloader module"
            }
        except Exception as e:
            self.logger.error(f"❌ Error in fallback document processing: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "source": "fallback_processing",
                "error": str(e)
            }
