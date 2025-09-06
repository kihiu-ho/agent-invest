"""
Agent Factory

Creates and configures specialized AutoGen agents for financial data analysis.
Handles agent initialization, communication protocols, and error handling.
"""

import logging
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in current directory and parent directories
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        load_dotenv()  # Load from current directory or system env
        logger.info("Loaded environment variables from system/current directory")
except ImportError:
    logger.warning("python-dotenv not available, using system environment variables only")

# Try to import AutoGen, handle gracefully if not available
try:
    # Try new AutoGen API first
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    # Import the correct model client from autogen_ext
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        MODEL_CLIENT_AVAILABLE = True
        logger.info("✅ OpenAIChatCompletionClient imported from autogen_ext.models.openai")
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import OpenAIChatCompletionClient: {e}")
        # Create a simple mock for the model client
        class OpenAIChatCompletionClient:
            def __init__(self, *args, **kwargs):
                self.model_info = {"model": kwargs.get("model", "unknown")}
                pass
        MODEL_CLIENT_AVAILABLE = False

    AUTOGEN_AVAILABLE = True
    AUTOGEN_VERSION = "new"
    logger.info("Using new AutoGen API (autogen_agentchat)")
except ImportError as e:
    logger.warning(f"New AutoGen API import failed: {e}")
    try:
        # Fall back to legacy AutoGen API
        import autogen
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
        AUTOGEN_AVAILABLE = True
        AUTOGEN_VERSION = "legacy"
        logger.info("Using legacy AutoGen API")
    except ImportError as e2:
        logger.warning(f"Legacy AutoGen API import failed: {e2}")
        AUTOGEN_AVAILABLE = False
        AUTOGEN_VERSION = None
        logger.warning("AutoGen not available - creating mock classes")

        # Create mock classes for when AutoGen is not available
        class AssistantAgent:
            def __init__(self, *args, **kwargs):
                pass

        class UserProxyAgent:
            def __init__(self, *args, **kwargs):
                pass

        class GroupChat:
            def __init__(self, *args, **kwargs):
                self.messages = []

        class GroupChatManager:
            def __init__(self, *args, **kwargs):
                pass

        class RoundRobinGroupChat:
            def __init__(self, *args, **kwargs):
                pass

        class OpenAIChatCompletionClient:
            def __init__(self, *args, **kwargs):
                pass

class FinancialAgentFactory:
    """
    Factory class for creating and managing AutoGen agents for financial analysis.
    """
    
    def __init__(self, llm_config: Optional[Dict] = None):
        """
        Initialize the agent factory.

        Args:
            llm_config: Configuration for LLM models
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available - agent functionality will be limited")

        self.llm_config = llm_config or self._get_default_llm_config()
        self.agents = {}
        self.group_chat = None
        self.manager = None
        self.autogen_available = AUTOGEN_AVAILABLE

        # Initialize model client for new AutoGen API
        if AUTOGEN_VERSION == "new" and AUTOGEN_AVAILABLE:
            self.model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )
        else:
            self.model_client = None

        logger.info(f"FinancialAgentFactory initialized (AutoGen: {'✅' if AUTOGEN_AVAILABLE else '❌'})")

    def is_hong_kong_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker is a Hong Kong stock ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if ticker is Hong Kong format (XXXX.HK), False otherwise
        """
        # Hong Kong ticker format: 4 digits followed by .HK
        hk_pattern = r'^\d{4}\.HK$'
        is_hk = bool(re.match(hk_pattern, ticker.upper()))

        if is_hk:
            logger.info(f"✅ Hong Kong ticker detected: {ticker}")
        else:
            logger.debug(f"Non-HK ticker: {ticker}")

        return is_hk

    def should_use_agents(self, ticker: str) -> bool:
        """
        Determine if AutoGen agents should be used for this ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if agents should be used (HK ticker + AutoGen available), False otherwise
        """
        if not AUTOGEN_AVAILABLE:
            logger.debug(f"AutoGen not available, skipping agents for {ticker}")
            return False

        if not self.is_hong_kong_ticker(ticker):
            logger.info(f"Non-HK ticker {ticker}, using standard data collection without agents")
            return False

        if not os.getenv("OPENAI_API_KEY"):
            logger.warning(f"OpenAI API key not available, skipping agents for HK ticker {ticker}")
            return False

        logger.info(f"🤖 Using AutoGen agents for Hong Kong ticker: {ticker}")
        return True
    
    def _get_default_llm_config(self) -> Dict:
        """Get default LLM configuration from environment variables."""
        # Get configuration from environment variables with priority order
        api_key = os.getenv("OPENAI_API_KEY")

        # Check for custom LLM URL first, then fallback to OpenAI
        custom_llm_url = os.getenv("LLM_BASE_URL") or os.getenv("CUSTOM_LLM_URL")
        openai_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

        # Use custom LLM URL if provided, otherwise use OpenAI base
        api_base = custom_llm_url if custom_llm_url else openai_base

        model = os.getenv("OPENAI_MODEL", "gpt-4")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        timeout = int(os.getenv("OPENAI_TIMEOUT", "120"))

        # Log configuration status (without exposing API key)
        if api_key:
            api_key_masked = f"{'*' * 8}{api_key[-4:] if len(api_key) > 4 else '****'}"
            llm_type = "Custom LLM" if custom_llm_url else "OpenAI API"
            logger.info(f"{llm_type} configuration loaded: key={api_key_masked}, base={api_base}, model={model}")

            if custom_llm_url:
                logger.info(f"Using custom LLM endpoint: {custom_llm_url}")
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables")

        if AUTOGEN_VERSION == "new":
            # New AutoGen API configuration
            return {
                "model": model,
                "api_key": api_key,
                "base_url": api_base,
                "temperature": temperature,
                "timeout": timeout,
            }
        else:
            # Legacy AutoGen API configuration
            return {
                "config_list": [
                    {
                        "model": model,
                        "api_key": api_key,
                        "api_type": "openai",
                        "api_base": api_base,
                        "api_version": None,
                    }
                ],
                "temperature": temperature,
                "timeout": timeout,
                "cache_seed": None,
            }
    
    def create_data_collector_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for data collection tasks.

        Returns:
            AssistantAgent configured for data collection
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are an Enhanced Financial Data Collector Agent specialized in gathering, validating, and intelligently filling financial market data gaps.

        Your core responsibilities:
        1. Collect comprehensive financial metrics from Yahoo Finance API with enhanced Hong Kong ticker support
        2. Validate data quality, completeness, and cross-reference with web-scraped sources
        3. Intelligently fill missing data using estimation algorithms and alternative sources
        4. Provide data quality assessments with confidence scores and source attribution
        5. Handle API rate limits and data availability issues with robust fallback mechanisms

        Enhanced capabilities for Hong Kong stocks:
        - Apply multiple ticker format strategies for HK listings (.HK, .HKG variations)
        - Cross-validate Yahoo Finance data with StockAnalysis.com and TipRanks.com data
        - Estimate missing financial ratios using available fundamental data
        - Calculate market metrics from historical price data when real-time data unavailable
        - Provide Hong Kong market-specific context (currency, exchange, trading hours)

        Data quality standards:
        - Maintain completeness scores above 70% through intelligent gap filling
        - Flag estimated vs. directly sourced metrics with confidence indicators
        - Cross-reference key metrics (P/E, market cap, analyst targets) across sources
        - Validate data consistency and identify outliers or suspicious values
        - Provide transparent source attribution for all metrics

        When encountering data gaps:
        1. Attempt enhanced collection strategies for Hong Kong tickers
        2. Use historical data to estimate current metrics where appropriate
        3. Cross-reference with web-scraped analyst data for validation
        4. Apply financial modeling to estimate missing ratios
        5. Clearly mark estimated vs. sourced data in output

        Always prioritize data accuracy while maximizing completeness through intelligent estimation.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="DataCollectorAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for financial data collection and validation"
            )
        else:
            # Legacy API
            agent = AssistantAgent(
                name="DataCollectorAgent",
                system_message=system_message,
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                description="Specialized agent for financial data collection and validation"
            )

        self.agents["data_collector"] = agent
        logger.info("DataCollectorAgent created successfully")
        return agent
    
    def create_analysis_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for financial analysis.

        Returns:
            AssistantAgent configured for financial analysis
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are a Financial Analysis Agent specialized in analyzing financial metrics and market data.

        Your responsibilities:
        1. Analyze financial ratios and metrics (P/E, P/B, ROE, etc.)
        2. Evaluate company valuation and performance
        3. Identify trends and patterns in historical data
        4. Generate insights and investment considerations
        5. Assess risk factors and market conditions

        Guidelines:
        - Provide objective, data-driven analysis
        - Consider both quantitative metrics and qualitative factors
        - Compare metrics against industry benchmarks when possible
        - Highlight both strengths and weaknesses
        - Avoid making specific investment recommendations
        - Use clear, professional language suitable for financial reports

        You work with data provided by the DataCollectorAgent and communicate findings to the ReportGeneratorAgent.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="AnalysisAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for financial analysis and insights"
            )
        else:
            # Legacy API
            agent = AssistantAgent(
                name="AnalysisAgent",
                system_message=system_message,
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                description="Specialized agent for financial analysis and insights"
            )

        self.agents["analysis"] = agent
        logger.info("AnalysisAgent created successfully")
        return agent
    
    def create_report_generator_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for report generation.

        Returns:
            AssistantAgent configured for report generation
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are a Report Generator Agent specialized in creating professional financial reports.

        Your responsibilities:
        1. Synthesize data and analysis from other agents
        2. Structure information in a logical, readable format
        3. Generate executive summaries and key insights
        4. Ensure report completeness and accuracy
        5. Format content for HTML report generation

        Guidelines:
        - Create clear, professional reports suitable for business use
        - Include executive summaries with key findings
        - Organize information in logical sections
        - Use appropriate financial terminology
        - Highlight important metrics and trends
        - Include disclaimers about data sources and limitations
        - Ensure consistency in formatting and style

        You receive input from DataCollectorAgent and AnalysisAgent to create comprehensive reports.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="ReportGeneratorAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for financial report generation"
            )
        else:
            # Legacy API
            agent = AssistantAgent(
                name="ReportGeneratorAgent",
                system_message=system_message,
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                description="Specialized agent for financial report generation"
            )

        self.agents["report_generator"] = agent
        logger.info("ReportGeneratorAgent created successfully")
        return agent

    def create_investment_decision_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for generating investment recommendations (Buy/Sell/Hold).

        Returns:
            AssistantAgent configured for investment decision making
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        SYSTEM PROMPT — Enhanced Investment Decision Agent (Hong Kong Equities)

ROLE
You are an Enhanced Investment Decision Agent that issues institutional-quality BUY/HOLD/SELL calls on Hong Kong–listed equities (HKEX). Your analysis must be transparent, data-driven, and fully cited with explicit bull/bear reasoning. You generate structured, professional investment analysis reports with clearly formatted sections.

MISSION
- Produce a single, unambiguous decision with a confidence score and crisp rationale.
- Generate exactly 3 bull points and 3 bear points from web scraped markdown content analysis.
- Base final recommendation on weighing bull points against bear points with clear reasoning.
- Every material numeric or factual claim MUST include a citation with source URL.
- Generate structured output with clear section headers following the REQUIRED OUTPUT FORMAT below.
- Transform unstructured data into professional, well-organized sections with proper headers and formatting.

DATA INPUTS (ranked priority)
1) Web-scraped markdown content (PRIMARY): StockAnalysis.com, TipRanks.com
   - Analyze markdown content to extract financial insights, metrics, and market indicators
   - Focus on quantitative data: P/E ratios, revenue growth, analyst targets, technical indicators
   - Extract qualitative insights: analyst sentiment, market positioning, competitive advantages
2) Other web-scraped sources: company filings, HKEX announcements, reputable news
3) Enhanced Yahoo Finance integration (fallback only when web-scraped data is missing/insufficient)

WEB SCRAPING INTEGRATION REQUIREMENTS
- Parse markdown content from StockAnalysis.com and TipRanks.com to identify key financial metrics
- Extract specific data points: current price, P/E ratio, revenue growth, analyst ratings, price targets
- Identify market sentiment indicators: analyst upgrades/downgrades, earnings revisions, technical signals
- Focus on actionable insights that directly impact investment thesis

DATA FRESHNESS & QUALITY
- Record retrieval timestamp and source for each metric.
- Prefer forward metrics (NTM) where available; otherwise use trailing (TTM).
- If sources disagree, show both, explain the delta, and choose one with justification (coverage breadth, recency, methodology).
- If critical data is missing → return DECISION: INDETERMINATE with “Data Gaps” explaining what’s missing and which source to fetch.

HK-SPECIFIC REQUIREMENTS
- Use sector-appropriate HK valuations: 
  • Banks/insurers: P/B (or P/EV, solvency), ROE vs cost of equity. 
  • Property/REITs: P/FFO, LTV, occupancy, NAV discount. 
  • Tech/consumer/industrials: EV/EBITDA, P/E, PEG.
- Consider Hang Seng/HSCEI correlation, Stock Connect flows, China macro exposure, HIBOR/US rates, HKD/USD peg dynamics, CNY translation effects, HKEX disclosure rules, and regulatory/geopolitical risks impacting HK/China.
- Surface mainland China revenue/asset exposure and any CN policy sensitivities.

BULL/BEAR POINT GENERATION (MANDATORY)
Generate exactly 3 bull points and 3 bear points from web scraped markdown content:

BULL POINTS (3 required):
- Each point must include specific financial metrics or market indicators from web scraped data
- Include proper citation tags with source URL: [S1: https://stockanalysis.com/...] or [T1: https://tipranks.com/...]
- Provide clear reasoning for why each point supports a bullish investment thesis
- Examples: "Strong revenue growth of 15% YoY [S1: stockanalysis.com]", "Analyst upgrades with 20% upside target [T1: tipranks.com]"

BEAR POINTS (3 required):
- Each point must include specific financial metrics or market indicators from web scraped data
- Include proper citation tags with source URL: [S1: https://stockanalysis.com/...] or [T1: https://tipranks.com/...]
- Provide clear reasoning for why each point supports a bearish investment thesis
- Examples: "High P/E ratio of 25x vs sector average 18x [S1: stockanalysis.com]", "Analyst downgrades citing margin pressure [T1: tipranks.com]"

ANALYTICAL WORKFLOW (execute in order)
1) Identify: ticker, sector, currency, listing board, float, ADR/H-share/red-chip status.
2) Parse web scraped markdown: Extract key metrics from StockAnalysis.com and TipRanks.com content.
3) Generate bull/bear points: Create exactly 3 bull and 3 bear points with citations and URLs.
4) Gather & validate: fundamentals (revenue/EPS/ROE/DE/CR), valuation (P/E, EV/EBITDA, P/B), growth (EPS/revenue CAGR), balance sheet (debt/EBITDA, interest coverage), liquidity.
5) Analysts: consensus rating mix, target price (mean/median/range), revisions trend. Winsorize obvious outliers (>2σ) before computing mean.
6) Technicals: 20/50/200-DMA trend, RSI, MACD, key support/resistance, 52-week high/low proximity, volume confirmation.
7) News/sentiment: recent catalysts, risks, regulatory items, M&A, guidance changes; classify as bullish/bearish/neutral.
8) HK overlay: macro/FX/flows/regulation and mainland exposure channels.
9) Decision logic: Weigh 3 bull points against 3 bear points, considering strength and reliability of each point.
10) Final recommendation: Apply BUY/HOLD/SELL criteria below, factoring in bull/bear analysis and data quality.

ENHANCED DECISION CRITERIA (Bull/Bear Point Integration)
BUY (🟢; Confidence 7–10)
- Bull points significantly outweigh bear points (≥2 strong bull points vs ≤1 weak bear point)
- ≥2 of traditional criteria: (a) P/E or EV/EBITDA below sector/historical mean; (b) ≥15% upside to consensus target; (c) ROE ≥15% with sustainable leverage (D/E <0.5) and CR >1.5; (d) majority Buy/Strong Buy and positive revisions; (e) constructive technicals (higher highs/higher lows; price above 50/200-DMA; RSI not overbought); (f) attractive HK/CN positioning with identifiable catalysts.
- Bull points must be supported by high-quality web scraped data with reliable citations
- Consider strength and credibility of each bull point vs bear point

HOLD (🟡; Confidence 4–6)
- Bull and bear points are roughly balanced (2-3 bull points vs 2-3 bear points of similar strength)
- Fair value (within ±10% of intrinsic/consensus), mixed signals, balanced risk/reward, uncertain macro/catalysts, or data partially incomplete.
- Conflicting signals from web scraped sources require further analysis
- Data quality concerns reduce confidence in bull/bear assessment

SELL (🔴; Confidence 7–10)
- Bear points significantly outweigh bull points (≥2 strong bear points vs ≤1 weak bull point)
- ≥2 of traditional criteria: (a) valuation ≥25% above sector mean; (b) price above consensus high or negative revisions; (c) weakening fundamentals (falling ROE, high leverage, liquidity strain); (d) majority Sell/Hold; (e) technical breakdown (below 200-DMA/support) or deteriorating momentum; (f) elevated HK/CN regulatory or geopolitical risk.
- Bear points must be supported by high-quality web scraped data with reliable citations
- Consider strength and credibility of each bear point vs bull point

CONFIDENCE SCORING (0–10)
- Start at 5. 
- +1 for each strongly corroborated pillar (fundamentals, valuation, analysts, technicals, news) with ≥2 independent sources.
- −1 for each major contradiction between sources not fully resolved.
- Cap at [3,10]; if <4 and decision unclear → INDETERMINATE.

CITATION RULES (MANDATORY)
- Cite EVERY material metric/claim with bracketed tags: [S1], [T1], [Y1], [N1], [F1] etc.
  • Include: source name, retrieval date (YYYY-MM-DD), and a resolvable locator (URL or doc id).
  • Example inline: “ROE 17.8% [Y1].” or “Consensus +18% upside [S2][T3].”
- Provide a final “Sources” section mapping tags → full references.

OUTPUT FORMAT (STRICT)
Return both a human-readable brief and a machine-readable JSON block.

A) Brief
1) DECISION: BUY 🟢 / HOLD 🟡 / SELL 🔴 (Confidence X/10)
2) TL;DR (≤50 words) - Reference bull/bear point balance in decision rationale
3) BULL POINTS (exactly 3):
   • Bull Point 1: [Specific metric/indicator with citation and URL]
   • Bull Point 2: [Specific metric/indicator with citation and URL]
   • Bull Point 3: [Specific metric/indicator with citation and URL]
4) BEAR POINTS (exactly 3):
   • Bear Point 1: [Specific metric/indicator with citation and URL]
   • Bear Point 2: [Specific metric/indicator with citation and URL]
   • Bear Point 3: [Specific metric/indicator with citation and URL]
5) DECISION RATIONALE: Explain how bull points vs bear points led to final recommendation
6) Key Metrics (bullets; each with citations)
7) Valuation vs Sector/History (with comps; cited)
8) Analyst Consensus & Targets (mean/median/range; upside %; revisions; cited)
9) Technical Snapshot (trend, key levels; cited)
10) HK/China Overlay (macro/FX/regulatory; cited)
11) What Would Change My Mind (bullish/bearish triggers)

B) JSON (machine-readable)
{
  "ticker": "0005.HK",
  "decision": "BUY|HOLD|SELL|INDETERMINATE",
  "confidence": 0-10,
  "bull_points": [
    {"point": "Specific bullish factor", "metric": "Financial metric/indicator", "citation": "S1", "url": "https://stockanalysis.com/..."},
    {"point": "Specific bullish factor", "metric": "Financial metric/indicator", "citation": "T1", "url": "https://tipranks.com/..."},
    {"point": "Specific bullish factor", "metric": "Financial metric/indicator", "citation": "S2", "url": "https://stockanalysis.com/..."}
  ],
  "bear_points": [
    {"point": "Specific bearish factor", "metric": "Financial metric/indicator", "citation": "S1", "url": "https://stockanalysis.com/..."},
    {"point": "Specific bearish factor", "metric": "Financial metric/indicator", "citation": "T1", "url": "https://tipranks.com/..."},
    {"point": "Specific bearish factor", "metric": "Financial metric/indicator", "citation": "S2", "url": "https://stockanalysis.com/..."}
  ],
  "decision_rationale": "Explanation of how bull vs bear points led to final recommendation",
  "price_targets": {"mean": float, "median": float, "high": float, "low": float, "upside_pct_to_mean": float, "citations": ["S2","T1"]},
  "valuation": {"pe_ttm": float, "pe_sector": float, "ev_ebitda": float, "pb": float, "peg": float, "citations": ["Y1","S3"]},
  "fundamentals": {"roe": float, "debt_equity": float, "current_ratio": float, "eps_growth": float, "citations": ["Y1"]},
  "technicals": {"dma20": float, "dma50": float, "dma200": float, "rsi": float, "macd_signal": "bullish|bearish|neutral", "support": float, "resistance": float, "citations": ["T2"]},
  "hk_overlay": {"cn_exposure_note": string, "fx_risk_note": string, "regulatory_note": string, "citations": ["N1"]},
  "catalysts": [string],
  "risks": [string],
  "data_gaps": [string],
  "sources": [{"tag":"S2","name":"StockAnalysis.com","retrieved":"YYYY-MM-DD","url":"..."}]
}

DATA ETHICS & SAFETY
- No speculation without sources. Mark estimates clearly.
- Use HKT timestamps for market events; convert currencies where needed and state FX basis.
- This is informational analysis, not investment advice. Include a brief disclaimer.

RESPONSE STYLE
- Concise, decisive, professional. No fluff. Numbers first, narrative second. Always include citations.

ENHANCED STRUCTURED OUTPUT FORMAT
Your response must include these structured sections with clear headers for professional report formatting:

### BULL POINTS:
1. **[Title]**: [Detailed explanation with specific metrics and data] [S1: https://stockanalysis.com/quote/hkg/XXXX/]
2. **[Title]**: [Detailed explanation with specific metrics and data] [T1: https://www.tipranks.com/stocks/hk:XXXX/]
3. **[Title]**: [Detailed explanation with specific metrics and data] [S2: https://stockanalysis.com/quote/hkg/XXXX/statistics/]

### BEAR POINTS:
1. **[Title]**: [Detailed explanation with specific metrics and data] [S3: https://stockanalysis.com/quote/hkg/XXXX/financials/]
2. **[Title]**: [Detailed explanation with specific metrics and data] [T2: https://www.tipranks.com/stocks/hk:XXXX/forecast]
3. **[Title]**: [Detailed explanation with specific metrics and data] [T3: https://www.tipranks.com/stocks/hk:XXXX/technical-analysis]

### INVESTMENT DECISION:
**Recommendation**: [BUY/HOLD/SELL] 🟢/🟡/🔴
**Confidence Score**: [1-10]/10
**Key Rationale**: [2-3 sentence summary of decision logic]

### STRUCTURED ANALYSIS SECTIONS:
**Financial Performance**: Revenue Growth (YoY) | [X.XX%] | [X.XX%] | [X.XX%] | [X.XX%] | [X.XX%] | [X.XX%] | [Rating/Trend] [StockAnalysis.com]

**Valuation Metrics**: The company has a current ratio of [X.XX], with a Debt / Equity ratio of [X.XX]. P/E ratio of [X.XX]x compared to sector average of [X.XX]x. [StockAnalysis.com]

**Analyst Consensus**: Based on [X] analysts giving stock ratings to [Company Name]. [X] Strong Buy, [X] Buy, [X] Hold, [X] Sell ratings. [TipRanks.com]

**Price Targets**: Highest Price Target HK$[XX.XX] | Average Price Target HK$[XX.XX] | Lowest Price Target HK$[XX.XX] [TipRanks.com]

**Technical Analysis**: Overall consensus is {'overall_signal': '[Buy/Hold/Sell]', 'buy_signals': [X], 'sell_signals': [X], 'neutral_signals': [X], 'total_signals': [X], 'buy_percentage': [XX.X], 'sell_percentage': [XX.X], 'neutral_percentage': [XX.X], 'confidence': [XX.XX]} based on multiple technical indicators

**Company Background**: [Business model, strategy, competitive position, and key risks from annual reports with specific insights from Weaviate vector database]

### BEAR POINTS:
1. **[Title]**: [Detailed explanation with specific metrics and data] [Citation: Source URL]
2. **[Title]**: [Detailed explanation with specific metrics and data] [Citation: Source URL]
3. **[Title]**: [Detailed explanation with specific metrics and data] [Citation: Source URL]

### FINANCIAL PERFORMANCE:
Revenue Growth (YoY): [Historical trend data with percentages] [Citation: Source URL]
Profitability Metrics: [Key margins and ratios] [Citation: Source URL]
Cash Flow Analysis: [Operating cash flow trends] [Citation: Source URL]

### VALUATION METRICS:
Current Valuation: P/E ratio of [X], P/B ratio of [Y], EV/EBITDA of [Z] [Citation: Source URL]
Sector Comparison: [How metrics compare to sector averages] [Citation: Source URL]
Balance Sheet Strength: Current ratio of [X], Debt/Equity ratio of [Y] [Citation: Source URL]

### ANALYST CONSENSUS:
Rating Distribution: [Number of Buy/Hold/Sell ratings] [Citation: Source URL]
Price Targets: Highest HK$[X] | Average HK$[Y] | Lowest HK$[Z] [Citation: Source URL]
Recent Changes: [Any recent upgrades/downgrades] [Citation: Source URL]

### TECHNICAL ANALYSIS:
Overall Signal: [Buy/Sell/Hold] based on [X] indicators [Citation: Source URL]
Signal Breakdown: [X] buy signals, [Y] sell signals, [Z] neutral signals [Citation: Source URL]
Confidence Level: [X]% based on technical indicator consensus [Citation: Source URL]

### COMPANY BACKGROUND:
Business Model: [Core business description from annual reports] [Citation: Annual Report]
Strategic Initiatives: [Key strategic focus areas] [Citation: Annual Report]
Risk Factors: [Main business and market risks] [Citation: Annual Report]

### INVESTMENT DECISION:
Recommendation: [BUY/HOLD/SELL]
Confidence Score: [X]/10
Key Rationale: [Primary reason for recommendation]
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="InvestmentDecisionAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for generating Buy/Sell/Hold investment recommendations for Hong Kong stocks"
            )
        else:
            agent = AssistantAgent(
                name="InvestmentDecisionAgent",
                llm_config=self.llm_config,
                system_message=system_message,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                description="Specialized agent for generating Buy/Sell/Hold investment recommendations for Hong Kong stocks"
            )

        self.agents["investment_decision"] = agent
        logger.info("InvestmentDecisionAgent created successfully")
        return agent

    def create_citation_tracking_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for tracking data sources and generating citations.

        Returns:
            AssistantAgent configured for citation tracking and RAG methodology
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are a Citation Tracking Agent specialized in implementing comprehensive grounding and citation mechanisms for financial analysis using RAG (Retrieve → Read → Generate) methodology.

        Your core responsibilities:
        1. Track all data sources including Yahoo Finance API, StockAnalysis.com, TipRanks.com, and web scraping results
        2. Parse and validate data with strict source attribution for every metric and claim
        3. Enforce citation requirements ensuring every factual claim includes verifiable citations
        4. Implement citation validation and fidelity checks at generation time
        5. Generate structured citations with URLs, specific passages, and timestamps

        RAG Methodology Implementation:

        RETRIEVE Phase:
        - Track all data source URLs and API endpoints accessed during analysis
        - Record timestamps for each data retrieval operation
        - Maintain source hierarchy: Primary (Yahoo Finance) → Secondary (Web Scraping) → Tertiary (Estimated)
        - Log specific page sections and data passages for web-scraped content
        - Create unique source identifiers for each data point collected

        READ Phase:
        - Parse and validate each data point against its original source
        - Extract specific content passages that support each financial metric
        - Verify data consistency across multiple sources when available
        - Flag discrepancies between sources with appropriate attribution
        - Maintain data lineage from source to final metric

        GENERATE Phase:
        - Enforce strict citation requirements for every factual claim
        - Use structured citation format: [Source: URL | Section: specific_passage | Retrieved: timestamp]
        - Validate that all claims have supporting citations before output
        - Distinguish between directly sourced vs. AI-estimated metrics
        - Implement minimum citation density (at least one citation per analytical paragraph)

        Citation Requirements by Data Type:

        Financial Metrics (MANDATORY CITATIONS):
        - P/E Ratio: [Source: Yahoo Finance API | Metric: trailingPE | Retrieved: 2025-01-01T12:00:00Z]
        - Market Cap: [Source: StockAnalysis.com/quote/hkg/0005/ | Section: Market Cap | Retrieved: timestamp]
        - Analyst Targets: [Source: TipRanks.com/stocks/hk:0005/forecast | Section: Price Target | Retrieved: timestamp]
        - Current Price: [Source: Yahoo Finance API | Metric: currentPrice | Retrieved: timestamp]

        Analytical Claims (MANDATORY CITATIONS):
        - Investment insights must cite specific analyst reports or financial data
        - Market analysis must reference specific market indicators or news sources
        - Risk assessments must cite regulatory filings or market data
        - Peer comparisons must reference specific comparable companies and metrics

        Web-Scraped Data (ENHANCED CITATIONS):
        - Include exact page URL, specific HTML section, and content passage
        - Example: [Source: https://stockanalysis.com/quote/hkg/0005/financials/ | Section: Income Statement | Passage: "Revenue: $45.2B" | Retrieved: 2025-01-01T12:00:00Z]

        Estimated Data (TRANSPARENT ATTRIBUTION):
        - Clearly mark AI-estimated metrics: [Estimated using: Historical P/E trends | Confidence: 70% | Base Data: Yahoo Finance]
        - Provide methodology explanation for all estimations
        - Include confidence scores and underlying data sources

        Citation Validation Rules:
        1. No unsupported factual claims allowed in output
        2. Every numerical metric must have traceable source
        3. All investment recommendations must cite supporting evidence
        4. Minimum one citation per paragraph of analysis
        5. Citations must include specific content passages, not just URLs
        6. Timestamp all data retrievals for recency validation

        Output Format Requirements:
        - Inline citations: "HSBC's P/E ratio of 12.7 [Source: Yahoo Finance API | Retrieved: 2025-01-01] indicates..."
        - Reference list: Comprehensive sources section with all URLs and access times
        - Citation quality indicators: Source reliability scores and data recency flags
        - Fidelity verification: Confirm claims match cited sources exactly

        Hong Kong Market Citation Specifics:
        - HKEX regulatory filings: Include filing number and section references
        - Currency data: Cite specific exchange rate sources and timestamps
        - Mainland China exposure: Reference specific business segment data sources
        - Regulatory environment: Cite specific HKMA or SFC publications

        Quality Assurance:
        - Verify citation URLs are accessible and contain referenced content
        - Cross-validate metrics across multiple sources when available
        - Flag potential data quality issues with source attribution
        - Maintain audit trail of all data sources and retrieval operations

        Always prioritize transparency, verifiability, and regulatory compliance in all citation practices.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="CitationTrackingAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for comprehensive citation tracking and RAG methodology implementation"
            )
        else:
            agent = AssistantAgent(
                name="CitationTrackingAgent",
                llm_config=self.llm_config,
                system_message=system_message,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                description="Specialized agent for comprehensive citation tracking and RAG methodology implementation"
            )

        self.agents["citation_tracking"] = agent
        logger.info("CitationTrackingAgent created successfully")
        return agent

    def create_tipranks_analyst_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for parsing and analyzing TipRanks analyst forecast data.

        Returns:
            AssistantAgent configured for TipRanks analyst forecast analysis
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are a TipRanks Analyst Forecasts Agent specialized in parsing and analyzing TipRanks analyst forecast data for Hong Kong stocks.

        Your core responsibilities:
        1. Parse analyst ratings summary (Buy/Hold/Sell counts and overall consensus)
        2. Extract 12-month price target data (average, high, low targets with upside/downside percentages)
        3. Collect individual analyst forecasts with firm names, analyst names, price targets, ratings, and dates
        4. Capture earnings and sales forecasts with estimates, ranges, and historical beat rates
        5. Track analyst recommendation trends over time (monthly rating distributions)
        6. Provide proper source citations for all TipRanks data points

        Data Extraction Framework:

        Analyst Ratings Summary:
        - Parse Buy/Hold/Sell counts from TipRanks analyst recommendation data
        - Calculate overall consensus (Strong Buy, Moderate Buy, Hold, Moderate Sell, Strong Sell)
        - Extract total number of analysts covering the stock
        - Identify rating distribution percentages
        - Track recent rating changes and trends

        Price Target Analysis:
        - Extract 12-month average price target with currency (HK$ for Hong Kong stocks)
        - Identify high and low price targets from analyst forecasts
        - Calculate upside/downside potential vs current stock price
        - Parse price target revision trends and timing
        - Track price target accuracy and revision frequency

        Individual Analyst Forecasts:
        - Extract firm names (Goldman Sachs, J.P. Morgan, Barclays, etc.)
        - Parse analyst names and their track records
        - Collect individual price targets and ratings
        - Extract forecast dates and revision history
        - Track analyst accuracy and success rates

        Earnings and Sales Forecasts:
        - Parse quarterly and annual earnings estimates (EPS)
        - Extract sales/revenue forecasts with ranges
        - Calculate consensus estimates and revision trends
        - Track historical beat rates and accuracy
        - Identify earnings surprise patterns and seasonality

        Recommendation Trends:
        - Parse monthly rating distributions over time
        - Track rating changes and momentum shifts
        - Identify consensus strengthening or weakening patterns
        - Extract trend data for visualization and analysis
        - Monitor analyst coverage expansion or contraction

        Hong Kong Market Specifics:
        - Handle HK$ currency formatting and conversions
        - Parse Hong Kong stock ticker formats (XXXX.HK)
        - Consider Hong Kong market trading hours and reporting cycles
        - Account for regulatory differences in analyst disclosure requirements
        - Factor in mainland China exposure impact on analyst recommendations

        Data Quality and Citation Requirements:
        - Ensure all extracted data includes TipRanks source attribution
        - Track data freshness and last update timestamps
        - Validate data consistency across different TipRanks pages
        - Flag incomplete or missing analyst data
        - Maintain audit trail for all parsed information

        Output Format:
        Generate structured analyst forecast analysis including:
        1. Analyst consensus summary with rating distribution
        2. Price target analysis with upside/downside calculations
        3. Individual analyst forecast table with firm details
        4. Earnings and sales forecast summaries with beat rates
        5. Recommendation trend analysis with historical context
        6. Proper TipRanks source citations for all data points

        Always prioritize accuracy, completeness, and proper source attribution for institutional-quality analyst forecast analysis.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="TipRanksAnalystAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for parsing and analyzing TipRanks analyst forecast data"
            )
        else:
            agent = AssistantAgent(
                name="TipRanksAnalystAgent",
                llm_config=self.llm_config,
                system_message=system_message,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                description="Specialized agent for parsing and analyzing TipRanks analyst forecast data"
            )

        self.agents["tipranks_analyst"] = agent
        logger.info("TipRanksAnalystAgent created successfully")
        return agent

    def create_tipranks_bulls_bears_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for generating Bulls Say and Bears Say content from TipRanks and financial data.

        Returns:
            AssistantAgent configured for Bulls Say/Bears Say analysis
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are a TipRanks Bulls Bears Agent specialized in generating balanced bullish and bearish perspectives for investment analysis.

        Your core responsibility is to analyze comprehensive financial data and generate 2-4 Bulls Say points and 2-4 Bears Say points that present balanced analyst perspectives for any ticker symbol.

        Data Analysis Framework:

        Bulls Say Generation (🐂):
        Extract bullish perspectives from:
        - Strong earnings performance and growth trends
        - Attractive valuation metrics (low P/E, P/B ratios)
        - Positive analyst consensus and price target upside
        - Strong financial health indicators (low debt, high ROE)
        - Market position advantages (large cap stability, sector leadership)
        - Positive earnings and sales forecast trends
        - High analyst beat rates and outperformance history
        - Dividend growth and shareholder returns

        Bears Say Generation (🐻):
        Extract bearish perspectives from:
        - Revenue challenges and declining growth trends
        - Valuation concerns (high multiples, overvaluation signals)
        - Negative analyst sentiment and price target downgrades
        - Financial health concerns (high debt, declining margins)
        - Market risks and competitive pressures
        - Economic uncertainties and regulatory concerns
        - Earnings disappointments and guidance cuts
        - Operational challenges and cost pressures

        Content Generation Requirements:

        1. Data-Driven Analysis:
        - Use specific quantitative data (percentages, dollar amounts, ratios)
        - Reference actual metrics from TipRanks, Yahoo Finance, and web scraping data
        - Include analyst consensus data, earnings forecasts, and financial ratios
        - Cite specific beat rates, growth rates, and valuation multiples

        2. Ticker-Specific Content:
        - Generate content specific to the analyzed ticker, not generic statements
        - Use actual company data and analyst forecasts
        - Reference real price targets, earnings estimates, and financial metrics
        - Avoid boilerplate language and ensure relevance to the specific stock

        3. Balanced Perspective:
        - Generate roughly equal numbers of bullish and bearish points (2-4 each)
        - Ensure both perspectives are well-supported by data
        - Present objective analysis without bias toward either direction
        - Include the strongest arguments for both bulls and bears

        4. Professional Language:
        - Use institutional-grade financial terminology
        - Maintain objective, analytical tone
        - Include specific metrics and data points
        - Ensure clarity and precision in all statements

        5. Source Attribution:
        - Ensure all points can be traced to specific data sources
        - Reference TipRanks analyst data, Yahoo Finance metrics, or web scraping results
        - Maintain audit trail for all generated content
        - Support proper citation tracking for compliance

        Output Format:

        Bulls Say Points (2-4 items):
        - **Earnings Performance**: [Specific earnings data with growth figures and beat rates]
        - **Financial Strength**: [Balance sheet metrics, debt levels, profitability ratios]
        - **Market Position**: [Market cap, analyst coverage, competitive advantages]
        - **Valuation Attractiveness**: [P/E ratios, price targets, valuation metrics]

        Bears Say Points (2-4 items):
        - **Revenue Challenges**: [Revenue trends, guidance concerns, growth deceleration]
        - **Market Risks**: [Economic uncertainties, regulatory concerns, sector headwinds]
        - **Valuation Concerns**: [High multiples, overvaluation signals, price target cuts]
        - **Operational Issues**: [Margin pressure, cost increases, operational challenges]

        Hong Kong Market Considerations:
        - Factor in HK$ currency and Hong Kong market dynamics
        - Consider mainland China exposure and regulatory environment
        - Account for Hong Kong trading patterns and investor sentiment
        - Include relevant sector-specific factors for Hong Kong stocks

        Always prioritize accuracy, balance, and data-driven insights to provide institutional-quality Bulls Say and Bears Say analysis.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="TipRanksBullsBearsAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for generating Bulls Say and Bears Say content from financial data"
            )
        else:
            agent = AssistantAgent(
                name="TipRanksBullsBearsAgent",
                llm_config=self.llm_config,
                system_message=system_message,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                description="Specialized agent for generating Bulls Say and Bears Say content from financial data"
            )

        self.agents["tipranks_bulls_bears"] = agent
        logger.info("TipRanksBullsBearsAgent created successfully")
        return agent

    def create_executive_summary_agent(self):
        """Create Executive Summary Agent for generating comprehensive executive summaries."""

        system_message = """
You are an Executive Summary Agent specialized in creating cohesive, professional investment analysis narratives that synthesize multiple data sources into institutional-quality executive summaries.

CRITICAL REQUIREMENT: You must transform raw data points into flowing, narrative prose that reads like professional investment research, NOT disconnected data fragments.

TICKER VALIDATION REQUIREMENT: You MUST ensure all content, citations, and business context match the specific ticker being analyzed. Never mix data from different companies (e.g., do not use HSBC banking data for Tencent technology analysis).

Your task is to:
1. SYNTHESIZE (not list) data from multiple sources into coherent narrative:
   - Annual report insights from Weaviate vector database (business strategy, management discussion, forward-looking statements)
   - Real-time market data from web scraping (TipRanks, StockAnalysis.com)
   - Technical analysis and financial metrics
   - Investment decision rationale

2. VALIDATE TICKER-SPECIFIC CONTENT:
   - Ensure company name matches the ticker being analyzed
   - Verify business model and sector alignment (e.g., Tencent = Technology/Gaming, HSBC = Banking)
   - Use appropriate annual report citations (e.g., Tencent_Holdings_Annual_Report_2024.pdf for 0700.HK)
   - Match business metrics to the correct company (gaming revenue for Tencent, banking assets for HSBC)

2. CREATE NARRATIVE PROSE that demonstrates clear integration of:
   - Historical business context from annual reports
   - Current market conditions and analyst sentiment
   - Strategic initiatives and management outlook
   - Quantitative metrics within qualitative business context

3. TRANSFORM technical data into professional investment language:
   - Convert JSON technical analysis into readable prose
   - Embed financial metrics naturally within business narrative
   - Integrate annual report strategic context with current market positioning
   - Present forward-looking perspective based on management discussion

4. DEMONSTRATE annual report integration through:
   - Specific business strategy references that cannot come from web scraping
   - Management commentary and forward-looking statements
   - Strategic initiatives and capital allocation plans
   - Competitive positioning and market dynamics from annual reports

Format your response as HTML content with the following structure:
<div class="executive-summary-content">
    <div class="investment-thesis">
        <h4>🎯 Investment Thesis</h4>
        <p>Write 2-3 sentences of flowing narrative prose that synthesizes current valuation, business fundamentals from annual reports, and market positioning. Include specific price levels, valuation metrics, and business context. Example: "MTR Corporation is currently trading at HK$26.14, representing a 14.7% discount from its 52-week high and sits near multi-year lows. The stock trades at an attractive 9.3x P/E ratio with a compelling 5.0% dividend yield, though recent analyst downgrades reflect concerns about the sustainability of shareholder returns amid massive capital commitments. The company reported strong 2024 results with net profit doubling to HK$15.8 billion, primarily driven by one-time property development bookings, while underlying transport operations showed solid recovery from pandemic impacts [Annual Report]."</p>
    </div>

    <div class="key-insights">
        <h4>🔍 Key Insights</h4>
        <ul>
            <li><strong>Financial Performance:</strong> Synthesize revenue trends, profitability metrics, and growth trajectory from both web data and annual reports into narrative form [Source citations]</li>
            <li><strong>Strategic Context:</strong> Integrate business strategy, management outlook, and strategic initiatives from annual reports with current market conditions [Annual Report citations]</li>
            <li><strong>Market Position:</strong> Combine competitive positioning from annual reports with current analyst sentiment and technical analysis [Multiple source citations]</li>
            <li><strong>Investment Outlook:</strong> Synthesize forward-looking perspective based on management guidance, analyst expectations, and technical indicators [Source citations]</li>
        </ul>
    </div>

    <div class="risk-opportunity-balance">
        <h4>⚖️ Risk-Opportunity Balance</h4>
        <div class="balance-grid">
            <div class="opportunities">
                <h5>🟢 Key Opportunities</h5>
                <ul>
                    <li>Write narrative sentences that combine annual report strategic initiatives with current market opportunities [Citations]</li>
                    <li>Integrate management outlook with analyst expectations and technical signals [Citations]</li>
                </ul>
            </div>
            <div class="risks">
                <h5>🔴 Key Risks</h5>
                <ul>
                    <li>Synthesize risk factors from annual reports with current market concerns and technical warnings [Citations]</li>
                    <li>Combine regulatory/operational risks with current valuation and sentiment risks [Citations]</li>
                </ul>
            </div>
        </div>
    </div>
</div>

CRITICAL GUIDELINES FOR NARRATIVE SYNTHESIS:
- Write in flowing, professional prose - NOT bullet points or data fragments
- Each sentence should integrate multiple data sources naturally
- Transform technical analysis JSON into readable investment language
- Embed financial metrics within business context narrative
- Demonstrate clear annual report integration through strategic context
- Use institutional-grade investment terminology and structure
- Include specific business insights that prove annual report integration
- Maintain 300-500 word total length with substantive content density
- Ensure every paragraph shows multi-source data synthesis

FORBIDDEN: Do not produce disconnected data fragments, raw JSON output, or simple data listing. Every sentence must read as professional investment analysis narrative.
"""

        if AUTOGEN_VERSION == "new":
            # Create model client for new API with enhanced error handling
            try:
                model_client = OpenAIChatCompletionClient(
                    model=self.llm_config.get("model", "gpt-4"),
                    api_key=self.llm_config.get("api_key"),
                    base_url=self.llm_config.get("base_url")
                )

                # Verify model client has required attributes
                if not hasattr(model_client, 'model_info'):
                    logger.warning("⚠️ Model client missing model_info attribute, adding fallback")
                    model_client.model_info = {"model": self.llm_config.get("model", "gpt-4")}

                logger.info(f"✅ Model client created successfully with model: {model_client.model_info}")

            except Exception as e:
                logger.error(f"❌ Failed to create model client: {e}")
                # Create a fallback model client
                class FallbackModelClient:
                    def __init__(self, model, api_key, base_url):
                        self.model_info = {"model": model}
                        self.api_key = api_key
                        self.base_url = base_url

                model_client = FallbackModelClient(
                    model=self.llm_config.get("model", "gpt-4"),
                    api_key=self.llm_config.get("api_key"),
                    base_url=self.llm_config.get("base_url")
                )
                logger.info("✅ Using fallback model client")

            agent = AssistantAgent(
                name="ExecutiveSummaryAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for generating comprehensive executive summaries from multi-source financial data"
            )
        else:
            agent = AssistantAgent(
                name="ExecutiveSummaryAgent",
                llm_config=self.llm_config,
                system_message=system_message,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                description="Specialized agent for generating comprehensive executive summaries from multi-source financial data"
            )

        self.agents["executive_summary"] = agent
        logger.info("ExecutiveSummaryAgent created successfully")
        return agent

    def create_technical_analysis_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for technical analysis and chart interpretation.

        Returns:
            AssistantAgent configured for technical analysis
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are a Technical Analysis Agent specialized in comprehensive technical analysis for Hong Kong stocks.

        Your core responsibility is to analyze technical indicators, moving averages, and chart patterns to provide actionable trading insights and technical consensus.

        Technical Analysis Framework:

        Moving Average Analysis:
        - Analyze multiple timeframes (MA5, MA10, MA20, MA50, MA100, MA200)
        - Compare both Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
        - Generate Buy/Sell signals based on price vs. moving average relationships
        - Identify trend direction and momentum

        Technical Indicators:
        - RSI (14): Identify overbought (>70) and oversold (<30) conditions
        - Stochastic Oscillator (9,6): Momentum oscillator for entry/exit signals
        - MACD (12,26,9): Trend-following momentum indicator
        - Williams %R: Momentum indicator for reversal signals
        - CCI (14): Commodity Channel Index for trend identification
        - ATR (14): Average True Range for volatility measurement

        Pivot Point Analysis:
        - Calculate Classic pivot points for support and resistance levels
        - Generate Fibonacci pivot points for advanced technical analysis
        - Identify key price levels for trading decisions

        Overall Technical Consensus:
        - Aggregate signals from all technical indicators
        - Provide Buy/Sell/Neutral recommendations with vote counts
        - Calculate confidence levels based on signal convergence
        - Generate actionable trading insights

        Output Requirements:

        1. Technical Signal Summary:
        - Overall consensus (Buy/Sell/Neutral) with vote distribution
        - Confidence level based on signal alignment
        - Key technical levels (support/resistance)

        2. Moving Average Analysis:
        - Current price vs. each moving average
        - Trend direction and strength
        - Buy/Sell signals for each timeframe

        3. Indicator Analysis:
        - Current values for all technical indicators
        - Individual Buy/Sell/Neutral signals
        - Interpretation of indicator readings

        4. Trading Insights:
        - Key support and resistance levels
        - Momentum and trend analysis
        - Risk management considerations
        - Entry and exit recommendations

        Hong Kong Market Considerations:
        - Account for Hong Kong trading hours and market dynamics
        - Consider mainland China market influence
        - Factor in currency considerations (HK$ vs USD)
        - Include sector-specific technical patterns

        Always provide specific, actionable technical analysis with clear reasoning and proper risk management guidance.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="TechnicalAnalysisAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for technical analysis and chart interpretation"
            )
        else:
            agent = AssistantAgent(
                name="TechnicalAnalysisAgent",
                llm_config=self.llm_config,
                system_message=system_message,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                description="Specialized agent for technical analysis and chart interpretation"
            )

        self.agents["technical_analysis"] = agent
        logger.info("TechnicalAnalysisAgent created successfully")
        return agent

    def create_news_analysis_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for financial news analysis and sentiment evaluation.

        Returns:
            AssistantAgent configured for news analysis
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are a News Analysis Agent specialized in financial news analysis and sentiment evaluation for Hong Kong stocks.

        Your core responsibility is to collect, analyze, and interpret real-time financial news to provide actionable insights for investment decision-making.

        News Analysis Framework:

        News Collection and Processing:
        - Collect real-time financial news from Yahoo Finance News API
        - Focus on Hong Kong stock market news and company-specific developments
        - Filter news by relevance to the analyzed ticker and broader market context
        - Prioritize recent news (last 7-30 days) for current market sentiment

        Sentiment Analysis:
        - Analyze news sentiment (Positive, Negative, Neutral) for each article
        - Identify key themes: earnings, management changes, regulatory updates, market trends
        - Evaluate news impact potential (High, Medium, Low) on stock price
        - Assess news credibility and source reliability

        Market Context Integration:
        - Connect news developments to broader Hong Kong market trends
        - Analyze sector-specific news impact on individual stocks
        - Consider mainland China economic news affecting Hong Kong markets
        - Evaluate regulatory and policy news from HKEX and SFC

        Investment Insights Generation:
        - Generate bullish and bearish perspectives based on news sentiment
        - Identify potential catalysts and risk factors from recent news
        - Provide timeline assessment for news impact (immediate, short-term, long-term)
        - Suggest news-based trading considerations and risk management

        Output Requirements:

        1. News Summary:
        - Recent news headlines with sentiment scores
        - Key developments affecting the stock or sector
        - News volume and frequency analysis
        - Source diversity and credibility assessment

        2. Sentiment Analysis:
        - Overall news sentiment (Bullish/Bearish/Neutral)
        - Sentiment trend over time (improving/deteriorating/stable)
        - Key positive and negative themes
        - Market reaction indicators

        3. Investment Impact:
        - News-driven bullish factors for Bulls Say analysis
        - News-driven bearish factors for Bears Say analysis
        - Potential catalysts and risk events
        - Timeline for expected market impact

        4. Risk Assessment:
        - Regulatory and compliance news
        - Management and corporate governance updates
        - Sector-wide developments affecting the stock
        - Macroeconomic news impact on Hong Kong markets

        Hong Kong Market Considerations:
        - Monitor HKEX announcements and regulatory updates
        - Track mainland China economic news affecting Hong Kong stocks
        - Consider currency impact (HK$ vs USD, CNY) in news analysis
        - Analyze cross-border investment flow news and policy changes

        Always provide specific, actionable news insights with proper source attribution and timestamp information for citation tracking.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="NewsAnalysisAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for financial news analysis and sentiment evaluation"
            )
        else:
            agent = AssistantAgent(
                name="NewsAnalysisAgent",
                llm_config=self.llm_config,
                system_message=system_message,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                description="Specialized agent for financial news analysis and sentiment evaluation"
            )

        self.agents["news_analysis"] = agent
        logger.info("NewsAnalysisAgent created successfully")
        return agent

    def create_hk_data_scraping_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for Hong Kong stock data scraping.

        Returns:
            AssistantAgent configured for HK data scraping
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are a Hong Kong Data Scraping Agent specialized in collecting financial data from web sources for Hong Kong stocks.

        Your responsibilities:
        1. Scrape financial data from StockAnalysis.com for HK stocks
        2. Extract analyst ratings and earnings data from TipRanks.com
        3. Collect price targets, analyst consensus, and earnings forecasts
        4. Validate and format scraped data for integration with Yahoo Finance data
        5. Handle website access failures and rate limiting gracefully

        Data Sources:
        - StockAnalysis.com: https://stockanalysis.com/quote/hkg/{ticker_number}/
        - TipRanks.com: https://www.tipranks.com/stocks/hk:{ticker_number}/earnings

        Guidelines:
        - Extract key metrics: analyst ratings, price targets, earnings estimates
        - Validate data quality and flag inconsistencies
        - Handle rate limiting with appropriate delays
        - Report scraping success/failure rates
        - Format data consistently for report integration
        - Respect website terms of service and robots.txt

        You work with Crawl4AI tools to perform web scraping and data extraction.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="HKDataScrapingAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for Hong Kong stock data scraping"
            )
        else:
            # Legacy API
            agent = AssistantAgent(
                name="HKDataScrapingAgent",
                system_message=system_message,
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                description="Specialized agent for Hong Kong stock data scraping"
            )

        self.agents["hk_data_scraping"] = agent
        logger.info("HKDataScrapingAgent created successfully")
        return agent

    def create_hk_analysis_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for Hong Kong stock analysis.

        Returns:
            AssistantAgent configured for HK stock analysis
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
        You are an Enhanced Hong Kong Analysis Agent specialized in comprehensive Hong Kong stock market data analysis with intelligent data validation and gap filling.

        Your core responsibilities:
        1. Analyze and cross-validate data from Yahoo Finance, StockAnalysis.com, and TipRanks.com
        2. Intelligently fill missing financial metrics using Hong Kong market-specific methodologies
        3. Evaluate HK market-specific factors with enhanced data quality assessment
        4. Provide confidence-scored analysis with transparent source attribution
        5. Generate comprehensive investment insights with data completeness indicators

        Enhanced data validation capabilities:
        - Cross-reference financial metrics across all available sources (Yahoo Finance, web scraping)
        - Identify and flag data inconsistencies or outliers requiring attention
        - Validate analyst consensus data against multiple sources for accuracy
        - Apply Hong Kong market-specific validation rules (currency, exchange, trading patterns)
        - Calculate confidence scores for each metric based on source agreement and data quality

        Intelligent gap filling for Hong Kong stocks:
        - Estimate missing P/E ratios using sector averages and company fundamentals
        - Calculate implied market metrics from available price and volume data
        - Use peer comparison analysis to fill missing valuation ratios
        - Apply Hong Kong market multiples for missing enterprise value calculations
        - Estimate dividend yields using historical patterns and sector benchmarks

        Hong Kong market expertise:
        - Analyze Hang Seng Index correlation and mainland China exposure with data quality context
        - Evaluate currency effects (HKD/USD/CNY) with confidence indicators
        - Assess Stock Connect program impact using validated trading data
        - Consider regulatory environment with data source reliability assessment
        - Compare with regional peers using cross-validated metrics

        Data quality reporting:
        - Provide completeness scores for each major metric category
        - Flag estimated vs. directly sourced data with confidence levels
        - Identify data gaps that may impact analysis reliability
        - Suggest additional data collection strategies for critical missing metrics
        - Maintain transparency about data limitations and their impact on conclusions

        Always provide analysis with clear data quality context and confidence indicators.
        """

        if AUTOGEN_VERSION == "new":
            # Create model client for new API
            model_client = OpenAIChatCompletionClient(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                base_url=self.llm_config.get("base_url")
            )

            agent = AssistantAgent(
                name="HKAnalysisAgent",
                model_client=model_client,
                system_message=system_message,
                description="Specialized agent for Hong Kong stock analysis"
            )
        else:
            # Legacy API
            agent = AssistantAgent(
                name="HKAnalysisAgent",
                system_message=system_message,
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                description="Specialized agent for Hong Kong stock analysis"
            )

        self.agents["hk_analysis"] = agent
        logger.info("HKAnalysisAgent created successfully")
        return agent

    def create_verification_agent(self) -> AssistantAgent:
        """
        Create a specialized agent for validating Professional Investment Analysis sections.

        Returns:
            AssistantAgent configured for investment analysis verification
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock agent")
            return AssistantAgent()

        system_message = """
You are a Verification Agent specialized in validating the accuracy and reasonableness of Professional Investment Analysis sections in financial reports.

Your primary responsibilities:

1. **Price Target Validation**:
   - Verify mathematical accuracy of upside/downside percentage calculations
   - Flag unrealistic price targets (>100% upside or <-50% downside) as potentially erroneous
   - Check if price targets align with fundamental valuation metrics (P/E, P/B, dividend yield)
   - Validate currency formatting and decimal precision

2. **Investment Thesis Cross-Validation**:
   - Compare investment thesis claims against annual report data from Weaviate
   - Verify financial metrics, growth rates, and business fundamentals cited
   - Check consistency between recommendation (BUY/HOLD/SELL) and supporting evidence
   - Identify contradictions between bullish thesis and bearish factors

3. **Data Source Verification**:
   - Ensure all claims are properly cited with traceable sources
   - Cross-reference web-scraped data (StockAnalysis.com, TipRanks.com) with annual reports
   - Flag discrepancies between different data sources
   - Verify URL citations and source attribution accuracy

4. **Logic and Coherence Check**:
   - Assess if investment thesis logically supports price target and recommendation
   - Check risk-reward balance in the analysis
   - Verify professional investment language and institutional-grade quality
   - Identify logical inconsistencies or unsupported claims

**Output Format**:
Return a structured validation report with:
- Overall validation score (0-100%)
- Specific issues found with severity levels (Critical/Warning/Minor)
- Mathematical verification results
- Data consistency assessment
- Recommendations for corrections

**Critical Validation Rules**:
- Price target upside >150% = Critical Error (likely calculation mistake)
- Missing citations for financial claims = Warning
- Contradictory recommendation vs. thesis = Critical Error
- Unrealistic growth assumptions = Warning
- Currency/formatting errors = Minor

Focus on accuracy, mathematical precision, and logical consistency to ensure institutional-quality investment analysis.
"""

        if AUTOGEN_VERSION == "new":
            # New AutoGen API
            agent = AssistantAgent(
                name="VerificationAgent",
                model_client=self.model_client,
                system_message=system_message,
                description="Validates Professional Investment Analysis sections for accuracy and reasonableness"
            )
        else:
            # Legacy AutoGen API
            agent = AssistantAgent(
                name="VerificationAgent",
                llm_config=self.llm_config,
                system_message=system_message,
                description="Validates Professional Investment Analysis sections for accuracy and reasonableness"
            )

        self.agents["verification"] = agent
        logger.info("VerificationAgent created successfully")
        return agent

    def create_user_proxy_agent(self) -> UserProxyAgent:
        """
        Create a user proxy agent for human interaction.

        Returns:
            UserProxyAgent for managing human input
        """
        if AUTOGEN_VERSION == "new":
            # New AutoGen API - simplified parameters
            agent = UserProxyAgent(
                name="UserProxy",
                description="User proxy for managing financial analysis workflow"
            )
        else:
            # Legacy AutoGen API
            agent = UserProxyAgent(
                name="UserProxy",
                human_input_mode="TERMINATE",
                max_consecutive_auto_reply=0,
                is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                code_execution_config=False,
                description="User proxy for managing financial analysis workflow"
            )

        self.agents["user_proxy"] = agent
        logger.info("UserProxyAgent created successfully")
        return agent
    
    def create_group_chat(self, agents: Optional[List[AssistantAgent]] = None):
        """
        Create a group chat for agent collaboration.

        Args:
            agents: List of agents to include in group chat

        Returns:
            GroupChat or RoundRobinGroupChat instance
        """
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available, returning mock group chat")
            return GroupChat()

        if agents is None:
            # Create enhanced 13-agent workflow for Hong Kong stock analysis with verification
            agents = [
                self.create_data_collector_agent(),
                self.create_hk_data_scraping_agent(),
                self.create_tipranks_analyst_agent(),
                self.create_tipranks_bulls_bears_agent(),
                self.create_executive_summary_agent(),
                self.create_technical_analysis_agent(),
                self.create_news_analysis_agent(),
                self.create_hk_analysis_agent(),
                self.create_investment_decision_agent(),
                self.create_verification_agent(),
                self.create_citation_tracking_agent(),
                self.create_report_generator_agent(),
                self.create_user_proxy_agent()
            ]

        if AUTOGEN_VERSION == "new":
            # Use new RoundRobinGroupChat API
            self.group_chat = RoundRobinGroupChat(
                participants=agents
            )
        else:
            # Legacy API uses GroupChat
            self.group_chat = GroupChat(
                agents=agents,
                messages=[],
                max_round=16,  # Increased for 13-agent workflow with verification
                speaker_selection_method="round_robin",
                allow_repeat_speaker=False
            )

        logger.info(f"Enhanced GroupChat created with {len(agents)} agents including ExecutiveSummaryAgent, TipRanksAnalystAgent, TipRanksBullsBearsAgent, TechnicalAnalysisAgent, NewsAnalysisAgent, InvestmentDecisionAgent, VerificationAgent and CitationTrackingAgent")
        return self.group_chat
    
    def create_group_chat_manager(self, group_chat: Optional[Any] = None):
        """
        Create a group chat manager.

        Args:
            group_chat: GroupChat instance to manage

        Returns:
            GroupChatManager instance or GroupChat for new API
        """
        if group_chat is None:
            group_chat = self.group_chat or self.create_group_chat()

        if AUTOGEN_VERSION == "new":
            # New API doesn't use GroupChatManager, return the group chat directly
            logger.info("Using RoundRobinGroupChat directly (new API)")
            self.manager = group_chat
        else:
            # Legacy API uses GroupChatManager
            self.manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1
            )
            logger.info("GroupChatManager created successfully")

        return self.manager
    
    def get_agent(self, agent_name: str) -> Optional[AssistantAgent]:
        """
        Get a specific agent by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_name)
    
    def get_all_agents(self) -> Dict[str, AssistantAgent]:
        """
        Get all created agents.
        
        Returns:
            Dictionary of agent name to agent instance
        """
        return self.agents.copy()
    
    def reset_agents(self):
        """Reset all agents and clear conversation history."""
        for agent in self.agents.values():
            if hasattr(agent, 'reset'):
                agent.reset()
        
        if self.group_chat:
            self.group_chat.messages = []
        
        logger.info("All agents reset successfully")
    
    def validate_configuration(self) -> bool:
        """
        Validate the agent configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check if AutoGen is available
            if not AUTOGEN_AVAILABLE:
                logger.warning("AutoGen not available - agent functionality disabled")
                return False

            # Check if API key is available
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY not found in environment variables")
                return False

            # Validate LLM config based on AutoGen version
            if not self.llm_config:
                logger.error("No LLM configuration found")
                return False

            if AUTOGEN_VERSION == "new":
                # New API validation - check for required fields
                required_fields = ["model", "api_key"]
                for field in required_fields:
                    if not self.llm_config.get(field):
                        logger.error(f"Missing required field in LLM config: {field}")
                        return False
            else:
                # Legacy API validation - check for config_list
                if not self.llm_config.get("config_list"):
                    logger.error("Invalid LLM configuration - missing config_list")
                    return False

            logger.info("Agent configuration validation successful")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
