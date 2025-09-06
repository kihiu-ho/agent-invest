"""
Hong Kong Financial Data Downloader

A comprehensive Python script for downloading both fundamental financial data 
and historical price data for Hong Kong stock tickers.

Features:
- Historical OHLCV price data download
- Financial statements (income, balance sheet, cash flow)
- Key financial metrics and ratios
- Structured data storage (CSV, JSON)
- Comprehensive error handling
- Rate limiting and retry mechanisms
- Production-ready logging

Author: Financial Metrics Agent
Date: 2025-09-02
"""

import yfinance as yf
import pandas as pd
import json
import logging
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import curl_cffi for browser impersonation
try:
    from curl_cffi import requests as cffi_requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False
    cffi_requests = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DownloadConfig:
    """Configuration class for data download parameters."""
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None  # Default to today
    output_dir: str = "hk_financial_data"
    rate_limit_delay: float = 2.0  # Seconds between requests
    max_retries: int = 3
    timeout: int = 30
    include_dividends: bool = True
    include_splits: bool = True
    enable_pdf_processing: bool = True
    enable_embeddings: bool = True
    download_directory: str = "./downloads"
    vector_store_path: str = "./vector_store"

@dataclass
class FinancialMetrics:
    """Data class for key financial metrics."""
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    book_value: Optional[float] = None
    revenue_ttm: Optional[float] = None
    profit_margin: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

class HKFinancialDataDownloader:
    """
    Comprehensive Hong Kong stock data downloader.
    
    Downloads historical price data, financial statements, and key metrics
    for Hong Kong stock tickers in XXXX.HK format.
    """
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        """
        Initialize the downloader with configuration.

        Args:
            config: DownloadConfig object with download parameters
        """
        self.config = config or DownloadConfig()
        self.session = self._create_session()
        self.yf_session = self._create_yfinance_session()
        self._setup_output_directory()

        logger.info(f"HK Financial Data Downloader initialized")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Date range: {self.config.start_date} to {self.config.end_date or 'today'}")
        logger.info(f"Browser impersonation: {'✅ curl_cffi' if CURL_CFFI_AVAILABLE else '❌ standard requests'}")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _create_yfinance_session(self):
        """Create a session for yfinance with browser impersonation if available."""
        if CURL_CFFI_AVAILABLE:
            try:
                # Create curl_cffi session with Chrome impersonation
                session = cffi_requests.Session(impersonate="chrome")
                logger.info("✅ Created curl_cffi session with Chrome impersonation")
                return session
            except Exception as e:
                logger.warning(f"⚠️ Failed to create curl_cffi session: {e}")
                logger.info("🔄 Falling back to standard requests session")
                return None
        else:
            logger.warning("⚠️ curl_cffi not available, using standard requests")
            return None

    def _setup_output_directory(self) -> None:
        """Create output directory structure."""
        base_dir = Path(self.config.output_dir)
        
        # Create subdirectories
        (base_dir / "price_data").mkdir(parents=True, exist_ok=True)
        (base_dir / "financial_data").mkdir(parents=True, exist_ok=True)
        (base_dir / "reports").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Output directory structure created: {base_dir}")
    
    def _validate_ticker(self, ticker: str) -> str:
        """
        Validate and format Hong Kong ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Formatted ticker symbol
            
        Raises:
            ValueError: If ticker format is invalid
        """
        ticker = ticker.upper().strip()
        
        # Handle different input formats
        if ticker.endswith('.HK'):
            return ticker
        elif ticker.isdigit() and len(ticker) == 4:
            return f"{ticker}.HK"
        elif ticker.isdigit() and len(ticker) < 4:
            return f"{ticker.zfill(4)}.HK"
        else:
            raise ValueError(f"Invalid Hong Kong ticker format: {ticker}")
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        time.sleep(self.config.rate_limit_delay)
    
    def download_price_data(self, ticker: str) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Download historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (success, dataframe, message)
        """
        try:
            formatted_ticker = self._validate_ticker(ticker)
            logger.info(f"📈 Downloading price data for {formatted_ticker}")
            
            # Set end date if not specified
            end_date = self.config.end_date or datetime.now().strftime("%Y-%m-%d")
            
            # Download data using yfinance with curl_cffi session
            if self.yf_session:
                stock = yf.Ticker(formatted_ticker, session=self.yf_session)
                logger.debug(f"Using curl_cffi session for {formatted_ticker}")
            else:
                stock = yf.Ticker(formatted_ticker)
                logger.debug(f"Using standard session for {formatted_ticker}")

            hist_data = stock.history(
                start=self.config.start_date,
                end=end_date,
                auto_adjust=True,
                prepost=True
            )
            
            if hist_data.empty:
                return False, None, f"No price data available for {formatted_ticker}"
            
            # Clean and format data
            hist_data = hist_data.round(4)
            hist_data.index = hist_data.index.strftime('%Y-%m-%d')
            
            # Add additional columns if requested
            if self.config.include_dividends:
                dividends = stock.dividends
                if not dividends.empty:
                    dividends.index = dividends.index.strftime('%Y-%m-%d')
                    hist_data = hist_data.join(dividends.rename('Dividends'), how='left')
                    hist_data['Dividends'] = hist_data['Dividends'].fillna(0)
            
            if self.config.include_splits:
                splits = stock.splits
                if not splits.empty:
                    splits.index = splits.index.strftime('%Y-%m-%d')
                    hist_data = hist_data.join(splits.rename('Stock_Splits'), how='left')
                    hist_data['Stock_Splits'] = hist_data['Stock_Splits'].fillna(1)
            
            logger.info(f"✅ Downloaded {len(hist_data)} days of price data for {formatted_ticker}")
            return True, hist_data, f"Successfully downloaded price data"
            
        except Exception as e:
            error_msg = f"Failed to download price data for {ticker}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def download_financial_data(self, ticker: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Download financial statements and key metrics.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (success, financial_data_dict, message)
        """
        try:
            formatted_ticker = self._validate_ticker(ticker)
            logger.info(f"📊 Downloading financial data for {formatted_ticker}")

            # Use curl_cffi session if available
            if self.yf_session:
                stock = yf.Ticker(formatted_ticker, session=self.yf_session)
                logger.debug(f"Using curl_cffi session for financial data: {formatted_ticker}")
            else:
                stock = yf.Ticker(formatted_ticker)
                logger.debug(f"Using standard session for financial data: {formatted_ticker}")
            
            # Get basic info
            info = stock.info
            
            # Get financial statements
            financial_data = {
                "ticker": formatted_ticker,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "download_date": datetime.now().isoformat(),
                "currency": info.get("currency", "HKD"),
                "financial_statements": {},
                "key_metrics": {}
            }
            
            # Download financial statements
            try:
                # Income Statement
                income_stmt = stock.financials
                if not income_stmt.empty:
                    financial_data["financial_statements"]["income_statement"] = income_stmt.to_dict()
                    logger.info(f"✅ Income statement: {income_stmt.shape[1]} periods")
                
                # Balance Sheet
                balance_sheet = stock.balance_sheet
                if not balance_sheet.empty:
                    financial_data["financial_statements"]["balance_sheet"] = balance_sheet.to_dict()
                    logger.info(f"✅ Balance sheet: {balance_sheet.shape[1]} periods")
                
                # Cash Flow Statement
                cash_flow = stock.cashflow
                if not cash_flow.empty:
                    financial_data["financial_statements"]["cash_flow"] = cash_flow.to_dict()
                    logger.info(f"✅ Cash flow: {cash_flow.shape[1]} periods")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not download financial statements: {e}")
            
            # Extract key metrics
            metrics = self._extract_key_metrics(info, stock)
            financial_data["key_metrics"] = asdict(metrics)
            
            logger.info(f"✅ Downloaded financial data for {formatted_ticker}")
            return True, financial_data, "Successfully downloaded financial data"

        except Exception as e:
            error_msg = f"Failed to download financial data for {ticker}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def _extract_key_metrics(self, info: Dict, stock: yf.Ticker) -> FinancialMetrics:
        """
        Extract key financial metrics from stock info and statements.

        Args:
            info: Stock info dictionary from yfinance
            stock: yfinance Ticker object

        Returns:
            FinancialMetrics object with extracted metrics
        """
        metrics = FinancialMetrics()

        try:
            # Basic valuation metrics
            metrics.market_cap = info.get("marketCap")
            metrics.pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            metrics.pb_ratio = info.get("priceToBook")
            metrics.dividend_yield = info.get("dividendYield")
            metrics.eps = info.get("trailingEps") or info.get("forwardEps")
            metrics.book_value = info.get("bookValue")

            # Revenue and profitability
            metrics.revenue_ttm = info.get("totalRevenue")
            metrics.profit_margin = info.get("profitMargins")
            metrics.roe = info.get("returnOnEquity")

            # Financial health ratios
            metrics.debt_to_equity = info.get("debtToEquity")
            metrics.current_ratio = info.get("currentRatio")
            metrics.quick_ratio = info.get("quickRatio")

            # Try to get additional metrics from financial statements
            try:
                balance_sheet = stock.balance_sheet
                if not balance_sheet.empty and len(balance_sheet.columns) > 0:
                    latest_bs = balance_sheet.iloc[:, 0]

                    # Calculate additional ratios if data is available
                    total_debt = latest_bs.get("Total Debt", 0) or 0
                    total_equity = latest_bs.get("Total Equity Gross Minority Interest", 0) or latest_bs.get("Stockholders Equity", 0) or 0

                    if total_equity and total_equity != 0:
                        if not metrics.debt_to_equity and total_debt:
                            metrics.debt_to_equity = total_debt / total_equity

            except Exception as e:
                logger.debug(f"Could not extract additional metrics from statements: {e}")

        except Exception as e:
            logger.warning(f"Error extracting key metrics: {e}")

        return metrics

    def save_price_data(self, ticker: str, price_data: pd.DataFrame) -> bool:
        """
        Save price data to CSV file.

        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with price data

        Returns:
            Success status
        """
        try:
            formatted_ticker = self._validate_ticker(ticker)
            filename = f"{formatted_ticker.replace('.', '_')}_price_data.csv"
            filepath = Path(self.config.output_dir) / "price_data" / filename

            price_data.to_csv(filepath, index=True)
            logger.info(f"💾 Saved price data to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save price data for {ticker}: {e}")
            return False

    def save_financial_data(self, ticker: str, financial_data: Dict) -> bool:
        """
        Save financial data to JSON file.

        Args:
            ticker: Stock ticker symbol
            financial_data: Dictionary with financial data

        Returns:
            Success status
        """
        try:
            formatted_ticker = self._validate_ticker(ticker)
            filename = f"{formatted_ticker.replace('.', '_')}_financial_data.json"
            filepath = Path(self.config.output_dir) / "financial_data" / filename

            # Convert any datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            # Clean the data for JSON serialization
            clean_data = json.loads(json.dumps(financial_data, default=convert_datetime))

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Saved financial data to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save financial data for {ticker}: {e}")
            return False

    def generate_summary_report(self, ticker: str, price_success: bool, financial_success: bool,
                              price_data: Optional[pd.DataFrame], financial_data: Optional[Dict]) -> bool:
        """
        Generate a summary report for the downloaded data.

        Args:
            ticker: Stock ticker symbol
            price_success: Whether price data download was successful
            financial_success: Whether financial data download was successful
            price_data: Price data DataFrame (if available)
            financial_data: Financial data dictionary (if available)

        Returns:
            Success status
        """
        try:
            formatted_ticker = self._validate_ticker(ticker)
            filename = f"{formatted_ticker.replace('.', '_')}_summary_report.txt"
            filepath = Path(self.config.output_dir) / "reports" / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"HONG KONG STOCK DATA SUMMARY REPORT\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Ticker: {formatted_ticker}\n")
                f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Date Range: {self.config.start_date} to {self.config.end_date or 'today'}\n\n")

                # Price data summary
                f.write(f"PRICE DATA SUMMARY\n")
                f.write(f"{'-' * 20}\n")
                f.write(f"Status: {'✅ SUCCESS' if price_success else '❌ FAILED'}\n")

                if price_success and price_data is not None:
                    f.write(f"Records: {len(price_data)}\n")
                    f.write(f"Date Range: {price_data.index[0]} to {price_data.index[-1]}\n")
                    f.write(f"Latest Close: {price_data['Close'].iloc[-1]:.4f}\n")
                    f.write(f"52-Week High: {price_data['High'].max():.4f}\n")
                    f.write(f"52-Week Low: {price_data['Low'].min():.4f}\n")
                    f.write(f"Average Volume: {price_data['Volume'].mean():.0f}\n")
                f.write(f"\n")

                # Financial data summary
                f.write(f"FINANCIAL DATA SUMMARY\n")
                f.write(f"{'-' * 25}\n")
                f.write(f"Status: {'✅ SUCCESS' if financial_success else '❌ FAILED'}\n")

                if financial_success and financial_data is not None:
                    f.write(f"Company: {financial_data.get('company_name', 'N/A')}\n")
                    f.write(f"Sector: {financial_data.get('sector', 'N/A')}\n")
                    f.write(f"Industry: {financial_data.get('industry', 'N/A')}\n")
                    f.write(f"Currency: {financial_data.get('currency', 'N/A')}\n\n")

                    # Key metrics
                    metrics = financial_data.get('key_metrics', {})
                    f.write(f"KEY FINANCIAL METRICS\n")
                    f.write(f"{'-' * 23}\n")

                    if metrics.get('market_cap'):
                        f.write(f"Market Cap: {metrics['market_cap']:,.0f}\n")
                    if metrics.get('pe_ratio'):
                        f.write(f"P/E Ratio: {metrics['pe_ratio']:.2f}\n")
                    if metrics.get('pb_ratio'):
                        f.write(f"P/B Ratio: {metrics['pb_ratio']:.2f}\n")
                    if metrics.get('dividend_yield'):
                        f.write(f"Dividend Yield: {metrics['dividend_yield']:.2%}\n")
                    if metrics.get('roe'):
                        f.write(f"ROE: {metrics['roe']:.2%}\n")
                    if metrics.get('debt_to_equity'):
                        f.write(f"Debt/Equity: {metrics['debt_to_equity']:.2f}\n")

                f.write(f"\n")
                f.write(f"FILES GENERATED\n")
                f.write(f"{'-' * 15}\n")
                if price_success:
                    f.write(f"✅ Price Data: price_data/{formatted_ticker.replace('.', '_')}_price_data.csv\n")
                if financial_success:
                    f.write(f"✅ Financial Data: financial_data/{formatted_ticker.replace('.', '_')}_financial_data.json\n")
                f.write(f"✅ Summary Report: reports/{filename}\n")

            logger.info(f"📋 Generated summary report: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate summary report for {ticker}: {e}")
            return False

    def download_single_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Download all data for a single ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with download results and status
        """
        logger.info(f"🚀 Starting download for {ticker}")
        start_time = time.time()

        results = {
            "ticker": ticker,
            "start_time": datetime.now().isoformat(),
            "price_data": {"success": False, "message": "", "records": 0},
            "financial_data": {"success": False, "message": "", "statements": 0},
            "files_saved": [],
            "total_time": 0
        }

        try:
            formatted_ticker = self._validate_ticker(ticker)
            results["formatted_ticker"] = formatted_ticker

            # Download price data
            price_success, price_data, price_msg = self.download_price_data(ticker)
            results["price_data"]["success"] = price_success
            results["price_data"]["message"] = price_msg

            if price_success and price_data is not None:
                results["price_data"]["records"] = len(price_data)
                if self.save_price_data(ticker, price_data):
                    results["files_saved"].append("price_data.csv")

            self._rate_limit()

            # Download financial data
            financial_success, financial_data, financial_msg = self.download_financial_data(ticker)
            results["financial_data"]["success"] = financial_success
            results["financial_data"]["message"] = financial_msg

            if financial_success and financial_data is not None:
                stmt_count = len(financial_data.get("financial_statements", {}))
                results["financial_data"]["statements"] = stmt_count
                if self.save_financial_data(ticker, financial_data):
                    results["files_saved"].append("financial_data.json")

            # Generate summary report
            if self.generate_summary_report(ticker, price_success, financial_success,
                                          price_data, financial_data):
                results["files_saved"].append("summary_report.txt")

            results["total_time"] = time.time() - start_time
            results["end_time"] = datetime.now().isoformat()

            logger.info(f"✅ Completed download for {formatted_ticker} in {results['total_time']:.2f}s")
            return results

        except Exception as e:
            error_msg = f"Failed to download data for {ticker}: {str(e)}"
            logger.error(error_msg)
            results["error"] = error_msg
            results["total_time"] = time.time() - start_time
            return results

    def download_multiple_tickers(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Download data for multiple tickers.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary with results for all tickers
        """
        logger.info(f"🚀 Starting batch download for {len(tickers)} tickers")
        start_time = time.time()

        batch_results = {
            "batch_start_time": datetime.now().isoformat(),
            "total_tickers": len(tickers),
            "successful_downloads": 0,
            "failed_downloads": 0,
            "ticker_results": {},
            "summary": {}
        }

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"📊 Processing ticker {i}/{len(tickers)}: {ticker}")

            try:
                ticker_result = self.download_single_ticker(ticker)
                batch_results["ticker_results"][ticker] = ticker_result

                if (ticker_result["price_data"]["success"] or
                    ticker_result["financial_data"]["success"]):
                    batch_results["successful_downloads"] += 1
                else:
                    batch_results["failed_downloads"] += 1

            except Exception as e:
                logger.error(f"❌ Failed to process {ticker}: {e}")
                batch_results["ticker_results"][ticker] = {"error": str(e)}
                batch_results["failed_downloads"] += 1

            # Rate limiting between tickers
            if i < len(tickers):
                self._rate_limit()

        batch_results["total_time"] = time.time() - start_time
        batch_results["batch_end_time"] = datetime.now().isoformat()

        # Generate batch summary
        batch_results["summary"] = {
            "success_rate": batch_results["successful_downloads"] / len(tickers) * 100,
            "average_time_per_ticker": batch_results["total_time"] / len(tickers),
            "total_price_records": sum(
                result.get("price_data", {}).get("records", 0)
                for result in batch_results["ticker_results"].values()
            ),
            "total_financial_statements": sum(
                result.get("financial_data", {}).get("statements", 0)
                for result in batch_results["ticker_results"].values()
            )
        }

        logger.info(f"✅ Batch download completed in {batch_results['total_time']:.2f}s")
        logger.info(f"📊 Success rate: {batch_results['summary']['success_rate']:.1f}%")

        return batch_results


def main():
    """
    Example usage of the HK Financial Data Downloader.
    """
    # Configuration
    config = DownloadConfig(
        start_date="2023-01-01",
        end_date="2024-12-31",
        output_dir="hk_stock_data",
        rate_limit_delay=1.5,
        max_retries=3,
        include_dividends=True,
        include_splits=True
    )

    # Initialize downloader
    downloader = HKFinancialDataDownloader(config)

    # Example 1: Download single ticker
    print("🔍 Example 1: Single Ticker Download")
    print("=" * 50)

    result = downloader.download_single_ticker("0700.HK")  # Tencent
    print(f"✅ Download completed for {result.get('formatted_ticker')}")
    print(f"📊 Price records: {result['price_data']['records']}")
    print(f"📋 Financial statements: {result['financial_data']['statements']}")
    print(f"⏱️ Total time: {result['total_time']:.2f}s")
    print()

    # Example 2: Download multiple tickers
    print("🔍 Example 2: Multiple Tickers Download")
    print("=" * 50)

    hk_tickers = ["0001.HK", "0005.HK", "0700.HK", "0941.HK", "1299.HK"]
    batch_result = downloader.download_multiple_tickers(hk_tickers)

    print(f"✅ Batch download completed")
    print(f"📊 Success rate: {batch_result['summary']['success_rate']:.1f}%")
    print(f"📈 Total price records: {batch_result['summary']['total_price_records']}")
    print(f"📋 Total financial statements: {batch_result['summary']['total_financial_statements']}")
    print(f"⏱️ Total time: {batch_result['total_time']:.2f}s")

    # Save batch results
    batch_file = Path(config.output_dir) / "batch_download_results.json"
    with open(batch_file, 'w') as f:
        json.dump(batch_result, f, indent=2, default=str)
    print(f"💾 Batch results saved to: {batch_file}")


if __name__ == "__main__":
    main()
