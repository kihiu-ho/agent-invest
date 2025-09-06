"""
TipRanks Data Models for Analyst Forecasts and Price Targets

This module defines structured data models for parsing and storing TipRanks analyst forecast data
including ratings, price targets, earnings forecasts, and analyst information.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AnalystRating(Enum):
    """Analyst rating categories."""
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


class ConsensusRating(Enum):
    """Overall consensus rating categories."""
    STRONG_BUY = "Strong Buy"
    MODERATE_BUY = "Moderate Buy"
    HOLD = "Hold"
    MODERATE_SELL = "Moderate Sell"
    STRONG_SELL = "Strong Sell"


@dataclass
class AnalystRatingsSummary:
    """Summary of analyst ratings and consensus."""
    ticker: str
    total_analysts: int
    buy_count: int
    hold_count: int
    sell_count: int
    strong_buy_count: int = 0
    strong_sell_count: int = 0
    consensus_rating: ConsensusRating = ConsensusRating.HOLD
    consensus_confidence: float = 0.0
    last_updated: Optional[datetime] = None
    
    @property
    def buy_percentage(self) -> float:
        """Calculate percentage of Buy ratings."""
        return (self.buy_count / max(1, self.total_analysts)) * 100
    
    @property
    def hold_percentage(self) -> float:
        """Calculate percentage of Hold ratings."""
        return (self.hold_count / max(1, self.total_analysts)) * 100
    
    @property
    def sell_percentage(self) -> float:
        """Calculate percentage of Sell ratings."""
        return (self.sell_count / max(1, self.total_analysts)) * 100


@dataclass
class PriceTarget:
    """12-month price target data."""
    ticker: str
    current_price: float
    average_target: float
    high_target: float
    low_target: float
    currency: str = "HK$"
    upside_potential: Optional[float] = None
    target_count: int = 0
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate upside potential after initialization."""
        if self.current_price and self.average_target:
            self.upside_potential = ((self.average_target - self.current_price) / self.current_price) * 100


@dataclass
class IndividualAnalystForecast:
    """Individual analyst forecast data."""
    analyst_name: str
    firm_name: str
    rating: AnalystRating
    price_target: float
    currency: str = "HK$"
    forecast_date: Optional[datetime] = None
    revision_date: Optional[datetime] = None
    accuracy_score: Optional[float] = None
    success_rate: Optional[float] = None
    track_record: Optional[str] = None


@dataclass
class EarningsForecast:
    """Earnings forecast data."""
    ticker: str
    period: str  # "Q1 2025", "FY 2025", etc.
    eps_estimate: float
    eps_high: Optional[float] = None
    eps_low: Optional[float] = None
    currency: str = "HK$"
    analyst_count: int = 0
    beat_rate: Optional[float] = None
    surprise_history: Optional[List[float]] = None
    last_updated: Optional[datetime] = None


@dataclass
class SalesForecast:
    """Sales/Revenue forecast data."""
    ticker: str
    period: str  # "Q1 2025", "FY 2025", etc.
    sales_estimate: float
    sales_high: Optional[float] = None
    sales_low: Optional[float] = None
    currency: str = "HK$"
    analyst_count: int = 0
    beat_rate: Optional[float] = None
    growth_rate: Optional[float] = None
    last_updated: Optional[datetime] = None


@dataclass
class RecommendationTrend:
    """Monthly recommendation trend data."""
    month: str  # "Aug 25", "Jul 25", etc.
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    total: int = 0
    
    @property
    def bullish_percentage(self) -> float:
        """Calculate percentage of bullish ratings (Strong Buy + Buy)."""
        return ((self.strong_buy + self.buy) / max(1, self.total)) * 100


@dataclass
class TipRanksAnalystData:
    """Complete TipRanks analyst forecast data structure."""
    ticker: str
    ratings_summary: AnalystRatingsSummary
    price_target: PriceTarget
    individual_forecasts: List[IndividualAnalystForecast]
    earnings_forecasts: List[EarningsForecast]
    sales_forecasts: List[SalesForecast]
    recommendation_trends: List[RecommendationTrend]
    data_quality_score: float = 0.0
    last_updated: Optional[datetime] = None
    source_urls: List[str] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.source_urls is None:
            self.source_urls = []
        if self.last_updated is None:
            self.last_updated = datetime.now()


class TipRanksDataParser:
    """Parser for extracting structured data from TipRanks web scraping results."""
    
    @staticmethod
    def parse_analyst_ratings(tipranks_data: Dict[str, Any]) -> Optional[AnalystRatingsSummary]:
        """
        Parse analyst ratings summary from TipRanks data.
        
        Args:
            tipranks_data: Raw TipRanks web scraping data
            
        Returns:
            AnalystRatingsSummary object or None if parsing fails
        """
        try:
            # Extract rating counts from TipRanks data structure
            # This would be implemented based on actual TipRanks HTML structure
            return None
        except Exception as e:
            return None
    
    @staticmethod
    def parse_price_targets(tipranks_data: Dict[str, Any], current_price: float) -> Optional[PriceTarget]:
        """
        Parse price target data from TipRanks data.
        
        Args:
            tipranks_data: Raw TipRanks web scraping data
            current_price: Current stock price for upside calculation
            
        Returns:
            PriceTarget object or None if parsing fails
        """
        try:
            # Extract price target data from TipRanks data structure
            # This would be implemented based on actual TipRanks HTML structure
            return None
        except Exception as e:
            return None
    
    @staticmethod
    def parse_individual_forecasts(tipranks_data: Dict[str, Any]) -> List[IndividualAnalystForecast]:
        """
        Parse individual analyst forecasts from TipRanks data.
        
        Args:
            tipranks_data: Raw TipRanks web scraping data
            
        Returns:
            List of IndividualAnalystForecast objects
        """
        try:
            # Extract individual analyst data from TipRanks data structure
            # This would be implemented based on actual TipRanks HTML structure
            return []
        except Exception as e:
            return []
    
    @staticmethod
    def parse_earnings_forecasts(tipranks_data: Dict[str, Any]) -> List[EarningsForecast]:
        """
        Parse earnings forecast data from TipRanks data.
        
        Args:
            tipranks_data: Raw TipRanks web scraping data
            
        Returns:
            List of EarningsForecast objects
        """
        try:
            # Extract earnings forecast data from TipRanks data structure
            # This would be implemented based on actual TipRanks HTML structure
            return []
        except Exception as e:
            return []
    
    @staticmethod
    def parse_sales_forecasts(tipranks_data: Dict[str, Any]) -> List[SalesForecast]:
        """
        Parse sales forecast data from TipRanks data.
        
        Args:
            tipranks_data: Raw TipRanks web scraping data
            
        Returns:
            List of SalesForecast objects
        """
        try:
            # Extract sales forecast data from TipRanks data structure
            # This would be implemented based on actual TipRanks HTML structure
            return []
        except Exception as e:
            return []
    
    @staticmethod
    def parse_recommendation_trends(tipranks_data: Dict[str, Any]) -> List[RecommendationTrend]:
        """
        Parse recommendation trend data from TipRanks data.
        
        Args:
            tipranks_data: Raw TipRanks web scraping data
            
        Returns:
            List of RecommendationTrend objects
        """
        try:
            # Extract recommendation trend data from TipRanks data structure
            # This would be implemented based on actual TipRanks HTML structure
            return []
        except Exception as e:
            return []
