"""
News Analysis Agent for Hong Kong Stock Financial Analysis System

This module provides comprehensive financial news analysis capabilities including:
- Real-time news collection from Yahoo Finance News API
- Sentiment analysis and impact assessment
- News-based investment insights generation
- Integration with Bulls Say/Bears Say analysis
"""

import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import re
logger = logging.getLogger(__name__)

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available. Sentiment analysis will use fallback method.")


class NewsAnalysisAgent:
    """
    News Analysis Agent for collecting and analyzing financial news.
    """
    
    def __init__(self, max_news_items: int = 20, days_back: int = 30):
        """
        Initialize the News Analysis Agent.
        
        Args:
            max_news_items: Maximum number of news items to collect
            days_back: Number of days to look back for news
        """
        self.max_news_items = max_news_items
        self.days_back = days_back
        
    def analyze_ticker_news(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive news analysis for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing news analysis results
        """
        try:
            logger.info(f"🔍 Starting news analysis for {ticker}")
            
            # Get news data from Yahoo Finance
            stock = yf.Ticker(ticker)
            news_data = stock.news
            
            if not news_data:
                logger.warning(f"No news data available for {ticker}")
                return {"success": False, "error": "No news data available"}
            
            # Process news articles
            processed_news = self._process_news_articles(news_data, ticker)
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_news_sentiment(processed_news)
            
            # Generate investment insights
            investment_insights = self._generate_investment_insights(processed_news, sentiment_analysis)
            
            # Create comprehensive news analysis
            news_analysis = {
                "ticker": ticker,
                "success": True,
                "analysis_date": datetime.now().isoformat(),
                "news_articles": processed_news[:self.max_news_items],
                "sentiment_analysis": sentiment_analysis,
                "investment_insights": investment_insights,
                "news_summary": self._generate_news_summary(processed_news, sentiment_analysis)
            }
            
            logger.info(f"✅ News analysis completed for {ticker} - {len(processed_news)} articles analyzed")
            return news_analysis
            
        except Exception as e:
            logger.error(f"❌ News analysis failed for {ticker}: {e}")
            return {"success": False, "error": str(e), "ticker": ticker}
    
    def _process_news_articles(self, news_data: List[Dict], ticker: str) -> List[Dict[str, Any]]:
        """Process raw news articles from Yahoo Finance."""
        processed_articles = []
        cutoff_date = datetime.now() - timedelta(days=self.days_back)
        
        for article in news_data:
            try:
                # Extract article information
                title = article.get('title', '')
                summary = article.get('summary', '')
                link = article.get('link', '')
                publisher = article.get('publisher', 'Unknown')
                publish_time = article.get('providerPublishTime', 0)
                
                # Convert timestamp to datetime
                if publish_time:
                    publish_date = datetime.fromtimestamp(publish_time)
                else:
                    publish_date = datetime.now()
                
                # Filter by date
                if publish_date < cutoff_date:
                    continue
                
                # Analyze article sentiment
                article_sentiment = self._analyze_article_sentiment(title, summary)
                
                # Assess relevance to ticker
                relevance_score = self._assess_article_relevance(title, summary, ticker)
                
                processed_article = {
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "publisher": publisher,
                    "publish_date": publish_date.isoformat(),
                    "sentiment": article_sentiment,
                    "relevance_score": relevance_score,
                    "impact_potential": self._assess_impact_potential(title, summary, article_sentiment)
                }
                
                processed_articles.append(processed_article)
                
            except Exception as e:
                logger.warning(f"Error processing article: {e}")
                continue
        
        # Sort by relevance and recency
        processed_articles.sort(key=lambda x: (x['relevance_score'], x['publish_date']), reverse=True)
        
        return processed_articles
    
    def _analyze_article_sentiment(self, title: str, summary: str) -> Dict[str, Any]:
        """Analyze sentiment of a single article."""
        try:
            # Combine title and summary for analysis
            text = f"{title}. {summary}"
            
            if TEXTBLOB_AVAILABLE:
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
            else:
                # Fallback sentiment analysis using keyword-based approach
                polarity = self._fallback_sentiment_analysis(text)
                subjectivity = 0.5  # Default subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment_label = "Positive"
            elif polarity < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            return {
                "label": sentiment_label,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "confidence": abs(polarity)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return {
                "label": "Neutral",
                "polarity": 0.0,
                "subjectivity": 0.0,
                "confidence": 0.0
            }

    def _fallback_sentiment_analysis(self, text: str) -> float:
        """Fallback sentiment analysis using keyword-based approach when TextBlob is unavailable."""
        text_lower = text.lower()

        # Positive keywords
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'strong', 'growth', 'profit', 'gain',
            'increase', 'rise', 'up', 'bullish', 'buy', 'upgrade', 'outperform', 'beat',
            'success', 'improve', 'better', 'optimistic', 'confident', 'robust'
        ]

        # Negative keywords
        negative_words = [
            'bad', 'poor', 'negative', 'weak', 'decline', 'loss', 'decrease', 'fall',
            'down', 'bearish', 'sell', 'downgrade', 'underperform', 'miss', 'fail',
            'worse', 'pessimistic', 'concern', 'risk', 'challenge', 'problem'
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0

        # Calculate polarity score (-1 to 1)
        net_sentiment = positive_count - negative_count
        polarity = max(-1.0, min(1.0, net_sentiment / max(total_words * 0.1, 1)))

        return polarity
    
    def _assess_article_relevance(self, title: str, summary: str, ticker: str) -> float:
        """Assess how relevant an article is to the specific ticker."""
        text = f"{title} {summary}".lower()
        
        # Extract company name from ticker (remove .HK suffix)
        base_ticker = ticker.replace('.HK', '').replace('.hk', '')
        
        relevance_score = 0.0
        
        # Direct ticker mention
        if base_ticker.lower() in text:
            relevance_score += 0.5
        
        # Company-specific keywords (for HSBC example)
        company_keywords = {
            '0005': ['hsbc', 'hongkong shanghai banking', 'hang seng'],
            '0700': ['tencent', 'wechat', 'qq'],
            '0941': ['china mobile', 'mobile', 'telecom'],
            '0388': ['hkex', 'hong kong exchanges', 'stock exchange']
        }
        
        if base_ticker in company_keywords:
            for keyword in company_keywords[base_ticker]:
                if keyword in text:
                    relevance_score += 0.3
        
        # Sector keywords
        sector_keywords = ['banking', 'financial', 'hong kong', 'china', 'market', 'stock', 'shares']
        for keyword in sector_keywords:
            if keyword in text:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _assess_impact_potential(self, title: str, summary: str, sentiment: Dict) -> str:
        """Assess the potential market impact of the news."""
        text = f"{title} {summary}".lower()
        
        # High impact keywords
        high_impact_keywords = [
            'earnings', 'profit', 'loss', 'revenue', 'acquisition', 'merger',
            'ceo', 'management', 'regulatory', 'investigation', 'lawsuit',
            'dividend', 'buyback', 'guidance', 'outlook', 'forecast'
        ]
        
        # Medium impact keywords
        medium_impact_keywords = [
            'partnership', 'contract', 'expansion', 'investment', 'upgrade',
            'downgrade', 'analyst', 'rating', 'target', 'recommendation'
        ]
        
        # Check for high impact
        for keyword in high_impact_keywords:
            if keyword in text:
                return "High"
        
        # Check for medium impact
        for keyword in medium_impact_keywords:
            if keyword in text:
                return "Medium"
        
        # Consider sentiment strength
        if abs(sentiment.get('polarity', 0)) > 0.5:
            return "Medium"
        
        return "Low"
    
    def _analyze_news_sentiment(self, articles: List[Dict]) -> Dict[str, Any]:
        """Analyze overall sentiment from all articles."""
        if not articles:
            return {
                "overall_sentiment": "Neutral",
                "sentiment_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "confidence": 0.0
            }
        
        sentiments = [article['sentiment'] for article in articles]
        
        positive_count = sum(1 for s in sentiments if s['label'] == 'Positive')
        negative_count = sum(1 for s in sentiments if s['label'] == 'Negative')
        neutral_count = sum(1 for s in sentiments if s['label'] == 'Neutral')
        
        # Calculate weighted sentiment score
        total_polarity = sum(s['polarity'] for s in sentiments)
        avg_polarity = total_polarity / len(sentiments)
        
        # Determine overall sentiment
        if avg_polarity > 0.1:
            overall_sentiment = "Positive"
        elif avg_polarity < -0.1:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(avg_polarity, 3),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total_articles": len(articles),
            "confidence": round(abs(avg_polarity), 3)
        }

    def _generate_investment_insights(self, articles: List[Dict], sentiment_analysis: Dict) -> Dict[str, Any]:
        """Generate investment insights based on news analysis."""
        bullish_factors = []
        bearish_factors = []
        catalysts = []
        risks = []

        for article in articles:
            if article['relevance_score'] < 0.3:
                continue

            title = article['title']
            sentiment = article['sentiment']
            impact = article['impact_potential']

            # Generate bullish factors
            if sentiment['label'] == 'Positive' and impact in ['High', 'Medium']:
                bullish_factors.append({
                    "factor": f"Positive news: {title[:100]}...",
                    "source": article['publisher'],
                    "date": article['publish_date'],
                    "impact": impact,
                    "sentiment_score": sentiment['polarity']
                })

            # Generate bearish factors
            elif sentiment['label'] == 'Negative' and impact in ['High', 'Medium']:
                bearish_factors.append({
                    "factor": f"Negative news: {title[:100]}...",
                    "source": article['publisher'],
                    "date": article['publish_date'],
                    "impact": impact,
                    "sentiment_score": sentiment['polarity']
                })

            # Identify potential catalysts
            if impact == 'High':
                catalysts.append({
                    "event": title[:100],
                    "type": "News Event",
                    "expected_impact": sentiment['label'],
                    "timeline": "Short-term",
                    "source": article['publisher']
                })

            # Identify risks
            if sentiment['label'] == 'Negative' and 'regulatory' in title.lower():
                risks.append({
                    "risk": f"Regulatory concern: {title[:100]}...",
                    "severity": impact,
                    "source": article['publisher']
                })

        return {
            "bullish_factors": bullish_factors[:5],  # Top 5
            "bearish_factors": bearish_factors[:5],  # Top 5
            "potential_catalysts": catalysts[:3],    # Top 3
            "identified_risks": risks[:3],          # Top 3
            "news_momentum": self._assess_news_momentum(articles),
            "sector_sentiment": self._assess_sector_sentiment(articles)
        }

    def _assess_news_momentum(self, articles: List[Dict]) -> Dict[str, Any]:
        """Assess news momentum over time."""
        if not articles:
            return {"trend": "Neutral", "momentum_score": 0.0}

        # Sort articles by date
        sorted_articles = sorted(articles, key=lambda x: x['publish_date'])

        # Calculate momentum based on recent vs older news sentiment
        recent_articles = [a for a in sorted_articles[-7:]]  # Last 7 articles
        older_articles = [a for a in sorted_articles[:-7]]   # Older articles

        if not recent_articles:
            return {"trend": "Neutral", "momentum_score": 0.0}

        recent_sentiment = sum(a['sentiment']['polarity'] for a in recent_articles) / len(recent_articles)

        if older_articles:
            older_sentiment = sum(a['sentiment']['polarity'] for a in older_articles) / len(older_articles)
            momentum_score = recent_sentiment - older_sentiment
        else:
            momentum_score = recent_sentiment

        if momentum_score > 0.1:
            trend = "Improving"
        elif momentum_score < -0.1:
            trend = "Deteriorating"
        else:
            trend = "Stable"

        return {
            "trend": trend,
            "momentum_score": round(momentum_score, 3),
            "recent_sentiment": round(recent_sentiment, 3)
        }

    def _assess_sector_sentiment(self, articles: List[Dict]) -> Dict[str, Any]:
        """Assess broader sector sentiment from news."""
        sector_keywords = {
            "banking": ["bank", "banking", "financial", "credit", "loan"],
            "technology": ["tech", "technology", "digital", "ai", "software"],
            "healthcare": ["health", "medical", "pharma", "drug", "hospital"],
            "energy": ["energy", "oil", "gas", "renewable", "power"]
        }

        sector_mentions = {}
        sector_sentiment = {}

        for article in articles:
            text = f"{article['title']} {article['summary']}".lower()

            for sector, keywords in sector_keywords.items():
                mentions = sum(1 for keyword in keywords if keyword in text)
                if mentions > 0:
                    if sector not in sector_mentions:
                        sector_mentions[sector] = 0
                        sector_sentiment[sector] = []

                    sector_mentions[sector] += mentions
                    sector_sentiment[sector].append(article['sentiment']['polarity'])

        # Calculate average sentiment by sector
        sector_analysis = {}
        for sector, sentiments in sector_sentiment.items():
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                sector_analysis[sector] = {
                    "mentions": sector_mentions[sector],
                    "sentiment": round(avg_sentiment, 3),
                    "articles_count": len(sentiments)
                }

        return sector_analysis

    def _generate_news_summary(self, articles: List[Dict], sentiment_analysis: Dict) -> Dict[str, Any]:
        """Generate a comprehensive news summary."""
        if not articles:
            return {"summary": "No recent news available"}

        # Key headlines
        top_headlines = [
            {
                "title": article['title'],
                "sentiment": article['sentiment']['label'],
                "impact": article['impact_potential'],
                "date": article['publish_date'][:10],  # Date only
                "publisher": article['publisher']
            }
            for article in articles[:5]  # Top 5 headlines
        ]

        # News volume analysis
        recent_count = len([a for a in articles if
                          (datetime.now() - datetime.fromisoformat(a['publish_date'].replace('Z', '+00:00').replace('+00:00', ''))).days <= 7])

        return {
            "total_articles": len(articles),
            "recent_articles_7days": recent_count,
            "overall_sentiment": sentiment_analysis['overall_sentiment'],
            "sentiment_distribution": {
                "positive": sentiment_analysis['positive_count'],
                "negative": sentiment_analysis['negative_count'],
                "neutral": sentiment_analysis['neutral_count']
            },
            "top_headlines": top_headlines,
            "news_volume": "High" if len(articles) > 15 else "Medium" if len(articles) > 5 else "Low",
            "analysis_quality": "High" if len(articles) >= 10 else "Medium" if len(articles) >= 5 else "Low"
        }
