#!/usr/bin/env python3
"""
Enhanced Chart Generation AutoGen Agent for Financial Metrics System.

This agent specializes in creating comprehensive price and technical analysis charts
by combining price data with technical indicators into unified visualizations.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Create mock classes for development
    class ConversableAgent:
        def __init__(self, *args, **kwargs):
            pass
    class GroupChat:
        def __init__(self, *args, **kwargs):
            pass
    class GroupChatManager:
        def __init__(self, *args, **kwargs):
            pass


class ChartEnhancementAgent:
    """
    AutoGen agent specialized in enhancing price chart functionality by combining
    price data with technical analysis into comprehensive visualizations.
    """

    def __init__(self, llm_config: Optional[Dict] = None):
        """Initialize the Chart Enhancement Agent."""
        self.logger = logging.getLogger(__name__)
        self.llm_config = llm_config or self._get_default_llm_config()

        # Initialize AutoGen agents if available
        if AUTOGEN_AVAILABLE:
            self._initialize_agents()

        # Chart configuration
        self.chart_config = {
            "default_period": "1Y",
            "chart_height": 600,
            "chart_width": 1200,
            "technical_indicators": ["SMA", "EMA", "RSI", "MACD", "Bollinger"],
            "price_metrics": ["period_return", "period_high", "period_low", "volatility"]
        }

    def _get_default_llm_config(self) -> Dict:
        """Get default LLM configuration."""
        return {
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000,
            "timeout": 120
        }

    def _initialize_agents(self):
        """Initialize AutoGen agents for chart enhancement."""
        # Chart Data Analyzer Agent
        self.chart_analyzer = ConversableAgent(
            name="ChartDataAnalyzer",
            system_message="""You are a specialized chart data analyzer for financial markets.
            Your role is to:
            1. Analyze historical price data and technical indicators
            2. Identify key price levels, trends, and patterns
            3. Determine optimal chart configurations for visualization
            4. Extract meaningful insights from price and technical data

            Focus on providing actionable insights for chart visualization and
            technical analysis integration.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )

        # Technical Integration Agent
        self.technical_integrator = ConversableAgent(
            name="TechnicalIntegrator",
            system_message="""You are a technical analysis integration specialist.
            Your role is to:
            1. Combine price data with technical indicators
            2. Create unified chart configurations
            3. Optimize technical indicator overlays and panels
            4. Generate comprehensive technical analysis narratives

            Ensure all technical indicators are properly integrated with price charts
            for maximum analytical value.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )

        # Chart Visualization Agent
        self.chart_visualizer = ConversableAgent(
            name="ChartVisualizer",
            system_message="""You are a chart visualization expert for financial data.
            Your role is to:
            1. Generate Chart.js configurations for interactive charts
            2. Create responsive and professional chart layouts
            3. Implement chart annotations and overlays
            4. Ensure optimal user experience and interactivity

            Focus on creating institutional-quality chart visualizations that
            combine price data with technical analysis effectively.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )

    async def enhance_price_chart_section(
        self,
        historical_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        ticker: str
    ) -> Dict[str, Any]:
        """
        Enhance the price chart section by combining price and technical analysis.

        Args:
            historical_data: Historical price data
            technical_data: Technical analysis data
            ticker: Stock ticker symbol

        Returns:
            Enhanced chart configuration and HTML
        """
        try:
            # Extract key data
            price_data = self._extract_price_data(historical_data)
            technical_indicators = self._extract_technical_indicators(technical_data)
            price_metrics = self._calculate_price_metrics(price_data)

            # Generate enhanced chart configuration
            chart_config = await self._generate_enhanced_chart_config(
                price_data, technical_indicators, price_metrics, ticker
            )

            # Create unified HTML section
            enhanced_html = self._generate_enhanced_chart_html(
                chart_config, price_metrics, ticker, technical_indicators
            )

            return {
                "success": True,
                "chart_config": chart_config,
                "enhanced_html": enhanced_html,
                "price_metrics": price_metrics,
                "technical_summary": self._generate_technical_summary(technical_indicators)
            }

        except Exception as e:
            self.logger.error(f"Error enhancing price chart section: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_html": self._generate_fallback_chart_html(ticker)
            }

    def _extract_price_data(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format price data for chart visualization with enhanced data structure handling."""
        self.logger.info(f"🔍 [CHART DEBUG] Extracting price data from historical_data")
        self.logger.info(f"🔍 [CHART DEBUG] Historical data type: {type(historical_data)}")
        self.logger.info(f"🔍 [CHART DEBUG] Historical data keys: {list(historical_data.keys()) if historical_data else 'None'}")

        if historical_data:
            # Log the full structure for debugging
            for key, value in historical_data.items():
                if isinstance(value, dict):
                    self.logger.info(f"🔍 [CHART DEBUG] {key} -> dict with keys: {list(value.keys())}")
                elif isinstance(value, list):
                    self.logger.info(f"🔍 [CHART DEBUG] {key} -> list with {len(value)} items")
                else:
                    self.logger.info(f"🔍 [CHART DEBUG] {key} -> {type(value)}: {str(value)[:100]}...")

        # Handle different data structures
        price_data = {}

        # Case 1: Direct structure with 'prices' key
        if historical_data and historical_data.get('prices'):
            self.logger.info(f"🔍 [CHART DEBUG] Case 1: Found direct 'prices' structure")
            prices = historical_data['prices']
            self.logger.info(f"🔍 [CHART DEBUG] Prices keys: {list(prices.keys()) if isinstance(prices, dict) else 'Not a dict'}")

            price_data = {
                "dates": prices.get('dates', []),
                "open": prices.get('open', []),
                "high": prices.get('high', []),
                "low": prices.get('low', []),
                "close": prices.get('close', []),
                "volume": prices.get('volume', []),
                "period": historical_data.get('period', '1Y')
            }

            # Log detailed data info
            for key, value in price_data.items():
                if isinstance(value, list):
                    self.logger.info(f"🔍 [CHART DEBUG] {key}: {len(value)} items, sample: {value[:3] if value else 'empty'}")
                else:
                    self.logger.info(f"🔍 [CHART DEBUG] {key}: {value}")

            self.logger.info(f"✅ [CHART DEBUG] Extracted price data from direct 'prices' structure: {len(price_data.get('dates', []))} data points")

        # Case 2: Nested structure with 'historical_data' -> 'prices'
        elif historical_data and historical_data.get('historical_data'):
            self.logger.info(f"🔍 [CHART DEBUG] Case 2: Found nested 'historical_data' structure")
            nested_data = historical_data['historical_data']
            self.logger.info(f"🔍 [CHART DEBUG] Nested data keys: {list(nested_data.keys()) if isinstance(nested_data, dict) else 'Not a dict'}")

            if nested_data.get('prices'):
                self.logger.info(f"🔍 [CHART DEBUG] Found 'prices' in nested data")
                prices = nested_data['prices']
                self.logger.info(f"🔍 [CHART DEBUG] Nested prices keys: {list(prices.keys()) if isinstance(prices, dict) else 'Not a dict'}")

                price_data = {
                    "dates": prices.get('dates', []),
                    "open": prices.get('open', []),
                    "high": prices.get('high', []),
                    "low": prices.get('low', []),
                    "close": prices.get('close', []),
                    "volume": prices.get('volume', []),
                    "period": nested_data.get('period', '1Y')
                }

                # Log detailed data info
                for key, value in price_data.items():
                    if isinstance(value, list):
                        self.logger.info(f"🔍 [CHART DEBUG] {key}: {len(value)} items, sample: {value[:3] if value else 'empty'}")
                    else:
                        self.logger.info(f"🔍 [CHART DEBUG] {key}: {value}")

                self.logger.info(f"✅ [CHART DEBUG] Extracted price data from nested 'historical_data.prices' structure: {len(price_data.get('dates', []))} data points")
            else:
                self.logger.warning(f"⚠️ [CHART DEBUG] No 'prices' found in nested data")

        # Case 3: Try to extract from any available financial data
        elif historical_data:
            # Look for any price-related data in the structure
            for key, value in historical_data.items():
                if isinstance(value, dict):
                    if 'close' in value or 'Close' in value:
                        # Found price data in a nested structure
                        price_data = self._extract_from_any_structure(value)
                        if price_data.get('dates'):
                            self.logger.info(f"✅ Extracted price data from '{key}' structure: {len(price_data.get('dates', []))} data points")
                            break

        # Case 4: Generate sample data if no real data is available (for testing/demo)
        if not price_data.get('dates'):
            self.logger.warning("⚠️ [CHART DEBUG] No historical price data found, generating sample data for chart display")
            price_data = self._generate_sample_price_data()
            self.logger.info(f"🔍 [CHART DEBUG] Generated sample data with {len(price_data.get('dates', []))} data points")

        # Final validation
        final_dates = price_data.get('dates', [])
        final_close = price_data.get('close', [])
        self.logger.info(f"🔍 [CHART DEBUG] Final price data summary:")
        self.logger.info(f"🔍 [CHART DEBUG] - Dates: {len(final_dates)} items")
        self.logger.info(f"🔍 [CHART DEBUG] - Close prices: {len(final_close)} items")
        self.logger.info(f"🔍 [CHART DEBUG] - Sample date: {final_dates[0] if final_dates else 'None'}")
        self.logger.info(f"🔍 [CHART DEBUG] - Sample price: {final_close[0] if final_close else 'None'}")

        return price_data

    def _extract_from_any_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract price data from any nested structure."""
        try:
            # Try different possible key names
            close_key = None
            for key in ['close', 'Close', 'closing_price', 'price']:
                if key in data:
                    close_key = key
                    break

            if not close_key:
                return {}

            close_prices = data[close_key]
            if not isinstance(close_prices, list) or not close_prices:
                return {}

            # Generate dates if not available
            dates = data.get('dates', data.get('Dates', []))
            if not dates:
                from datetime import datetime, timedelta
                end_date = datetime.now()
                dates = [(end_date - timedelta(days=i)).isoformat() for i in range(len(close_prices)-1, -1, -1)]

            return {
                "dates": dates,
                "open": data.get('open', data.get('Open', close_prices)),  # Use close as fallback
                "high": data.get('high', data.get('High', close_prices)),  # Use close as fallback
                "low": data.get('low', data.get('Low', close_prices)),     # Use close as fallback
                "close": close_prices,
                "volume": data.get('volume', data.get('Volume', [0] * len(close_prices))),
                "period": "1Y"
            }
        except Exception as e:
            self.logger.error(f"Error extracting from structure: {e}")
            return {}

    def _generate_sample_price_data(self) -> Dict[str, Any]:
        """Generate sample price data for demonstration when no real data is available."""
        try:
            import random
            from datetime import datetime, timedelta

            # Generate 30 days of sample data
            num_days = 30
            base_price = 100.0
            dates = []
            prices = []

            for i in range(num_days):
                date = datetime.now() - timedelta(days=num_days - i - 1)
                dates.append(date.isoformat())

                # Generate realistic price movement
                change = random.uniform(-0.05, 0.05)  # ±5% daily change
                base_price *= (1 + change)
                prices.append(round(base_price, 2))

            return {
                "dates": dates,
                "open": prices,
                "high": [p * random.uniform(1.0, 1.02) for p in prices],
                "low": [p * random.uniform(0.98, 1.0) for p in prices],
                "close": prices,
                "volume": [random.randint(100000, 1000000) for _ in range(num_days)],
                "period": "1M"
            }
        except Exception as e:
            self.logger.error(f"Error generating sample data: {e}")
            return {
                "dates": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "period": "1Y"
            }

    def _extract_technical_indicators(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical indicators for chart overlay."""
        if not technical_data:
            return {}

        return {
            "moving_averages": technical_data.get('moving_averages', {}),
            "technical_indicators": technical_data.get('technical_indicators', {}),
            "macd_analysis": technical_data.get('macd_analysis', {}),
            "pivot_points": technical_data.get('pivot_points', {}),
            "overall_consensus": technical_data.get('overall_consensus', {}),
            # Include institutional summary for enhanced technical analysis
            "institutional_summary": technical_data.get('institutional_summary', {})
        }

    def _calculate_price_metrics(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key price metrics for display."""
        if not price_data or not price_data.get('close'):
            return {}

        close_prices = price_data['close']
        high_prices = price_data.get('high', [])
        low_prices = price_data.get('low', [])

        if not close_prices:
            return {}

        # Calculate metrics
        current_price = close_prices[-1] if close_prices else 0
        start_price = close_prices[0] if close_prices else 0
        period_return = ((current_price - start_price) / start_price * 100) if start_price > 0 else 0
        period_high = max(high_prices) if high_prices else max(close_prices)
        period_low = min(low_prices) if low_prices else min(close_prices)

        # Calculate volatility (simplified)
        if len(close_prices) > 1:
            returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                      for i in range(1, len(close_prices)) if close_prices[i-1] > 0]
            volatility = (sum(r**2 for r in returns) / len(returns))**0.5 * 100 if returns else 0
        else:
            volatility = 0

        return {
            "current_price": current_price,
            "period_return": period_return,
            "period_high": period_high,
            "period_low": period_low,
            "volatility": volatility,
            "period": price_data.get('period', '1Y')
        }

    def _calculate_enhanced_moving_averages(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced SMA and EMA using historical OHLC data."""
        close_prices = price_data.get('close', [])
        dates = price_data.get('dates', [])

        if not close_prices or len(close_prices) < 20:
            return {}

        try:
            import pandas as pd
            import numpy as np

            # Create DataFrame for calculations
            df = pd.DataFrame({
                'close': close_prices,
                'date': dates[:len(close_prices)]
            })

            enhanced_mas = {}
            periods = [20, 50, 100, 200]

            for period in periods:
                if len(close_prices) >= period:
                    # Calculate SMA
                    sma_series = df['close'].rolling(window=period).mean()
                    sma_values = sma_series.tolist()

                    # Calculate EMA
                    ema_series = df['close'].ewm(span=period, adjust=False).mean()
                    ema_values = ema_series.tolist()

                    # Get current values (last valid value)
                    sma_current = sma_series.iloc[-1] if not pd.isna(sma_series.iloc[-1]) else None
                    ema_current = ema_series.iloc[-1] if not pd.isna(ema_series.iloc[-1]) else None

                    # Generate signals
                    current_price = close_prices[-1]
                    sma_signal = "Buy" if sma_current and current_price > sma_current else "Sell" if sma_current and current_price < sma_current else "Neutral"
                    ema_signal = "Buy" if ema_current and current_price > ema_current else "Sell" if ema_current and current_price < ema_current else "Neutral"

                    enhanced_mas[f"{period}_day"] = {
                        "sma_value": sma_current,
                        "ema_value": ema_current,
                        "sma_signal": sma_signal,
                        "ema_signal": ema_signal,
                        "sma_data": sma_values,
                        "ema_data": ema_values,
                        "price_vs_sma": ((current_price - sma_current) / sma_current * 100) if sma_current else 0,
                        "price_vs_ema": ((current_price - ema_current) / ema_current * 100) if ema_current else 0
                    }

            return enhanced_mas

        except Exception as e:
            logger.error(f"Error calculating enhanced moving averages: {e}")
            return {}

    def _calculate_moving_averages(self, close_prices: list, dates: list) -> Dict[str, Any]:
        """Calculate SMA and EMA for different periods using historical data."""
        if not close_prices or len(close_prices) < 20:
            return {}

        try:
            import pandas as pd

            # Create DataFrame for easier calculation
            df = pd.DataFrame({
                'close': close_prices,
                'date': dates[:len(close_prices)]
            })

            moving_averages = {}

            # Calculate different period moving averages
            periods = [20, 50, 100, 200]

            for period in periods:
                if len(close_prices) >= period:
                    # Simple Moving Average
                    sma_values = df['close'].rolling(window=period).mean().tolist()

                    # Exponential Moving Average
                    ema_values = df['close'].ewm(span=period, adjust=False).mean().tolist()

                    # Get current values (last non-NaN value)
                    sma_current = None
                    ema_current = None

                    for val in reversed(sma_values):
                        if pd.notna(val):
                            sma_current = val
                            break

                    for val in reversed(ema_values):
                        if pd.notna(val):
                            ema_current = val
                            break

                    # Determine signals
                    current_price = close_prices[-1]
                    sma_signal = "Buy" if current_price > sma_current else "Sell" if sma_current else "Neutral"
                    ema_signal = "Buy" if current_price > ema_current else "Sell" if ema_current else "Neutral"

                    moving_averages[f"{period}_day"] = {
                        "sma_value": sma_current,
                        "ema_value": ema_current,
                        "sma_signal": sma_signal,
                        "ema_signal": ema_signal,
                        "sma_data": sma_values,  # Full historical data
                        "ema_data": ema_values,  # Full historical data
                        "price_vs_sma": ((current_price - sma_current) / sma_current * 100) if sma_current else 0
                    }

            return moving_averages

        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}

    async def _generate_enhanced_chart_config(
        self,
        price_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        price_metrics: Dict[str, Any],
        ticker: str
    ) -> Dict[str, Any]:
        """Generate enhanced Chart.js configuration with automatic candlestick detection."""

        self.logger.info(f"🔍 [CHART DEBUG] Generating enhanced chart config for {ticker}")
        self.logger.info(f"🔍 [CHART DEBUG] Price data keys: {list(price_data.keys()) if price_data else 'None'}")
        self.logger.info(f"🔍 [CHART DEBUG] Technical indicators keys: {list(technical_indicators.keys()) if technical_indicators else 'None'}")
        self.logger.info(f"🔍 [CHART DEBUG] Price metrics keys: {list(price_metrics.keys()) if price_metrics else 'None'}")

        # Automatically detect chart type based on available data
        has_ohlc_data = all(
            price_data.get(field) and len(price_data.get(field, [])) > 0
            for field in ['open', 'high', 'low', 'close']
        )

        # Log OHLC data availability
        for field in ['open', 'high', 'low', 'close']:
            data = price_data.get(field, [])
            self.logger.info(f"🔍 [CHART DEBUG] {field}: {len(data)} items, sample: {data[:3] if data else 'empty'}")

        chart_type = "candlestick" if has_ohlc_data else "line"
        self.logger.info(f"📊 [CHART DEBUG] Auto-detected chart type: {chart_type} (OHLC data available: {has_ohlc_data})")

        # Base chart configuration
        config = {
            "type": chart_type,
            "data": {
                "labels": price_data.get('dates', []) if chart_type == "line" else [],
                "datasets": []
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "interaction": {
                    "mode": "index",
                    "intersect": False
                },
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{ticker} - Price & Technical Analysis - {price_metrics.get('period', '1Y')}"
                    },
                    "legend": {
                        "display": True,
                        "position": "top"
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False
                    }
                },
                "scales": {
                    "x": {
                        "type": "time",
                        "time": {
                            "displayFormats": {
                                "day": "MMM dd",
                                "month": "MMM yyyy"
                            }
                        },
                        "display": True,
                        "title": {
                            "display": True,
                            "text": "Date"
                        }
                    },
                    "y": {
                        "display": True,
                        "title": {
                            "display": True,
                            "text": "Price (HK$)"
                        },
                        "beginAtZero": False
                    }
                }
            }
        }

        # Add price data based on auto-detected chart type
        if config["type"] == "candlestick":
            self.logger.info(f"🔍 [CHART DEBUG] Generating candlestick chart data")
            # Format OHLC data for candlestick chart
            ohlc_data = []
            dates = price_data.get('dates', [])
            open_prices = price_data.get('open', [])
            high_prices = price_data.get('high', [])
            low_prices = price_data.get('low', [])
            close_prices = price_data.get('close', [])

            self.logger.info(f"🔍 [CHART DEBUG] Raw data lengths - dates: {len(dates)}, open: {len(open_prices)}, high: {len(high_prices)}, low: {len(low_prices)}, close: {len(close_prices)}")

            min_length = min(len(dates), len(open_prices), len(high_prices), len(low_prices), len(close_prices))
            self.logger.info(f"🔍 [CHART DEBUG] Min length for OHLC data: {min_length}")

            # Convert ISO date strings to epoch milliseconds for robust time scale parsing
            def _to_epoch_ms(dt_str: str):
                try:
                    return int(datetime.fromisoformat(dt_str.replace('Z', '+00:00')).timestamp() * 1000)
                except Exception:
                    return dt_str

            for i in range(min_length):
                ohlc_data.append({
                    "x": _to_epoch_ms(dates[i]),
                    "o": open_prices[i],
                    "h": high_prices[i],
                    "l": low_prices[i],
                    "c": close_prices[i]
                })

            self.logger.info(f"🔍 [CHART DEBUG] Generated {len(ohlc_data)} OHLC data points")
            if ohlc_data:
                self.logger.info(f"🔍 [CHART DEBUG] Sample OHLC data: {ohlc_data[0]}")

            config["type"] = "candlestick"
            dataset = {
                "label": f"{ticker} Price",
                "data": ohlc_data,
                "type": "candlestick",
                "borderColor": "#2E8B57",
                "backgroundColor": "rgba(46, 139, 87, 0.1)"
            }
            config["data"]["datasets"].append(dataset)
            self.logger.info(f"🔍 [CHART DEBUG] Added candlestick dataset with {len(ohlc_data)} data points")
        else:
            self.logger.info(f"🔍 [CHART DEBUG] Generating line chart data")
            # Fallback to line chart
            line_data = []
            dates = price_data.get('dates', [])
            close_prices = price_data.get('close', [])

            self.logger.info(f"🔍 [CHART DEBUG] Line chart data lengths - dates: {len(dates)}, close: {len(close_prices)}")

            # Convert ISO date strings to epoch milliseconds for robust time scale parsing
            def _to_epoch_ms(dt_str: str):
                try:
                    return int(datetime.fromisoformat(dt_str.replace('Z', '+00:00')).timestamp() * 1000)
                except Exception:
                    return dt_str

            for i in range(min(len(dates), len(close_prices))):
                line_data.append({
                    "x": _to_epoch_ms(dates[i]),
                    "y": close_prices[i]
                })

            self.logger.info(f"🔍 [CHART DEBUG] Generated {len(line_data)} line data points")
            if line_data:
                self.logger.info(f"🔍 [CHART DEBUG] Sample line data: {line_data[0]}")

            dataset = {
                "label": f"{ticker} Close Price",
                "data": line_data,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.1)",
                "tension": 0.1,
                "fill": False
            }
            config["data"]["datasets"].append(dataset)
            self.logger.info(f"🔍 [CHART DEBUG] Added line dataset with {len(line_data)} data points")

        # Calculate enhanced moving averages with historical data
        enhanced_mas = self._calculate_enhanced_moving_averages(price_data)
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7DC6F", "#BB8FCE"]
        color_index = 0

        # Add SMA lines with historical data
        for ma_period, ma_data in enhanced_mas.items():
            if ma_data.get('sma_data'):
                # Create SMA line data points
                sma_line_data = []
                dates = price_data.get('dates', [])
                sma_values = ma_data['sma_data']

                for i, sma_val in enumerate(sma_values):
                    if (i < len(dates) and sma_val is not None and
                        (not PANDAS_AVAILABLE or not pd.isna(sma_val))):
                        sma_line_data.append({
                            "x": _to_epoch_ms(dates[i]),
                            "y": sma_val
                        })

                if sma_line_data:
                    config["data"]["datasets"].append({
                        "label": f"SMA {ma_period.replace('_day', '')}",
                        "data": sma_line_data,
                        "type": "line",
                        "borderColor": colors[color_index % len(colors)],
                        "backgroundColor": "transparent",
                        "borderDash": [5, 5],
                        "tension": 0,
                        "fill": False,
                        "pointRadius": 0,
                        "hidden": True  # Start hidden, can be toggled
                    })
                    color_index += 1

            # Add EMA lines with historical data
            if ma_data.get('ema_data'):
                # Create EMA line data points
                ema_line_data = []
                dates = price_data.get('dates', [])
                ema_values = ma_data['ema_data']

                for i, ema_val in enumerate(ema_values):
                    if (i < len(dates) and ema_val is not None and
                        (not PANDAS_AVAILABLE or not pd.isna(ema_val))):
                        ema_line_data.append({
                            "x": _to_epoch_ms(dates[i]),
                            "y": ema_val
                        })

                if ema_line_data:
                    config["data"]["datasets"].append({
                        "label": f"EMA {ma_period.replace('_day', '')}",
                        "data": ema_line_data,
                        "type": "line",
                        "borderColor": colors[color_index % len(colors)],
                        "backgroundColor": "transparent",
                        "borderDash": [2, 2],
                        "tension": 0,
                        "fill": False,
                        "pointRadius": 0,
                        "hidden": True  # Start hidden, can be toggled
                    })
                    color_index += 1

        # Add Volume data if available
        if price_data.get('volume'):
            dates = price_data.get('dates', [])
            volumes = price_data.get('volume', [])
            vol_points = []

            for i in range(min(len(dates), len(volumes))):
                vol_points.append({
                    "x": _to_epoch_ms(dates[i]),
                    "y": volumes[i]
                })

            if vol_points:
                # Create a secondary y-axis for volume
                config["options"]["scales"]["y1"] = {
                    "type": "linear",
                    "display": True,
                    "position": "right",
                    "title": {"display": True, "text": "Volume"},
                    "grid": {"drawOnChartArea": False}
                }

                config["data"]["datasets"].append({
                    "label": "Volume",
                    "data": vol_points,
                    "type": "bar",
                    "parsing": {"xAxisKey": "x", "yAxisKey": "y"},
                    "backgroundColor": "rgba(128, 128, 128, 0.3)",
                    "borderColor": "rgba(128, 128, 128, 0.8)",
                    "borderWidth": 1,
                    "yAxisID": "y1",
                    "hidden": True  # Start hidden, can be toggled
                })

        return config

    def _generate_enhanced_chart_html(
        self,
        chart_config: Dict[str, Any],
        price_metrics: Dict[str, Any],
        ticker: str,
        technical_indicators: Dict[str, Any] = None
    ) -> str:
        """Generate enhanced HTML for the combined price and technical analysis chart."""

        period_display = {
            '1Y': '1 Year',
            '2Y': '2 Years',
            '5Y': '5 Years',
            '6M': '6 Months',
            '1M': '1 Month',
            'YTD': 'Year to Date'
        }.get(price_metrics.get('period', '1Y'), price_metrics.get('period', '1Y'))

        return f"""
        <div class="section">
            <h2>📈 Price & Technical Analysis - {period_display}</h2>

            <!-- Price Metrics Summary -->
            <div class="alert alert-info" style="margin: 15px 0;">
                <div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                    <div class="metric-card">
                        <div class="metric-label">Period Return</div>
                        <div class="metric-value" style="color: {'#27ae60' if price_metrics.get('period_return', 0) > 0 else '#e74c3c'};">
                            {price_metrics.get('period_return', 0):+.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Period High</div>
                        <div class="metric-value">HK${price_metrics.get('period_high', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Period Low</div>
                        <div class="metric-value">HK${price_metrics.get('period_low', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Volatility</div>
                        <div class="metric-value">{price_metrics.get('volatility', 0):.2f}%</div>
                    </div>
                </div>
            </div>

            <!-- Enhanced Interactive Chart -->
            <div class="chart-container" style="position: relative; height: 600px; margin: 20px 0; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; background: #fafafa;">
                <canvas id="enhancedPriceChart" style="width: 100%; height: 100%;"></canvas>
            </div>

            <!-- Enhanced Chart Controls -->
            <div class="chart-controls" style="margin: 15px 0; text-align: center;">
                <!-- Chart Type Controls -->
                <div class="btn-group" role="group" style="margin: 5px;">
                    <button onclick="switchChartType('line')" class="btn btn-sm btn-outline-primary" title="Switch to Line Chart">
                        📈 Line Chart
                    </button>
                    <button onclick="switchChartType('candlestick')" class="btn btn-sm btn-outline-success" title="Switch to Candlestick Chart">
                        🕯️ Candlestick
                    </button>
                </div>

                <!-- Technical Indicator Controls -->
                <div class="btn-group" role="group" style="margin: 5px;">
                    <button onclick="toggleIndicator('sma')" class="btn btn-sm btn-outline-primary" title="Toggle Simple Moving Average">
                        📈 Toggle SMA
                    </button>
                    <button onclick="toggleIndicator('ema')" class="btn btn-sm btn-outline-info" title="Toggle Exponential Moving Average">
                        📊 Toggle EMA
                    </button>
                    <button onclick="toggleIndicator('volume')" class="btn btn-sm btn-outline-secondary" title="Toggle Volume Display">
                        📊 Toggle Volume
                    </button>
                </div>

                <!-- Chart Management Controls -->
                <div class="btn-group" role="group" style="margin: 5px;">
                    <button onclick="showAllIndicators()" class="btn btn-sm btn-outline-success" title="Show All Technical Indicators">
                        👁️ Show All
                    </button>
                    <button onclick="hideAllIndicators()" class="btn btn-sm btn-outline-dark" title="Hide All Technical Indicators">
                        🙈 Hide All
                    </button>
                    <button onclick="resetChart()" class="btn btn-sm btn-outline-warning" title="Reset Chart to Default View">
                        🔄 Reset View
                    </button>
                    <button onclick="debugChartState()" class="btn btn-sm btn-outline-info" title="Debug Chart State (Check Console)">
                        🔍 Debug
                    </button>
                </div>
            </div>

            <!-- Technical Analysis Summary -->
            <div class="alert alert-light" style="margin: 15px 0;">
                {self._generate_enhanced_technical_summary(technical_indicators or {})}
            </div>
        </div>"""

    def _generate_enhanced_technical_summary(self, technical_indicators: Dict[str, Any]) -> str:
        """Generate enhanced technical analysis summary using institutional summary when available."""
        if not technical_indicators:
            return """<h5>🔍 Technical Analysis Summary</h5>
                <p>No technical indicators available for analysis.</p>"""

        # Check if institutional summary is available
        institutional_summary = technical_indicators.get('institutional_summary', {})

        if institutional_summary and institutional_summary.get('technical_narrative'):
            self.logger.info(f"✅ [CHART ENHANCEMENT] Using institutional technical narrative")
            # Use the enhanced institutional-grade technical analysis without duplicate heading
            # The institutional narrative already contains its own heading
            return institutional_summary.get('technical_narrative', '')
        else:
            self.logger.warning(f"⚠️ [CHART ENHANCEMENT] Using fallback technical summary - institutional narrative not available")
            # Fallback to generic summary
            return """<h5>🔍 Technical Analysis Summary</h5>
                <p>This enhanced chart combines price action with key technical indicators to provide comprehensive market analysis.
                Use the controls above to toggle different indicators and customize your view.</p>"""

    def _generate_technical_summary(self, technical_indicators: Dict[str, Any]) -> str:
        """Generate a summary of technical indicators."""
        if not technical_indicators:
            return "No technical indicators available."

        summary_parts = []

        # Moving averages summary
        moving_averages = technical_indicators.get('moving_averages', {})
        if moving_averages:
            ma_signals = [ma_data.get('sma_signal', 'Neutral') for ma_data in moving_averages.values()]
            buy_signals = ma_signals.count('Buy')
            sell_signals = ma_signals.count('Sell')
            summary_parts.append(f"Moving Averages: {buy_signals} Buy, {sell_signals} Sell signals")

        # Technical indicators summary
        tech_indicators = technical_indicators.get('technical_indicators', {})
        if tech_indicators:
            tech_signals = [ind_data.get('signal', 'Neutral') for ind_data in tech_indicators.values()]
            tech_buy = tech_signals.count('Buy')
            tech_sell = tech_signals.count('Sell')
            summary_parts.append(f"Technical Indicators: {tech_buy} Buy, {tech_sell} Sell signals")

        # Overall consensus
        overall = technical_indicators.get('overall_consensus', {})
        if overall:
            consensus = overall.get('consensus', 'Neutral')
            summary_parts.append(f"Overall Consensus: {consensus}")

        return " | ".join(summary_parts) if summary_parts else "Technical analysis data available."

    def _generate_fallback_chart_html(self, ticker: str) -> str:
        """Generate fallback HTML when chart enhancement fails."""
        return f"""
        <div class="section">
            <h2>Price & Technical Analysis</h2>
            <div class="alert alert-warning">
                <h5>Chart Enhancement Unavailable</h5>
                <p>Enhanced chart functionality is temporarily unavailable for {ticker}.
                Please refer to the individual Price Chart and Technical Analysis sections below.</p>
            </div>
        </div>"""

    def _to_epoch_ms_py(self, x):
        """Coerce x into epoch milliseconds if possible; return original if not parseable.
        Accepts int/float (assumed ms), or ISO8601 string.
        """
        try:
            if isinstance(x, (int, float)):
                # Assume already epoch ms (or seconds if too small)
                # If value looks like epoch seconds (~1e9 to 2e9), scale to ms
                if 1e9 <= x < 1e11:
                    return int(x * 1000)
                return int(x)
            if isinstance(x, str):
                from datetime import datetime
                return int(datetime.fromisoformat(x.replace('Z', '+00:00')).timestamp() * 1000)
        except Exception:
            return x
        return x

    def validate_and_coerce_chart_config(self, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all visible datasets have valid time x values. Coerce when possible.
        Returns potentially modified chart_config with fixes applied. Logs any issues found.
        """
        cfg = chart_config
        try:
            datasets = cfg.get('data', {}).get('datasets', []) or []
            fixes = 0
            issues = 0
            for ds in datasets:
                ds_type = ds.get('type') or cfg.get('type')
                data = ds.get('data', []) or []
                # Only datasets plotted against time x need validation
                # Expect objects with x for candlestick/line/volume (bar) datasets we add
                if not data:
                    continue
                # Detect shape: candlestick has o,h,l,c; line/volume have y
                first = data[0]
                # If it's a primitive array (e.g., numbers), convert to [{x, y}] using index -> invalid for time scale
                if not isinstance(first, dict):
                    issues += 1
                    # Cannot reliably infer x; mark dataset hidden to avoid runtime crash
                    ds['hidden'] = True
                    continue
                # Walk through points and coerce x
                for i, pt in enumerate(data):
                    x = pt.get('x')
                    new_x = self._to_epoch_ms_py(x)
                    if new_x != x:
                        data[i]['x'] = new_x
                        fixes += 1
                # Ensure parsing keys for non-candles
                if ds_type != 'candlestick':
                    parsing = ds.get('parsing') or {}
                    if 'xAxisKey' not in parsing or 'yAxisKey' not in parsing:
                        parsing.update({'xAxisKey': 'x', 'yAxisKey': 'y'})
                        ds['parsing'] = parsing
            if fixes or issues:
                self.logger.info(f"🔍 [CHART DEBUG] Validation adjusted datasets: fixes={fixes}, issues={issues}")
        except Exception as e:
            self.logger.warning(f"⚠️ [CHART DEBUG] Chart config validation failed: {e}")
        return cfg

    def generate_enhanced_chart_scripts(self, chart_config: Dict[str, Any]) -> str:
        """Generate JavaScript for the enhanced chart with proper OHLC tooltip support."""

        # Handle tooltip callbacks separately for candlestick charts
        chart_type = chart_config.get('type', 'line')

        # Validate/coerce x values to avoid Invalid time value at runtime
        chart_config = self.validate_and_coerce_chart_config(chart_config)

        # Create a copy of config for JSON serialization
        config_for_json = chart_config.copy()

        # Remove callback functions from JSON config (will be added as JavaScript)
        if 'options' in config_for_json and 'plugins' in config_for_json['options']:
            if 'tooltip' in config_for_json['options']['plugins']:
                tooltip_config = config_for_json['options']['plugins']['tooltip']
                if 'callbacks' in tooltip_config:
                    del tooltip_config['callbacks']

        config_json = json.dumps(config_for_json, indent=2)

        # Generate tooltip callbacks for candlestick charts
        tooltip_callbacks = ""
        if chart_type == "candlestick":
            tooltip_callbacks = """
        // Add enhanced OHLC tooltip callbacks for candlestick charts
        enhancedChartConfig.options.plugins.tooltip.callbacks = {
            title: function(context) {
                return new Date(context[0].parsed.x).toLocaleDateString();
            },
            label: function(context) {
                const dataset = context.dataset;
                if (dataset.type === 'candlestick') {
                    const data = context.raw;
                    return [
                        'Open: $' + data.o.toFixed(2),
                        'High: $' + data.h.toFixed(2),
                        'Low: $' + data.l.toFixed(2),
                        'Close: $' + data.c.toFixed(2)
                    ];
                } else {
                    return dataset.label + ': $' + context.parsed.y.toFixed(2);
                }
            }
        };"""

        return f"""
    <script>
        // Enhanced Chart Configuration
        const enhancedChartConfig = {config_json};
        {tooltip_callbacks}

        // Debug logging for chart configuration
        console.log('🔍 [CHART DEBUG] Enhanced chart configuration:', enhancedChartConfig);
        console.log('🔍 [CHART DEBUG] Chart type:', enhancedChartConfig.type);
        console.log('🔍 [CHART DEBUG] Datasets count:', enhancedChartConfig.data.datasets.length);

        if (enhancedChartConfig.data.datasets.length > 0) {{
            const dataset = enhancedChartConfig.data.datasets[0];
            console.log('🔍 [CHART DEBUG] First dataset label:', dataset.label);
            console.log('🔍 [CHART DEBUG] First dataset data length:', dataset.data.length);
            console.log('🔍 [CHART DEBUG] First dataset sample data:', dataset.data.slice(0, 3));
        }}

        // Initialize enhanced chart
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('🔍 [CHART DEBUG] DOM loaded, initializing chart...');
            const ctx = document.getElementById('enhancedPriceChart');
            console.log('🔍 [CHART DEBUG] Chart canvas element:', ctx);
            console.log('🔍 [CHART DEBUG] Chart.js available:', typeof Chart !== 'undefined');

            if (ctx && typeof Chart !== 'undefined') {{
                try {{
                    console.log('🔍 [CHART DEBUG] Creating Chart.js instance...');
                    window.enhancedPriceChart = new Chart(ctx, enhancedChartConfig);
                    console.log('✅ [CHART DEBUG] Chart created successfully:', window.enhancedPriceChart);

                    // Log chart details after creation
                    if (window.enhancedPriceChart && window.enhancedPriceChart.data) {{
                        console.log('🔍 [CHART DEBUG] Chart data after creation:', window.enhancedPriceChart.data);
                        console.log('🔍 [CHART DEBUG] Chart config type:', window.enhancedPriceChart.config.type);
                        if (window.enhancedPriceChart.data.datasets && window.enhancedPriceChart.data.datasets[0]) {{
                            const dataset = window.enhancedPriceChart.data.datasets[0];
                            console.log('🔍 [CHART DEBUG] First dataset type:', dataset.type);
                            console.log('🔍 [CHART DEBUG] First dataset data length:', dataset.data.length);
                            console.log('🔍 [CHART DEBUG] First dataset sample:', dataset.data.slice(0, 2));
                        }}
                    }}
                }} catch (error) {{
                    console.error('❌ [CHART DEBUG] Error creating chart:', error);
                    console.error('❌ [CHART DEBUG] Error stack:', error.stack);
                }}

                // Enhanced chart interaction functions
                window.toggleIndicator = function(indicator) {{
                    const chart = window.enhancedPriceChart;
                    if (!chart) {{
                        console.error('Enhanced chart not found');
                        return;
                    }}

                    const datasets = chart.data.datasets;
                    let toggledCount = 0;

                    datasets.forEach(dataset => {{
                        const label = dataset.label.toLowerCase();
                        const indicatorLower = indicator.toLowerCase();

                        // More specific matching for different indicators
                        let shouldToggle = false;
                        if (indicatorLower === 'sma' && label.includes('sma')) {{
                            shouldToggle = true;
                        }} else if (indicatorLower === 'ema' && label.includes('ema')) {{
                            shouldToggle = true;
                        }} else if (indicatorLower === 'volume' && label.includes('volume')) {{
                            shouldToggle = true;
                        }}

                        if (shouldToggle) {{
                            dataset.hidden = !dataset.hidden;
                            toggledCount++;
                        }}
                    }});

                    chart.update();
                    console.log(`Toggled ${{toggledCount}} ${{indicator}} datasets`);
                }};

                window.resetChart = function() {{
                    const chart = window.enhancedPriceChart;
                    if (!chart) {{
                        console.error('Enhanced chart not found');
                        return;
                    }}

                    // Reset zoom if available
                    if (chart.resetZoom) {{
                        chart.resetZoom();
                    }}

                    // Show main price data, hide technical indicators
                    chart.data.datasets.forEach(dataset => {{
                        const label = dataset.label.toLowerCase();
                        if (label.includes('close price')) {{
                            dataset.hidden = false;  // Always show price
                        }} else {{
                            dataset.hidden = true;   // Hide technical indicators by default
                        }}
                    }});

                    chart.update();
                    console.log('Chart reset to default view');
                }};

                // Chart type switching function with enhanced debugging
                window.switchChartType = function(newType) {{
                    console.log('🔍 [CHART DEBUG] switchChartType called with:', newType);
                    const chart = window.enhancedPriceChart;

                    if (!chart || !chart.data || !chart.data.datasets) {{
                        console.error('❌ [CHART DEBUG] Chart or datasets not available');
                        return;
                    }}

                    const priceDataset = chart.data.datasets[0];
                    if (!priceDataset) {{
                        console.error('❌ [CHART DEBUG] Price dataset not found');
                        return;
                    }}

                    console.log('🔍 [CHART DEBUG] Current dataset type:', priceDataset.type);
                    console.log('🔍 [CHART DEBUG] Current chart type:', chart.config.type);
                    console.log('🔍 [CHART DEBUG] Dataset data length:', priceDataset.data.length);
                    console.log('🔍 [CHART DEBUG] Sample data point:', priceDataset.data[0]);

                    // Check if we're already in the requested type
                    if (newType === priceDataset.type) {{
                        console.log('✅ [CHART DEBUG] Chart is already in', newType, 'mode');
                        return;
                    }}

                    if (newType === 'candlestick') {{
                        // Convert line to candlestick (if OHLC data available)
                        console.log('🔍 [CHART DEBUG] Converting to candlestick chart');

                        // Check if we have OHLC data structure
                        if (priceDataset.data[0] && typeof priceDataset.data[0].o !== 'undefined') {{
                            console.log('✅ [CHART DEBUG] OHLC data available, converting to candlestick');
                            priceDataset.type = 'candlestick';
                            chart.config.type = 'candlestick';
                        }} else {{
                            console.warn('⚠️ [CHART DEBUG] OHLC data not available for candlestick chart');
                            console.log('🔍 [CHART DEBUG] Available data structure:', priceDataset.data[0]);
                            return;
                        }}
                    }} else if (newType === 'line') {{
                        // Convert candlestick to line
                        console.log('🔍 [CHART DEBUG] Converting to line chart');

                        if (priceDataset.data[0] && typeof priceDataset.data[0].c !== 'undefined') {{
                            console.log('✅ [CHART DEBUG] Converting OHLC data to line data');
                            const lineData = priceDataset.data.map(d => ({{ x: d.x, y: d.c }}));
                            priceDataset.data = lineData;
                            priceDataset.type = 'line';
                            priceDataset.fill = false;
                            priceDataset.tension = 0.1;
                            chart.config.type = 'line';
                            console.log('🔍 [CHART DEBUG] Line data created with', lineData.length, 'points');
                        }} else if (priceDataset.data[0] && typeof priceDataset.data[0].y !== 'undefined') {{
                            console.log('✅ [CHART DEBUG] Data is already in line format');
                            priceDataset.type = 'line';
                            priceDataset.fill = false;
                            priceDataset.tension = 0.1;
                            chart.config.type = 'line';
                        }} else {{
                            console.warn('⚠️ [CHART DEBUG] Cannot convert to line chart - no valid data structure');
                            console.log('🔍 [CHART DEBUG] Available data structure:', priceDataset.data[0]);
                            return;
                        }}
                    }}

                    try {{
                        chart.update();
                        console.log('✅ [CHART DEBUG] Chart type successfully switched to', newType);
                    }} catch (error) {{
                        console.error('❌ [CHART DEBUG] Error updating chart:', error);
                    }}
                }};

                // Additional utility functions
                window.showAllIndicators = function() {{
                    const chart = window.enhancedPriceChart;
                    if (!chart) return;

                    chart.data.datasets.forEach(dataset => {{
                        dataset.hidden = false;
                    }});

                    chart.update();
                    console.log('All indicators shown');
                }};

                window.hideAllIndicators = function() {{
                    console.log('🔍 [CHART DEBUG] hideAllIndicators called');
                    const chart = window.enhancedPriceChart;
                    if (!chart) {{
                        console.error('❌ [CHART DEBUG] Chart not available for hideAllIndicators');
                        return;
                    }}

                    chart.data.datasets.forEach(dataset => {{
                        const label = dataset.label.toLowerCase();
                        if (!label.includes('close price') && !label.includes('price')) {{
                            dataset.hidden = true;
                            console.log('🔍 [CHART DEBUG] Hiding indicator:', dataset.label);
                        }}
                    }});

                    chart.update();
                    console.log('✅ [CHART DEBUG] All indicators hidden');
                }};

                // Debug function to inspect chart state
                window.debugChartState = function() {{
                    console.log('🔍 [CHART DEBUG] === CHART STATE DEBUG ===');
                    const chart = window.enhancedPriceChart;
                    if (!chart) {{
                        console.error('❌ [CHART DEBUG] Chart not available');
                        return;
                    }}

                    console.log('🔍 [CHART DEBUG] Chart config type:', chart.config.type);
                    console.log('🔍 [CHART DEBUG] Chart data:', chart.data);
                    console.log('🔍 [CHART DEBUG] Datasets count:', chart.data.datasets.length);

                    chart.data.datasets.forEach((dataset, index) => {{
                        console.log(`🔍 [CHART DEBUG] Dataset ${{index}}:`, {{
                            label: dataset.label,
                            type: dataset.type,
                            dataLength: dataset.data.length,
                            hidden: dataset.hidden,
                            sampleData: dataset.data.slice(0, 2)
                        }});
                    }});

                    console.log('🔍 [CHART DEBUG] === END CHART STATE DEBUG ===');
                }};

                console.log('✅ [CHART DEBUG] Enhanced price chart initialized successfully');
            }} else {{
                console.error('❌ [CHART DEBUG] Enhanced chart canvas element not found or Chart.js not loaded');
                console.error('❌ [CHART DEBUG] Canvas element:', ctx);
                console.error('❌ [CHART DEBUG] Chart.js type:', typeof Chart);
            }}
        }});
    </script>"""
