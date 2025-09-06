"""
Technical Analysis Agent for Hong Kong Stock Financial Analysis System

This module provides comprehensive technical analysis capabilities including:
- Moving averages (SMA and EMA)
- Technical indicators (RSI, MACD, Stochastic, etc.)
- Pivot points (Classic and Fibonacci)
- Overall technical consensus generation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import yfinance as yf

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgent:
    """
    Technical Analysis Agent for generating comprehensive technical analysis data.
    """
    
    def __init__(self, period: str = "1y"):
        """
        Initialize the Technical Analysis Agent.
        
        Args:
            period: Data period for analysis (default: 1 year)
        """
        self.period = period
        
    def analyze_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing technical analysis results
        """
        try:
            logger.info(f"🔍 Starting technical analysis for {ticker}")
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=self.period)
            
            if hist.empty:
                logger.warning(f"No historical data available for {ticker}")
                return self._create_fallback_analysis(ticker)
            
            # Calculate technical indicators
            technical_data = {
                "ticker": ticker,
                "success": True,
                "data_period": self.period,
                "last_updated": datetime.now().isoformat(),
                "current_price": float(hist['Close'].iloc[-1]),
                "moving_averages": self._calculate_moving_averages(hist),
                "technical_indicators": self._calculate_technical_indicators(hist),
                "pivot_points": self._calculate_pivot_points(hist),
                "macd_analysis": self._calculate_macd_detailed(hist),
                "overall_consensus": None  # Will be calculated after all indicators
            }
            
            # Calculate overall technical consensus
            technical_data["overall_consensus"] = self._calculate_overall_consensus(technical_data)

            # Generate institutional-grade technical analysis summary
            technical_data["institutional_summary"] = self._generate_institutional_technical_summary(
                technical_data, ticker, hist
            )

            logger.info(f"✅ Technical analysis completed for {ticker}")
            return technical_data
            
        except Exception as e:
            logger.error(f"❌ Technical analysis failed for {ticker}: {e}")
            return {"success": False, "error": str(e), "ticker": ticker}
    
    def _calculate_moving_averages(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate moving averages and their signals."""
        current_price = hist['Close'].iloc[-1]
        ma_data = {}
        
        # Define moving average periods
        ma_periods = [5, 10, 20, 50, 100, 200]
        
        for period in ma_periods:
            if len(hist) >= period:
                # Simple Moving Average
                sma = hist['Close'].rolling(window=period).mean()
                sma_current = sma.iloc[-1]
                
                # Exponential Moving Average
                ema = hist['Close'].ewm(span=period).mean()
                ema_current = ema.iloc[-1]
                
                # Determine signals
                sma_signal = "Buy" if current_price > sma_current else "Sell"
                ema_signal = "Buy" if current_price > ema_current else "Sell"
                
                ma_data[f"MA{period}"] = {
                    "sma_value": float(sma_current),
                    "ema_value": float(ema_current),
                    "sma_signal": sma_signal,
                    "ema_signal": ema_signal,
                    "price_vs_sma": float((current_price - sma_current) / sma_current * 100),
                    "price_vs_ema": float((current_price - ema_current) / ema_current * 100)
                }
        
        return ma_data
    
    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various technical indicators."""
        indicators = {}
        
        # RSI (14)
        rsi = self._calculate_rsi(hist['Close'], 14)
        rsi_current = rsi.iloc[-1] if not rsi.empty else None
        if rsi_current:
            if rsi_current > 70:
                rsi_signal = "Sell"
            elif rsi_current < 30:
                rsi_signal = "Buy"
            else:
                rsi_signal = "Neutral"
            
            indicators["RSI"] = {
                "value": float(rsi_current),
                "signal": rsi_signal,
                "period": 14
            }
        
        # Stochastic Oscillator (9,6)
        stoch_k, stoch_d = self._calculate_stochastic(hist, 9, 6)
        if not stoch_k.empty and not stoch_d.empty:
            stoch_k_current = stoch_k.iloc[-1]
            stoch_d_current = stoch_d.iloc[-1]
            
            if stoch_k_current > 80:
                stoch_signal = "Sell"
            elif stoch_k_current < 20:
                stoch_signal = "Buy"
            else:
                stoch_signal = "Neutral"
            
            indicators["STOCH"] = {
                "k_value": float(stoch_k_current),
                "d_value": float(stoch_d_current),
                "signal": stoch_signal,
                "k_period": 9,
                "d_period": 6
            }
        
        # Williams %R
        williams_r = self._calculate_williams_r(hist, 14)
        if not williams_r.empty:
            wr_current = williams_r.iloc[-1]
            if wr_current > -20:
                wr_signal = "Sell"
            elif wr_current < -80:
                wr_signal = "Buy"
            else:
                wr_signal = "Neutral"
            
            indicators["Williams_R"] = {
                "value": float(wr_current),
                "signal": wr_signal,
                "period": 14
            }
        
        # CCI (14)
        cci = self._calculate_cci(hist, 14)
        if not cci.empty:
            cci_current = cci.iloc[-1]
            if cci_current > 100:
                cci_signal = "Sell"
            elif cci_current < -100:
                cci_signal = "Buy"
            else:
                cci_signal = "Neutral"
            
            indicators["CCI"] = {
                "value": float(cci_current),
                "signal": cci_signal,
                "period": 14
            }
        
        # ATR (14)
        atr = self._calculate_atr(hist, 14)
        if not atr.empty:
            indicators["ATR"] = {
                "value": float(atr.iloc[-1]),
                "period": 14
            }
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, hist: pd.DataFrame, k_period: int = 9, d_period: int = 6) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = hist['Low'].rolling(window=k_period).min()
        high_max = hist['High'].rolling(window=k_period).max()
        k_percent = 100 * ((hist['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, hist: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = hist['High'].rolling(window=period).max()
        low_min = hist['Low'].rolling(window=period).min()
        williams_r = -100 * ((high_max - hist['Close']) / (high_max - low_min))
        return williams_r
    
    def _calculate_cci(self, hist: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
    
    def _calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = hist['High'] - hist['Low']
        high_close = np.abs(hist['High'] - hist['Close'].shift())
        low_close = np.abs(hist['Low'] - hist['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr

    def _calculate_pivot_points(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate pivot points (Classic and Fibonacci)."""
        # Use last complete trading day
        last_day = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
        high = last_day['High']
        low = last_day['Low']
        close = last_day['Close']

        # Classic Pivot Points
        pivot = (high + low + close) / 3

        classic_pivots = {
            "pivot": float(pivot),
            "resistance_1": float(2 * pivot - low),
            "resistance_2": float(pivot + (high - low)),
            "resistance_3": float(high + 2 * (pivot - low)),
            "support_1": float(2 * pivot - high),
            "support_2": float(pivot - (high - low)),
            "support_3": float(low - 2 * (high - pivot))
        }

        # Fibonacci Pivot Points
        fib_pivots = {
            "pivot": float(pivot),
            "resistance_1": float(pivot + 0.382 * (high - low)),
            "resistance_2": float(pivot + 0.618 * (high - low)),
            "resistance_3": float(pivot + 1.000 * (high - low)),
            "support_1": float(pivot - 0.382 * (high - low)),
            "support_2": float(pivot - 0.618 * (high - low)),
            "support_3": float(pivot - 1.000 * (high - low))
        }

        return {
            "classic": classic_pivots,
            "fibonacci": fib_pivots
        }

    def _calculate_macd_detailed(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed MACD analysis."""
        # MACD (12, 26, 9)
        exp1 = hist['Close'].ewm(span=12).mean()
        exp2 = hist['Close'].ewm(span=26).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        if not macd_line.empty and not signal_line.empty:
            macd_current = macd_line.iloc[-1]
            signal_current = signal_line.iloc[-1]
            histogram_current = histogram.iloc[-1]

            # Determine MACD signal
            if macd_current > signal_current and histogram_current > 0:
                macd_signal = "Buy"
            elif macd_current < signal_current and histogram_current < 0:
                macd_signal = "Sell"
            else:
                macd_signal = "Neutral"

            return {
                "macd_value": float(macd_current),
                "signal_value": float(signal_current),
                "histogram_value": float(histogram_current),
                "signal": macd_signal,
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            }

        return {}

    def _calculate_overall_consensus(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall technical consensus from all indicators."""
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        total_signals = 0

        # Count moving average signals
        ma_data = technical_data.get("moving_averages", {})
        for ma_key, ma_info in ma_data.items():
            if ma_info.get("sma_signal") == "Buy":
                buy_signals += 1
            elif ma_info.get("sma_signal") == "Sell":
                sell_signals += 1
            else:
                neutral_signals += 1
            total_signals += 1

            if ma_info.get("ema_signal") == "Buy":
                buy_signals += 1
            elif ma_info.get("ema_signal") == "Sell":
                sell_signals += 1
            else:
                neutral_signals += 1
            total_signals += 1

        # Count technical indicator signals
        indicators = technical_data.get("technical_indicators", {})
        for indicator_name, indicator_data in indicators.items():
            signal = indicator_data.get("signal")
            if signal == "Buy":
                buy_signals += 1
            elif signal == "Sell":
                sell_signals += 1
            elif signal == "Neutral":
                neutral_signals += 1

            if signal:  # Only count if signal exists
                total_signals += 1

        # Count MACD signal
        macd_data = technical_data.get("macd_analysis", {})
        if macd_data.get("signal"):
            if macd_data["signal"] == "Buy":
                buy_signals += 1
            elif macd_data["signal"] == "Sell":
                sell_signals += 1
            else:
                neutral_signals += 1
            total_signals += 1

        # Determine overall consensus
        if total_signals == 0:
            overall_signal = "Neutral"
        elif buy_signals > sell_signals and buy_signals > neutral_signals:
            overall_signal = "Buy"
        elif sell_signals > buy_signals and sell_signals > neutral_signals:
            overall_signal = "Sell"
        else:
            overall_signal = "Neutral"

        # Calculate confidence based on signal distribution
        if total_signals > 0:
            buy_percentage = (buy_signals / total_signals) * 100
            sell_percentage = (sell_signals / total_signals) * 100
            neutral_percentage = (neutral_signals / total_signals) * 100
        else:
            buy_percentage = sell_percentage = neutral_percentage = 0

        return {
            "overall_signal": overall_signal,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "neutral_signals": neutral_signals,
            "total_signals": total_signals,
            "buy_percentage": round(buy_percentage, 1),
            "sell_percentage": round(sell_percentage, 1),
            "neutral_percentage": round(neutral_percentage, 1),
            "confidence": max(buy_percentage, sell_percentage, neutral_percentage)
        }

    def _create_fallback_analysis(self, ticker: str) -> Dict[str, Any]:
        """Create a fallback analysis when historical data is unavailable."""
        return {
            "success": True,
            "ticker": ticker,
            "data_available": False,
            "message": "Technical analysis unavailable - no historical price data found",
            "indicators": {},
            "moving_averages": {},
            "pivot_points": {},
            "summary": {
                "overall_signal": "Neutral",
                "buy_signals": 0,
                "sell_signals": 0,
                "neutral_signals": 1,
                "total_signals": 1,
                "buy_percentage": 0.0,
                "sell_percentage": 0.0,
                "neutral_percentage": 100.0,
                "confidence": 0.0
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "fallback_reason": "Historical data unavailable (possible rate limiting or delisted ticker)"
        }

    def _generate_institutional_technical_summary(self, technical_data: Dict[str, Any], ticker: str, hist: pd.DataFrame) -> Dict[str, Any]:
        """Generate institutional-grade technical analysis summary."""

        try:
            # Extract key technical components
            current_price = technical_data.get('current_price', 0)
            moving_averages = technical_data.get('moving_averages', {})
            technical_indicators = technical_data.get('technical_indicators', {})
            macd_analysis = technical_data.get('macd_analysis', {})
            overall_consensus = technical_data.get('overall_consensus', {})
            pivot_points = technical_data.get('pivot_points', {})

            # Generate momentum assessment
            momentum_analysis = self._generate_momentum_assessment(
                current_price, moving_averages, technical_indicators, macd_analysis, hist
            )

            # Generate institutional accumulation analysis
            institutional_flow = self._generate_institutional_flow_analysis(
                hist, current_price, moving_averages
            )

            # Generate support/resistance analysis
            support_resistance = self._generate_support_resistance_analysis(
                current_price, pivot_points, hist
            )

            # Generate professional technical narrative
            technical_narrative = self._generate_professional_technical_narrative(
                ticker, current_price, moving_averages, technical_indicators,
                macd_analysis, overall_consensus, momentum_analysis,
                institutional_flow, support_resistance
            )

            return {
                "momentum_analysis": momentum_analysis,
                "institutional_flow": institutional_flow,
                "support_resistance": support_resistance,
                "technical_narrative": technical_narrative,
                "confidence_level": overall_consensus.get('confidence', 50),
                "signal_strength": self._calculate_signal_strength(overall_consensus),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error generating institutional technical summary for {ticker}: {e}")
            return {
                "error": str(e),
                "fallback_narrative": self._generate_fallback_technical_narrative(ticker),
                "timestamp": datetime.now().isoformat()
            }

    def _generate_momentum_assessment(self, current_price: float, moving_averages: Dict,
                                    technical_indicators: Dict, macd_analysis: Dict,
                                    hist: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive momentum assessment with institutional focus."""

        # Calculate moving average positioning
        ma_positioning = {}
        for ma_key, ma_data in moving_averages.items():
            if 'price_vs_sma' in ma_data:
                ma_positioning[ma_key] = {
                    'percentage_above_sma': ma_data['price_vs_sma'],
                    'percentage_above_ema': ma_data.get('price_vs_ema', 0),
                    'signal': ma_data.get('sma_signal', 'NEUTRAL')
                }

        # Determine momentum direction
        momentum_signals = []
        if moving_averages.get('MA20', {}).get('price_vs_sma', 0) > 0:
            momentum_signals.append('SHORT_TERM_BULLISH')
        if moving_averages.get('MA50', {}).get('price_vs_sma', 0) > 0:
            momentum_signals.append('MEDIUM_TERM_BULLISH')
        if moving_averages.get('MA200', {}).get('price_vs_sma', 0) > 0:
            momentum_signals.append('LONG_TERM_BULLISH')

        # MACD momentum assessment
        macd_momentum = 'NEUTRAL'
        if macd_analysis.get('signal') == 'Buy' and macd_analysis.get('histogram_value', 0) > 0:
            macd_momentum = 'BULLISH_CROSSOVER'
        elif macd_analysis.get('signal') == 'Sell' and macd_analysis.get('histogram_value', 0) < 0:
            macd_momentum = 'BEARISH_CROSSOVER'

        # RSI momentum assessment
        rsi_data = technical_indicators.get('RSI', {})
        rsi_momentum = 'NEUTRAL'
        if rsi_data.get('value', 50) > 70:
            rsi_momentum = 'OVERBOUGHT'
        elif rsi_data.get('value', 50) < 30:
            rsi_momentum = 'OVERSOLD'
        elif 50 <= rsi_data.get('value', 50) <= 70:
            rsi_momentum = 'BULLISH_MOMENTUM'
        elif 30 <= rsi_data.get('value', 50) <= 50:
            rsi_momentum = 'BEARISH_MOMENTUM'

        # Calculate momentum strength
        bullish_signals = len([s for s in momentum_signals if 'BULLISH' in s])
        momentum_strength = 'STRONG' if bullish_signals >= 2 else 'MODERATE' if bullish_signals == 1 else 'WEAK'

        return {
            'ma_positioning': ma_positioning,
            'momentum_signals': momentum_signals,
            'macd_momentum': macd_momentum,
            'rsi_momentum': rsi_momentum,
            'momentum_strength': momentum_strength,
            'overall_direction': 'BULLISH' if bullish_signals >= 2 else 'BEARISH' if bullish_signals == 0 else 'NEUTRAL'
        }

    def _generate_institutional_flow_analysis(self, hist: pd.DataFrame, current_price: float,
                                            moving_averages: Dict) -> Dict[str, Any]:
        """Generate institutional accumulation/distribution analysis."""

        # Calculate volume-weighted metrics
        if 'Volume' in hist.columns and len(hist) >= 20:
            # Calculate average volume over different periods
            volume_20d = hist['Volume'].tail(20).mean()
            volume_50d = hist['Volume'].tail(50).mean() if len(hist) >= 50 else volume_20d
            recent_volume = hist['Volume'].tail(5).mean()

            # Volume trend analysis
            volume_trend = 'INCREASING' if recent_volume > volume_20d * 1.2 else 'DECREASING' if recent_volume < volume_20d * 0.8 else 'STABLE'

            # Price-volume relationship
            price_change_5d = (current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6] * 100 if len(hist) >= 6 else 0
            volume_change_5d = (recent_volume - volume_20d) / volume_20d * 100

            # Institutional accumulation patterns
            accumulation_pattern = 'NEUTRAL'
            if price_change_5d > 2 and volume_change_5d > 20:
                accumulation_pattern = 'STRONG_ACCUMULATION'
            elif price_change_5d > 0 and volume_change_5d > 0:
                accumulation_pattern = 'MODERATE_ACCUMULATION'
            elif price_change_5d < -2 and volume_change_5d > 20:
                accumulation_pattern = 'DISTRIBUTION'
            elif price_change_5d < 0 and volume_change_5d < -20:
                accumulation_pattern = 'WEAK_SELLING'

            # On-Balance Volume approximation
            obv_trend = self._calculate_obv_trend(hist)

        else:
            volume_trend = 'DATA_UNAVAILABLE'
            accumulation_pattern = 'DATA_UNAVAILABLE'
            volume_20d = volume_50d = recent_volume = 0
            price_change_5d = volume_change_5d = 0
            obv_trend = 'DATA_UNAVAILABLE'

        # Moving average support analysis
        ma_support_levels = []
        for ma_key, ma_data in moving_averages.items():
            if ma_data.get('price_vs_sma', 0) > -5:  # Within 5% of MA
                ma_support_levels.append({
                    'level': ma_key,
                    'price': ma_data.get('sma_value', 0),
                    'distance_percent': ma_data.get('price_vs_sma', 0)
                })

        return {
            'volume_analysis': {
                'volume_20d_avg': volume_20d,
                'volume_50d_avg': volume_50d,
                'recent_volume_avg': recent_volume,
                'volume_trend': volume_trend,
                'volume_change_5d_percent': volume_change_5d
            },
            'accumulation_pattern': accumulation_pattern,
            'price_volume_relationship': {
                'price_change_5d_percent': price_change_5d,
                'volume_change_5d_percent': volume_change_5d
            },
            'obv_trend': obv_trend,
            'ma_support_levels': ma_support_levels,
            'institutional_signal': self._determine_institutional_signal(accumulation_pattern, obv_trend, volume_trend)
        }

    def _calculate_obv_trend(self, hist: pd.DataFrame) -> str:
        """Calculate On-Balance Volume trend."""
        try:
            if 'Volume' not in hist.columns or len(hist) < 10:
                return 'DATA_UNAVAILABLE'

            # Simple OBV calculation
            obv = []
            obv_value = 0

            for i in range(1, len(hist)):
                if hist['Close'].iloc[i] > hist['Close'].iloc[i-1]:
                    obv_value += hist['Volume'].iloc[i]
                elif hist['Close'].iloc[i] < hist['Close'].iloc[i-1]:
                    obv_value -= hist['Volume'].iloc[i]
                obv.append(obv_value)

            if len(obv) >= 10:
                recent_obv = np.mean(obv[-5:])
                earlier_obv = np.mean(obv[-10:-5])

                if recent_obv > earlier_obv * 1.05:
                    return 'BULLISH'
                elif recent_obv < earlier_obv * 0.95:
                    return 'BEARISH'
                else:
                    return 'NEUTRAL'

            return 'INSUFFICIENT_DATA'

        except Exception:
            return 'CALCULATION_ERROR'

    def _determine_institutional_signal(self, accumulation_pattern: str, obv_trend: str, volume_trend: str) -> str:
        """Determine overall institutional signal."""
        bullish_signals = 0
        bearish_signals = 0

        if accumulation_pattern in ['STRONG_ACCUMULATION', 'MODERATE_ACCUMULATION']:
            bullish_signals += 1
        elif accumulation_pattern in ['DISTRIBUTION', 'WEAK_SELLING']:
            bearish_signals += 1

        if obv_trend == 'BULLISH':
            bullish_signals += 1
        elif obv_trend == 'BEARISH':
            bearish_signals += 1

        if volume_trend == 'INCREASING':
            bullish_signals += 0.5
        elif volume_trend == 'DECREASING':
            bearish_signals += 0.5

        if bullish_signals > bearish_signals:
            return 'INSTITUTIONAL_ACCUMULATION'
        elif bearish_signals > bullish_signals:
            return 'INSTITUTIONAL_DISTRIBUTION'
        else:
            return 'NEUTRAL_FLOW'

    def _generate_support_resistance_analysis(self, current_price: float, pivot_points: Dict,
                                            hist: pd.DataFrame) -> Dict[str, Any]:
        """Generate support and resistance level analysis."""

        support_levels = []
        resistance_levels = []

        # Extract pivot point levels
        classic_pivots = pivot_points.get('classic', {})
        fib_pivots = pivot_points.get('fibonacci', {})

        # Add classic pivot supports
        for level in ['support_1', 'support_2', 'support_3']:
            if level in classic_pivots:
                support_levels.append({
                    'price': classic_pivots[level],
                    'type': f'Classic Pivot {level.replace("_", " ").title()}',
                    'strength': 'HIGH' if level == 'support_1' else 'MEDIUM',
                    'distance_percent': ((classic_pivots[level] - current_price) / current_price * 100)
                })

        # Add classic pivot resistances
        for level in ['resistance_1', 'resistance_2', 'resistance_3']:
            if level in classic_pivots:
                resistance_levels.append({
                    'price': classic_pivots[level],
                    'type': f'Classic Pivot {level.replace("_", " ").title()}',
                    'strength': 'HIGH' if level == 'resistance_1' else 'MEDIUM',
                    'distance_percent': ((classic_pivots[level] - current_price) / current_price * 100)
                })

        # Add Fibonacci levels
        for level in ['support_1', 'support_2']:
            if level in fib_pivots:
                support_levels.append({
                    'price': fib_pivots[level],
                    'type': f'Fibonacci {level.replace("_", " ").title()}',
                    'strength': 'MEDIUM',
                    'distance_percent': ((fib_pivots[level] - current_price) / current_price * 100)
                })

        for level in ['resistance_1', 'resistance_2']:
            if level in fib_pivots:
                resistance_levels.append({
                    'price': fib_pivots[level],
                    'type': f'Fibonacci {level.replace("_", " ").title()}',
                    'strength': 'MEDIUM',
                    'distance_percent': ((fib_pivots[level] - current_price) / current_price * 100)
                })

        # Calculate price targets
        nearest_resistance = min([r for r in resistance_levels if r['distance_percent'] > 0],
                                key=lambda x: x['distance_percent'], default=None)
        nearest_support = max([s for s in support_levels if s['distance_percent'] < 0],
                             key=lambda x: x['distance_percent'], default=None)

        return {
            'support_levels': sorted(support_levels, key=lambda x: x['price'], reverse=True),
            'resistance_levels': sorted(resistance_levels, key=lambda x: x['price']),
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_price': current_price,
            'price_position': self._determine_price_position(current_price, support_levels, resistance_levels)
        }

    def _determine_price_position(self, current_price: float, support_levels: List, resistance_levels: List) -> str:
        """Determine current price position relative to support/resistance."""
        supports_below = [s for s in support_levels if s['price'] < current_price]
        resistances_above = [r for r in resistance_levels if r['price'] > current_price]

        if supports_below and resistances_above:
            return 'BETWEEN_LEVELS'
        elif supports_below and not resistances_above:
            return 'ABOVE_RESISTANCE'
        elif not supports_below and resistances_above:
            return 'BELOW_SUPPORT'
        else:
            return 'NEUTRAL_ZONE'

    def _calculate_signal_strength(self, overall_consensus: Dict) -> str:
        """Calculate technical signal strength."""
        confidence = overall_consensus.get('confidence', 50)

        if confidence >= 80:
            return 'VERY_STRONG'
        elif confidence >= 65:
            return 'STRONG'
        elif confidence >= 50:
            return 'MODERATE'
        else:
            return 'WEAK'

    def _generate_professional_technical_narrative(self, ticker: str, current_price: float,
                                                 moving_averages: Dict, technical_indicators: Dict,
                                                 macd_analysis: Dict, overall_consensus: Dict,
                                                 momentum_analysis: Dict, institutional_flow: Dict,
                                                 support_resistance: Dict) -> str:
        """Generate institutional-grade technical analysis narrative."""

        # Extract key data points
        ma_20 = moving_averages.get('MA20', {})
        ma_50 = moving_averages.get('MA50', {})
        ma_200 = moving_averages.get('MA200', {})

        rsi_data = technical_indicators.get('RSI', {})
        macd_signal = macd_analysis.get('signal', 'NEUTRAL')

        momentum_direction = momentum_analysis.get('overall_direction', 'NEUTRAL')
        momentum_strength = momentum_analysis.get('momentum_strength', 'MODERATE')

        institutional_signal = institutional_flow.get('institutional_signal', 'NEUTRAL_FLOW')
        accumulation_pattern = institutional_flow.get('accumulation_pattern', 'NEUTRAL')

        nearest_resistance = support_resistance.get('nearest_resistance')
        nearest_support = support_resistance.get('nearest_support')

        # Determine currency
        currency = "HK$" if ".HK" in ticker else "$"

        # Generate momentum and positioning narrative
        ma_narrative = self._generate_ma_positioning_narrative(ma_20, ma_50, ma_200, currency, current_price)

        # Generate technical indicators narrative
        indicators_narrative = self._generate_indicators_narrative(rsi_data, macd_analysis, technical_indicators)

        # Generate institutional flow narrative
        flow_narrative = self._generate_flow_narrative(institutional_signal, accumulation_pattern, institutional_flow)

        # Generate support/resistance narrative
        levels_narrative = self._generate_levels_narrative(nearest_support, nearest_resistance, currency)

        # Combine into professional narrative
        narrative = f"""
        <div style="margin-bottom: 20px;">
            <h5 style="color: #0c5460; margin-bottom: 15px;">📈 Technical Analysis and Momentum Indicators</h5>

            <p style="line-height: 1.6; margin-bottom: 15px;">
                <strong>{ticker.replace('.HK', '')}</strong> demonstrates {momentum_direction.lower()} momentum with {momentum_strength.lower()}
                signal strength, reflecting {institutional_signal.replace('_', ' ').lower()} patterns in institutional flow analysis.
                {ma_narrative}
                <small>[Source: Yahoo Finance Technical Data, Real-time MA Analysis]</small>
            </p>

            <p style="line-height: 1.6; margin-bottom: 15px;">
                {indicators_narrative}
                <small>[Source: Technical Indicators Analysis, {datetime.now().strftime('%Y-%m-%d')}]</small>
            </p>

            <p style="line-height: 1.6; margin-bottom: 15px;">
                {flow_narrative}
                <small>[Source: Market Volume Analysis, {datetime.now().strftime('%Y-%m-%d %H:%M')}]</small>
            </p>

            <p style="line-height: 1.6; margin-bottom: 10px;">
                {levels_narrative}
                <small>[Source: Chart Pattern Analysis, Technical Signals]</small>
            </p>
        </div>
        """

        return narrative

    def _generate_ma_positioning_narrative(self, ma_20: Dict, ma_50: Dict, ma_200: Dict,
                                         currency: str, current_price: float) -> str:
        """Generate moving average positioning narrative."""

        ma_20_pct = ma_20.get('price_vs_sma', 0)
        ma_50_pct = ma_50.get('price_vs_sma', 0)
        ma_200_pct = ma_200.get('price_vs_sma', 0)

        if ma_50_pct > 0 and ma_200_pct > 0:
            return (f"The stock trades {ma_50_pct:+.1f}% above its 50-day moving average and "
                   f"{ma_200_pct:+.1f}% above its 200-day average, indicating sustained upward momentum "
                   f"supported by institutional accumulation patterns and bullish technical breakout levels.")
        elif ma_50_pct > 0:
            return (f"Current positioning shows {ma_50_pct:+.1f}% above the 50-day moving average "
                   f"while {ma_200_pct:+.1f}% relative to the 200-day average, reflecting "
                   f"medium-term momentum continuation patterns with mixed long-term signals.")
        elif ma_200_pct > 0:
            return (f"The stock maintains {ma_200_pct:+.1f}% above its 200-day moving average "
                   f"despite {ma_50_pct:+.1f}% positioning relative to the 50-day average, "
                   f"indicating long-term uptrend preservation amid short-term consolidation.")
        else:
            return (f"Technical positioning reflects {ma_50_pct:+.1f}% relative to 50-day and "
                   f"{ma_200_pct:+.1f}% relative to 200-day moving averages, suggesting "
                   f"bearish momentum with potential support level testing.")

    def _generate_indicators_narrative(self, rsi_data: Dict, macd_analysis: Dict,
                                     technical_indicators: Dict) -> str:
        """Generate technical indicators narrative."""

        rsi_value = rsi_data.get('value', 50)
        rsi_signal = rsi_data.get('signal', 'NEUTRAL')
        macd_signal = macd_analysis.get('signal', 'NEUTRAL')
        macd_histogram = macd_analysis.get('histogram_value', 0)

        # RSI analysis
        if rsi_value > 70:
            rsi_narrative = f"RSI at {rsi_value:.1f} indicates overbought conditions with potential reversal signals"
        elif rsi_value < 30:
            rsi_narrative = f"RSI at {rsi_value:.1f} reflects oversold conditions presenting institutional accumulation opportunities"
        else:
            rsi_narrative = f"RSI at {rsi_value:.1f} maintains neutral momentum within normal trading ranges"

        # MACD analysis
        if macd_signal == 'Buy' and macd_histogram > 0:
            macd_narrative = "bullish MACD crossover signals with expanding histogram momentum"
        elif macd_signal == 'Sell' and macd_histogram < 0:
            macd_narrative = "bearish MACD crossover with contracting histogram indicating selling pressure"
        else:
            macd_narrative = "neutral MACD positioning with sideways momentum patterns"

        return f"{rsi_narrative}, while {macd_narrative} support the overall technical assessment."

    def _generate_flow_narrative(self, institutional_signal: str, accumulation_pattern: str,
                               institutional_flow: Dict) -> str:
        """Generate institutional flow narrative."""

        volume_trend = institutional_flow.get('volume_analysis', {}).get('volume_trend', 'STABLE')
        price_volume = institutional_flow.get('price_volume_relationship', {})
        price_change = price_volume.get('price_change_5d_percent', 0)
        volume_change = price_volume.get('volume_change_5d_percent', 0)

        if institutional_signal == 'INSTITUTIONAL_ACCUMULATION':
            return (f"Institutional accumulation patterns emerge with {volume_change:+.1f}% volume increase "
                   f"accompanying {price_change:+.1f}% price movement, indicating smart money positioning "
                   f"and above-average institutional flow supporting momentum continuation.")
        elif institutional_signal == 'INSTITUTIONAL_DISTRIBUTION':
            return (f"Distribution patterns suggest institutional selling with {volume_change:+.1f}% volume change "
                   f"during {price_change:+.1f}% price movement, reflecting profit-taking activities "
                   f"and potential resistance level formation.")
        else:
            return (f"Neutral institutional flow with {volume_trend.lower()} volume patterns and "
                   f"{price_change:+.1f}% price movement, indicating balanced supply-demand dynamics "
                   f"without significant directional bias from institutional participants.")

    def _generate_levels_narrative(self, nearest_support: Dict, nearest_resistance: Dict,
                                 currency: str) -> str:
        """Generate support/resistance levels narrative."""

        if nearest_resistance and nearest_support:
            resistance_price = nearest_resistance['price']
            resistance_distance = nearest_resistance['distance_percent']
            support_price = nearest_support['price']
            support_distance = abs(nearest_support['distance_percent'])

            return (f"Key technical levels identify {currency}{resistance_price:.2f} as immediate resistance "
                   f"({resistance_distance:+.1f}% upside target) with {currency}{support_price:.2f} providing "
                   f"primary support ({support_distance:.1f}% downside protection), establishing clear "
                   f"risk-reward parameters for institutional position sizing.")
        elif nearest_resistance:
            resistance_price = nearest_resistance['price']
            resistance_distance = nearest_resistance['distance_percent']
            return (f"Technical resistance at {currency}{resistance_price:.2f} ({resistance_distance:+.1f}% upside) "
                   f"represents key breakout level for momentum continuation patterns.")
        elif nearest_support:
            support_price = nearest_support['price']
            support_distance = abs(nearest_support['distance_percent'])
            return (f"Primary support at {currency}{support_price:.2f} ({support_distance:.1f}% downside) "
                   f"provides institutional accumulation opportunity on technical pullbacks.")
        else:
            return ("Current price action operates in neutral technical territory without clearly defined "
                   "support or resistance levels, suggesting range-bound consolidation patterns.")

    def _generate_fallback_technical_narrative(self, ticker: str) -> str:
        """Generate fallback technical narrative when analysis fails."""
        return f"""
        <div style="margin-bottom: 20px;">
            <h5 style="color: #0c5460; margin-bottom: 15px;">📈 Technical Analysis</h5>
            <p style="line-height: 1.6; margin-bottom: 10px;">
                Technical analysis for {ticker} is currently unavailable due to insufficient historical data
                or data processing limitations. Please refer to fundamental analysis and market data for
                investment decision-making.
                <small>[Source: Technical Analysis System, {datetime.now().strftime('%Y-%m-%d')}]</small>
            </p>
        </div>
        """
