// Enhanced Candlestick Chart with SMA and EMA
// Requires Chart.js and Chart.js Financial plugin

// Function to calculate SMA (Simple Moving Average)
function calculateSMA(data, period) {
    let sma = [];
    for (let i = period - 1; i < data.length; i++) {
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += data[i - j].c;
        }
        sma.push({ 
            x: data[i].x, 
            y: sum / period 
        });
    }
    return sma;
}

// Function to calculate EMA (Exponential Moving Average)
function calculateEMA(data, period) {
    let ema = [];
    let k = 2 / (period + 1);
    let prevEma = data[0].c;
    
    for (let i = 0; i < data.length; i++) {
        let price = data[i].c;
        if (i === 0) {
            prevEma = price;
        } else {
            prevEma = (price - prevEma) * k + prevEma;
        }
        ema.push({ 
            x: data[i].x, 
            y: prevEma 
        });
    }
    return ema;
}

// Function to format OHLC data for Chart.js
function formatOHLCData(historicalData) {
    const dates = historicalData.dates || [];
    const open = historicalData.open || [];
    const high = historicalData.high || [];
    const low = historicalData.low || [];
    const close = historicalData.close || [];
    
    let candleData = [];
    const minLength = Math.min(dates.length, open.length, high.length, low.length, close.length);
    
    for (let i = 0; i < minLength; i++) {
        candleData.push({
            x: dates[i],
            o: open[i],
            h: high[i],
            l: low[i],
            c: close[i]
        });
    }
    
    return candleData;
}

// Enhanced Chart Configuration Generator
function createEnhancedCandlestickConfig(historicalData, ticker, chartType = 'candlestick') {
    const candleData = formatOHLCData(historicalData);
    
    // Calculate moving averages
    const sma20 = calculateSMA(candleData, 20);
    const sma50 = calculateSMA(candleData, 50);
    const ema20 = calculateEMA(candleData, 20);
    const ema50 = calculateEMA(candleData, 50);
    
    // Base datasets
    let datasets = [];
    
    // Add candlestick or line chart data
    if (chartType === 'candlestick' && candleData.length > 0) {
        datasets.push({
            label: `${ticker} Price`,
            data: candleData,
            type: 'candlestick',
            borderColor: '#2E8B57',
            backgroundColor: 'rgba(46, 139, 87, 0.1)'
        });
    } else {
        // Fallback to line chart
        const lineData = candleData.map(d => ({ x: d.x, y: d.c }));
        datasets.push({
            label: `${ticker} Close Price`,
            data: lineData,
            type: 'line',
            borderColor: '#2E8B57',
            backgroundColor: 'rgba(46, 139, 87, 0.1)',
            fill: false,
            tension: 0.1
        });
    }
    
    // Add SMA datasets
    if (sma20.length > 0) {
        datasets.push({
            label: 'SMA 20',
            data: sma20,
            type: 'line',
            borderColor: '#FF6B6B',
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
            hidden: true
        });
    }
    
    if (sma50.length > 0) {
        datasets.push({
            label: 'SMA 50',
            data: sma50,
            type: 'line',
            borderColor: '#4ECDC4',
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
            hidden: true
        });
    }
    
    // Add EMA datasets
    if (ema20.length > 0) {
        datasets.push({
            label: 'EMA 20',
            data: ema20,
            type: 'line',
            borderColor: '#45B7D1',
            backgroundColor: 'transparent',
            borderDash: [2, 2],
            fill: false,
            pointRadius: 0,
            hidden: true
        });
    }
    
    if (ema50.length > 0) {
        datasets.push({
            label: 'EMA 50',
            data: ema50,
            type: 'line',
            borderColor: '#F7DC6F',
            backgroundColor: 'transparent',
            borderDash: [2, 2],
            fill: false,
            pointRadius: 0,
            hidden: true
        });
    }
    
    // Chart configuration
    const config = {
        type: 'candlestick',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: true,
                    text: `${ticker} - Enhanced Price Chart with Technical Analysis`
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        title: function(context) {
                            return new Date(context[0].parsed.x).toLocaleDateString();
                        },
                        label: function(context) {
                            const dataset = context.dataset;
                            if (dataset.type === 'candlestick') {
                                const data = context.raw;
                                return [
                                    `Open: $${data.o.toFixed(2)}`,
                                    `High: $${data.h.toFixed(2)}`,
                                    `Low: $${data.l.toFixed(2)}`,
                                    `Close: $${data.c.toFixed(2)}`
                                ];
                            } else {
                                return `${dataset.label}: $${context.parsed.y.toFixed(2)}`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        parser: 'yyyy-MM-dd\'T\'HH:mm:ss.SSSxxx',
                        displayFormats: {
                            day: 'MMM dd',
                            month: 'MMM yyyy'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price (HK$)'
                    },
                    beginAtZero: false
                }
            }
        }
    };
    
    return config;
}

// Chart type switching function
function switchChartType(chart, newType) {
    if (!chart || !chart.data || !chart.data.datasets) return;
    
    const priceDataset = chart.data.datasets[0];
    if (!priceDataset) return;
    
    if (newType === 'candlestick' && priceDataset.type !== 'candlestick') {
        // Convert line to candlestick (if OHLC data available)
        if (priceDataset.data[0] && typeof priceDataset.data[0].o !== 'undefined') {
            priceDataset.type = 'candlestick';
            chart.config.type = 'candlestick';
        }
    } else if (newType === 'line' && priceDataset.type !== 'line') {
        // Convert candlestick to line
        if (priceDataset.data[0] && typeof priceDataset.data[0].c !== 'undefined') {
            const lineData = priceDataset.data.map(d => ({ x: d.x, y: d.c }));
            priceDataset.data = lineData;
            priceDataset.type = 'line';
            priceDataset.fill = false;
            priceDataset.tension = 0.1;
            chart.config.type = 'line';
        }
    }
    
    chart.update();
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        calculateSMA,
        calculateEMA,
        formatOHLCData,
        createEnhancedCandlestickConfig,
        switchChartType
    };
}
