import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { Play, AlertCircle, CheckCircle, Clock, BarChart3 } from 'lucide-react';
import { reportService } from '../services/reportService';
import ProgressIndicator from './ProgressIndicator';

const ReportGenerator = ({ onReportGenerated, onReportUpdated }) => {
  const [ticker, setTicker] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentReport, setCurrentReport] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState('');
  
  const navigate = useNavigate();
  const wsRef = useRef(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    // Cleanup WebSocket on unmount
    return () => {
      isMountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const validateTicker = (value) => {
    const formatted = reportService.formatTicker(value);
    if (!formatted) return '';
    
    if (!reportService.validateHKTicker(formatted)) {
      setError('Please enter a valid Hong Kong stock ticker (format: XXXX.HK where XXXX is 1-5 digits)');
      return formatted;
    }
    
    setError('');
    return formatted;
  };

  const handleTickerChange = (e) => {
    const value = e.target.value;
    const formatted = validateTicker(value);
    setTicker(formatted);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!ticker) {
      setError('Please enter a ticker symbol');
      return;
    }

    if (!reportService.validateHKTicker(ticker)) {
      setError('Please enter a valid Hong Kong stock ticker (format: XXXX.HK where XXXX is 1-5 digits)');
      return;
    }

    try {
      setIsGenerating(true);
      setError('');
      setProgress(0);
      setStatusMessage('Initializing report generation...');

      // Start report generation
      const report = await reportService.generateReport(ticker);
      setCurrentReport(report);
      
      if (onReportGenerated) {
        onReportGenerated(report);
      }

      // Set up WebSocket for real-time updates
      wsRef.current = reportService.createWebSocket(
        report.report_id,
        handleWebSocketMessage,
        handleWebSocketError,
        handleWebSocketClose
      );

      toast.success(`Report generation started for ${ticker}`);

    } catch (error) {
      console.error('Failed to start report generation:', error);
      setError(error.message || 'Failed to start report generation');
      setIsGenerating(false);
      toast.error('Failed to start report generation');
    }
  };

  const handleWebSocketMessage = (data) => {
    console.log('WebSocket message:', data);

    // Check if component is still mounted
    if (!isMountedRef.current) {
      console.log('Component unmounted, ignoring WebSocket message');
      return;
    }

    // Ignore ping messages
    if (data.type === 'ping') {
      return;
    }

    setProgress(data.progress || 0);
    setStatusMessage(data.message || '');

    if (data.status === 'completed') {
      setIsGenerating(false);
      toast.success('Report generated successfully!');

      // Update the report with completion data
      if (currentReport && onReportUpdated) {
        const updatedReport = {
          ...currentReport,
          status: 'completed',
          charts_generated: data.charts_generated,
          processing_time: data.processing_time
        };
        onReportUpdated(updatedReport);
      }

      // Navigate to the report after a short delay, with null check
      const reportIdToNavigate = data.report_id || (currentReport && currentReport.report_id);
      if (reportIdToNavigate) {
        setTimeout(() => {
          // Check if component is still mounted before navigating
          if (!isMountedRef.current) {
            console.log('Component unmounted, skipping navigation');
            return;
          }

          // Double-check that we still have a valid report ID before navigating
          const finalReportId = currentReport?.report_id || reportIdToNavigate;
          if (finalReportId) {
            navigate(`/report/${finalReportId}`);
          } else {
            console.warn('Cannot navigate: report ID is null');
            toast.error('Report completed but navigation failed. Please check the report list.');
          }
        }, 2000);
      } else {
        console.warn('Cannot navigate: no report ID available');
        toast.error('Report completed but navigation failed. Please check the report list.');
      }

    } else if (data.status === 'failed') {
      setIsGenerating(false);
      setError(data.message || 'Report generation failed');
      toast.error('Report generation failed');

      if (currentReport && onReportUpdated) {
        const updatedReport = {
          ...currentReport,
          status: 'failed',
          error_message: data.message
        };
        onReportUpdated(updatedReport);
      }
    }
  };

  const handleWebSocketError = (error) => {
    console.error('WebSocket error:', error);

    // Only show error if component is still mounted
    if (isMountedRef.current) {
      toast.error('Connection error during report generation');
    }
  };

  const handleWebSocketClose = (event) => {
    console.log('WebSocket closed:', event);

    // Only handle close events if component is still mounted
    if (isMountedRef.current && isGenerating && event.code !== 1000) {
      // Unexpected close, try to reconnect or handle gracefully
      toast.error('Connection lost. Please check the report status.');
    }
  };

  const resetForm = () => {
    setTicker('');
    setIsGenerating(false);
    setCurrentReport(null);
    setProgress(0);
    setStatusMessage('');
    setError('');
    
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Generate Financial Report</h1>
        <p className="text-gray-600">
          Enter a Hong Kong stock ticker to generate a comprehensive financial analysis report
          with interactive charts and multi-agent insights.
        </p>
      </div>

      {/* Form */}
      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="ticker" className="block text-sm font-medium text-gray-700 mb-2">
              Hong Kong Stock Ticker
            </label>
            <div className="relative">
              <input
                type="text"
                id="ticker"
                value={ticker}
                onChange={handleTickerChange}
                placeholder="e.g., 0005.HK (HSBC Holdings)"
                className={`input-field ${error ? 'border-error-500 focus:border-error-500 focus:ring-error-500' : ''}`}
                disabled={isGenerating}
                maxLength={7}
              />
              <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                <BarChart3 className="w-5 h-5 text-gray-400" />
              </div>
            </div>
            {error && (
              <div className="mt-2 flex items-center space-x-2 text-error-600">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
            )}
            <p className="mt-2 text-sm text-gray-500">
              Format: XXXX.HK where XXXX is a 4-digit number (e.g., 0005.HK, 0700.HK, 0941.HK)
            </p>
          </div>

          <button
            type="submit"
            disabled={isGenerating || !ticker || !!error}
            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {isGenerating ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Generating Report...</span>
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                <span>Generate Report</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Progress Indicator */}
      {isGenerating && currentReport && (
        <div className="card">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">
                Generating Report for {currentReport.ticker}
              </h3>
              <span className="text-sm text-gray-500">
                Report ID: {currentReport.report_id.slice(0, 8)}...
              </span>
            </div>
            
            <ProgressIndicator
              progress={progress}
              message={statusMessage}
              status={currentReport.status}
            />

            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-2">What's happening:</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-success-500" />
                  <span>Multi-region market data collection</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-success-500" />
                  <span>Web content extraction and analysis</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-warning-500" />
                  <span>Multi-agent financial analysis</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <span>Interactive chart generation (4 charts)</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <span>HTML report compilation</span>
                </li>
              </ul>
            </div>

            <button
              onClick={resetForm}
              className="w-full btn-secondary"
            >
              Cancel and Start Over
            </button>
          </div>
        </div>
      )}

      {/* Information */}
      <div className="card bg-blue-50 border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">Report Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <h4 className="font-medium text-blue-800">Analysis Components:</h4>
            <ul className="text-blue-700 space-y-1">
              <li>• Multi-region market data</li>
              <li>• Web content extraction</li>
              <li>• Yahoo Finance integration</li>
              <li>• Multi-agent AI analysis</li>
            </ul>
          </div>
          <div className="space-y-2">
            <h4 className="font-medium text-blue-800">Interactive Charts:</h4>
            <ul className="text-blue-700 space-y-1">
              <li>• Price trend analysis</li>
              <li>• Technical indicators (RSI, MACD)</li>
              <li>• Volume analysis</li>
              <li>• Performance comparison</li>
            </ul>
          </div>
        </div>
        <p className="text-blue-600 text-sm mt-4">
          <strong>Estimated time:</strong> 60-90 seconds for complete analysis
        </p>
      </div>
    </div>
  );
};

export default ReportGenerator;
