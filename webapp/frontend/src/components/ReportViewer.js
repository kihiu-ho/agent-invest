import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Download, ExternalLink, Clock, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { reportService } from '../services/reportService';
import ProgressIndicator from './ProgressIndicator';
import FeedbackWidget from './FeedbackWidget';

const ReportViewer = () => {
  const { reportId } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isPolling, setIsPolling] = useState(false);

  const loadReport = useCallback(async () => {
    try {
      setError('');
      const data = await reportService.getReport(reportId);
      setReport(data);
      if (data.status === 'completed' || data.status === 'failed') {
        setIsPolling(false);
      }
    } catch (error) {
      console.error('Failed to load report:', error);
      setError(error.message || 'Failed to load report');
      toast.error('Failed to load report');
    } finally {
      setLoading(false);
    }
  }, [reportId]);

  useEffect(() => {
    loadReport();
  }, [reportId, loadReport]);

  useEffect(() => {
    // Set up polling for processing reports
    if (report && (report.status === 'processing' || report.status === 'starting' || report.status === 'queued')) {
      setIsPolling(true);
      const interval = setInterval(loadReport, 5000); // Poll every 5 seconds

      return () => {
        clearInterval(interval);
        setIsPolling(false);
      };
    } else {
      setIsPolling(false);
    }
  }, [report?.status, loadReport]);

  const handleRefresh = () => {
    setLoading(true);
    loadReport();
  };

  const getReportFileUrl = () => {
    return reportService.getReportFileUrl(reportId);
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="h-96 bg-gray-200 rounded-lg"></div>
        </div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-2xl mx-auto text-center space-y-6">
        <div className="card">
          <XCircle className="w-16 h-16 text-error-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Report Not Found</h2>
          <p className="text-gray-600 mb-6">
            {error || 'The requested report could not be found or loaded.'}
          </p>
          <div className="flex justify-center space-x-4">
            <Link to="/" className="btn-primary">
              Back to Dashboard
            </Link>
            <button onClick={handleRefresh} className="btn-secondary">
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link
            to="/"
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Dashboard</span>
          </Link>
          <div className="h-6 border-l border-gray-300"></div>
          <h1 className="text-2xl font-bold text-gray-900">
            Report for {report.ticker}
          </h1>
        </div>

        <div className="flex items-center space-x-3">
          {isPolling && (
            <button
              onClick={handleRefresh}
              className="btn-secondary inline-flex items-center space-x-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
          )}
          
          {report.status === 'completed' && (
            <a
              href={getReportFileUrl()}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-primary inline-flex items-center space-x-2"
            >
              <ExternalLink className="w-4 h-4" />
              <span>Open Full Report</span>
            </a>
          )}
        </div>
      </div>

      {/* Report Status */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Report Status</h2>
          <div className="flex items-center space-x-2">
            {report.status === 'completed' && <CheckCircle className="w-5 h-5 text-success-500" />}
            {report.status === 'failed' && <XCircle className="w-5 h-5 text-error-500" />}
            {(report.status === 'processing' || report.status === 'starting') && <Clock className="w-5 h-5 text-warning-500" />}
            <span className="font-medium capitalize">{report.status}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Created:</span>
            <p className="font-medium">{new Date(report.created_at).toLocaleString()}</p>
          </div>
          {report.charts_generated !== undefined && (
            <div>
              <span className="text-gray-600">Charts Generated:</span>
              <p className="font-medium">{report.charts_generated}</p>
            </div>
          )}
          {report.processing_time && (
            <div>
              <span className="text-gray-600">Processing Time:</span>
              <p className="font-medium">{report.processing_time.toFixed(1)}s</p>
            </div>
          )}
        </div>

        {report.error_message && (
          <div className="mt-4 p-4 bg-error-50 border border-error-200 rounded-lg">
            <p className="text-error-700">{report.error_message}</p>
          </div>
        )}
      </div>

      {/* Progress Indicator for Processing Reports */}
      {(report.status === 'processing' || report.status === 'starting' || report.status === 'queued') && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Generation Progress</h2>
          <ProgressIndicator
            progress={report.status === 'starting' ? 10 : report.status === 'processing' ? 50 : 0}
            message={`Generating comprehensive financial analysis for ${report.ticker}...`}
            status={report.status}
          />
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <p className="text-blue-700 text-sm">
              <strong>Please wait:</strong> Report generation typically takes 60-90 seconds. 
              The page will automatically update when the report is ready.
            </p>
          </div>
        </div>
      )}

      {/* Report Content */}
      {report.status === 'completed' && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Financial Analysis Report</h2>
            <a
              href={getReportFileUrl()}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-600 hover:text-primary-700 text-sm font-medium inline-flex items-center space-x-1"
            >
              <Download className="w-4 h-4" />
              <span>Download Report</span>
            </a>
          </div>

          {/* Embedded Report */}
          <div className="border border-gray-200 rounded-lg overflow-hidden">
            <iframe
              src={getReportFileUrl()}
              className="w-full h-screen border-0"
              title={`Financial Report for ${report.ticker}`}
              sandbox="allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox"
              referrerPolicy="strict-origin-when-cross-origin"
            />
          </div>

          <div className="mt-4 text-sm text-gray-600">
            <p>
              This report includes interactive charts and comprehensive financial analysis.
              For the best experience, open the report in a new tab using the button above.
            </p>
          </div>

          {/* Feedback Widget */}
          <FeedbackWidget reportId={reportId} />
        </div>
      )}

      {/* Failed Report */}
      {report.status === 'failed' && (
        <div className="card text-center">
          <XCircle className="w-16 h-16 text-error-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Report Generation Failed</h3>
          <p className="text-gray-600 mb-6">
            The report generation process encountered an error. Please try generating a new report.
          </p>
          <Link to="/generate" className="btn-primary">
            Generate New Report
          </Link>
        </div>
      )}
    </div>
  );
};

export default ReportViewer;
