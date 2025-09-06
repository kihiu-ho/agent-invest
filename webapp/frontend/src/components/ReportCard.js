import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Eye, Clock, CheckCircle, XCircle, BarChart3, Calendar } from 'lucide-react';
import { reportService } from '../services/reportService';

const ReportCard = ({ report, onReportUpdated }) => {
  const [isPolling, setIsPolling] = useState(false);

  useEffect(() => {
    // Poll for status updates if report is still processing
    if (report.status === 'processing' || report.status === 'starting' || report.status === 'queued') {
      setIsPolling(true);
      const interval = setInterval(async () => {
        try {
          const updatedReport = await reportService.getReport(report.report_id);
          if (onReportUpdated) {
            onReportUpdated(updatedReport);
          }
          
          // Stop polling if completed or failed
          if (updatedReport.status === 'completed' || updatedReport.status === 'failed') {
            setIsPolling(false);
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Failed to poll report status:', error);
        }
      }, 5000); // Poll every 5 seconds

      return () => {
        clearInterval(interval);
        setIsPolling(false);
      };
    }
  }, [report.status, report.report_id, onReportUpdated]);

  const getStatusBadge = () => {
    switch (report.status) {
      case 'completed':
        return (
          <span className="status-badge status-completed">
            <CheckCircle className="w-3 h-3 mr-1" />
            Completed
          </span>
        );
      case 'processing':
      case 'starting':
        return (
          <span className="status-badge status-processing">
            <Clock className="w-3 h-3 mr-1" />
            {isPolling && <div className="animate-spin w-3 h-3 mr-1 border border-warning-600 border-t-transparent rounded-full" />}
            Processing
          </span>
        );
      case 'failed':
        return (
          <span className="status-badge status-failed">
            <XCircle className="w-3 h-3 mr-1" />
            Failed
          </span>
        );
      default:
        return (
          <span className="status-badge status-queued">
            <Clock className="w-3 h-3 mr-1" />
            Queued
          </span>
        );
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const formatProcessingTime = (seconds) => {
    if (!seconds) return 'N/A';
    return `${seconds.toFixed(1)}s`;
  };

  return (
    <div className="card hover:shadow-md transition-shadow duration-200">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex-shrink-0">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-primary-600" />
            </div>
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-3">
              <h3 className="text-lg font-semibold text-gray-900">
                {report.ticker}
              </h3>
              {getStatusBadge()}
            </div>
            
            <div className="mt-1 flex items-center space-x-4 text-sm text-gray-600">
              <div className="flex items-center space-x-1">
                <Calendar className="w-4 h-4" />
                <span>{formatDate(report.created_at)}</span>
              </div>
              
              {report.charts_generated !== undefined && (
                <div className="flex items-center space-x-1">
                  <BarChart3 className="w-4 h-4" />
                  <span>{report.charts_generated} charts</span>
                </div>
              )}
              
              {report.processing_time && (
                <div className="flex items-center space-x-1">
                  <Clock className="w-4 h-4" />
                  <span>{formatProcessingTime(report.processing_time)}</span>
                </div>
              )}
            </div>
            
            {report.error_message && (
              <div className="mt-2 text-sm text-error-600 bg-error-50 rounded-lg p-2">
                {report.error_message}
              </div>
            )}
          </div>
        </div>
        
        <div className="flex-shrink-0">
          {report.status === 'completed' ? (
            <Link
              to={`/report/${report.report_id}`}
              className="btn-primary inline-flex items-center space-x-2"
            >
              <Eye className="w-4 h-4" />
              <span>View Report</span>
            </Link>
          ) : report.status === 'processing' || report.status === 'starting' ? (
            <Link
              to={`/report/${report.report_id}`}
              className="btn-secondary inline-flex items-center space-x-2"
            >
              <Clock className="w-4 h-4" />
              <span>View Progress</span>
            </Link>
          ) : (
            <button
              disabled
              className="btn-secondary opacity-50 cursor-not-allowed inline-flex items-center space-x-2"
            >
              <XCircle className="w-4 h-4" />
              <span>Unavailable</span>
            </button>
          )}
        </div>
      </div>
      
      {/* Progress bar for processing reports */}
      {(report.status === 'processing' || report.status === 'starting') && (
        <div className="mt-4">
          <div className="progress-bar">
            <div className="progress-fill bg-primary-500 animate-pulse" style={{ width: '60%' }} />
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Report generation in progress... This may take 60-90 seconds.
          </p>
        </div>
      )}
    </div>
  );
};

export default ReportCard;
