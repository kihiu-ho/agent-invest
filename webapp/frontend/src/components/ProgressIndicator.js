import React from 'react';
import { Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

const ProgressIndicator = ({ progress, message, status }) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-success-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-error-500" />;
      case 'processing':
      case 'starting':
        return <Clock className="w-5 h-5 text-warning-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'text-success-600';
      case 'failed':
        return 'text-error-600';
      case 'processing':
      case 'starting':
        return 'text-warning-600';
      default:
        return 'text-gray-600';
    }
  };

  const getProgressBarColor = () => {
    switch (status) {
      case 'completed':
        return 'bg-success-500';
      case 'failed':
        return 'bg-error-500';
      case 'processing':
      case 'starting':
        return 'bg-primary-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="space-y-4">
      {/* Status and Progress */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {getStatusIcon()}
          <span className={`font-medium ${getStatusColor()}`}>
            {status === 'starting' && 'Starting...'}
            {status === 'processing' && 'Processing...'}
            {status === 'completed' && 'Completed'}
            {status === 'failed' && 'Failed'}
            {!['starting', 'processing', 'completed', 'failed'].includes(status) && 'Queued'}
          </span>
        </div>
        <span className="text-sm font-medium text-gray-600">
          {Math.round(progress)}%
        </span>
      </div>

      {/* Progress Bar */}
      <div className="progress-bar">
        <div
          className={`progress-fill ${getProgressBarColor()}`}
          style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
        />
      </div>

      {/* Status Message */}
      {message && (
        <p className="text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
          {message}
        </p>
      )}

      {/* Progress Steps */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-xs">
        <div className={`flex items-center space-x-1 ${progress >= 25 ? 'text-success-600' : 'text-gray-400'}`}>
          <div className={`w-2 h-2 rounded-full ${progress >= 25 ? 'bg-success-500' : 'bg-gray-300'}`} />
          <span>Data Collection</span>
        </div>
        <div className={`flex items-center space-x-1 ${progress >= 50 ? 'text-success-600' : 'text-gray-400'}`}>
          <div className={`w-2 h-2 rounded-full ${progress >= 50 ? 'bg-success-500' : 'bg-gray-300'}`} />
          <span>Analysis</span>
        </div>
        <div className={`flex items-center space-x-1 ${progress >= 75 ? 'text-success-600' : 'text-gray-400'}`}>
          <div className={`w-2 h-2 rounded-full ${progress >= 75 ? 'bg-success-500' : 'bg-gray-300'}`} />
          <span>Chart Generation</span>
        </div>
        <div className={`flex items-center space-x-1 ${progress >= 100 ? 'text-success-600' : 'text-gray-400'}`}>
          <div className={`w-2 h-2 rounded-full ${progress >= 100 ? 'bg-success-500' : 'bg-gray-300'}`} />
          <span>Report Ready</span>
        </div>
      </div>
    </div>
  );
};

export default ProgressIndicator;
