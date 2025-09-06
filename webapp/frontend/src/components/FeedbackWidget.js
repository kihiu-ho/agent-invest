import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, MessageCircle, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { reportService } from '../services/reportService';

const FeedbackWidget = ({ reportId }) => {
  const [feedbackState, setFeedbackState] = useState('initial'); // 'initial', 'loading', 'submitted', 'error'
  const [selectedFeedback, setSelectedFeedback] = useState(null); // 'thumbs_up' or 'thumbs_down'
  const [showDetailedOption, setShowDetailedOption] = useState(false);

  // Generate a unique session ID for this user session
  const getUserSessionId = () => {
    let sessionId = sessionStorage.getItem('agentinvest_session_id');
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('agentinvest_session_id', sessionId);
    }
    return sessionId;
  };

  const submitFeedback = async (feedbackType) => {
    try {
      setFeedbackState('loading');
      setSelectedFeedback(feedbackType);

      const feedbackData = {
        report_id: reportId,
        feedback_type: feedbackType,
        category: 'overall_quality',
        rating: feedbackType === 'thumbs_up' ? 5 : 1, // Convert thumbs to rating scale
        user_session_id: getUserSessionId(),
        feedback_context: {
          source: 'quick_feedback_widget',
          timestamp: new Date().toISOString(),
          user_agent: navigator.userAgent,
          page_url: window.location.href
        }
      };

      await reportService.submitFeedback(feedbackData);
      
      setFeedbackState('submitted');
      setShowDetailedOption(true);
      
      toast.success(
        feedbackType === 'thumbs_up' 
          ? 'Thank you for your positive feedback!' 
          : 'Thank you for your feedback. We\'ll work to improve our reports.'
      );

    } catch (error) {
      console.error('Failed to submit feedback:', error);
      setFeedbackState('error');
      toast.error('Failed to submit feedback. Please try again.');
    }
  };

  const handleDetailedFeedback = () => {
    // For now, just show a message. In Phase 2, this would open a detailed feedback modal
    toast.success('Detailed feedback feature coming soon! Your quick feedback has been recorded.');
  };

  const handleRetry = () => {
    setFeedbackState('initial');
    setSelectedFeedback(null);
    setShowDetailedOption(false);
  };

  if (feedbackState === 'submitted') {
    return (
      <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
        <div className="flex items-center space-x-3">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <div className="flex-1">
            <h4 className="text-sm font-medium text-green-800">
              Feedback Submitted Successfully
            </h4>
            <p className="text-sm text-green-700 mt-1">
              Your feedback helps us improve our AI investment analysis. Thank you!
            </p>
          </div>
        </div>
        
        {showDetailedOption && (
          <div className="mt-3 pt-3 border-t border-green-200">
            <button
              onClick={handleDetailedFeedback}
              className="text-sm text-green-700 hover:text-green-800 font-medium inline-flex items-center space-x-1"
            >
              <MessageCircle className="w-4 h-4" />
              <span>Provide detailed feedback</span>
            </button>
          </div>
        )}
      </div>
    );
  }

  if (feedbackState === 'error') {
    return (
      <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-red-600" />
          <div className="flex-1">
            <h4 className="text-sm font-medium text-red-800">
              Failed to Submit Feedback
            </h4>
            <p className="text-sm text-red-700 mt-1">
              There was an error submitting your feedback. Please try again.
            </p>
          </div>
          <button
            onClick={handleRetry}
            className="text-sm text-red-700 hover:text-red-800 font-medium"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
      <div className="text-center">
        <h4 className="text-lg font-medium text-gray-900 mb-4">
          Was this report helpful?
        </h4>
        
        <div className="flex justify-center space-x-4">
          <button
            onClick={() => submitFeedback('thumbs_up')}
            disabled={feedbackState === 'loading'}
            className={`
              inline-flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200
              ${feedbackState === 'loading' && selectedFeedback === 'thumbs_up'
                ? 'bg-green-100 text-green-700 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white hover:scale-105'
              }
              ${feedbackState === 'loading' ? 'opacity-75' : ''}
            `}
          >
            {feedbackState === 'loading' && selectedFeedback === 'thumbs_up' ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-600"></div>
                <span>Submitting...</span>
              </>
            ) : (
              <>
                <ThumbsUp className="w-5 h-5" />
                <span>Yes, helpful</span>
              </>
            )}
          </button>

          <button
            onClick={() => submitFeedback('thumbs_down')}
            disabled={feedbackState === 'loading'}
            className={`
              inline-flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200
              ${feedbackState === 'loading' && selectedFeedback === 'thumbs_down'
                ? 'bg-red-100 text-red-700 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700 text-white hover:scale-105'
              }
              ${feedbackState === 'loading' ? 'opacity-75' : ''}
            `}
          >
            {feedbackState === 'loading' && selectedFeedback === 'thumbs_down' ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-red-600"></div>
                <span>Submitting...</span>
              </>
            ) : (
              <>
                <ThumbsDown className="w-5 h-5" />
                <span>Needs improvement</span>
              </>
            )}
          </button>
        </div>

        <p className="text-sm text-gray-600 mt-3">
          Your feedback helps improve our AI investment analysis
        </p>
      </div>
    </div>
  );
};

export default FeedbackWidget;
