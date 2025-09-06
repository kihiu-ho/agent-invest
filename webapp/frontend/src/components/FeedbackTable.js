import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  ThumbsUp, 
  ThumbsDown, 
  ExternalLink, 
  Search, 
  Filter,
  ChevronLeft,
  ChevronRight,
  Star
} from 'lucide-react';

const FeedbackTable = ({ feedbackData, onRefresh }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // Early return if no feedback data is available
  if (!feedbackData || !feedbackData.feedback || !Array.isArray(feedbackData.feedback)) {
    return (
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Recent Feedback</h3>
        </div>
        <div className="p-6 text-center">
          <div className="text-gray-500">
            <div className="animate-pulse">Loading feedback data...</div>
          </div>
        </div>
      </div>
    );
  }

  // Filter feedback based on search and filter criteria
  const filteredFeedback = feedbackData.feedback.filter(feedback => {
    // Safety checks for feedback properties
    if (!feedback) return false;

    const reportId = feedback.report_id || '';
    const sessionId = feedback.user_session_id || '';
    const source = feedback.source || '';
    const feedbackType = feedback.feedback_type || '';

    const matchesSearch = searchTerm === '' ||
      reportId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sessionId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      source.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesFilter =
      filterType === 'all' ||
      feedbackType === filterType;

    return matchesSearch && matchesFilter;
  });

  // Pagination
  const totalPages = Math.ceil(filteredFeedback.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedFeedback = filteredFeedback.slice(startIndex, startIndex + itemsPerPage);

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getFeedbackIcon = (type) => {
    return type === 'thumbs_up' ? (
      <ThumbsUp className="w-4 h-4 text-green-600" />
    ) : (
      <ThumbsDown className="w-4 h-4 text-red-600" />
    );
  };

  const getFeedbackBadge = (type) => {
    return type === 'thumbs_up' ? (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
        <ThumbsUp className="w-3 h-3 mr-1" />
        Positive
      </span>
    ) : (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
        <ThumbsDown className="w-3 h-3 mr-1" />
        Negative
      </span>
    );
  };

  const getRatingStars = (rating) => {
    if (!rating) return <span className="text-gray-400">N/A</span>;
    
    return (
      <div className="flex items-center">
        {[...Array(5)].map((_, i) => (
          <Star
            key={i}
            className={`w-4 h-4 ${
              i < rating ? 'text-yellow-400 fill-current' : 'text-gray-300'
            }`}
          />
        ))}
        <span className="ml-1 text-sm text-gray-600">({rating})</span>
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Table Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Recent Feedback</h3>
            <p className="mt-1 text-sm text-gray-500">
              {filteredFeedback.length} of {feedbackData.total_recent || feedbackData.feedback?.length || 0} feedback submissions
            </p>
          </div>
          
          {/* Search and Filter Controls */}
          <div className="mt-4 sm:mt-0 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-3">
            {/* Search */}
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-gray-400" />
              </div>
              <input
                type="text"
                placeholder="Search feedback..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md text-sm placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            
            {/* Filter */}
            <div className="relative">
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="block w-full pl-3 pr-10 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="all">All Feedback</option>
                <option value="thumbs_up">Positive Only</option>
                <option value="thumbs_down">Negative Only</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Report
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Feedback
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Rating
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                User Session
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Date
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Source
              </th>
              <th className="relative px-6 py-3">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {paginatedFeedback.map((feedback) => {
              // Safety checks for feedback properties
              const feedbackId = feedback?.feedback_id || 'unknown';
              const reportId = feedback?.report_id || 'unknown';
              const feedbackType = feedback?.feedback_type || 'unknown';
              const rating = feedback?.rating;
              const sessionId = feedback?.user_session_id || 'unknown';
              const createdAt = feedback?.created_at;
              const source = feedback?.source || 'unknown';

              return (
                <tr key={feedbackId} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div>
                        <div className="text-sm font-medium text-gray-900">
                          {reportId.length > 8 ? `${reportId.substring(0, 8)}...` : reportId}
                        </div>
                        <div className="text-sm text-gray-500">
                          ID: {feedbackId.length > 8 ? `${feedbackId.substring(0, 8)}...` : feedbackId}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {getFeedbackBadge(feedbackType)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {getRatingStars(rating)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <code className="bg-gray-100 px-2 py-1 rounded text-xs">
                      {sessionId}
                    </code>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {createdAt ? formatDate(createdAt) : 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {source}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <Link
                      to={`/report/${reportId}`}
                      className="text-blue-600 hover:text-blue-900 inline-flex items-center"
                    >
                      <ExternalLink className="w-4 h-4 mr-1" />
                      View Report
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="px-6 py-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-700">
              Showing {startIndex + 1} to {Math.min(startIndex + itemsPerPage, filteredFeedback.length)} of{' '}
              {filteredFeedback.length} results
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                Previous
              </button>
              
              <span className="text-sm text-gray-700">
                Page {currentPage} of {totalPages}
              </span>
              
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight className="w-4 h-4 ml-1" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {filteredFeedback.length === 0 && (
        <div className="px-6 py-12 text-center">
          <div className="text-gray-500">
            <Filter className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No feedback found</h3>
            <p className="text-sm">
              {searchTerm || filterType !== 'all' 
                ? 'Try adjusting your search or filter criteria.'
                : 'No feedback submissions yet.'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeedbackTable;
