import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Plus, RefreshCw, Search, Filter, BarChart3 } from 'lucide-react';
import ReportCard from './ReportCard';

const ReportHistory = ({ reports, loading, onRefresh }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortBy, setSortBy] = useState('newest');

  // Filter and sort reports
  const filteredReports = React.useMemo(() => {
    let filtered = reports;

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(report =>
        report.ticker.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Filter by status
    if (statusFilter !== 'all') {
      filtered = filtered.filter(report => report.status === statusFilter);
    }

    // Sort reports
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'newest':
          return new Date(b.created_at) - new Date(a.created_at);
        case 'oldest':
          return new Date(a.created_at) - new Date(b.created_at);
        case 'ticker':
          return a.ticker.localeCompare(b.ticker);
        default:
          return 0;
      }
    });

    return filtered;
  }, [reports, searchTerm, statusFilter, sortBy]);

  const getStatusCounts = () => {
    return {
      all: reports.length,
      completed: reports.filter(r => r.status === 'completed').length,
      processing: reports.filter(r => r.status === 'processing' || r.status === 'starting').length,
      failed: reports.filter(r => r.status === 'failed').length,
    };
  };

  const statusCounts = getStatusCounts();

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="h-16 bg-gray-200 rounded-lg mb-6"></div>
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Report History</h1>
          <p className="mt-2 text-gray-600">
            View and manage all your generated financial analysis reports
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex items-center space-x-3">
          <button
            onClick={onRefresh}
            className="btn-secondary inline-flex items-center space-x-2"
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <Link
            to="/generate"
            className="btn-primary inline-flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>New Report</span>
          </Link>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="card">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search by ticker..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input-field pl-10"
            />
          </div>

          {/* Status Filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="input-field pl-10 appearance-none"
            >
              <option value="all">All Status ({statusCounts.all})</option>
              <option value="completed">Completed ({statusCounts.completed})</option>
              <option value="processing">Processing ({statusCounts.processing})</option>
              <option value="failed">Failed ({statusCounts.failed})</option>
            </select>
          </div>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="input-field"
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
            <option value="ticker">By Ticker</option>
          </select>
        </div>
      </div>

      {/* Results Summary */}
      <div className="flex items-center justify-between text-sm text-gray-600">
        <span>
          Showing {filteredReports.length} of {reports.length} reports
        </span>
        {(searchTerm || statusFilter !== 'all') && (
          <button
            onClick={() => {
              setSearchTerm('');
              setStatusFilter('all');
            }}
            className="text-primary-600 hover:text-primary-700 font-medium"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Reports List */}
      {filteredReports.length === 0 ? (
        <div className="card text-center py-12">
          <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {reports.length === 0 ? 'No reports yet' : 'No reports match your filters'}
          </h3>
          <p className="text-gray-600 mb-6">
            {reports.length === 0
              ? 'Generate your first financial analysis report to get started'
              : 'Try adjusting your search terms or filters'
            }
          </p>
          {reports.length === 0 ? (
            <Link to="/generate" className="btn-primary">
              Generate Your First Report
            </Link>
          ) : (
            <button
              onClick={() => {
                setSearchTerm('');
                setStatusFilter('all');
              }}
              className="btn-secondary"
            >
              Clear Filters
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredReports.map((report) => (
            <ReportCard
              key={report.report_id}
              report={report}
              onReportUpdated={(updatedReport) => {
                // This will be handled by the parent component
                // through the reports prop update
              }}
            />
          ))}
        </div>
      )}

      {/* Load More (if needed for pagination in the future) */}
      {filteredReports.length > 0 && filteredReports.length < reports.length && (
        <div className="text-center">
          <p className="text-sm text-gray-600">
            Showing {filteredReports.length} of {reports.length} reports
          </p>
        </div>
      )}
    </div>
  );
};

export default ReportHistory;
