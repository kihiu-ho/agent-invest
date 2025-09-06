import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  BarChart3,
  TrendingUp,
  Users,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  Download,
  Filter,
  RefreshCw,
  Calendar,
  FileText
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import FeedbackCharts from './FeedbackCharts';
import FeedbackTable from './FeedbackTable';
import reportService from '../services/reportService';

const FeedbackAnalytics = () => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [trendsData, setTrendsData] = useState(null);
  const [reportsData, setReportsData] = useState(null);
  const [recentFeedback, setRecentFeedback] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeFilter, setTimeFilter] = useState(7); // Default to 7 days
  const [refreshing, setRefreshing] = useState(false);

  const fetchAnalyticsData = async () => {
    try {
      setRefreshing(true);

      // Fetch all analytics data in parallel using reportService
      const [overview, trends, reports, recent] = await Promise.all([
        reportService.getAnalyticsOverview(),
        reportService.getAnalyticsTrends(timeFilter),
        reportService.getAnalyticsReports(10),
        reportService.getRecentFeedback(20)
      ]);

      setAnalyticsData(overview);
      setTrendsData(trends);
      setReportsData(reports);
      setRecentFeedback(recent);

    } catch (error) {
      console.error('Failed to fetch analytics data:', error);
      toast.error('Failed to load analytics data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleExport = async (format) => {
    try {
      const response = await reportService.exportAnalytics(format, timeFilter);

      if (format === 'csv') {
        const blob = response.data;
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `feedback_export_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success('CSV export downloaded successfully');
      } else {
        const blob = response.data;
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `feedback_export_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success('JSON export downloaded successfully');
      }
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export data');
    }
  };

  const handleTimeFilterChange = (days) => {
    setTimeFilter(days);
  };

  useEffect(() => {
    fetchAnalyticsData();
  }, [timeFilter]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading analytics data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-6">
            {/* Breadcrumb */}
            <nav className="flex mb-4" aria-label="Breadcrumb">
              <ol className="inline-flex items-center space-x-1 md:space-x-3">
                <li className="inline-flex items-center">
                  <Link to="/" className="text-gray-500 hover:text-gray-700">
                    Dashboard
                  </Link>
                </li>
                <li>
                  <div className="flex items-center">
                    <span className="text-gray-400">/</span>
                    <span className="ml-1 text-gray-900 font-medium">Feedback Analytics</span>
                  </div>
                </li>
              </ol>
            </nav>

            {/* Page Header */}
            <div className="flex justify-between items-center">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                  <BarChart3 className="w-8 h-8 mr-3 text-blue-600" />
                  Feedback Analytics
                </h1>
                <p className="mt-2 text-gray-600">
                  Insights into user feedback and report quality metrics
                </p>
              </div>
              
              <div className="flex items-center space-x-3">
                {/* Time Filter */}
                <div className="flex items-center space-x-2">
                  <Calendar className="w-4 h-4 text-gray-500" />
                  <select
                    value={timeFilter}
                    onChange={(e) => handleTimeFilterChange(parseInt(e.target.value))}
                    className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value={7}>Last 7 days</option>
                    <option value={30}>Last 30 days</option>
                    <option value={90}>Last 90 days</option>
                    <option value={365}>Last year</option>
                  </select>
                </div>

                {/* Export Buttons */}
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleExport('csv')}
                    className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    CSV
                  </button>
                  <button
                    onClick={() => handleExport('json')}
                    className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    JSON
                  </button>
                </div>

                {/* Refresh Button */}
                <button
                  onClick={fetchAnalyticsData}
                  disabled={refreshing}
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
                  Refresh
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Overview Stats */}
        {analyticsData && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <MessageSquare className="h-8 w-8 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Total Feedback</p>
                  <p className="text-2xl font-bold text-gray-900">{analyticsData.total_feedback}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <ThumbsUp className="h-8 w-8 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Positive Feedback</p>
                  <p className="text-2xl font-bold text-gray-900">{analyticsData.thumbs_up_count}</p>
                  <p className="text-sm text-green-600">{analyticsData.feedback_rate}% positive</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <FileText className="h-8 w-8 text-purple-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Reports with Feedback</p>
                  <p className="text-2xl font-bold text-gray-900">{analyticsData.unique_reports}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <Users className="h-8 w-8 text-orange-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Unique Users</p>
                  <p className="text-2xl font-bold text-gray-900">{analyticsData.unique_sessions}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Charts Section */}
        {trendsData && analyticsData && reportsData && (
          <FeedbackCharts 
            trendsData={trendsData}
            overviewData={analyticsData}
            reportsData={reportsData}
          />
        )}

        {/* Recent Feedback Table */}
        {recentFeedback && (
          <div className="mt-8">
            <FeedbackTable 
              feedbackData={recentFeedback}
              onRefresh={fetchAnalyticsData}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default FeedbackAnalytics;
