import React from 'react';
import { Link } from 'react-router-dom';
import { Plus, BarChart3, Clock, CheckCircle, XCircle, TrendingUp } from 'lucide-react';
import ReportCard from './ReportCard';
import StatsCard from './StatsCard';

const Dashboard = ({ reports, loading, onReportGenerated, onReportUpdated }) => {
  // Calculate statistics
  const stats = React.useMemo(() => {
    const total = reports.length;
    const completed = reports.filter(r => r.status === 'completed').length;
    const processing = reports.filter(r => r.status === 'processing' || r.status === 'starting').length;
    const failed = reports.filter(r => r.status === 'failed').length;
    
    return { total, completed, processing, failed };
  }, [reports]);

  const recentReports = reports.slice(0, 5);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Monitor your financial analysis reports and generate new insights
          </p>
        </div>
        <Link
          to="/generate"
          className="mt-4 sm:mt-0 btn-primary inline-flex items-center space-x-2"
        >
          <Plus className="w-5 h-5" />
          <span>Generate Report</span>
        </Link>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Reports"
          value={stats.total}
          icon={BarChart3}
          color="primary"
        />
        <StatsCard
          title="Completed"
          value={stats.completed}
          icon={CheckCircle}
          color="success"
        />
        <StatsCard
          title="Processing"
          value={stats.processing}
          icon={Clock}
          color="warning"
        />
        <StatsCard
          title="Failed"
          value={stats.failed}
          icon={XCircle}
          color="error"
        />
      </div>

      {/* Recent Reports */}
      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <TrendingUp className="w-6 h-6 text-primary-600" />
              <h2 className="text-xl font-semibold text-gray-900">Recent Reports</h2>
            </div>
            {reports.length > 5 && (
              <Link
                to="/history"
                className="text-primary-600 hover:text-primary-700 text-sm font-medium"
              >
                View All
              </Link>
            )}
          </div>
        </div>

        {reports.length === 0 ? (
          <div className="text-center py-12">
            <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No reports yet</h3>
            <p className="text-gray-600 mb-6">
              Generate your first financial analysis report to get started
            </p>
            <Link to="/generate" className="btn-primary">
              Generate Your First Report
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {recentReports.map((report) => (
              <ReportCard
                key={report.report_id}
                report={report}
                onReportUpdated={onReportUpdated}
              />
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
          <div className="space-y-3">
            <Link
              to="/generate"
              className="flex items-center space-x-3 p-3 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors duration-200"
            >
              <Plus className="w-5 h-5 text-primary-600" />
              <div>
                <p className="font-medium text-gray-900">Generate New Report</p>
                <p className="text-sm text-gray-600">Create a comprehensive financial analysis</p>
              </div>
            </Link>
            <Link
              to="/history"
              className="flex items-center space-x-3 p-3 rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors duration-200"
            >
              <BarChart3 className="w-5 h-5 text-primary-600" />
              <div>
                <p className="font-medium text-gray-900">View All Reports</p>
                <p className="text-sm text-gray-600">Browse your report history</p>
              </div>
            </Link>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">API Status</span>
              <span className="flex items-center space-x-2 text-success-600">
                <div className="w-2 h-2 bg-success-500 rounded-full"></div>
                <span className="text-sm font-medium">Online</span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Chart Generation</span>
              <span className="flex items-center space-x-2 text-success-600">
                <div className="w-2 h-2 bg-success-500 rounded-full"></div>
                <span className="text-sm font-medium">Available</span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Data Sources</span>
              <span className="flex items-center space-x-2 text-success-600">
                <div className="w-2 h-2 bg-success-500 rounded-full"></div>
                <span className="text-sm font-medium">Connected</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
