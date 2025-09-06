import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useParams } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import ReportGenerator from './components/ReportGenerator';
import ReportViewer from './components/ReportViewer';
import ReportHistory from './components/ReportHistory';
import DocumentManager from './components/DocumentManager';
import FeedbackAnalytics from './components/FeedbackAnalytics';
import { reportService } from './services/reportService';

// Component to redirect API routes to backend
function RedirectToBackend() {
  const { reportId } = useParams();

  useEffect(() => {
    // Redirect to the backend API endpoint
    const backendUrl = reportService.getReportFileUrl(reportId);
    window.location.href = backendUrl;
  }, [reportId]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Redirecting to report file...</p>
      </div>
    </div>
  );
}

function App() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    try {
      setLoading(true);
      const data = await reportService.getReports();
      setReports(data);
    } catch (error) {
      console.error('Failed to load reports:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReportGenerated = (newReport) => {
    setReports(prev => [newReport, ...prev]);
  };

  const handleReportUpdated = (updatedReport) => {
    setReports(prev => 
      prev.map(report => 
        report.report_id === updatedReport.report_id ? updatedReport : report
      )
    );
  };

  return (
    <Router future={{ v7_relativeSplatPath: true }}>
      <div className="min-h-screen bg-gray-50">
        <Header />
        
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route 
              path="/" 
              element={
                <Dashboard 
                  reports={reports}
                  loading={loading}
                  onReportGenerated={handleReportGenerated}
                  onReportUpdated={handleReportUpdated}
                />
              } 
            />
            <Route 
              path="/generate" 
              element={
                <ReportGenerator 
                  onReportGenerated={handleReportGenerated}
                  onReportUpdated={handleReportUpdated}
                />
              } 
            />
            <Route 
              path="/report/:reportId" 
              element={<ReportViewer />} 
            />
            <Route
              path="/history"
              element={
                <ReportHistory
                  reports={reports}
                  loading={loading}
                  onRefresh={loadReports}
                />
              }
            />
            <Route
              path="/documents"
              element={<DocumentManager />}
            />
            <Route
              path="/feedback-analytics"
              element={<FeedbackAnalytics />}
            />
            {/* Redirect API routes to backend */}
            <Route
              path="/api/reports/:reportId/file"
              element={<RedirectToBackend />}
            />
          </Routes>
        </main>

        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#22c55e',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
