import axios from 'axios';

// Configure API and WebSocket URLs with environment-specific logic
const getApiBaseUrl = () => {
  // If explicit environment variable is set, use it
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }

  // For local development (localhost:3000), use direct backend connection
  // Return base URL without /api since axios will concatenate with endpoint paths
  if (window.location.hostname === 'localhost' && window.location.port === '3000') {
    return 'http://localhost:8000';
  }

  // For production/K8s deployment, use relative URLs for nginx proxy
  // Return empty string since nginx proxy handles the full path
  return '';
};

const getWsBaseUrl = () => {
  // If explicit environment variable is set, use it
  if (process.env.REACT_APP_WS_URL) {
    return process.env.REACT_APP_WS_URL;
  }

  // For local development (localhost:3000), use direct backend connection
  if (window.location.hostname === 'localhost' && window.location.port === '3000') {
    return 'ws://localhost:8000';
  }

  // For production/K8s deployment, use relative URLs for nginx proxy
  return '';
};

const API_BASE_URL = getApiBaseUrl();
const WS_BASE_URL = getWsBaseUrl();

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes timeout for report generation
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      throw new Error(data.detail || `Server error: ${status}`);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check your connection.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An unexpected error occurred');
    }
  }
);

export const reportService = {
  /**
   * Generate a new financial report
   * @param {string} ticker - Hong Kong stock ticker (XXXX.HK format)
   * @returns {Promise<Object>} Report generation response
   */
  async generateReport(ticker) {
    try {
      const response = await api.post('/api/reports', { ticker });
      return response.data;
    } catch (error) {
      console.error('Failed to generate report:', error);
      throw error;
    }
  },

  /**
   * Get report details by ID
   * @param {string} reportId - Report ID
   * @returns {Promise<Object>} Report details
   */
  async getReport(reportId) {
    try {
      const response = await api.get(`/api/reports/${reportId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get report:', error);
      throw error;
    }
  },

  /**
   * Get report generation status
   * @param {string} reportId - Report ID
   * @returns {Promise<Object>} Report status
   */
  async getReportStatus(reportId) {
    try {
      const response = await api.get(`/api/reports/${reportId}/status`);
      return response.data;
    } catch (error) {
      console.error('Failed to get report status:', error);
      throw error;
    }
  },

  /**
   * Get list of all reports
   * @returns {Promise<Array>} List of reports
   */
  async getReports() {
    try {
      const response = await api.get('/api/reports');
      return response.data;
    } catch (error) {
      console.error('Failed to get reports:', error);
      throw error;
    }
  },

  /**
   * Get report file URL
   * @param {string} reportId - Report ID
   * @returns {string} Report file URL
   */
  getReportFileUrl(reportId) {
    return `${API_BASE_URL}/api/reports/${reportId}/file`;
  },

  /**
   * Get chart file URL
   * @param {string} reportId - Report ID
   * @param {string} chartFilename - Chart filename
   * @returns {string} Chart file URL
   */
  getChartFileUrl(reportId, chartFilename) {
    return `${API_BASE_URL}/api/reports/${reportId}/charts/${chartFilename}`;
  },

  /**
   * Submit user feedback for a report
   * @param {Object} feedbackData - Feedback submission data
   * @returns {Promise<Object>} Feedback submission response
   */
  async submitFeedback(feedbackData) {
    try {
      const response = await api.post('/api/feedback/submit', feedbackData);
      return response.data;
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      throw error;
    }
  },

  /**
   * Get feedback summary for a report
   * @param {string} reportId - Report ID
   * @returns {Promise<Object>} Feedback summary
   */
  async getFeedbackSummary(reportId) {
    try {
      const response = await api.get(`/api/feedback/summary/${reportId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get feedback summary:', error);
      throw error;
    }
  },

  /**
   * Get analytics overview
   * @returns {Promise<Object>} Analytics overview data
   */
  async getAnalyticsOverview() {
    try {
      const response = await api.get('/api/feedback/analytics/overview');
      return response.data;
    } catch (error) {
      console.error('Failed to get analytics overview:', error);
      throw error;
    }
  },

  /**
   * Get analytics trends
   * @param {number} days - Number of days for trends
   * @returns {Promise<Object>} Analytics trends data
   */
  async getAnalyticsTrends(days = 7) {
    try {
      const response = await api.get(`/api/feedback/analytics/trends?days=${days}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get analytics trends:', error);
      throw error;
    }
  },

  /**
   * Get analytics reports
   * @param {number} limit - Number of reports to fetch
   * @returns {Promise<Object>} Analytics reports data
   */
  async getAnalyticsReports(limit = 10) {
    try {
      const response = await api.get(`/api/feedback/analytics/reports?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get analytics reports:', error);
      throw error;
    }
  },

  /**
   * Get recent feedback
   * @param {number} limit - Number of recent feedback entries to fetch
   * @returns {Promise<Object>} Recent feedback data
   */
  async getRecentFeedback(limit = 20) {
    try {
      const response = await api.get(`/api/feedback/analytics/recent?limit=${limit}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get recent feedback:', error);
      throw error;
    }
  },

  /**
   * Export analytics data
   * @param {string} format - Export format (csv or json)
   * @param {number} days - Number of days for export
   * @returns {Promise<Response>} Export response
   */
  async exportAnalytics(format = 'csv', days = 7) {
    try {
      const response = await api.get(`/api/feedback/analytics/export?format=${format}&days=${days}`, {
        responseType: 'blob'
      });
      return response;
    } catch (error) {
      console.error('Failed to export analytics:', error);
      throw error;
    }
  },

  /**
   * Create WebSocket connection for real-time updates
   * @param {string} reportId - Report ID
   * @param {Function} onMessage - Message handler
   * @param {Function} onError - Error handler
   * @param {Function} onClose - Close handler
   * @returns {WebSocket} WebSocket instance
   */
  createWebSocket(reportId, onMessage, onError, onClose) {
    // Use the same intelligent URL configuration as API calls
    const wsBaseUrl = getWsBaseUrl();
    const wsUrl = `${wsBaseUrl}/ws/reports/${reportId}`;

    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log(`WebSocket connected for report ${reportId}`);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Validate that the message has the expected structure
          if (data && typeof data === 'object') {
            onMessage(data);
          } else {
            console.warn('Received invalid WebSocket message format:', event.data);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error, 'Raw data:', event.data);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) onError(error);
      };
      
      ws.onclose = (event) => {
        console.log(`WebSocket closed for report ${reportId}:`, event.code, event.reason);
        if (onClose) onClose(event);
      };
      
      return ws;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      throw error;
    }
  },

  /**
   * Validate Hong Kong stock ticker format
   * @param {string} ticker - Ticker to validate
   * @returns {boolean} True if valid
   */
  validateHKTicker(ticker) {
    // Allow 1-5 digits followed by .HK (backend will normalize)
    const pattern = /^\d{1,5}\.HK$/i;
    return pattern.test(ticker);
  },

  /**
   * Format ticker to standard format
   * @param {string} ticker - Ticker to format
   * @returns {string} Formatted ticker
   */
  formatTicker(ticker) {
    return ticker.toUpperCase().trim();
  },

  /**
   * Check API health
   * @returns {Promise<Object>} Health status
   */
  async checkHealth() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }
};

export default reportService;
