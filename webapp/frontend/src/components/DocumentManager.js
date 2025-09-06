import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { toast } from 'react-hot-toast';

const DocumentManager = () => {
  const [ticker, setTicker] = useState('');
  const [documents, setDocuments] = useState([]);
  const [downloadStatus, setDownloadStatus] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [embeddingStatus, setEmbeddingStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('download');

  // Polling for status updates
  useEffect(() => {
    let interval;
    if (downloadStatus?.operation_id && downloadStatus?.status === 'in_progress') {
      interval = setInterval(() => {
        checkOperationStatus(downloadStatus.operation_id, 'download');
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [downloadStatus, checkOperationStatus]);

  useEffect(() => {
    let interval;
    if (processingStatus?.operation_id && processingStatus?.status === 'in_progress') {
      interval = setInterval(() => {
        checkOperationStatus(processingStatus.operation_id, 'processing');
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [processingStatus, checkOperationStatus]);

  const validateTicker = (ticker) => {
    const pattern = /^\d{4}\.HK$/;
    return pattern.test(ticker.toUpperCase());
  };

  const checkOperationStatus = useCallback(async (operationId, type) => {
    try {
      const response = await axios.get(`/api/documents/status/${operationId}`);
      if (response.data.success) {
        if (type === 'download') {
          setDownloadStatus(response.data);
        } else if (type === 'processing') {
          setProcessingStatus(response.data);
        }

        if (response.data.status === 'completed' || response.data.status === 'failed') {
          if (type === 'download' && response.data.status === 'completed') {
            loadDocuments(ticker);
          }
        }
      }
    } catch (error) {
      console.error('Status check failed:', error);
    }
  }, [ticker]);

  const downloadDocuments = async () => {
    if (!validateTicker(ticker)) {
      toast.error('Please enter a valid Hong Kong stock ticker (e.g., 0005.HK)');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`/api/documents/download/${ticker.toUpperCase()}`, {
        max_reports: 3
      });

      if (response.data.success) {
        setDownloadStatus({
          operation_id: response.data.download_id,
          status: 'in_progress',
          progress: 0,
          message: 'Starting download...'
        });
        toast.success('Document download started');
      }
    } catch (error) {
      toast.error(`Download failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadDocuments = async (tickerSymbol = ticker) => {
    if (!validateTicker(tickerSymbol)) return;

    try {
      const response = await axios.get(`/api/documents/list/${tickerSymbol.toUpperCase()}`);
      if (response.data.success) {
        setDocuments(response.data.documents);
      }
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  const processDocuments = async () => {
    if (!validateTicker(ticker)) {
      toast.error('Please enter a valid Hong Kong stock ticker');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`/api/documents/process/${ticker.toUpperCase()}`, {
        chunk_strategy: 'hybrid'
      });

      if (response.data.success) {
        setProcessingStatus({
          operation_id: `processing_${Date.now()}`,
          status: 'completed',
          progress: 100,
          message: `Processed ${response.data.processed_successfully} documents successfully`
        });
        toast.success(`Processing completed: ${response.data.processed_successfully} documents processed`);
      }
    } catch (error) {
      toast.error(`Processing failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const createEmbeddings = async () => {
    if (!validateTicker(ticker)) {
      toast.error('Please enter a valid Hong Kong stock ticker');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`/api/documents/embed/${ticker.toUpperCase()}`);

      if (response.data.success) {
        setEmbeddingStatus({
          operation_id: `embedding_${Date.now()}`,
          status: 'completed',
          progress: 100,
          message: response.data.message
        });
        toast.success('Embedding creation completed');
      }
    } catch (error) {
      toast.error(`Embedding creation failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const ProgressBar = ({ progress, status, message }) => (
    <div className="mt-4">
      <div className="flex justify-between text-sm text-gray-600 mb-1">
        <span>{message}</span>
        <span>{progress}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${
            status === 'completed' ? 'bg-green-500' : 
            status === 'failed' ? 'bg-red-500' : 'bg-blue-500'
          }`}
          style={{ width: `${progress}%` }}
        ></div>
      </div>
    </div>
  );

  const StatusIndicator = ({ status }) => {
    const getStatusColor = (status) => {
      switch (status) {
        case 'completed': return 'text-green-600 bg-green-100';
        case 'failed': return 'text-red-600 bg-red-100';
        case 'in_progress': return 'text-blue-600 bg-blue-100';
        default: return 'text-gray-600 bg-gray-100';
      }
    };

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(status)}`}>
        {status?.replace('_', ' ').toUpperCase()}
      </span>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900">Document Processing Manager</h2>
          <p className="text-gray-600 mt-2">
            Download, process, and analyze financial documents for Hong Kong stocks
          </p>
        </div>

        {/* Ticker Input */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <label htmlFor="ticker" className="block text-sm font-medium text-gray-700 mb-2">
                Stock Ticker Symbol
              </label>
              <input
                type="text"
                id="ticker"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g., 0005.HK"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button
              onClick={() => loadDocuments()}
              disabled={!validateTicker(ticker)}
              className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Load Documents
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {['download', 'process', 'embed', 'documents'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {activeTab === 'download' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Download Annual Reports</h3>
              <p className="text-gray-600 mb-4">
                Download the latest annual reports from HKEX for the specified ticker.
              </p>
              
              <button
                onClick={downloadDocuments}
                disabled={loading || !validateTicker(ticker)}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Downloading...' : 'Download Documents'}
              </button>

              {downloadStatus && (
                <div className="mt-4 p-4 bg-gray-50 rounded-md">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Download Status</span>
                    <StatusIndicator status={downloadStatus.status} />
                  </div>
                  <ProgressBar 
                    progress={downloadStatus.progress || 0}
                    status={downloadStatus.status}
                    message={downloadStatus.message}
                  />
                </div>
              )}
            </div>
          )}

          {activeTab === 'process' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Process PDF Documents</h3>
              <p className="text-gray-600 mb-4">
                Extract text content and create intelligent chunks from downloaded PDF documents.
              </p>
              
              <button
                onClick={processDocuments}
                disabled={loading || !validateTicker(ticker)}
                className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Processing...' : 'Process Documents'}
              </button>

              {processingStatus && (
                <div className="mt-4 p-4 bg-gray-50 rounded-md">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Processing Status</span>
                    <StatusIndicator status={processingStatus.status} />
                  </div>
                  <ProgressBar 
                    progress={processingStatus.progress || 0}
                    status={processingStatus.status}
                    message={processingStatus.message}
                  />
                </div>
              )}
            </div>
          )}

          {activeTab === 'embed' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Create Vector Embeddings</h3>
              <p className="text-gray-600 mb-4">
                Generate vector embeddings and store them in Weaviate for semantic search.
              </p>
              
              <button
                onClick={createEmbeddings}
                disabled={loading || !validateTicker(ticker)}
                className="px-6 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Creating Embeddings...' : 'Create Embeddings'}
              </button>

              {embeddingStatus && (
                <div className="mt-4 p-4 bg-gray-50 rounded-md">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Embedding Status</span>
                    <StatusIndicator status={embeddingStatus.status} />
                  </div>
                  <ProgressBar 
                    progress={embeddingStatus.progress || 0}
                    status={embeddingStatus.status}
                    message={embeddingStatus.message}
                  />
                </div>
              )}
            </div>
          )}

          {activeTab === 'documents' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Available Documents</h3>
              
              {documents.length === 0 ? (
                <p className="text-gray-500">No documents found. Try downloading documents first.</p>
              ) : (
                <div className="space-y-3">
                  {documents.map((doc, index) => (
                    <div key={index} className="p-4 border border-gray-200 rounded-md">
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium text-gray-900">{doc.filename}</h4>
                          <p className="text-sm text-gray-600">
                            Size: {(doc.file_size / 1024 / 1024).toFixed(2)} MB
                          </p>
                          <p className="text-sm text-gray-600">
                            Modified: {new Date(doc.modified_date).toLocaleDateString()}
                          </p>
                        </div>
                        <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                          {doc.file_type}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentManager;
