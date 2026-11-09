import React, { useState } from 'react';
import Navbar from './components/Navbar';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import { compareSignaturesApi, MOCK_COMPARISON_RESULT } from './services/api';

export default function App() {
  const [currentPage, setCurrentPage] = useState('upload'); // 'upload' | 'result'
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileSelect = (file, url) => {
    setUploadedFile(file);
    setPreviewUrl(url);
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    setPreviewUrl(null);
    setComparisonResult(null);
  };

  const handleCompare = async () => {
    setIsAnalyzing(true);
    try {
      // Simulate calling the signature comparison API
      const result = await compareSignaturesApi(uploadedFile, previewUrl);
      setComparisonResult(result);
      setCurrentPage('result');
    } catch (err) {
      console.error("Comparison analysis error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleNavigate = (page) => {
    if (page === 'result' && !comparisonResult) {
      // If user navigates directly to result, prepare mock comparison result
      setComparisonResult({
        ...MOCK_COMPARISON_RESULT,
        documentPreviewUrl: previewUrl || null
      });
    }
    setCurrentPage(page);
  };

  const handleBackToUpload = () => {
    setCurrentPage('upload');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 selection:bg-blue-100 selection:text-blue-900">
      {/* Navigation Bar */}
      <Navbar
        currentPage={currentPage}
        onNavigate={handleNavigate}
      />

      {/* Main Content View */}
      <main className="flex-1 pb-16">
        {currentPage === 'upload' ? (
          <UploadPage
            uploadedFile={uploadedFile}
            previewUrl={previewUrl}
            onFileSelect={handleFileSelect}
            onRemove={handleRemoveFile}
            onCompare={handleCompare}
            isAnalyzing={isAnalyzing}
          />
        ) : (
          <ResultPage
            comparisonResult={comparisonResult}
            previewUrl={previewUrl}
            onBackToUpload={handleBackToUpload}
          />
        )}
      </main>

      {/* Subtle Corporate Footer */}
      <footer className="border-t border-slate-200/80 bg-white py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>© {new Date().getFullYear()} SignVerify AI Systems. All rights reserved.</p>
          <div className="flex items-center space-x-4 text-slate-400">
            <span>FastAPI Integration Ready</span>
            <span>•</span>
            <span>Biometric Standard ISO/IEC 19794-7</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
