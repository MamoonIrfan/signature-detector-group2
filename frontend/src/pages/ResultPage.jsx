import React, { useState } from 'react';
import { ArrowLeft, FileText, CheckCircle2, XCircle, Scan, Eye, SlidersHorizontal, ShieldCheck } from 'lucide-react';
import SignatureCard from '../components/SignatureCard';
import SimilarityScore from '../components/SimilarityScore';
import { MOCK_COMPARISON_RESULT, MOCK_NO_MATCH_RESULT } from '../services/api';

export default function ResultPage({ 
  comparisonResult, 
  previewUrl, 
  onBackToUpload 
}) {
  // Allow toggling between Match and No Match mock views for comprehensive evaluation
  const [activeScenario, setActiveScenario] = useState('match');

  // Active result data: use props comparisonResult, but allow manual simulation toggle
  const currentData = activeScenario === 'no_match' 
    ? { ...MOCK_NO_MATCH_RESULT, documentPreviewUrl: previewUrl || MOCK_NO_MATCH_RESULT.documentPreviewUrl }
    : (comparisonResult || MOCK_COMPARISON_RESULT);

  const docPreview = previewUrl || currentData.documentPreviewUrl;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 space-y-8">
      {/* Top Action & Breadcrumb Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div className="flex items-center space-x-3">
          <button
            type="button"
            onClick={onBackToUpload}
            className="inline-flex items-center px-3.5 py-2 rounded-lg text-sm font-semibold text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 hover:border-slate-400 shadow-xs transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2 text-slate-600" />
            Back to Upload
          </button>
          <div>
            <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">
              Signature Comparison Result
            </h1>
            <p className="text-xs text-slate-500">
              Automated Forensic Extraction &amp; Similarity Report
            </p>
          </div>
        </div>

        {/* Demo Switcher for Match vs No Match Evaluation */}
        <div className="flex items-center space-x-2 bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs self-start sm:self-auto">
          <span className="text-slate-500 px-2 font-medium flex items-center">
            <SlidersHorizontal className="w-3.5 h-3.5 mr-1" /> Mock Scenario:
          </span>
          <button
            type="button"
            onClick={() => setActiveScenario('match')}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              activeScenario === 'match'
                ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Match (94.8%)
          </button>
          <button
            type="button"
            onClick={() => setActiveScenario('no_match')}
            className={`px-3 py-1 rounded-md font-medium transition-colors ${
              activeScenario === 'no_match'
                ? 'bg-white text-rose-800 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            No Match (38.4%)
          </button>
        </div>
      </div>

      {/* Main Grid: Left Side Document Preview, Right Side Similarity Score */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Uploaded Document Preview with Bounding Box Highlights (5 cols on lg) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col h-full">
            <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/70">
              <div className="flex items-center space-x-2">
                <FileText className="w-4 h-4 text-slate-600" />
                <h2 className="text-sm font-bold text-slate-900">Uploaded Document Preview</h2>
              </div>
              <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200">
                2 Signatures Detected
              </span>
            </div>

            {/* Document Render Area with Detection Bounding Box Overlays */}
            <div className="p-4 flex-1 flex flex-col justify-center items-center bg-slate-50/40">
              <div className="relative rounded-lg border border-slate-200 shadow-xs overflow-hidden max-h-[460px] max-w-full bg-white flex items-center justify-center">
                {docPreview ? (
                  <div className="relative inline-block">
                    <img
                      src={docPreview}
                      alt="Uploaded Document"
                      className="max-h-[440px] w-auto max-w-full object-contain block"
                    />

                    {/* Visual Detection Box 1 (Signature A) */}
                    <div 
                      className="absolute border-2 border-blue-500 bg-blue-500/10 rounded pointer-events-none transition-all duration-300"
                      style={{
                        left: '12%',
                        top: '60%',
                        width: '36%',
                        height: '18%'
                      }}
                    >
                      <div className="absolute -top-6 left-0 px-1.5 py-0.5 bg-blue-600 text-white text-[9px] font-bold rounded flex items-center shadow-xs">
                        <Scan className="w-2.5 h-2.5 mr-1" /> Sig A (98.4%)
                      </div>
                    </div>

                    {/* Visual Detection Box 2 (Signature B) */}
                    <div 
                      className={`absolute border-2 rounded pointer-events-none transition-all duration-300 ${
                        activeScenario === 'no_match'
                          ? 'border-rose-500 bg-rose-500/10'
                          : 'border-cyan-600 bg-cyan-600/10'
                      }`}
                      style={{
                        left: '54%',
                        top: '60%',
                        width: '36%',
                        height: '18%'
                      }}
                    >
                      <div className={`absolute -top-6 left-0 px-1.5 py-0.5 text-white text-[9px] font-bold rounded flex items-center shadow-xs ${
                        activeScenario === 'no_match' ? 'bg-rose-600' : 'bg-cyan-700'
                      }`}>
                        <Scan className="w-2.5 h-2.5 mr-1" /> Sig B ({activeScenario === 'no_match' ? '84.2%' : '96.7%'})
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-20 text-center text-slate-400">
                    <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Document preview unavailable</p>
                  </div>
                )}
              </div>

              {/* Document Metadata Details */}
              <div className="w-full mt-4 p-3 rounded-lg bg-white border border-slate-200 text-xs space-y-1.5">
                <div className="flex justify-between text-slate-500">
                  <span>File:</span>
                  <span className="font-medium text-slate-800 truncate max-w-[200px]">
                    {currentData.documentMetadata?.fileName || 'document.png'}
                  </span>
                </div>
                <div className="flex justify-between text-slate-500">
                  <span>Detection Model:</span>
                  <span className="font-mono text-slate-700">YOLOv8-SignatureDet</span>
                </div>
                <div className="flex justify-between text-slate-500">
                  <span>Extraction Resolution:</span>
                  <span className="text-slate-700 font-medium">300 DPI Native</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Similarity Score & Decision Engine (7 cols on lg) */}
        <div className="lg:col-span-7 space-y-6">
          <SimilarityScore
            score={currentData.similarityScore}
            threshold={currentData.threshold}
            isMatch={currentData.isMatch}
            statusLabel={currentData.statusLabel}
            summary={currentData.summaryMessage}
            metrics={currentData.metrics}
            processingTimeMs={currentData.processingTimeMs}
          />
        </div>
      </div>

      {/* Detected Signature Crops: Signature A and Signature B Displayed Side by Side */}
      <div className="space-y-4">
        <div className="flex items-center justify-between pb-2 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">
              Detected Signature Crops (Side by Side)
            </h2>
            <p className="text-xs text-slate-500">
              Isolated bounding regions extracted from document coordinates for comparative vector analysis
            </p>
          </div>
          <span className="text-xs font-medium text-slate-500">
            Biometric Alignment: <strong className="text-slate-800">Normalized 500x200</strong>
          </span>
        </div>

        {/* Side-by-side cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Signature A */}
          <SignatureCard
            title={currentData.signatures.signatureA.label}
            tag={currentData.signatures.signatureA.tag}
            imageUrl={currentData.signatures.signatureA.imageUrl}
            confidence={currentData.signatures.signatureA.confidence}
            dimensions={currentData.signatures.signatureA.dimensions}
            location={currentData.signatures.signatureA.detectedLocation}
            qualityScore={currentData.signatures.signatureA.qualityScore}
            isReference={true}
          />

          {/* Signature B */}
          <SignatureCard
            title={currentData.signatures.signatureB.label}
            tag={currentData.signatures.signatureB.tag}
            imageUrl={currentData.signatures.signatureB.imageUrl}
            confidence={currentData.signatures.signatureB.confidence}
            dimensions={currentData.signatures.signatureB.dimensions}
            location={currentData.signatures.signatureB.detectedLocation}
            qualityScore={currentData.signatures.signatureB.qualityScore}
            isReference={false}
            isMismatch={!currentData.isMatch}
          />
        </div>
      </div>

      {/* Bottom Navigation Call to Action */}
      <div className="pt-6 border-t border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-4">
        <p className="text-xs text-slate-500">
          SignVerify AI Model Pipeline • Verification Session <span className="font-mono text-slate-700">#{currentData.jobId}</span>
        </p>

        <button
          type="button"
          onClick={onBackToUpload}
          className="w-full sm:w-auto inline-flex items-center justify-center px-6 py-2.5 rounded-xl text-sm font-semibold text-white bg-slate-900 hover:bg-slate-800 shadow-sm transition-colors"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Verify Another Document
        </button>
      </div>
    </div>
  );
}
