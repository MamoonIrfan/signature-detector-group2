import React, { useState } from 'react';
import { Sparkles, ArrowRight, ShieldCheck, CheckCircle2, FileText, Cpu, AlertTriangle } from 'lucide-react';
import UploadBox from '../components/UploadBox';

// High-fidelity sample agreement document as an SVG data URL for instant zero-friction demo testing
const SAMPLE_DOCUMENT_PREVIEW = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1000" width="800" height="1000"><rect width="800" height="1000" fill="%23ffffff"/><rect x="40" y="40" width="720" height="920" fill="none" stroke="%23e2e8f0" stroke-width="2"/><text x="80" y="110" font-family="sans-serif" font-size="22" font-weight="bold" fill="%230f172a">COMMERCIAL CONTRACT &amp; SETTLEMENT AGREEMENT</text><text x="80" y="140" font-family="sans-serif" font-size="12" fill="%2364748b">Document ID: AGR-2025-99824 • Executed on 14 May 2025</text><line x1="80" y1="160" x2="720" y2="160" stroke="%23cbd5e1" stroke-width="1.5"/><rect x="80" y="190" width="640" height="12" rx="4" fill="%23f1f5f9"/><rect x="80" y="215" width="580" height="12" rx="4" fill="%23f1f5f9"/><rect x="80" y="240" width="620" height="12" rx="4" fill="%23f1f5f9"/><rect x="80" y="265" width="480" height="12" rx="4" fill="%23f1f5f9"/><rect x="80" y="320" width="640" height="12" rx="4" fill="%23f1f5f9"/><rect x="80" y="345" width="610" height="12" rx="4" fill="%23f1f5f9"/><rect x="80" y="370" width="540" height="12" rx="4" fill="%23f1f5f9"/><text x="80" y="580" font-family="sans-serif" font-size="15" font-weight="bold" fill="%231e293b">SECTION 7: EXECUTION AND PARTY SIGNATURES</text><rect x="80" y="610" width="290" height="150" fill="%23f8fafc" stroke="%23cbd5e1" rx="6"/><text x="95" y="635" font-family="sans-serif" font-size="12" font-weight="600" fill="%23475569">Party A: Reference Signatory</text><path d="M 100,720 C 120,660 145,630 170,640 C 190,650 180,725 165,735 C 150,745 145,720 175,705 C 205,690 235,680 250,700 C 265,720 255,735 275,725 C 295,715 305,680 330,670 C 345,665 355,700 375,695" fill="none" stroke="%230f2b5c" stroke-width="3" stroke-linecap="round"/><rect x="430" y="610" width="290" height="150" fill="%23f8fafc" stroke="%23cbd5e1" rx="6"/><text x="445" y="635" font-family="sans-serif" font-size="12" font-weight="600" fill="%23475569">Party B: Endorsement Signatory</text><path d="M 450,718 C 470,662 498,632 522,642 C 542,652 530,723 514,733 C 498,743 496,720 524,705 C 554,690 580,680 596,702 C 612,722 602,735 622,725 C 642,715 652,682 676,672 C 692,667 702,702 720,697" fill="none" stroke="%230f2b5c" stroke-width="3" stroke-linecap="round"/><text x="80" y="860" font-family="sans-serif" font-size="11" fill="%2394a3b8">SignVerify AI • Verified Document Cryptographic Hash #92401-AA</text></svg>`;

export default function UploadPage({ 
  uploadedFile, 
  previewUrl, 
  onFileSelect, 
  onRemove, 
  onCompare, 
  isAnalyzing = false 
}) {
  const [error, setError] = useState(null);

  const handleCompareClick = () => {
    if (!uploadedFile && !previewUrl) {
      setError("Please select or drop a signature document before comparing.");
      return;
    }
    setError(null);
    if (onCompare) {
      onCompare();
    }
  };

  const handleLoadSample = () => {
    // Helper to test with high-res sample document immediately
    const sampleBlob = new Blob(["sample document content"], { type: "image/png" });
    const sampleFile = new File([sampleBlob], "standard_commercial_contract.png", { type: "image/png" });
    if (onFileSelect) {
      onFileSelect(sampleFile, SAMPLE_DOCUMENT_PREVIEW);
      setError(null);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-8">
      {/* Dashboard Heading & Description */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5 text-blue-600" />
          <span>Automated Signature Intelligence</span>
        </div>
        
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900">
          AI-Based Signature Detection &amp; Verification
        </h1>
        
        <p className="text-base sm:text-lg text-slate-600 leading-relaxed">
          Upload documents or signature files to automatically detect bounding regions,
          extract signature crops, and verify authenticity using Siamese biometrics.
        </p>

        {/* Feature Badges */}
        <div className="pt-2 flex flex-wrap items-center justify-center gap-2 sm:gap-4 text-xs text-slate-600 font-medium">
          <span className="flex items-center px-3 py-1 rounded-md bg-white border border-slate-200 shadow-2xs">
            <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 text-emerald-600" />
            YOLO Crop Extraction
          </span>
          <span className="flex items-center px-3 py-1 rounded-md bg-white border border-slate-200 shadow-2xs">
            <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 text-emerald-600" />
            Siamese Similarity Engine
          </span>
          <span className="flex items-center px-3 py-1 rounded-md bg-white border border-slate-200 shadow-2xs">
            <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 text-emerald-600" />
            Confidence &amp; Discrepancy Scoring
          </span>
        </div>
      </div>

      {/* Main Upload Card */}
      <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-2">
          <div>
            <h2 className="text-lg font-bold text-slate-900">Document Upload Area</h2>
            <p className="text-xs text-slate-500">
              Provide a document with signatures or a signature image (PNG, JPG, JPEG)
            </p>
          </div>

          {!uploadedFile && !previewUrl && (
            <button
              type="button"
              onClick={handleLoadSample}
              className="inline-flex items-center text-xs font-medium text-blue-600 hover:text-blue-800 bg-blue-50/70 hover:bg-blue-100/80 px-3 py-1.5 rounded-lg border border-blue-200 transition-colors self-start sm:self-auto"
            >
              <FileText className="w-3.5 h-3.5 mr-1.5" />
              Load Sample Document
            </button>
          )}
        </div>

        {/* Drag-and-Drop Image Upload Area */}
        <UploadBox
          file={uploadedFile}
          previewUrl={previewUrl}
          onFileSelect={(file, url) => {
            onFileSelect(file, url);
            setError(null);
          }}
          onRemove={() => {
            onRemove();
            setError(null);
          }}
          error={error}
          onError={setError}
        />

        {/* Compare Signatures Call-to-Action Bar */}
        <div className="pt-4 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-xs text-slate-500 flex items-center">
            <Cpu className="w-4 h-4 mr-1.5 text-slate-400" />
            <span>AI will detect and compare extracted signatures against forensic standards.</span>
          </div>

          <button
            type="button"
            onClick={handleCompareClick}
            disabled={isAnalyzing || (!uploadedFile && !previewUrl)}
            className={`w-full sm:w-auto inline-flex items-center justify-center px-6 py-3 rounded-xl text-sm font-semibold shadow-sm transition-all duration-150 ${
              isAnalyzing || (!uploadedFile && !previewUrl)
                ? 'bg-slate-200 text-slate-400 cursor-not-allowed border border-slate-200'
                : 'bg-slate-900 text-white hover:bg-slate-800 hover:shadow active:scale-[0.99] border border-slate-900'
            }`}
          >
            {isAnalyzing ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Extracting &amp; Comparing...
              </>
            ) : (
              <>
                <span>Compare Signatures</span>
                <ArrowRight className="w-4 h-4 ml-2" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
