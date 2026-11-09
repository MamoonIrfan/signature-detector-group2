import React from 'react';
import { Scan, Sparkles, CheckCircle2, ShieldAlert } from 'lucide-react';

export default function SignatureCard({
  title = "Signature Sample",
  tag = "Detected Crop",
  imageUrl,
  confidence = 98.0,
  dimensions = "310 x 120 px",
  location = "Document Body",
  qualityScore = "High (98%)",
  isReference = false,
  isMismatch = false,
}) {
  return (
    <div className={`rounded-xl border bg-white shadow-sm transition-all duration-200 overflow-hidden ${
      isMismatch 
        ? 'border-rose-200 hover:border-rose-300' 
        : isReference
        ? 'border-blue-200 hover:border-blue-300'
        : 'border-slate-200 hover:border-slate-300'
    }`}>
      {/* Card Header */}
      <div className={`px-5 py-3.5 border-b flex items-center justify-between ${
        isReference 
          ? 'bg-blue-50/50 border-blue-100' 
          : isMismatch
          ? 'bg-rose-50/40 border-rose-100'
          : 'bg-slate-50/80 border-slate-100'
      }`}>
        <div className="flex items-center space-x-2.5">
          <div className={`w-7 h-7 rounded-md flex items-center justify-center text-xs font-bold ${
            isReference 
              ? 'bg-blue-600 text-white' 
              : isMismatch
              ? 'bg-rose-600 text-white'
              : 'bg-slate-800 text-white'
          }`}>
            {isReference ? 'A' : 'B'}
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
            <p className="text-[11px] text-slate-500">{location}</p>
          </div>
        </div>

        {/* Badge */}
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${
          isReference
            ? 'bg-blue-50 text-blue-700 border-blue-200'
            : isMismatch
            ? 'bg-rose-50 text-rose-700 border-rose-200'
            : 'bg-emerald-50 text-emerald-700 border-emerald-200'
        }`}>
          {tag}
        </span>
      </div>

      {/* Signature Crop Canvas Display */}
      <div className="p-5">
        <div className="relative rounded-lg border border-slate-200 bg-slate-50/60 p-4 flex items-center justify-center min-h-[170px] overflow-hidden group">
          {/* Subtle grid background to simulate forensic analysis canvas */}
          <div 
            className="absolute inset-0 opacity-40 pointer-events-none"
            style={{
              backgroundImage: 'radial-gradient(circle, #cbd5e1 1px, transparent 1px)',
              backgroundSize: '16px 16px'
            }}
          />

          {imageUrl ? (
            <img
              src={imageUrl}
              alt={title}
              className="relative z-10 max-h-[140px] w-auto max-w-full object-contain filter drop-shadow-sm transition-transform duration-200 group-hover:scale-105"
            />
          ) : (
            <div className="text-center text-slate-400 py-6">
              <Scan className="w-8 h-8 mx-auto mb-1 opacity-50" />
              <span className="text-xs">No signature crop</span>
            </div>
          )}

          {/* Crop dimension indicator overlay */}
          <div className="absolute top-2 right-2 z-10 px-2 py-0.5 rounded bg-white/90 backdrop-blur border border-slate-200 text-[10px] font-mono text-slate-600 shadow-xs">
            {dimensions}
          </div>
        </div>

        {/* Forensic Metadata Breakdown */}
        <div className="mt-4 pt-3 border-t border-slate-100 grid grid-cols-2 gap-3 text-xs">
          <div className="space-y-1">
            <span className="text-slate-500 text-[11px]">Detection Confidence</span>
            <div className="flex items-center space-x-1.5">
              <span className="font-semibold text-slate-800">{confidence}%</span>
              <div className="w-16 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                <div 
                  className={`h-full rounded-full ${
                    confidence >= 90 ? 'bg-emerald-500' : confidence >= 70 ? 'bg-amber-500' : 'bg-rose-500'
                  }`} 
                  style={{ width: `${confidence}%` }}
                />
              </div>
            </div>
          </div>

          <div className="space-y-1">
            <span className="text-slate-500 text-[11px]">Stroke Quality</span>
            <div className="font-medium text-slate-800 flex items-center space-x-1">
              {isMismatch ? (
                <span className="text-amber-700 flex items-center">
                  <ShieldAlert className="w-3.5 h-3.5 mr-1 text-amber-600" /> Discrepancies
                </span>
              ) : (
                <span className="text-emerald-700 flex items-center">
                  <CheckCircle2 className="w-3.5 h-3.5 mr-1 text-emerald-600" /> {qualityScore}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
