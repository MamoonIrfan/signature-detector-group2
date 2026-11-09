import React from 'react';
import { CheckCircle2, XCircle, Activity, Gauge, Zap, FileSpreadsheet, ShieldAlert } from 'lucide-react';

export default function SimilarityScore({
  score = 94.8,
  threshold = 80.0,
  isMatch = true,
  statusLabel = "Authentic Match",
  summary = "High biometric correlation detected.",
  metrics = {},
  processingTimeMs = 428
}) {
  const normalizedScore = Math.min(100, Math.max(0, score));

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-6">
      {/* Top Header: Score & Match Status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-100">
        <div>
          <span className="text-xs font-semibold tracking-wider text-slate-500 uppercase">
            AI Similarity Verification
          </span>
          <div className="flex items-baseline space-x-3 mt-1">
            <span className="text-4xl sm:text-5xl font-extrabold tracking-tight text-slate-900">
              {normalizedScore.toFixed(1)}%
            </span>
            <span className="text-xs text-slate-500">
              Confidence Index
            </span>
          </div>
        </div>

        {/* Match / No Match Status Badge */}
        <div>
          {isMatch ? (
            <div className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 shadow-xs">
              <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
              <div>
                <p className="text-xs font-bold uppercase tracking-wide text-emerald-900">Match Confirmed</p>
                <p className="text-[11px] text-emerald-700">{statusLabel}</p>
              </div>
            </div>
          ) : (
            <div className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 shadow-xs">
              <XCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
              <div>
                <p className="text-xs font-bold uppercase tracking-wide text-rose-900">No Match</p>
                <p className="text-[11px] text-rose-700">{statusLabel}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Attractive Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-slate-700">Similarity Score Gradient</span>
          <div className="flex items-center space-x-1.5 text-slate-500">
            <span className="w-2 h-2 rounded-full bg-slate-400"></span>
            <span>Threshold: <strong className="text-slate-800">{threshold}%</strong></span>
          </div>
        </div>

        <div className="relative w-full h-4 bg-slate-100 rounded-full overflow-hidden p-0.5 border border-slate-200/80">
          {/* Threshold marker tick */}
          <div 
            className="absolute top-0 bottom-0 w-0.5 bg-slate-400 z-10 opacity-70"
            style={{ left: `${threshold}%` }}
            title={`Threshold: ${threshold}%`}
          />
          
          {/* Progress fill */}
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${
              isMatch 
                ? 'bg-gradient-to-r from-blue-600 via-emerald-500 to-emerald-600 shadow-sm'
                : 'bg-gradient-to-r from-amber-500 to-rose-600 shadow-sm'
            }`}
            style={{ width: `${normalizedScore}%` }}
          />
        </div>

        <div className="flex justify-between text-[11px] text-slate-400 px-0.5">
          <span>0% Disparate</span>
          <span className="text-slate-600 font-medium">{threshold}% Acceptance Cutoff</span>
          <span>100% Identical</span>
        </div>
      </div>

      {/* Result Information Narrative */}
      <div className={`p-4 rounded-lg border text-sm leading-relaxed ${
        isMatch 
          ? 'bg-slate-50/70 border-slate-200 text-slate-700' 
          : 'bg-rose-50/30 border-rose-100 text-slate-800'
      }`}>
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
          Analysis Result Summary
        </p>
        <p className="text-xs sm:text-sm text-slate-700">
          {summary}
        </p>
      </div>

      {/* Verification Metrics Breakdown */}
      <div>
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
          Algorithmic Feature Analysis
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3 rounded-lg border border-slate-200/80 bg-slate-50/50">
            <span className="text-[11px] text-slate-500 block">Curvature Match</span>
            <span className="text-sm font-bold text-slate-800">
              {metrics.strokeCurvature ? `${metrics.strokeCurvature}%` : '96.1%'}
            </span>
          </div>

          <div className="p-3 rounded-lg border border-slate-200/80 bg-slate-50/50">
            <span className="text-[11px] text-slate-500 block">Geometry Congruence</span>
            <span className="text-sm font-bold text-slate-800">
              {metrics.structuralGeometry ? `${metrics.structuralGeometry}%` : '95.3%'}
            </span>
          </div>

          <div className="p-3 rounded-lg border border-slate-200/80 bg-slate-50/50">
            <span className="text-[11px] text-slate-500 block">Aspect Ratio</span>
            <span className="text-sm font-bold text-slate-800">
              {metrics.aspectRatioConsistency ? `${metrics.aspectRatioConsistency}%` : '98.0%'}
            </span>
          </div>

          <div className="p-3 rounded-lg border border-slate-200/80 bg-slate-50/50">
            <span className="text-[11px] text-slate-500 block">Latency</span>
            <span className="text-sm font-bold text-slate-800">
              {processingTimeMs} ms
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
