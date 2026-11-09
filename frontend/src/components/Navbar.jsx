import React from 'react';
import { ShieldCheck, Cpu } from 'lucide-react';

export default function Navbar({ currentPage, onNavigate }) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200/80 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Name */}
          <div 
            className="flex items-center space-x-3 cursor-pointer group"
            onClick={() => onNavigate && onNavigate('upload')}
          >
            <div className="w-10 h-10 rounded-lg bg-slate-900 flex items-center justify-center text-white shadow-sm transition-transform duration-150 group-hover:bg-slate-800">
              <ShieldCheck className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-lg font-bold tracking-tight text-slate-900">
                  SignVerify <span className="text-blue-600">AI</span>
                </span>
                <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
                  v2.4
                </span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">
                AI-Based Signature Detection & Verification System
              </p>
            </div>
          </div>

          {/* Navigation & Engine Status */}
          <div className="flex items-center space-x-4">
            <div className="hidden md:flex items-center px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200 text-xs text-slate-600 space-x-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="font-medium text-slate-700">Detection Engine:</span>
              <span className="text-emerald-700 font-semibold">Active (YOLO + Siamese)</span>
            </div>

            {/* Quick Page indicator */}
            <nav className="flex items-center space-x-1 sm:space-x-2">
              <button
                type="button"
                onClick={() => onNavigate && onNavigate('upload')}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  currentPage === 'upload'
                    ? 'bg-slate-100 text-slate-900 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                Upload
              </button>
              <button
                type="button"
                onClick={() => onNavigate && onNavigate('result')}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  currentPage === 'result'
                    ? 'bg-slate-100 text-slate-900 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                Comparison Result
              </button>
            </nav>
          </div>
        </div>
      </div>
    </header>
  );
}
