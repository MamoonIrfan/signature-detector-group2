import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Trash2, RefreshCw, AlertCircle, FileCheck, CheckCircle2 } from 'lucide-react';

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
const MAX_FILE_SIZE_MB = 15;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

export default function UploadBox({ 
  file, 
  previewUrl, 
  onFileSelect, 
  onRemove, 
  error, 
  onError 
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const validateAndProcessFile = (selectedFile) => {
    if (!selectedFile) return;

    // Validate type
    const fileType = selectedFile.type?.toLowerCase();
    const fileName = selectedFile.name?.toLowerCase();
    const isValidExtension = fileName.endsWith('.png') || fileName.endsWith('.jpg') || fileName.endsWith('.jpeg');
    const isValidMime = ACCEPTED_TYPES.includes(fileType);

    if (!isValidMime && !isValidExtension) {
      if (onError) {
        onError(`Unsupported file format (${selectedFile.name.split('.').pop()}). Please upload a valid PNG, JPG, or JPEG image.`);
      }
      return;
    }

    // Validate size
    if (selectedFile.size > MAX_FILE_SIZE_BYTES) {
      if (onError) {
        onError(`File size exceeds ${MAX_FILE_SIZE_MB}MB limit. Please upload a smaller image.`);
      }
      return;
    }

    // Clear previous error
    if (onError) onError(null);

    // Generate local preview URL
    const url = URL.createObjectURL(selectedFile);
    if (onFileSelect) {
      onFileSelect(selectedFile, url);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const handleBrowseClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  return (
    <div className="w-full">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".png,.jpg,.jpeg,image/png,image/jpeg"
        onChange={handleFileInputChange}
        className="hidden"
      />

      {/* Upload Box Container */}
      {!file ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleBrowseClick}
          className={`relative border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 bg-white ${
            isDragging
              ? 'border-blue-600 bg-blue-50/50 shadow-inner ring-4 ring-blue-500/10'
              : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50/60 shadow-sm'
          }`}
        >
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center transition-colors ${
              isDragging ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600'
            }`}>
              <UploadCloud className="w-8 h-8" />
            </div>

            <div className="space-y-1">
              <p className="text-base font-semibold text-slate-800">
                {isDragging ? 'Drop your document or signature here' : 'Drag & drop image here, or browse files'}
              </p>
              <p className="text-sm text-slate-500">
                Supports <span className="font-medium text-slate-700">PNG, JPG, JPEG</span> up to {MAX_FILE_SIZE_MB}MB
              </p>
            </div>

            <div className="pt-2">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleBrowseClick();
                }}
                className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg text-slate-700 bg-white border border-slate-300 hover:bg-slate-50 hover:border-slate-400 shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
              >
                Browse Files
              </button>
            </div>
          </div>
        </div>
      ) : (
        /* Image Preview & Management Mode */
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="p-4 sm:p-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600">
                  <FileCheck className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-slate-900 truncate max-w-xs sm:max-w-md">
                    {file.name}
                  </h4>
                  <div className="flex items-center space-x-2 text-xs text-slate-500 mt-0.5">
                    <span>{formatFileSize(file.size)}</span>
                    <span>•</span>
                    <span className="uppercase">{file.name.split('.').pop()}</span>
                    <span>•</span>
                    <span className="text-emerald-600 font-medium inline-flex items-center">
                      <CheckCircle2 className="w-3 h-3 mr-1" /> Ready for AI Analysis
                    </span>
                  </div>
                </div>
              </div>

              {/* Action buttons: Replace and Remove */}
              <div className="flex items-center space-x-2 w-full sm:w-auto justify-end">
                <button
                  type="button"
                  onClick={handleBrowseClick}
                  className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 hover:text-slate-900 shadow-sm transition-colors"
                  title="Choose another image"
                >
                  <RefreshCw className="w-3.5 h-3.5 mr-1.5 text-slate-500" />
                  Replace Image
                </button>
                <button
                  type="button"
                  onClick={onRemove}
                  className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-rose-700 bg-rose-50 border border-rose-200 rounded-lg hover:bg-rose-100 transition-colors"
                  title="Remove selected image"
                >
                  <Trash2 className="w-3.5 h-3.5 mr-1.5 text-rose-600" />
                  Remove
                </button>
              </div>
            </div>

            {/* Document Image Preview Box */}
            <div className="mt-4 relative rounded-lg border border-slate-200 bg-slate-50/50 p-3 sm:p-4 flex items-center justify-center overflow-hidden min-h-[260px] max-h-[460px]">
              {previewUrl ? (
                <div className="relative group max-h-[420px] max-w-full flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="Uploaded signature document preview"
                    className="max-h-[400px] w-auto max-w-full object-contain rounded border border-slate-200/80 shadow-sm bg-white"
                  />
                  <div className="absolute bottom-2 right-2 px-2 py-1 bg-slate-900/80 backdrop-blur text-white text-[10px] font-mono rounded pointer-events-none">
                    Preview Mode
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-400">
                  <ImageIcon className="w-10 h-10 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Preview unavailable</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Validation Error Message */}
      {error && (
        <div className="mt-3 flex items-start space-x-2.5 p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-sm">
          <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium text-rose-900">Upload Validation Error</p>
            <p className="text-rose-700 text-xs mt-0.5">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
