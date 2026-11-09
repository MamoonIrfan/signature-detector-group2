/**
 * API Service & Data Layer for SignVerify AI
 * 
 * NOTE ON FASTAPI BACKEND INTEGRATION:
 * This file is prepared for future FastAPI backend integration.
 * The backend operations will eventually encompass:
 *  1. Image upload (POST /api/v1/upload)
 *  2. Job status checking (GET /api/v1/jobs/{job_id}/status)
 *  3. Comparison result retrieval (GET /api/v1/jobs/{job_id}/result)
 * 
 * For frontend development and demonstration, realistic mock data is used below.
 * Mock data is strictly separated from actual API handlers so it can be swapped
 * seamlessly once the official FastAPI endpoints and schemas are deployed.
 */

// Placeholder configuration for future backend
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

/**
 * High-fidelity vector signatures for mock visualization
 */
const SAMPLE_SIGNATURE_A_SVG = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 200" width="500" height="200"><path d="M 40,140 C 60,70 95,25 120,40 C 145,55 130,150 110,165 C 90,180 85,150 120,130 C 160,110 200,90 220,115 C 240,140 225,160 250,150 C 275,140 290,95 320,80 C 350,65 370,120 400,110 C 430,100 460,70 480,90" fill="none" stroke="%230f2b5c" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M 90,110 Q 230,90 440,135" fill="none" stroke="%231e3a8a" stroke-width="2.5" stroke-linecap="round"/><circle cx="455" cy="120" r="3" fill="%231e3a8a"/></svg>`;

const SAMPLE_SIGNATURE_B_SVG = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 200" width="500" height="200"><path d="M 45,138 C 65,72 98,28 122,42 C 146,56 128,148 108,162 C 88,176 86,148 122,128 C 162,108 198,92 218,118 C 238,144 223,162 248,152 C 273,142 288,98 318,82 C 348,66 368,122 398,112 C 428,102 458,72 478,92" fill="none" stroke="%230f2b5c" stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M 95,112 Q 232,92 438,137" fill="none" stroke="%231e3a8a" stroke-width="2.4" stroke-linecap="round"/><circle cx="452" cy="122" r="3" fill="%231e3a8a"/></svg>`;

const SAMPLE_SIGNATURE_MISMATCH_SVG = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 200" width="500" height="200"><path d="M 50,150 L 90,40 L 130,140 L 170,50 L 210,130 C 230,130 260,110 300,110 C 340,110 370,140 420,130" fill="none" stroke="%23991b1b" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M 60,160 L 440,155" fill="none" stroke="%23991b1b" stroke-width="2" stroke-linecap="round"/></svg>`;

// ==========================================
// MOCK DATA DEFINITION
// ==========================================
export const MOCK_COMPARISON_RESULT = {
  jobId: "job_sv_89410329",
  processedAt: new Date().toISOString(),
  processingTimeMs: 428,
  
  // Similarity score (percentage 0 - 100)
  similarityScore: 94.8,
  threshold: 80.0,
  isMatch: true,
  status: "MATCH", // "MATCH" or "NO_MATCH"
  statusLabel: "Authentic Match",
  summaryMessage: "High biometric correlation detected. The stroke dynamics, slant angle, and spatial dimensions match the reference signature within normal intra-writer variance.",
  
  // Detected signature crops
  signatures: {
    signatureA: {
      id: "sig_crop_01",
      label: "Signature A (Reference)",
      tag: "Reference Sample",
      imageUrl: SAMPLE_SIGNATURE_A_SVG,
      confidence: 98.4,
      boundingBox: {
        x: 140,
        y: 620,
        width: 310,
        height: 125,
      },
      dimensions: "310 x 125 px",
      dpi: 300,
      qualityScore: "Excellent (99.1%)",
      detectedLocation: "Page 1 - Execution Clause (Party A)"
    },
    signatureB: {
      id: "sig_crop_02",
      label: "Signature B (Detected)",
      tag: "Questioned Sample",
      imageUrl: SAMPLE_SIGNATURE_B_SVG,
      confidence: 96.7,
      boundingBox: {
        x: 520,
        y: 625,
        width: 305,
        height: 120,
      },
      dimensions: "305 x 120 px",
      dpi: 300,
      qualityScore: "Very Good (97.5%)",
      detectedLocation: "Page 1 - Verification Endorsement (Party B)"
    }
  },

  // Detailed verification analytics
  metrics: {
    strokeCurvature: 96.1,
    structuralGeometry: 95.3,
    aspectRatioConsistency: 98.0,
    penPressureDistribution: 91.8,
    featureVectorDistance: "0.052 (Cosine Distance)"
  },

  // Document metadata
  documentMetadata: {
    fileName: "commercial_agreement_signed.pdf",
    detectedSignaturesCount: 2,
    detectionModel: "YOLOv8-SignatureDet-v2",
    verificationModel: "Siamese-ResNet50-Biometric"
  }
};

/**
 * Alternative mock result demonstrating a NO MATCH scenario
 */
export const MOCK_NO_MATCH_RESULT = {
  ...MOCK_COMPARISON_RESULT,
  jobId: "job_sv_99182371",
  similarityScore: 38.4,
  isMatch: false,
  status: "NO_MATCH",
  statusLabel: "Discrepancy / No Match",
  summaryMessage: "Significant structural disparity detected. Pen stroke paths, loops, and baseline alignment diverge significantly from the reference signature threshold.",
  signatures: {
    ...MOCK_COMPARISON_RESULT.signatures,
    signatureB: {
      ...MOCK_COMPARISON_RESULT.signatures.signatureB,
      imageUrl: SAMPLE_SIGNATURE_MISMATCH_SVG,
      confidence: 84.2,
      qualityScore: "Moderate (84.2%)",
      tag: "Suspect Sample"
    }
  },
  metrics: {
    strokeCurvature: 34.2,
    structuralGeometry: 41.0,
    aspectRatioConsistency: 52.8,
    penPressureDistribution: 36.5,
    featureVectorDistance: "0.618 (Cosine Distance)"
  }
};

// ==========================================
// SERVICE LAYER (MOCK IMPLEMENTATION FOR NOW)
// ==========================================

/**
 * Simulates analyzing an uploaded document or signature file
 * @param {File} file - The uploaded image file
 * @param {string} filePreviewUrl - Object URL or base64 preview of the file
 * @param {boolean} simulateNoMatch - Option to test No Match scenario
 * @returns {Promise<typeof MOCK_COMPARISON_RESULT>}
 */
export async function compareSignaturesApi(file, filePreviewUrl = null, simulateNoMatch = false) {
  // Simulate network latency (500ms) for realistic UX
  await new Promise((resolve) => setTimeout(resolve, 550));

  const baseResult = simulateNoMatch ? MOCK_NO_MATCH_RESULT : MOCK_COMPARISON_RESULT;

  return {
    ...baseResult,
    processedAt: new Date().toISOString(),
    documentPreviewUrl: filePreviewUrl || null,
    documentMetadata: {
      ...baseResult.documentMetadata,
      fileName: file?.name || "uploaded_signature_document.png",
      fileSize: file?.size ? `${(file.size / 1024).toFixed(1)} KB` : "1.2 MB",
    }
  };
}

// ==========================================
// FUTURE FASTAPI BACKEND INTEGRATION STUBS
// ==========================================
/*
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error('Upload failed');
  return response.json(); // { job_id: "..." }
}

export async function checkJobStatus(jobId) {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/status`);
  if (!response.ok) throw new Error('Failed to fetch job status');
  return response.json(); // { status: "processing" | "completed" | "failed" }
}

export async function getComparisonResult(jobId) {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/result`);
  if (!response.ok) throw new Error('Failed to fetch comparison result');
  return response.json();
}
*/
