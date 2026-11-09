# SignVerify AI — Signature Detection & Verification System

SignVerify AI is a modern, production-grade frontend built with **React**, **Vite**, **Tailwind CSS**, and **shadcn/ui** design principles for an AI-Based Signature Detection and Biometric Verification System.

---

## 🌟 Key Features

### 1. Signature Upload Page (`src/pages/UploadPage.jsx`)
- **Modern Navigation Bar**: Clean brand header with model status indicators.
- **Drag & Drop Upload Area (`src/components/UploadBox.jsx`)**:
  - Drag-and-drop file upload with hover/drag-over visual feedback.
  - Native **Browse Files** button.
  - Strict validation for **PNG**, **JPG**, and **JPEG** formats with descriptive error alerts.
  - Live image preview with file details (size, format, status).
  - Quick **Replace Image** and **Remove** controls.
  - Instant **Load Sample Document** button for quick verification demos.
- **Compare Signatures Button**: Prompts analysis and transitions to the result view.

### 2. Signature Comparison Result Page (`src/pages/ResultPage.jsx`)
- **Uploaded Document Preview**: Visualizes the original document with bounding box overlays for detected signature regions.
- **Detected Signature Crops (`src/components/SignatureCard.jsx`)**: Side-by-side comparative cards for **Signature A (Reference)** and **Signature B (Questioned)** with dimensions, confidence, and quality scores.
- **Similarity & Biometrics (`src/components/SimilarityScore.jsx`)**:
  - High-impact percentage similarity score.
  - Progress bar with an **80% threshold acceptance marker**.
  - **Match Confirmed** / **No Match** verification badges.
  - Feature metrics: Curvature match, structural geometry, and aspect ratio consistency.
  - Interactive scenario toggle to preview both **Match (94.8%)** and **No Match (38.4%)** states.
- **Back to Upload Navigation**: Return to upload another document.

---

## 📁 Project Structure

```
src/
├── components/
│   ├── Navbar.jsx            # Navigation bar with SignVerify AI branding
│   ├── UploadBox.jsx         # Drag-and-drop box, validation, preview & replace
│   ├── SignatureCard.jsx     # Side-by-side signature comparison card
│   └── SimilarityScore.jsx   # Percentage score, progress bar & Match/No Match badge
├── pages/
│   ├── UploadPage.jsx        # Signature upload interface
│   └── ResultPage.jsx        # Full comparison analysis & forensic preview
├── services/
│   └── api.js                # Separated mock data & prepared FastAPI endpoints
├── App.jsx                   # Main layout & page view routing
├── main.jsx                  # React DOM root entry
└── index.css                 # Base styling & Tailwind directives
```

---

## 🚀 Getting Started

### 1. Installation
```bash
npm install
```

### 2. Run Development Server
```bash
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

### 3. Production Build
```bash
npm run build
```

---

## 🔌 Backend Integration (FastAPI Ready)

All future API integration hooks are pre-structured inside [`src/services/api.js`](src/services/api.js) for:
- Image upload (`POST /api/v1/upload`)
- Job status checking (`GET /api/v1/jobs/{job_id}/status`)
- Comparison result retrieval (`GET /api/v1/jobs/{job_id}/result`)
