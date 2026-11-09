# Signature Detection & Similarity System
### Group 2 | 3-Day Development Challenge

A team-based machine learning project focused on detecting signatures in document images and comparing signature samples using similarity scores.

The system is designed around a React frontend, FastAPI backend, asynchronous job processing, and a machine-learning inference pipeline. The planned training workflow supports dataset updates, remote training, experiment tracking, model evaluation, and controlled model promotion.

> **Project status:** This repository guide describes the team plan, responsibilities, and target architecture. A feature should be considered implemented only when its code, tests, and integration have been completed and verified.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Goals and Scope](#goals-and-scope)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [End-to-End Workflow](#end-to-end-workflow)
- [Team Responsibilities](#team-responsibilities)
- [Three-Day Execution Plan](#three-day-execution-plan)
- [Backend API](#backend-api)
- [Detection and Similarity](#detection-and-similarity)
- [Asynchronous Processing](#asynchronous-processing)
- [Automated Training Pipeline](#automated-training-pipeline)
- [Deployment](#deployment)
- [Security and Configuration](#security-and-configuration)
- [Licensing Policy](#licensing-policy)
- [Testing and Validation](#testing-and-validation)
- [Documentation and Demo](#documentation-and-demo)
- [Development Guidelines](#development-guidelines)
- [Contributors](#contributors)

---

## Project Overview

The Signature Detection & Similarity System aims to provide two related capabilities:

1. **Signature Detection:** Locate signature regions in uploaded document images using YOLOX-S.
2. **Signature Similarity:** Compare two signature samples and return a similarity score between 0 and 1.

The backend exposes authenticated REST APIs. Long-running detection tasks are processed asynchronously through Redis and Celery, while the frontend displays job progress and results.

A planned model-training pipeline will support dataset updates, remote training, experiment tracking, evaluation, and model promotion.

### Important distinction

A similarity score is a model output, not proof of identity or authenticity. Thresholds must be calibrated and evaluated on representative genuine and forged signature pairs before the system is used for consequential decisions.

---

## Goals and Scope

### Core application

- Detect signature regions in document images.
- Compare two signature samples.
- Return a normalized similarity score from 0 to 1.
- Provide authenticated API access using JWT.
- Process detection jobs asynchronously.
- Track job status and results.
- Display detection boxes and similarity results in a web interface.

### Planned ML operations

- Dataset hosting through Hugging Face.
- Webhook-triggered training.
- Remote training execution.
- Experiment tracking with MLflow or Comet.
- Evaluation gates before model promotion.
- ONNX export for inference.

### Target delivery

The team is working toward a three-day challenge deliverable. Features should be prioritized by dependency and integration risk. A working end-to-end detection flow takes priority over optional automation.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Detection | YOLOX-S | Signature-region detection |
| Model inference | ONNX Runtime | ONNX model execution |
| Similarity | PyTorch / pretrained CNN | Signature embeddings and comparison |
| Backend | FastAPI | REST API |
| Validation | Pydantic | Request and response schemas |
| Authentication | JWT | Protected API access |
| ORM | SQLAlchemy | Database interaction |
| Database | PostgreSQL | Persistent application and job data |
| Message broker | Redis | Background task delivery |
| Worker | Celery | Asynchronous job execution |
| Frontend | React + Vite | Web application |
| Styling | Tailwind CSS | UI styling |
| UI components | shadcn/ui | Reusable interface components |
| Dataset hosting | Hugging Face | Dataset storage and update events |
| Experiment tracking | MLflow or Comet | Training parameters, metrics, and artifacts |
| CI | GitHub Actions | Automated checks and Docker builds |
| Cloud | AWS or GCP | Application deployment |

> Select one experiment-tracking platform and one cloud provider for the initial delivery. Supporting every alternative during a three-day challenge increases integration work without improving the core demonstration.

---

## System Architecture

```text
                         +----------------------+
                         |     React Frontend   |
                         |    Vite + Tailwind   |
                         +----------+-----------+
                                    |
                              HTTPS / REST
                                    |
                         +----------v-----------+
                         |     FastAPI API      |
                         | JWT / CORS / Pydantic|
                         +---+--------------+---+
                             |              |
                      Create job       Read / compare
                             |              |
                    +--------v-----+   +----v---------+
                    | Redis Broker |   | PostgreSQL   |
                    |  Task Queue  |   | Jobs / Data  |
                    +--------+-----+   +--------------+
                             |
                    +--------v---------+
                    |   Celery Worker  |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Detection Model  |
                    |     YOLOX-S      |
                    +--------+---------+
                             |
                    +--------v---------+
                    | ONNX Runtime /   |
                    | Detection Result |
                    +------------------+

       Hugging Face Dataset
                |
             Webhook
                |
          FastAPI Webhook
                |
        Training Trigger
                |
         Remote Training
                |
        MLflow / Comet
                |
        Evaluation Gate
                |
         Model Promotion
                |
        Deployment Process
```

This is a logical architecture. The final cloud architecture depends on the selected provider and the actual deployment configuration.

---

## End-to-End Workflow

### Signature detection

1. User signs in.
2. Frontend receives a JWT access token.
3. User uploads a document or image.
4. FastAPI validates the request.
5. Backend creates a job with a unique job ID.
6. Job is submitted to Redis through Celery.
7. Worker loads the image and runs YOLOX-S inference.
8. Detection results are saved.
9. Frontend retrieves job status.
10. Result page displays bounding boxes and detection information.

### Signature comparison

1. User provides two signature samples.
2. Backend validates both images.
3. The similarity model extracts embeddings.
4. Cosine similarity is calculated.
5. The result is normalized to a score between 0 and 1.
6. API returns the score and relevant comparison information.

The score must not be interpreted as a calibrated probability unless the model has been explicitly calibrated for that purpose.

---

## Team Responsibilities

The work is divided among five members. Each member owns a primary area but must coordinate with the others on shared interfaces and integration.

### 1. Ali Murad — Team Lead & FastAPI Backend

**Primary responsibility:** Backend architecture, API development, authentication, integration, and project coordination.

**Learning topics**

- JWT: header, payload, signature, expiry, and refresh tokens.
- CORS and cross-origin requests.
- Webhooks versus polling.
- Message brokers versus message queues.
- Worker concurrency and CPU cores.
- Idempotency.
- Dead-letter queues.

**Development deliverables**

- FastAPI project structure.
- `POST /auth/login` — authenticate users and issue JWT.
- `POST /detect` — submit a detection job.
- `POST /compare` — compare two signature samples.
- `GET /jobs/{id}` — retrieve job status and results.
- `GET /health` — application health check.
- `POST /webhooks/hf` — receive verified Hugging Face events.
- Pydantic request and response schemas.
- CORS middleware.
- JWT authentication dependency.
- SQLAlchemy job model.
- Pytest API and integration tests.
- Docker Compose integration with API, Redis, and PostgreSQL.

**Database job fields**

| Field | Description |
|---|---|
| `job_id` | Unique job identifier |
| `status` | Current processing status |
| `result_json` | Detection result or output |
| `error` | Failure details, if any |
| `created_at` | Job creation timestamp |

**Team coordination deliverables**

- Jira-compatible task CSV.
- Architecture coordination.
- Integration tracking.
- Ten-minute Tuesday presentation.
- Preparation for technical questions and project demonstration.

**Suggested Jira epics**

Architecture, Detection, Similarity, API and Auth, Queue and Worker, Deployment, Training Pipeline, Documentation and Demo.

### 2. Zubair Saeed — Signature Detection & YOLOX-S

**Primary responsibility:** Model training, evaluation, export, and inference integration.

**Learning topics**

- Anchor-free object detection.
- Decoupled detection head.
- IoU and Non-Maximum Suppression (NMS).
- Confidence threshold.
- Precision and recall.
- mAP@0.5.
- Pretrained weights and dataset licensing.

**Development deliverables**

1. Convert the signature dataset to COCO format when required.
2. Create a custom YOLOX experiment configuration with `num_classes=1`.
3. Fine-tune YOLOX-S using Google Colab.
4. Save checkpoints to Google Drive.
5. Evaluate mAP, precision, and recall.
6. Export the trained model to ONNX.
7. Implement an inference function:

```python
def detect(image_bytes) -> list:
    """
    Returns detected signature regions.

    Each detection should contain:
    - bbox
    - score
    - crop
    """
```

8. Integrate the inference function with the Celery worker.

**Training pipeline deliverables**

- Non-interactive training script.
- CLI arguments for training configuration.
- Hugging Face dataset download.
- MLflow or Comet experiment logging.
- Model checkpoint and ONNX export.
- Evaluation gate comparing the candidate model against the currently approved model.
- Model promotion only when the agreed evaluation criteria pass.

**License requirement**

Use the official YOLOX project and independently verify the licenses of the source code, pretrained weights, datasets, and dependencies. A permissive repository license does not automatically establish the license of every associated artifact.

### 3. Umme Kalsoom — React Frontend & Signature Similarity

**Primary responsibility:** User interface, API integration, and signature comparison.

**Learning topics**

- Siamese networks.
- CNN embeddings.
- Cosine similarity.
- Contrastive loss.
- Equal Error Rate (EER).
- Receiver Operating Characteristic (ROC).
- False Acceptance Rate (FAR).
- False Rejection Rate (FRR).
- Genuine versus forged signatures.

**Frontend deliverables**

Build a React + Vite application using Tailwind CSS and shadcn/ui.

**Page 1: Login and Upload**

- Login form.
- JWT authentication.
- Image upload.
- Drag-and-drop support.
- API integration.
- Upload validation and error messages.

**Page 2: Results**

- Job status display.
- Polling for job updates.
- Detection image with bounding boxes.
- Similarity score progress indicator.
- Loading, success, and failure states.

**Expected job states**

`queued` → `processing` → `done` or `failed`

**Frontend API module**

Keep HTTP requests in a separate `api.js` module rather than placing API logic inside UI components.

**Similarity deliverables**

- Extract image embeddings using a pretrained ResNet18 or MobileNetV3 backbone.
- Apply consistent preprocessing to both signature images.
- Calculate cosine similarity.
- Return a score in the range 0–1.
- Evaluate using genuine and forged signature pairs from a properly licensed public dataset such as CEDAR, subject to availability and license verification.
- Generate ROC and EER metrics.
- Determine a validation-based decision threshold.
- Export the similarity model to ONNX if required for backend inference.

**Example interface**

```python
def compare(img_a, img_b) -> float:
    """
    Compare two signature images.

    Returns:
        Similarity score between 0 and 1.
    """
```

### 4. Farhan Ahmed — Celery Worker, Redis & Webhooks

**Primary responsibility:** Asynchronous task execution, job reliability, and training webhook integration.

**Learning topics**

- Message broker versus queue.
- Celery producer, broker, worker, and result backend.
- Worker concurrency and CPU cores.
- Exponential backoff.
- Dead-letter queue concepts.
- Idempotency.
- At-least-once delivery.
- Redis broker advantages and limitations.
- Upstash free-tier constraints.

**Worker deliverables**

Build a Celery worker using Redis.

**Required behavior**

- Accept detection jobs.
- Update job states: `queued`, `processing`, `done`, and `failed`.
- Retry transient failures up to three times.
- Use exponential backoff.
- Record permanently failed jobs separately.
- Prevent duplicate execution through idempotency controls.
- Produce structured logs.
- Support local Redis and Upstash Redis configurations.
- Include worker-flow tests.

**Dummy inference**

Initially, the worker may use a dummy `detect()` function. Replace it with Zubair's validated YOLOX inference implementation during integration.

**Webhook deliverables**

- Understand Hugging Face webhook events.
- Create a FastAPI webhook endpoint.
- Verify the configured secret.
- Accept only supported dataset-update events.
- Trigger remote training using an approved mechanism, such as GitHub Actions `workflow_dispatch` or Kaggle API, subject to credentials and platform support.
- Test the local webhook using a temporary public tunnel such as ngrok.

### 5. Muhammad Mamoon — DevOps & Documentation

**Primary responsibility:** Containerization, CI/CD, cloud deployment, architecture documentation, and demo preparation.

**Learning topics**

- AWS services: ECS Fargate, SQS, S3, RDS, ECR, CloudWatch, and API Gateway.
- GCP equivalents.
- Containers versus virtual machines.
- CI/CD.
- Infrastructure as Code.
- Secrets management.
- Cloud free tiers and potential costs.

**DevOps deliverables**

1. Dockerfile for FastAPI.
2. Dockerfile or shared image configuration for Celery worker.
3. Docker Compose services:
   - API
   - Worker
   - Redis
   - PostgreSQL
4. GitHub Actions workflow:
   - Lint
   - Test
   - Docker build
5. Local MLflow setup or Comet integration guide.
6. Cloud deployment configuration.
7. AWS-style architecture diagram.

**Architecture diagram components**

- React frontend.
- FastAPI backend.
- Redis or managed queue.
- Celery worker.
- Object storage.
- PostgreSQL database.
- Model registry.
- Training pipeline.
- Monitoring and logging.

**Documentation deliverables**

- Project README.
- Setup instructions.
- API documentation.
- Dependency license audit.
- YOLOX-S technical justification.
- Demo instructions.
- Three-minute demo script.

---

## Three-Day Execution Plan

The exact schedule should be adjusted after checking task dependencies and actual progress.

### Day 1 — Foundations and Individual Development

| Owner | Priority deliverable |
|---|---|
| Ali | API skeleton, authentication, schemas, database |
| Zubair | Dataset inspection, YOLOX setup, baseline inference |
| Umme | React setup, login and upload interface |
| Farhan | Redis setup, Celery worker and dummy job |
| Mamoon | Docker setup, CI workflow, architecture draft |

**Day 1 integration checkpoint**

- API starts successfully.
- Frontend can communicate with the API.
- Redis and PostgreSQL are accessible.
- Dummy background job can be submitted and tracked.
- Detection dataset and model approach are confirmed.

### Day 2 — Model Integration and End-to-End Testing

| Owner | Priority deliverable |
|---|---|
| Ali | Integrate detection and comparison endpoints |
| Zubair | Fine-tuned model, evaluation and ONNX inference |
| Umme | Result page, polling and similarity integration |
| Farhan | Retries, idempotency, webhook implementation |
| Mamoon | Docker integration and deployment preparation |

**Day 2 integration checkpoint**

- Real or clearly labelled baseline detection works.
- Worker processes submitted jobs.
- Job statuses are returned correctly.
- Frontend displays detection results.
- Similarity endpoint has a defined input/output contract.

### Day 3 — Deployment, Validation and Presentation

| Owner | Priority deliverable |
|---|---|
| Ali | End-to-end integration and API validation |
| Zubair | Final model evaluation and export |
| Umme | Frontend integration and UI fixes |
| Farhan | Worker reliability and webhook tests |
| Mamoon | Cloud deployment, documentation and demo preparation |

**Final delivery checklist**

- [ ] Frontend and backend communicate successfully.
- [ ] JWT authentication works.
- [ ] Detection endpoint accepts valid images.
- [ ] Worker processes jobs asynchronously.
- [ ] Job status and result retrieval work.
- [ ] Similarity endpoint returns a documented score.
- [ ] Model and dataset licenses are reviewed.
- [ ] Docker build succeeds.
- [ ] Automated tests pass.
- [ ] Deployment and demo instructions are documented.

---

## Backend API

The following endpoints define the target API contract.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/auth/login` | Authenticate and obtain JWT |
| POST | `/detect` | Submit an image for signature detection |
| POST | `/compare` | Compare two signature images |
| GET | `/jobs/{id}` | Retrieve job status and result |
| GET | `/health` | Check service health |
| POST | `/webhooks/hf` | Receive supported Hugging Face events |

### API documentation

FastAPI provides interactive API documentation at:

```text
http://localhost:8000/docs
```

The endpoint and schema details should be verified against the actual running application.

---

## Detection and Similarity

### Signature detection — YOLOX-S

YOLOX-S is the selected object-detection model for locating signature regions.

Evaluation should include:

- Precision.
- Recall.
- IoU.
- mAP@0.5.
- Confidence threshold.
- NMS behavior.
- Inference latency.

The model should be evaluated on held-out data rather than only the training dataset.

### Signature similarity

The similarity component extracts embeddings from two signature samples and compares them using cosine similarity.

For embeddings \(A\) and \(B\):

\[
\text{Cosine Similarity} =
\frac{A \cdot B}{\|A\|\|B\|}
\]

If cosine similarity is in the range \([-1,1]\), a simple normalization to \([0,1]\) is:

\[
\text{Score} = \frac{\text{Cosine Similarity}+1}{2}
\]

This normalization changes the scale; it does not calibrate the score into a probability.

### Evaluation metrics

| Metric | Meaning |
|---|---|
| FAR | Rate at which forged or impostor samples are accepted |
| FRR | Rate at which genuine samples are rejected |
| EER | Operating point where FAR and FRR are approximately equal |
| ROC | Relationship between true positive and false positive rates |
| Threshold | Decision boundary selected using validation data |

The threshold must be selected using validation data and reported with the evaluation protocol. Test data must remain separate for final performance reporting.

---

## Asynchronous Processing

Celery handles background jobs while Redis acts as the message broker.

### Job lifecycle

```text
queued
  |
  v
processing
  |
  +-------> done
  |
  +-------> retrying
                 |
                 +-----> processing
                 |
                 +-----> failed
```

### Reliability requirements

- Late task acknowledgement where appropriate.
- Bounded retries with exponential backoff.
- Idempotent job handling.
- Duplicate submission protection.
- Structured logging.
- Failed-job inspection and recovery procedures.
- Shared access to uploaded files and job state.

> Celery commonly provides at-least-once task delivery under suitable acknowledgement and broker settings. This does not guarantee exactly-once execution. Tasks and their side effects must be designed to be idempotent.

A Redis list or database table used to record failed jobs is a dead-letter-queue-like mechanism; it should not be described as a fully managed DLQ without implementing the required delivery, retention, inspection, and replay behavior.

---

## Automated Training Pipeline

The planned training workflow is:

```text
Dataset Update
      |
      v
Hugging Face Dataset
      |
      v
Webhook Event
      |
      v
Secret Verification
      |
      v
Training Trigger
      |
      v
Remote Training
      |
      v
Experiment Tracking
      |
      v
Model Evaluation
      |
      v
Evaluation Gate
      |
      +---- Evaluation fails ---> Keep current model
      |
      +---- Evaluation passes --> Promote candidate model
                                      |
                                      v
                                Deployment
```

### Pipeline requirements

1. Receive a supported dataset-update event.
2. Verify webhook authenticity.
3. Trigger a controlled training workflow.
4. Download a versioned dataset.
5. Record dataset and model versions.
6. Log parameters, metrics, and artifacts.
7. Evaluate the candidate model against the approved baseline.
8. Promote only when predefined evaluation criteria pass.
9. Retain the previous approved model for rollback.

A higher mAP alone may not be sufficient for promotion. The evaluation gate should also consider precision, recall, validation data consistency, and any required latency or model-size constraints.

---

## Deployment

The target cloud provider is **AWS or GCP**, subject to the team's final decision and available credits.

### AWS service mapping

| AWS service | Purpose |
|---|---|
| ECS Fargate | Run API and worker containers |
| SQS | Managed message queue alternative |
| S3 | Store uploaded images and model artifacts |
| RDS | Managed relational database |
| ECR | Store Docker images |
| CloudWatch | Logs and monitoring |
| API Gateway | API entry point and routing |

### GCP equivalents

| AWS | GCP option |
|---|---|
| ECS Fargate | Cloud Run or GKE Autopilot |
| SQS | Cloud Tasks or Pub/Sub, depending on delivery semantics |
| S3 | Cloud Storage |
| RDS | Cloud SQL |
| ECR | Artifact Registry |
| CloudWatch | Cloud Logging and Monitoring |
| API Gateway | API Gateway |

These services are not exact one-to-one equivalents. Select services according to workload, queue semantics, deployment model, and cost.

### Deployment requirements

- Separate API and worker processes.
- Configure production environment variables securely.
- Use managed database and queue services where appropriate.
- Use persistent object storage for uploaded images.
- Configure health checks.
- Collect application and worker logs.
- Restrict network access and permissions.
- Monitor cloud usage and costs.

### Cost warning

Free tiers and promotional credits change over time. Verify current regional pricing, resource quotas, storage charges, network egress, and credit eligibility before deployment. Do not assume that an entire architecture will remain free.

---

## Security and Configuration

Create a local `.env` file and configure the corresponding environment variables in the deployment platform.

Example:

```env
APP_ENV=development

DATABASE_URL=your_postgresql_url
REDIS_URL=your_redis_url

JWT_SECRET=replace_with_a_strong_secret
HF_WEBHOOK_SECRET=replace_with_a_webhook_secret
PIPELINE_TOKEN=replace_with_a_pipeline_token

CORS_ORIGINS=http://localhost:5173
```

### Security guidelines

- Never commit `.env` files.
- Never expose JWT secrets, webhook secrets, or database credentials.
- Use HTTPS in production.
- Validate uploaded file types and sizes.
- Restrict CORS to trusted frontend origins.
- Verify webhook signatures or secrets before triggering workflows.
- Apply authentication and authorization to protected endpoints.
- Avoid storing sensitive signature images longer than necessary.
- Define access, retention, and deletion policies for uploaded documents.

---

## Licensing Policy

The project follows a permissive-license preference:

- MIT
- Apache License 2.0
- BSD licenses

### License audit

| Component | Planned license review | Verification required |
|---|---|---|
| YOLOX source code | Apache 2.0 | Yes |
| YOLOX pretrained weights | Artifact-specific terms | Yes |
| ONNX Runtime | MIT | Verify selected package |
| FastAPI | MIT | Verify version |
| Celery | BSD-3-Clause | Verify version |
| Redis client | MIT | Verify selected package |
| React | MIT | Verify version |
| Vite | MIT | Verify version |
| Tailwind CSS | MIT | Verify version |
| PyTorch | BSD-style license | Verify version and dependencies |
| Hugging Face datasets | Dataset-specific | Yes |
| CEDAR dataset | Dataset-specific | Yes |
| MLflow | Apache 2.0 | Verify version |
| PostgreSQL | PostgreSQL License | Verify version |

**Important:** This is an initial review checklist, not a completed legal audit. Check the exact release, transitive dependencies, pretrained weights, and dataset terms before redistribution or commercial use. A library's license does not automatically cover its model weights or training data.

### Why YOLOX-S?

YOLOX-S is selected as a lightweight object-detection baseline for the signature-localization task. Its anchor-free design and decoupled detection head provide a practical architecture to fine-tune and evaluate on a custom signature dataset. The official YOLOX repository uses the Apache License 2.0, which is compatible with the project's preferred permissive-license policy, subject to separate verification of pretrained weights, datasets, and dependencies. Its suitability must ultimately be established through accuracy, inference speed, resource usage, and deployment testing.

---

## Testing and Validation

Testing should cover individual modules and the complete integrated workflow.

### Backend

- Authentication success and failure.
- Invalid and missing JWT.
- Request validation.
- Image upload validation.
- Detection job creation.
- Job status retrieval.
- Database connectivity.
- Webhook secret verification.

### Worker

- Successful job execution.
- Retry after a transient failure.
- Failure after maximum retries.
- Duplicate job submission.
- Worker restart and task redelivery.
- Redis connection failure.
- Structured error logging.

### Machine learning

- Dataset format validation.
- Training/validation/test separation.
- Detection metrics on held-out data.
- ONNX output consistency against the reference model.
- Similarity ROC and EER evaluation.
- Threshold selection on validation data.
- Inference latency and memory usage.

### Frontend

- Login and logout.
- Upload validation.
- API error handling.
- Job polling.
- Loading and failure states.
- Bounding-box rendering.
- Similarity score display.

### End-to-end acceptance flow

A successful demonstration should show:

1. User login.
2. Image upload.
3. Job creation.
4. Background worker processing.
5. Detection result retrieval.
6. Bounding-box visualization.
7. Signature comparison.
8. Similarity score display.

Do not present dummy inference output as a trained model result.

---

## Documentation and Demo

### Three-minute demo outline

| Time | Demonstration |
|---|---|
| 0:00–0:25 | Introduce the problem and system |
| 0:25–0:55 | Login and upload a document |
| 0:55–1:30 | Show asynchronous job processing |
| 1:30–2:05 | Display detected signature regions |
| 2:05–2:35 | Compare two signature samples |
| 2:35–3:00 | Explain architecture, evaluation, and limitations |

### Tuesday presentation

Suggested ten-minute flow:

1. Problem statement and motivation.
2. Proposed solution and use cases.
3. System architecture.
4. YOLOX-S detection approach.
5. Signature similarity approach.
6. FastAPI and asynchronous processing.
7. Training and deployment pipeline.
8. Demonstration.
9. Evaluation and limitations.
10. Future improvements.

### Technical defense preparation

Every team member should be able to explain:

- Why the selected model and framework were chosen.
- How data moves through the system.
- How authentication protects the APIs.
- Why asynchronous processing is used.
- How worker failures and duplicate jobs are handled.
- How detection and similarity models are evaluated.
- How licensing was verified.
- What is implemented, what is experimental, and what remains planned.

---

## Development Guidelines

1. Understand generated code before integrating it.
2. Agree on API contracts before frontend/backend integration.
3. Keep secrets and credentials outside Git.
4. Use Git branches and meaningful commits.
5. Test modules independently before merging.
6. Share complete error messages when requesting debugging help.
7. Record model versions, dataset versions, and evaluation metrics.
8. Do not claim unverified accuracy or production readiness.
9. Keep dummy implementations clearly labelled.
10. Prioritize a working end-to-end flow over optional features.

### Recommended Git workflow

```bash
git checkout -b feature/your-feature
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature
```

Open a pull request and request review before merging into the main branch.

---

## Contributors

| Team Member | Role |
|---|---|
| Ali Murad | Team Lead & FastAPI Backend |
| Zubair Saeed | Signature Detection & YOLOX-S |
| Umme Kalsoom | React Frontend & Signature Similarity |
| Farhan Ahmed | Celery Worker, Redis & Webhooks |
| Muhammad Mamoon | DevOps & Documentation |

---

## Project Status

This project is under active development as part of a three-day collaborative challenge.

The README describes the intended architecture and delivery plan. Update the implementation status, setup instructions, API contracts, test results, model metrics, and deployment details as the team completes and verifies each component.

**Disclaimer:** This system provides machine-learning-based signature localization and similarity analysis. It does not independently establish the legal authenticity of a signature or the identity of the person who created it.
