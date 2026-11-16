# RakshaSite — Frontend API & WebSocket Contract Specification

**Document Version**: 1.0.0  
**Target Consumer**: Ashmeet (Frontend Lead — React Dashboard)  
**Target Provider**: FastAPI Backend (`backend/api.py`)  
**Base URL**: `http://localhost:8000`  
**WebSocket URL**: `ws://localhost:8000/ws/video`  
**Static Assets Base URL**: `http://localhost:8000/reports`  
**Date**: November 2026  

---

## 1. Overview & Architecture

RakshaSite provides an intelligent construction safety monitoring platform. The backend exposes both synchronous RESTful endpoints for telemetry, reporting, and statutory compliance, alongside a full-duplex WebSocket stream for low-latency live camera analysis.

All models adhere to the Building and Other Construction Workers (Regulation of Employment and Conditions of Service) Act, 1996 (**BOCW Act 1996**) and Central Rules 1998.

```
+-------------------------------------------------------------------------+
|                  Ashmeet's React Dashboard (frontend/)                   |
+--------------------+--------------------------------+-------------------+
                     |                                |
       HTTP REST API |                                | Full-Duplex WS
                     v                                v
+-------------------------------------------------------------------------+
|                      FastAPI Backend (backend/api.py)                   |
|  - /api/v1/analyze         - /api/v1/demo/summary   - /ws/video         |
|  - /api/v1/analytics/*     - /api/v1/demo/status    - /reports/*        |
|  - /api/v1/report/*        - /api/v1/demo/run                           |
+--------------------+----------------------------------------------------+
                     |
       In-Memory &   v
    Filesystem Layer +----------------------------------------------------+
                     |  reports/                                          |
                     |  - demo_annotated_output.mp4                       |
                     |  - demo_snapshot_violation.jpg                     |
                     |  - safety_audit.pdf                                |
                     |  - demo_audit_log.json                             |
                     +----------------------------------------------------+
```

---

## 2. TypeScript Interfaces & Data Models

```typescript
/**
 * Worker Safety Violation Record
 */
export interface ViolationResponse {
  worker_id: string;               // e.g. "w-01" or "worker-02"
  violation_type: string;          // "no_helmet" | "no_vest" | "no_harness"
  confidence: number;              // Float 0.0 - 1.0
  risk_score: number;              // Integer 0 - 100
  risk_label: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  bocw_section: string;            // e.g. "Section 38(1)(a) & BOCW Rule 46 (IS 2925:1984)"
  penalty: string;                 // e.g. "Section 47: Fine up to Rs. 2,000 / 3 mo imprisonment"
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
}

/**
 * Single Frame Analysis Request & Response
 */
export interface AnalyzeRequest {
  frame_b64: string;               // Base64-encoded image string (JPEG/PNG)
  camera_id?: string;              // Optional camera ID, default "cam_01"
}

export interface AnalyzeResponse {
  violations: ViolationResponse[];
  compliance_score: number;        // Float 0.0 - 100.0
  alert_messages: string[];        // Array of formatted alert strings
}

/**
 * Historical Analytics Aggregate
 */
export interface TrendPoint {
  date: string;                    // ISO date string "YYYY-MM-DD"
  score: number;                   // Compliance score 0.0 - 100.0
}

export interface AnalyticsResponse {
  compliance_score: number;
  violation_counts: Record<string, number>; // e.g. {"No Helmet": 10, "No Vest": 4}
  trend_data: TrendPoint[];
  top_offenders: string[];         // Persistent worker IDs
}

/**
 * Shift Inspection Report
 */
export interface ShiftReportDetail {
  type: string;                    // Violation category
  count: number;                   // Frequency during shift
}

export interface ShiftReportResponse {
  shift_id: string;
  total_violations: number;
  compliance_rating: 'Compliant' | 'Good' | 'Fair' | 'Critical Hazard';
  details: ShiftReportDetail[];
}

/**
 * Demo Viewer & Pipeline Status Models
 */
export interface ArtifactMeta {
  exists: boolean;
  path: string;                    // Relative URL path (e.g. "/reports/demo_annotated_output.mp4")
  size_bytes: number;
}

export interface DemoStatusResponse {
  status: 'ready' | 'pending_execution';
  pipeline_ready: boolean;
  mode: 'offline' | 'online';
  artifacts: {
    video: ArtifactMeta;
    snapshot: ArtifactMeta;
    pdf_report: ArtifactMeta;
    audit_log: ArtifactMeta;
  };
}

export interface DemoSummaryResponse {
  compliance_score: number;        // Float 0.0 - 100.0
  compliance_rating: string;       // "Compliant" | "Fair" | "Critical Hazard"
  overall_risk_tier: 'SAFE' | 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  total_violations: number;
  active_workers_tracked: number;
  bocw_sections_cited: string[];
  video_url: string;               // "/reports/demo_annotated_output.mp4"
  snapshot_url: string;            // "/reports/demo_snapshot_violation.jpg"
  pdf_url: string;                 // "/reports/safety_audit.pdf"
  audit_log_url: string;           // "/reports/demo_audit_log.json"
}

export interface CitedRegulation {
  worker_id: string;
  violation_type: string;
  zone: string;
  risk_score: number;
  risk_label: string;
  bocw_section: string;
  penalty: string;
}

export interface DemoAuditLogResponse {
  video_source: string;
  frames_processed: number;
  total_violation_detections: number;
  annotated_video_output: string;
  annotated_snapshot_output: string;
  pdf_audit_report: string;
  sample_agent_output: {
    compliance_score: number;
    alert_message: string;
    regulations_cited: CitedRegulation[];
  };
}

/**
 * WebSocket Streaming Frame Payloads
 */
export interface WSFrameClientPayload {
  frame_b64: string;               // Base64 image
  timestamp?: string;              // ISO timestamp
}

export interface WSFrameServerPayload {
  annotated_frame: string;         // Base64 annotated image
  alerts: string[];                // Real-time violation alerts
  timestamp: string;
}
```

---

## 3. REST API Endpoints Specification

### 3.1 Single Frame Live Analysis
- **Method**: `POST`
- **Route**: `/api/v1/analyze`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "frame_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "camera_id": "CAM-01-SCAFFOLDING"
  }
  ```
- **Response** (`200 OK`):
  ```json
  {
    "violations": [
      {
        "worker_id": "w-01",
        "violation_type": "no_helmet",
        "confidence": 0.88,
        "risk_score": 85,
        "risk_label": "CRITICAL",
        "bocw_section": "Section 38(1)(a) & BOCW Rule 46 (IS 2925:1984)",
        "penalty": "Section 47: Fine up to Rs. 2,000 or imprisonment up to 3 months, or both.",
        "bbox": [183, 62, 299, 325]
      }
    ],
    "compliance_score": 70.0,
    "alert_messages": [
      "CRITICAL risk: Worker w-01 (no_helmet)"
    ]
  }
  ```

---

### 3.2 Shift Safety Report
- **Method**: `GET`
- **Route**: `/api/v1/report/{shift_id}`
- **Path Parameter**: `shift_id` (string, e.g. `shift_morning_01`)
- **Response** (`200 OK`):
  ```json
  {
    "shift_id": "shift_morning_01",
    "total_violations": 12,
    "compliance_rating": "Fair",
    "details": [
      {"type": "No Harness", "count": 2},
      {"type": "No Helmet", "count": 10}
    ]
  }
  ```

---

### 3.3 Site Safety Analytics & Trend Telemetry
- **Method**: `GET`
- **Route**: `/api/v1/analytics/{camera_id}`
- **Path Parameter**: `camera_id` (string, e.g. `cam_01`)
- **Response** (`200 OK`):
  ```json
  {
    "compliance_score": 82.0,
    "violation_counts": {
      "No Helmet": 10,
      "No Harness": 2
    },
    "trend_data": [
      {"date": "2026-09-14", "score": 80.0},
      {"date": "2026-09-15", "score": 82.0}
    ],
    "top_offenders": ["w-001", "w-045"]
  }
  ```

---

### 3.4 BOCW Act Statutory Reference Catalog
- **Method**: `GET`
- **Route**: `/api/v1/regulations`
- **Response** (`200 OK`):
  ```json
  {
    "Section 38(1)(a)": "Mandates provision and mandatory wearing of protective headgear...",
    "Section 40(2)(r)": "Mandates high-visibility safety clothing on active vehicular and crane pathways...",
    "Section 47": "General penalty for contravention: Fine up to Rs. 2,000 or imprisonment up to 3 months..."
  }
  ```

---

### 3.5 Demo Pipeline Status & Readiness
- **Method**: `GET`
- **Route**: `/api/v1/demo/status`
- **Response** (`200 OK`):
  ```json
  {
    "status": "ready",
    "pipeline_ready": true,
    "mode": "offline",
    "artifacts": {
      "video": {
        "exists": true,
        "path": "/reports/demo_annotated_output.mp4",
        "size_bytes": 190652
      },
      "snapshot": {
        "exists": true,
        "path": "/reports/demo_snapshot_violation.jpg",
        "size_bytes": 279955
      },
      "pdf_report": {
        "exists": true,
        "path": "/reports/safety_audit.pdf",
        "size_bytes": 2402
      },
      "audit_log": {
        "exists": true,
        "path": "/reports/demo_audit_log.json",
        "size_bytes": 1301
      }
    }
  }
  ```

---

### 3.6 Demo KPI Summary
- **Method**: `GET`
- **Route**: `/api/v1/demo/summary`
- **Response** (`200 OK`):
  ```json
  {
    "compliance_score": 70.0,
    "compliance_rating": "Fair",
    "overall_risk_tier": "CRITICAL",
    "total_violations": 2,
    "active_workers_tracked": 2,
    "bocw_sections_cited": [
      "Section 38(1)(a) & BOCW Rule 46 (IS 2925:1984)",
      "Section 38(1)(a), Section 40(2)(r) & BOCW Rule 45"
    ],
    "video_url": "/reports/demo_annotated_output.mp4",
    "snapshot_url": "/reports/demo_snapshot_violation.jpg",
    "pdf_url": "/reports/safety_audit.pdf",
    "audit_log_url": "/reports/demo_audit_log.json"
  }
  ```

---

### 3.7 Demo Machine-Readable Audit Log
- **Method**: `GET`
- **Route**: `/api/v1/demo/audit-log` (or `/api/v1/demo/results`)
- **Response** (`200 OK`): Returns the complete JSON schema documented in `DemoAuditLogResponse`.
- **Error Response** (`404 Not Found`): Returned if the demo runner has not executed yet.

---

### 3.8 On-Demand Pipeline Execution Trigger
- **Method**: `POST`
- **Route**: `/api/v1/demo/run`
- **Response** (`200 OK`):
  ```json
  {
    "message": "Demo pipeline executed successfully",
    "summary": { ... }
  }
  ```

---

### 3.9 Static Asset Streaming Routes
FastAPI mounts the `reports/` folder at `/reports`:
- `GET /reports/demo_annotated_output.mp4`: Streams H.264 video with HTTP Range header support.
- `GET /reports/demo_snapshot_violation.jpg`: Direct JPEG image stream for high-res forensic evidence.
- `GET /reports/safety_audit.pdf`: Binary PDF download stream.
- `GET /reports/demo_audit_log.json`: Direct download of raw audit log.

Convenience API aliases:
- `GET /api/v1/demo/video`
- `GET /api/v1/demo/snapshot`
- `GET /api/v1/demo/report/pdf`

---

## 4. WebSocket Video Streaming Protocol

### Connection
- **Endpoint**: `ws://localhost:8000/ws/video`
- **Protocol**: Standard WebSocket (JSON messages over text frame)

### Workflow
1. Client establishes WebSocket connection.
2. Client sends JSON payload:
   ```json
   {
     "frame_b64": "data:image/jpeg;base64,...",
     "timestamp": "2026-11-06T12:00:00Z"
   }
   ```
3. Server processes frame through perception + tracking + risk pipeline and responds:
   ```json
   {
     "annotated_frame": "data:image/jpeg;base64,...",
     "alerts": ["Alert: Missing helmet detected"],
     "timestamp": "2026-11-06T12:00:00Z"
   }
   ```
4. Client updates HTML5 canvas or image element with `annotated_frame`.

---

## 5. UI Component Integration Map for Ashmeet

| Dashboard Component | Data Source / Endpoint | Polling Interval / Trigger |
|---------------------|------------------------|----------------------------|
| **Executive KPI Cards** (Compliance Score, Risk Tier, Violations Count) | `GET /api/v1/demo/summary` | 10 seconds or on pipeline run |
| **Annotated Video Feed Player** | `<video src="/reports/demo_annotated_output.mp4" controls autoplay loop>` | Static streaming |
| **Forensic Snapshot Viewer** | `<img src="/reports/demo_snapshot_violation.jpg">` | Static streaming with cache buster |
| **BOCW Statutory Audit Table** | `GET /api/v1/demo/audit-log` (`sample_agent_output.regulations_cited`) | On load and poll |
| **PDF Report Download Button** | `<a href="/reports/safety_audit.pdf" download>` | User click |
| **Export Audit Log Button** | `<a href="/reports/demo_audit_log.json" download>` | User click |
| **Live WebCam Streamer** | `WebSocket /ws/video` | Continuous frame loop (15-25 FPS) |
| **Run Pipeline Trigger** | `POST /api/v1/demo/run` | Button click with loading spinner |

---

## 6. Offline Guarantees & Fault Tolerance
- Zero external CDN dependencies: All core fonts and styles are self-contained.
- 100% Offline execution: If remote LLMs or Hugging Face are unavailable, all BOCW sections map deterministically to valid statutory clauses with Section 47 penalties.
- Non-breaking error handling: When `reports/` artifacts do not exist, status endpoints return `pending_execution` and clean default state (100% baseline compliance) without raising server exceptions.
