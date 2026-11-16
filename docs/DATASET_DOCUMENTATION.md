# RakshaSite: Comprehensive Dataset Documentation & Datasheet
## Dataset Architecture, Preprocessing, Forensic Validation, and Ethical Governance

---

## 1. Executive Overview & Problem Context

Construction environments represent unstructured, highly dynamic industrial domains characterized by severe visual occlusion, variable illumination, dust, weather fluctuations, and diverse worker apparel. 

To train and rigorously evaluate the **RakshaSite** multimodal compliance framework, this project establishes a unified dataset architecture that resolves the standard 25 MB Git repository constraint through deterministic, SHA-256 verified streaming distribution (`scripts/fetch_data.py`).

---

## 2. Dataset Composition & Sub-Corpora

The RakshaSite data ecosystem comprises five complementary sub-corpora:

### 2.1 Roboflow Construction Site Safety (CSS) Foundation
- **Role**: Base pre-training foundation for general PPE visual features.
- **Source**: Roboflow Universe Public Repository.
- **Scope**: ~10,000 multi-class annotated worksite frames.
- **Original Classes**: `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`.
- **Training Artifact**: 100-epoch baseline model checkpoint (`models/best_roboflow_baseline.pt`, 6.25 MB).

### 2.2 Ultralytics Construction-PPE Benchmark
- **Role**: Domain-adaptation fine-tuning and strict held-out evaluation.
- **Source**: Ultralytics Public Research Assets (CC BY 4.0).
- **Split Breakdown**:
  - **Training Split**: 800 real worksite images (`data/raw/construction-ppe/construction-ppe/images/train`).
  - **Held-Out Validation Split**: 200 real worksite images (`data/raw/construction-ppe/construction-ppe/images/val`) containing **1,424 annotated instances**.
- **Distribution**: Packaged in `construction-ppe.zip` and unpacked on demand.

### 2.3 Arshi's 50-Photo Real Validation Test Suite (`tests/test_cases.json`)
- **Role**: Functional verification test suite for end-to-end agentic pipeline regression.
- **Composition**: 50 authentic worksite photographs (`TC_001` to `TC_050`) capturing diverse real-world risk scenarios:
  - Excavation pits, high-rise scaffolding, active heavy machinery, and ground-level staging.
  - Multi-worker groups, compliant workers, lone violators, and partial occlusions.
- **Ground Truth**: Each scenario registers exact bounding box counts, hazard zones, expected severity levels, and applicable BOCW statutory clauses.

### 2.4 Worksite Operational Risk Dataset (`data/raw/construction_safety_risk_data.csv`)
- **Role**: Training corpus for the Agent 2 XGBoost hazard severity scoring model.
- **Scale**: 100,000 tabular operational incident records.
- **Feature Set**:
  - `violation_type`: Categorical (`no_helmet`, `no_vest`, `no_harness`, etc.).
  - `zone_danger_level`: Ordinal scale (1 = ground staging, 4 = scaffolding > 2m, 5 = active crane radius).
  - `height_meters`: Continuous (0.0 m to 45.0 m).
  - `duration_seconds`: Consecutive persistence (1 s to 3600 s).
  - `proximity_to_hazard`: Continuous distance to nearest machinery (0.2 m to 25.0 m).
  - `time_of_day`: Environmental lighting proxy (Morning, Midday, Twilight, Night Shift).
- **Model Output**: Severity score (0.0 to 100.0) and tier (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

### 2.5 BOCW Statutory Regulatory Corpus (`knowledge_base/bocw_act_1996_and_rules_1998.md`)
- **Role**: Vector database knowledge base for Agent 3 RAG compliance retrieval.
- **Corpus**:
  - *Building and Other Construction Workers (BOCW) Act, 1996* (Sections 38, 39, 40, 44, 45, 47).
  - *BOCW Central Rules, 1998* (Rules 41, 45, 46, 49, 54, 175–185).
  - *Bureau of Indian Standards (BIS)*: IS 2925:1984 (Helmets), IS 3521:1999 (Harnesses), IS 3696:1991 (Scaffolds).

---

## 3. Class Ontology & Normalization Mapping

To maintain strict cross-dataset consistency, raw dataset annotations are normalized into five canonical monitored classes:

| Canonical Monitored Class | Model Class ID | Raw Roboflow Label | Raw Construction-PPE Label | Spatial Validation Rule |
|---|---|---|---|---|
| **`person`** | 5 | `Person` | `person` | Root anchor entity ($area \ge 1\%$) |
| **`helmet`** | 0 | `Hardhat` | `helmet` | Must overlap upper 25% of `person` box |
| **`no_helmet`** | 2 | `NO-Hardhat` | `no_helmet` | Explicit detector output or vacant head zone |
| **`vest`** | 7 | `Safety Vest` | `vest` | Must overlap torso region (20%–70% height) |
| **`no_vest`** | 4 | `NO-Safety Vest` | `no_vest` | Explicit detector output or vacant torso zone |

*Auxiliary Classes (Ignored during compliance scoring)*: `Mask` (1), `NO-Mask` (3), `Safety Cone` (6), `machinery` (8), `vehicle` (9).

---

## 4. Annotation Format & Coordinate System

All visual annotations adhere to standard **YOLO text format**:
```text
<class_id> <x_center> <y_center> <width> <height>
```
Where all coordinates are normalized floats within the range `[0.0, 1.0]` relative to image width and height.

Example annotation (`data/unified/val/labels/real_source-6be0ee79a4c7364262f7c9b4820fde0c.txt`):
```text
5 0.512500 0.428125 0.187500 0.543750
2 0.515625 0.184375 0.081250 0.093750
4 0.514062 0.359375 0.171875 0.281250
```
This represents: `person` at center (0.51, 0.43), `no_helmet` at center (0.52, 0.18) situated at the head region, and `no_vest` at center (0.51, 0.36) situated at the torso region.

---

## 5. Preprocessing & Augmentation Pipeline

Training images underwent stochastic augmentation designed to simulate real Indian construction conditions:
1. **Geometric Distortions**: Random horizontal flip ($p=0.5$), random scaling ($\pm 15\%$), and mild perspective transforms ($\pm 5\%$) to account for camera mounting angles.
2. **Photometric Perturbations**: HSV color space jitter (Hue $\pm 0.015$, Saturation $\pm 0.7$, Value $\pm 0.4$) to simulate direct tropical sunlight and heavy twilight shadows.
3. **Environmental Occlusion**: Random mosaic augmentation (4-image composition) during early epochs, disabled during final 10 epochs for fine localization convergence.
4. **Resolution Normalization**: Padded resize to $640 \times 640$ pixels maintaining aspect ratio with stride-32 compatibility.

---

## 6. Zero-Git-Bloat Distribution Protocol

To comply with GitHub's 25 MB file limit while ensuring strict academic reproducibility:
- **Zero Binary Bloat**: Git tracks only code, configs, labels, and small metadata. All image archives and heavy model weights are excluded via `.gitignore`.
- **Automated Manifest (`scripts/manifest.json`)**: Formulates exact asset specifications:
  - `weights`: `models/best.pt` (6.2 MB, SHA-256: `48fc9bfe7c62ab039052f54bb536935dfebb5f4ee07f73df0895f8a8e6ac6a0e`).
  - `demo_assets`: `data/raw/hardhat_demo.mp4` (4.1 MB, SHA-256: `e4ea71aa6ffc460c430debe677e9cecea6e2a1712d45ce26d48ace0a1aa76a83`).
  - `val_images`: 50 real photos (9.7 MB, SHA-256: `248013424bfb5ed9d9028a4b61b36303a75890accef5142809a045909c2455bd`).
- **Fetch Script (`scripts/fetch_data.py`)**:
  - Automatically retrieves archives from GitHub Release mirrors or local fallback.
  - Verifies cryptographically matching SHA-256 digests before unpacking.
  - Implements Zip Slip path traversal defenses and atomic extraction.

---

## 7. Data Quality & Forensic Integrity Forensics

To guarantee zero synthetic data leakage:
- **Automated Color Variance Forensics (`tests/test_real_images.py`)**:
  - Every photograph in the 50-image test suite is analyzed for color entropy:
    $$\text{Unique 24-bit Colors} > 5,000 \quad (\text{Observed min: } 18,079, \text{ mean: } 84,210)$$
    $$\text{Pixel Standard Deviation } \sigma > 20.0 \quad (\text{Observed min: } 38.4, \text{ mean: } 54.2)$$
  - Confirms zero flat synthetic canvases or mocked rectangular masks exist in the test split.
- **Strict Partition Independence**: Zero image overlap exists between the 800 training images, 200 held-out validation images, and the 50 test case images.

---

## 8. Ethics, Privacy, and Licensing

1. **Licensing**:
   - Ultralytics Construction-PPE: Creative Commons Attribution 4.0 International (CC BY 4.0).
   - Roboflow CSS: Open Research Community License.
   - Statutory Texts: Public Domain (Government of India legislative publications).
2. **Privacy & Anonymization**:
   - Worksites captured in training datasets feature natural low-resolution face regions ($< 30 \times 30$ pixels) that render individual biometric facial identification impossible.
   - The framework detects safety apparel (helmets, vests) and does not perform facial recognition or individual identity tracking beyond transient session tracking IDs (`w-1`, `w-2`).
