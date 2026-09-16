# Arshi — Data & Research Lead

## Your Role
You are the FOUNDATION of RakshaSite. Without your work:
- Chaitanya can't train the AI model (he needs YOUR clean datasets)
- The Compliance Agent can't cite laws (it needs YOUR BOCW Act knowledge base)
- The project report won't exist (YOU write it)
- The demo won't work (YOU find the demo video)

Your work is less about coding and more about research, data preparation, and documentation. But it's equally important — garbage data in = garbage AI out.

## Files You Own
- data/raw/ — downloaded raw datasets
- data/unified/ — merged and cleaned training data
- knowledge_base/ — BOCW Act PDFs and parsed sections
- src/rag/bocw_parser.py — the Python file containing manually tagged BOCW sections
- All documentation (project report, presentation slides)
- Test cases file

## What You Need to Learn First (Week 1)

### Python Basics
- You need to be comfortable running Python scripts
- Key concepts: variables, lists, dictionaries, loops, reading/writing files
- YouTube: Any "Python for Beginners" crash course (2-3 hours)

### Kaggle
- Kaggle is a website where people share datasets
- You'll download datasets from here
- Create a free account at kaggle.com
- Install kaggle CLI: pip install kaggle

### Basic Git
- You need to push your files to GitHub
- Learn: clone, branch, add, commit, push
- YouTube: "Git & GitHub for Beginners" (30-60 min)

## Your Task Breakdown (Week by Week)

### Week 1-2: Download and Prepare Datasets
This is your MOST CRITICAL task. Chaitanya is waiting for this data to train the model.

- [ ] Create a Kaggle account at kaggle.com
- [ ] Download datasets:
  ```bash
  # Install kaggle CLI
  pip install kaggle
  
  # Download Hard Hat Workers dataset
  kaggle datasets download -d andrewmvd/hard-hat-detection
  # Unzip into data/raw/hardhat/
  
  # Download Safety Helmet dataset
  kaggle datasets download -d vodan37/safety-helmet-detection-dataset
  # Unzip into data/raw/safety/
  ```
- [ ] Download from Roboflow:
  1. Go to https://universe.roboflow.com/
  2. Search "construction safety PPE"
  3. Pick a dataset with 1000+ images
  4. Click Download → Format: YOLOv8 → Download zip
  5. Unzip into data/raw/roboflow/
- [ ] Download Pictor-v3:
  1. Go to https://github.com/ciber-lab/pictor-ppe
  2. Download the dataset
  3. Unzip into data/raw/pictor/
- [ ] Tell Chaitanya when datasets are downloaded (he'll run the merge script)
- [ ] Push to your branch

### Week 3-4: BOCW Act Knowledge Base (MOST IMPORTANT FOR RAG)

This is what makes RakshaSite special. You need to read the BOCW Act and tag the important sections.

- [ ] Download BOCW Act 1996 PDF:
  1. Go to https://www.indiacode.nic.in/ or https://legislative.gov.in/
  2. Search "Building and Other Construction Workers Act 1996"
  3. Download the PDF
  4. Save to knowledge_base/bocw_act_1996.pdf

- [ ] Download BOCW Central Rules 1998:
  1. Same website
  2. Save to knowledge_base/bocw_rules_1998.pdf

- [ ] READ the BOCW Act carefully (focus on Chapters VI, VII, VIII — Safety, Health, and Welfare)

- [ ] Create the manual sections dictionary in src/rag/bocw_parser.py
  For each relevant section, you need to fill in:
  ```python
  MANUAL_BOCW_SECTIONS = {
      "Section 38": {
          "content": "Paste the exact text of Section 38 here...",
          "tags": ["helmet", "safety equipment", "PPE", "protective equipment"],
          "violation_types": ["NO_HELMET", "NO_VEST", "NO_HARNESS"],
          "penalty": "Section 47: Fine up to Rs.2,000 and/or imprisonment up to 3 months"
      },
      # ... more sections
  }
  ```

- [ ] Sections you MUST tag (minimum):
  - Section 35: Safety measures
  - Section 36: Provision of safety equipment
  - Section 38: Safety of workers (THE most important one for PPE)
  - Section 39: Safety measures for height work
  - Section 40: Safety measures for scaffolding
  - Section 44: Duties of employers
  - Section 45: Rights of workers
  - Section 46: Duties of architects/designers
  - Section 47: Penalties for violations
  - Section 48: Penalties for repeat offenses
  Note: Double-check these section numbers against the actual PDF — they may vary slightly.

- [ ] For each section, write:
  1. The exact text (copy from PDF)
  2. 5-8 tags (keywords that relate to this section)
  3. Which violation types it applies to (NO_HELMET, NO_VEST, etc.)
  4. The applicable penalty

### Week 5-6: Demo Video + Test Cases

- [ ] Find a good YouTube demo video:
  - Search: "construction site workers India" or "construction safety violations"
  - Requirements:
    - At least 1-2 minutes long
    - Workers clearly visible
    - Some workers with helmets, some without (ideally)
    - Decent video quality (720p+)
    - Indian construction site preferred (for authenticity)
  - Download using yt-dlp:
    ```bash
    pip install yt-dlp
    yt-dlp -f "best[height<=720]" -o "data/demo_video.mp4" "YOUTUBE_URL"
    ```

- [ ] Create test cases file (tests/test_cases.json):
  Create 50 test scenarios. For each:
  ```json
  [
    {
      "id": 1,
      "description": "Worker without helmet on ground level",
      "violation": true,
      "violation_type": "NO_HELMET",
      "expected_severity": "MEDIUM",
      "expected_section": "Section 38(1)(a)"
    },
    {
      "id": 2,
      "description": "Worker with full PPE (helmet + vest)",
      "violation": false,
      "violation_type": null,
      "expected_severity": null,
      "expected_section": null
    }
  ]
  ```
  Include scenarios: no helmet, no vest, both missing, compliant worker, worker with cap (not helmet), multiple workers, worker on scaffolding, etc.

### Week 7-8: Documentation

- [ ] Write the project report (Chaitanya will give you the template)
  Sections to write:
  1. Abstract
  2. Introduction (background, motivation, objectives)
  3. Literature survey (existing systems, research papers, gaps)
  4. Proposed system (architecture, modules, DFDs)
  5. Implementation details (tech stack, data, training)
  6. Results and evaluation
  7. Conclusion and future scope
  8. References

- [ ] Create presentation slides (Google Slides or PowerPoint):
  - Slide 1: Title + team names
  - Slide 2: Problem statement (38 deaths/day statistic)
  - Slide 3: Existing solutions and their gaps
  - Slide 4: Our solution (one-line + architecture diagram)
  - Slide 5: Tech stack
  - Slide 6: Demo screenshot/video
  - Slide 7: Results (ablation study table)
  - Slide 8: USP + competitive advantage
  - Slide 9: Future scope
  - Slide 10: Thank you + Q&A

### Week 9-10: Testing Support
- [ ] Help Chaitanya run the system on your test cases
- [ ] Note which test cases fail and why
- [ ] Update documentation with final results

### Week 11-12: Final Polish
- [ ] Finalize project report with actual results and screenshots
- [ ] Finalize presentation slides
- [ ] Proofread everything
- [ ] Print report if required

## Your Git Workflow (Step by Step)

### First Time Setup
```bash
# 1. Clone the repo
git clone https://github.com/Chaitanya1914/rakshasite.git
cd rakshasite

# 2. Create your branch
git checkout -b arshi/data-docs

# 3. Install Python dependencies (you need this for yt-dlp and kaggle)
python -m venv venv
venv\Scripts\activate
pip install kaggle yt-dlp pdfplumber
```

### Daily Workflow
```bash
# 1. Make sure you're on your branch
git checkout arshi/data-docs

# 2. Add your files (datasets, PDFs, documentation)

# 3. Save and push
git add .
git commit -m "data: download hard hat workers dataset"
git push origin arshi/data-docs
```

### Important: Large Files
Dataset images are large. If Git rejects your push (file too large):
- Don't push the raw image datasets to GitHub
- Instead, push only the data.yaml file and a README explaining where to download
- Share datasets via Google Drive link with the team

### Commit Message Format
- data: download hard hat workers dataset
- data: clean and merge datasets
- kb: add BOCW Act Section 38 to knowledge base
- docs: write literature survey chapter
- test: add 50 test cases

### When Your Work is Complete
1. GitHub → Pull Requests → New Pull Request
2. Base: main, Compare: arshi/data-docs
3. Describe what you added
4. Chaitanya reviews and merges

## CRITICAL DEADLINES
| Deadline | What Chaitanya Needs From You | Why |
|---|---|---|
| END OF WEEK 2 | All 4 datasets downloaded and organized in data/raw/ | He can't train the model without data |
| END OF WEEK 4 | BOCW Act sections tagged in bocw_parser.py | He can't build the Compliance Agent without this |
| END OF WEEK 6 | Demo video downloaded + test cases file ready | He needs these for testing |
| END OF WEEK 10 | First draft of project report | Time needed for review and corrections |

If you miss these deadlines, the entire project gets delayed. Your work is the FOUNDATION.

## Common Problems and Solutions
| Problem | Solution |
|---|---|
| Kaggle download fails | Check API key: kaggle.com → Settings → API → Create New Token → save kaggle.json to ~/.kaggle/ |
| Can't find BOCW Act PDF | Try: indiacode.nic.in or Google "BOCW Act 1996 full text PDF" |
| Git push fails (file too large) | Don't push image datasets. Share via Google Drive. |
| Not sure which BOCW section to tag | Read the section title first. If it mentions "safety", "equipment", "protection", "penalty" — tag it. |

## Need Help?
- Ask Chaitanya for technical issues
- Google "BOCW Act 1996 simplified" for easier-to-read versions of the Act
- For Git: https://rogerdudler.github.io/git-guide/ (simplest Git guide ever)
