# Ashmeet — Frontend Developer

## Your Role
You are building the FACE of RakshaSite — the dashboard that professors and judges will see. Everything the AI detects, every risk score, every legal citation will be displayed on YOUR dashboard. If your UI looks amazing, the whole project looks amazing.

Think of it this way:
- Chaitanya's AI detects violations
- Sejal's API serves the data
- YOUR dashboard displays it beautifully
- You are what the user actually SEES.

## What the Dashboard Should Show
1. Live video feed panel (left side) — showing the construction video with bounding boxes around workers
2. Agent reasoning panel (right side) — showing the detection → risk → compliance → report chain in real-time
3. Violation alert cards — red alert cards that pop up when a violation is detected, showing violation type, worker ID, BOCW section, risk level
4. Compliance score gauge — a 0-100 gauge showing current compliance level (green 80-100, yellow 50-79, red 0-49)
5. Analytics charts:
   - Bar chart: violations by type (NO_HELMET, NO_VEST)
   - Line chart: compliance score trend over time
   - Table: repeat offender list (worker IDs with violation counts)
6. PDF download button — downloads the compliance report as PDF

## Files You Own
- frontend/ — the entire React application
- You'll create: App.jsx, components/, pages/, styles

## What You Need to Learn First (Week 1-3)

### HTML + CSS Basics (if needed)
- If you don't know HTML/CSS, spend 2-3 days learning basics
- YouTube: "HTML & CSS Full Course for Beginners" (any 2-3 hour course)

### JavaScript Basics (if needed)
- Variables, functions, arrays, objects, arrow functions, template literals, async/await, fetch()
- YouTube: "JavaScript Crash Course" (2-3 hours)

### React (THIS IS THE MAIN THING)
React is a JavaScript library for building user interfaces using components.

**What to learn:**
- What is a component? (A reusable piece of UI)
- JSX (HTML inside JavaScript)
- Props (passing data to a component)
- State (useState hook — data that changes)
- useEffect (fetching data when page loads)
- Conditional rendering (show/hide things based on state)

**Resources (in this order):**
1. YouTube: "React in 100 Seconds" by Fireship (get the overview)
2. YouTube: "React Full Course for Beginners 2024" by Bro Code or Net Ninja (pick one, follow along)
3. Official tutorial: https://react.dev/learn

**Practice exercise:** Build a simple Todo app in React:
- Input field + Add button
- List of todos below
- Click to mark as done (strikethrough)
- Delete button
This teaches: components, state, props, event handling — everything you need.

### Tailwind CSS (Styling)
- Tailwind lets you style directly in your HTML/JSX using utility classes
- Example: `<div className="bg-blue-500 text-white p-4 rounded-lg">Hello</div>`
- YouTube: "Tailwind CSS in 15 Minutes" (any short tutorial)
- Docs: https://tailwindcss.com/docs

### Recharts (Charts)
- Simple chart library for React
- YouTube: "Recharts Tutorial" (any 20-30 min tutorial)
- Docs: https://recharts.org/

## Your Task Breakdown (Week by Week)

### Week 1-2: Learn React Basics
- [ ] Complete React Todo app practice exercise
- [ ] Learn Tailwind CSS basics
- [ ] Study 2-3 admin dashboard designs on Dribbble/Behance for inspiration

### Week 3-4: Set Up React Project & Build Layout
- [ ] Initialize React project:
  ```bash
  cd C:\Users\chait\Desktop\rakshasite
  npm create vite@latest frontend -- --template react
  cd frontend
  npm install
  npm install tailwindcss @tailwindcss/vite
  npm install recharts axios react-icons
  npm run dev
  ```
- [ ] Set up Tailwind CSS (follow Vite + Tailwind setup guide)
- [ ] Build the dashboard layout:
  - Sidebar navigation (logo, menu items)
  - Main content area (split into grid sections)
  - Header with project name and compliance score
- [ ] Push to your branch

### Week 5-6: Build Dashboard Components
- [ ] Build VideoPlayer component — displays the video/stream (can use HTML5 <video> tag initially)
- [ ] Build ViolationCard component — shows violation type, worker ID, risk level, BOCW section with color coding
- [ ] Build ComplianceGauge component — circular or bar gauge showing 0-100 score
- [ ] Build ViolationChart component — bar chart using Recharts showing violations by type
- [ ] Build TrendChart component — line chart showing compliance score over time
- [ ] Build OffenderTable component — table showing repeat offenders
- [ ] Build ReportButton component — button that downloads PDF report

### Week 7-8: Connect to Backend API
- [ ] Use axios or fetch() to call Sejal's API endpoints:
  ```javascript
  // Example: Fetch analytics data
  const fetchAnalytics = async () => {
    const response = await fetch('http://localhost:8000/api/v1/analytics/CAM-01');
    const data = await response.json();
    setViolations(data.violation_counts);
    setComplianceScore(data.compliance_score);
  };
  ```
- [ ] Handle loading states (show spinner while data loads)
- [ ] Handle error states (show error message if API is down)
- [ ] Test with mock data first, then switch to real API

### Week 9-10: Polish & Responsive Design
- [ ] Make the dashboard look professional:
  - Dark theme or clean white theme
  - Smooth animations on alerts
  - Color coding: green for safe, red for violations
  - Professional fonts (Inter, Poppins from Google Fonts)
- [ ] Make it responsive (works on laptop screens)
- [ ] Add loading skeletons
- [ ] Final UI review with team

## Your Git Workflow (Step by Step)

### First Time Setup
```bash
# 1. Clone the repo
git clone https://github.com/Chaitanya1914/rakshasite.git
cd rakshasite

# 2. Create your branch
git checkout -b ashmeet/frontend

# 3. Set up the React project (if not already done)
cd frontend
npm install
npm run dev
# Open http://localhost:5173 in browser
```

### Daily Workflow
```bash
# 1. Make sure you're on your branch
git checkout ashmeet/frontend

# 2. Write code

# 3. Save and push
git add .
git commit -m "feat: add compliance gauge component"
git push origin ashmeet/frontend
```

### Commit Message Format
- feat: add ViolationCard component
- fix: fix chart not rendering correctly
- style: improve dashboard spacing and colors
- docs: add comments to components

### When Your Dashboard is Complete
1. Go to GitHub → Pull Requests → New Pull Request
2. Base: main, Compare: ashmeet/frontend
3. Add screenshots of your dashboard in the PR description
4. Chaitanya will review and merge

## Mock Data (Use This Before API is Ready)
```javascript
const mockData = {
  compliance_score: 72,
  violations: [
    { id: 1, worker_id: 'W-003', type: 'NO_HELMET', risk_label: 'CRITICAL', risk_score: 87, bocw_section: 'Section 38(1)(a)', timestamp: '2026-09-15T14:23:07' },
    { id: 2, worker_id: 'W-007', type: 'NO_VEST', risk_label: 'MEDIUM', risk_score: 45, bocw_section: 'Section 38(1)(a)', timestamp: '2026-09-15T14:25:12' },
  ],
  violation_counts: { NO_HELMET: 5, NO_VEST: 3 },
  trend_data: [
    { time: '08:00', score: 95 },
    { time: '09:00', score: 88 },
    { time: '10:00', score: 72 },
    { time: '11:00', score: 65 },
  ],
  top_offenders: [
    { worker_id: 'W-003', violation_count: 4 },
    { worker_id: 'W-012', violation_count: 3 },
  ]
};
```

## Dashboard Design Reference
Search these for inspiration:
- Dribbble: "admin dashboard dark theme"
- Dribbble: "monitoring dashboard"
- Behance: "safety dashboard UI"
- Google: "Tailwind admin dashboard free template"

## Common Problems and Solutions
| Problem | Solution |
|---|---|
| npm command not found | Install Node.js from nodejs.org |
| CORS error when calling API | Tell Sejal to add CORS middleware in FastAPI |
| Chart not showing | Check if data format matches Recharts expected format |
| Component not rendering | Check for typos in component name, make sure it's exported |
| Git push rejected | Run: git pull origin ashmeet/frontend first |

## Need Help?
- Ask Chaitanya (team lead)
- React docs: https://react.dev/
- Tailwind docs: https://tailwindcss.com/
- Recharts docs: https://recharts.org/
- Stack Overflow for specific errors
