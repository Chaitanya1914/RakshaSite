# Sejal — Backend Developer

## Your Role
You are building the backend of RakshaSite — the database that stores all violation records and the API that connects Chaitanya's AI pipeline to Ashmeet's frontend dashboard.

Think of it this way:
- Chaitanya's AI detects violations → YOUR API receives that data → YOUR database stores it → Ashmeet's frontend displays it
- You are the BRIDGE between the AI brain and the user interface.

## Files You Own
List and explain each file:
- `src/database/models.py` — SQLAlchemy database table definitions
- `src/database/crud.py` — Create, Read, Update, Delete operations
- `backend/api.py` — FastAPI web server with REST endpoints
- `backend/schemas.py` — Pydantic models defining request/response formats

## What You Need to Learn First (Week 1-2)

### Python Basics (if needed)
- You should be comfortable with: functions, classes, dictionaries, lists, f-strings, imports
- If not, watch: "Python for Beginners" on YouTube (any 2-hour crash course)

### SQLAlchemy (Database ORM)
SQLAlchemy lets you interact with a database using Python code instead of writing raw SQL.

**What to learn:**
- What is an ORM?
- How to define a table as a Python class
- How to add, query, update, delete records
- Sessions and engines

**Resources:**
- YouTube: "SQLAlchemy Tutorial for Beginners" (30-60 min)
- Official tutorial: https://docs.sqlalchemy.org/en/20/tutorial/
- Just focus on the basics: Engine, Session, Model classes, CRUD

**Practice exercise:** Create a simple Student database with name, roll_no, marks columns. Add 5 students. Query students with marks > 80. This takes 30 minutes and teaches you everything you need.

### FastAPI (Web Framework)
FastAPI is a Python web framework for building APIs. An API is just a URL that returns data instead of a webpage.

**What to learn:**
- What is a REST API? (GET, POST, PUT, DELETE)
- How to create endpoints in FastAPI
- Path parameters and query parameters
- Request body (Pydantic models)
- How to return JSON responses

**Resources:**
- YouTube: "FastAPI Tutorial for Beginners" (1-2 hours)
- Official tutorial: https://fastapi.tiangolo.com/tutorial/
- The official tutorial is THE BEST resource. Follow it step by step.

**Practice exercise:** Build a simple Todo API with FastAPI:
- `POST /todos` — add a new todo
- `GET /todos` — list all todos
- `GET /todos/{id}` — get one todo
- `DELETE /todos/{id}` — delete a todo
This takes 1-2 hours and teaches you everything you need for RakshaSite.

## Your Task Breakdown (Week by Week)

### Week 1-2: Learn SQLAlchemy + FastAPI
- [ ] Complete SQLAlchemy practice exercise
- [ ] Complete FastAPI practice exercise (Todo API)
- [ ] Read through the existing code in `src/database/models.py` — understand the Violation and ShiftReport tables
- [ ] Read through `backend/api.py` — understand the endpoint structure

### Week 3-4: Build Database Layer
- [ ] Set up SQLite database locally
- [ ] Test creating a Violation record manually
- [ ] Implement all functions in `src/database/crud.py`:
  - `create_violation(session, data)`
  - `get_violations_by_timerange(session, start, end)`
  - `get_violation_stats(session)` — returns counts by type, average risk
  - `get_repeat_offenders(session, min_count=3)`
- [ ] Write simple test scripts to verify each function works
- [ ] Push to your branch: `sejal/backend-api`

### Week 5-6: Build FastAPI Endpoints
- [ ] Implement `POST /api/v1/analyze` — this receives detection data from Chaitanya's pipeline and stores it in the database
- [ ] Implement `GET /api/v1/analytics/{camera_id}` — returns violation counts, compliance score, trends
- [ ] Implement `GET /api/v1/report/{shift_id}` — returns shift compliance report
- [ ] Implement `GET /api/v1/regulations` — returns list of BOCW Act sections
- [ ] Add CORS middleware (so Ashmeet's React frontend can call your API)
- [ ] Test every endpoint using FastAPI's built-in Swagger UI (go to `http://localhost:8000/docs`)

### Week 7-8: Integration
- [ ] Work with Chaitanya to connect the agent pipeline output to your `POST /analyze` endpoint
- [ ] Work with Ashmeet to make sure his frontend can call your API
- [ ] Debug any issues together
- [ ] Test the full flow: AI detects violation → API stores it → frontend displays it

### Week 9-10: Polish & Test
- [ ] Run pytest tests in `tests/test_detection.py` and `tests/test_agents.py`
- [ ] Fix any failing tests
- [ ] Add error handling to all endpoints (what if the database is empty? what if the camera_id doesn't exist?)

## Your Git Workflow (Step by Step)

### First Time Setup (Do This Once)
```bash
# 1. Clone the repo (Chaitanya will give you the URL)
git clone https://github.com/Chaitanya1914/rakshasite.git
cd rakshasite

# 2. Create your own branch
git checkout -b sejal/backend-api

# 3. Set up Python environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Daily Workflow (Do This Every Day You Work)
```bash
# 1. Make sure you're on your branch
git checkout sejal/backend-api

# 2. Write code, make changes

# 3. Save your changes
git add .
git commit -m "feat: add create_violation function in crud.py"

# 4. Push to GitHub
git push origin sejal/backend-api
```

### Commit Message Format
Use this format: `type: short description`
- `feat`: add new feature (e.g., "feat: add analytics endpoint")
- `fix`: fix a bug (e.g., "fix: handle empty database in get_violations")
- `docs`: update documentation
- `test`: add tests

### When Your Module is Complete
1. Go to GitHub → your repo → Pull Requests → New Pull Request
2. Set base: `main`, compare: `sejal/backend-api`
3. Write a description of what you built
4. Chaitanya will review and merge it

## How to Test Your Code

### Test Database Functions
```python
# Create a test script: test_my_db.py
from src.database.models import init_db, save_violation
from datetime import datetime

session = init_db("sqlite:///test.db")

# Add a test violation
test_data = {
    "timestamp": datetime.now(),
    "camera_id": "CAM-01",
    "worker_id": "W-001",
    "violation_type": "NO_HELMET",
    "confidence": 0.89,
    "risk_score": 75.0,
    "risk_label": "HIGH",
    "bocw_section": "Section 38(1)(a)",
    "penalty": "Fine up to Rs.2000",
    "zone": "elevated_platform",
    "report_text": "Worker W-001 detected without helmet"
}

save_violation(session, test_data)
print("Violation saved successfully!")
```

### Test FastAPI Endpoints
```bash
# Run the server
uvicorn backend.api:app --reload

# Open browser: http://localhost:8000/docs
# This shows a beautiful interactive UI where you can test every endpoint
```

## Common Problems and Solutions
| Problem | Solution |
|---|---|
| ModuleNotFoundError | Run: `pip install -r requirements.txt` |
| Database table doesn't exist | Run `init_db()` first |
| CORS error from frontend | Make sure CORS middleware is configured in `api.py` |
| Git push rejected | Run: `git pull origin sejal/backend-api` first, then push again |

## Need Help?
- Ask Chaitanya (he's the team lead)
- Google the error message (Stack Overflow usually has the answer)
- FastAPI docs: https://fastapi.tiangolo.com/
- SQLAlchemy docs: https://docs.sqlalchemy.org/
