# Agentic Database RAG

> **Chop Kong Hin · Intelligent Business Intelligence Assistant**
>
> A production-ready agentic AI system that answers natural-language business questions
> by orchestrating four specialised AI agents over your PostgreSQL database.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  Question Understanding Agent   │  ← Classifies: general | database
└─────────────────────────────────┘
     │
 ┌───┴──────────────────────────────┐
 │ general                          │ database
 ▼                                  ▼
Direct answer            ┌──────────────────────┐
                         │  SQL Query Agent      │  ← RAG on Bilingual README.txt
                         │  (max 3 retries)      │
                         └──────────────────────┘
                                    │
                         ┌──────────────────────┐
                         │  Data Evaluation      │  ← Is data sufficient?
                         │  Agent               │
                         └──────────────────────┘
                             │              │
                       sufficient     insufficient → retry SQL Agent
                             │
                         ┌──────────────────────┐
                         │  Summary Agent        │  ← RAG on knowledge.txt
                         └──────────────────────┘
                                    │
                              Final Answer → UI
```

---

## 📁 Folder Structure

```
Database_RAG/
├── app.py                          # Streamlit entry point
├── .env                            # DB credentials + OpenAI API key (fill this in)
├── requirements.txt
├── README.md
│
├── agents/
│   ├── __init__.py
│   ├── question_understanding.py   # Agent 1: classify & restructure question
│   ├── sql_query_agent.py          # Agent 2: RAG schema + generate + execute SQL
│   ├── data_evaluation_agent.py    # Agent 3: quality gate for DB results
│   └── summary_agent.py            # Agent 4: RAG knowledge + synthesise answer
│
├── db/
│   ├── __init__.py
│   └── connection.py               # psycopg2 connection with Streamlit caching
│
├── utils/
│   ├── __init__.py
│   ├── rag.py                      # TF-IDF RAG retrieval (no external vector DB)
│   └── logging_utils.py            # Rotating file + console logging
│
├── knowledge/
│   ├── Bilingual README.txt        # Database schema (bilingual EN/ZH) — used by SQL Agent
│   └── knowledge.txt               # Store policies & product info — used by Summary Agent
│
├── logs/
│   └── app.log                     # Auto-created rotating log file
│
└── tests/
    └── test_rag.py                 # Smoke tests for the RAG module
```

---

## ⚙️ Setup & Running Locally

### Prerequisites

- **Python 3.10+**
- **PostgreSQL** database running and accessible
- **OpenAI API key**

---

### 1. Clone / Open the project

```bash
git clone https://github.com/KJQ2000/DatabaseRAG.git
```

### 2. Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Open `.env` and fill in your credentials:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password

OPENAI_API_KEY=sk-your-key-here

# Optional overrides
OPENAI_MODEL=gpt-4o-mini   # or gpt-4o
MAX_SQL_RETRIES=3
LOG_LEVEL=INFO
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## 🚀 Usage

| Scenario | Example Question | Pipeline path |
|---|---|---|
| General knowledge | "What is your refund policy?" | Question Agent → direct answer |
| Database query | "How many rings are in stock?" | All 4 agents → SQL + Summary |
| Financial summary | "What is total sales revenue?" | All 4 agents → SQL + Summary |
| Customer lookup | "How many customers do we have?" | All 4 agents → SQL + Summary |

### Features in the UI

- **🔬 Reasoning Trace** — see every agent step, verdict, and SQL query
- **📊 Raw DB Results** — expandable table of database rows
- **🛢️ Generated SQL** — view the exact query sent to PostgreSQL
- **📚 Store Knowledge Used** — the RAG chunks pulled from `knowledge.txt`
- **⚡ Query Cache** — repeated questions are answered instantly
- **🕑 History** — last 20 questions shown at the bottom

---

## 🧪 Running Tests

```bash
# From the project root, with venv activated
python -m pytest tests/ -v
```

---

## 📋 Database Tables

The following PostgreSQL tables are supported (detailed schema in `knowledge/Bilingual README.txt`):

| Table | Description |
|---|---|
| `BOOKING` | Customer gold item bookings & deposit tracking |
| `BOOK_PAYMENT` | Payments linked to bookings |
| `CATEGORY_PATTERN_MAPPING` | Product category ↔ pattern mappings |
| `CUSTOMER` | Customer records (name, contact, e-invoice info) |
| `PURCHASE` | Stock purchases from salesmen |
| `SALE` | Customer sale transactions |
| `SALESMAN` | Supplier/salesman records |
| `STOCK` | Inventory items (type, weight, status, pricing) |

---

## 🔐 Security Notes

- Only `SELECT` / `WITH … SELECT` queries are permitted. All DML/DDL is blocked.
- Database credentials are read from `.env` and never hard-coded.
- Add `.env` to your `.gitignore` before committing.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `DB Disconnected` in sidebar | Check `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` in `.env` |
| `OPENAI_API_KEY is not set` | Make sure `.env` contains a valid `OPENAI_API_KEY` |
| Import errors on startup | Ensure venv is activated and `pip install -r requirements.txt` ran successfully |
| 0 rows from SQL Agent | The agent will auto-retry up to `MAX_SQL_RETRIES` times with refined prompts |
| Logs | Check `logs/app.log` for detailed debug information |
