# Odoo ISO Reference Embedding Service (FastAPI + Qdrant)

FastAPI service designed to vectorize, index, and query atomic **Odoo ISO Reference documents** using **Qdrant Vector DB** and **FastEmbed**.

Each document is stored as an **independent atomic fact** with rich metadata headings (ISO Standard, Clause, Sub-clause, Topic, Odoo Module, Odoo Model, Process, and Tags).

---

## 📁 Project Structure

```text
iso_fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Application entry point & lifespan handler
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            # Pydantic BaseSettings & .env configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models for ISO facts, metadata, search, ingestion
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py # Text vectorization & contextual header structuring
│   │   └── qdrant_service.py    # Qdrant client, collections, payload indexing & search
│   └── api/
│       ├── __init__.py
│       └── iso_router.py        # /api/v1/iso endpoints (embed, search, stats)
├── data/
│   └── sample_iso_facts.json    # Ready-to-use sample dataset
├── deploy/
│   ├── docker-compose.yml       # Qdrant container compose file
│   └── iso-fastapi.service      # Linux systemd production service file
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Step 1: Push from Local to GitHub

Inside your local folder (`e:/iso_fastapi`):

```bash
git init
git add .
git commit -m "Initial commit: FastAPI Odoo ISO Qdrant embedding service"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

---

## 🖥️ Step 2: Server Setup Instructions (Linux / Ubuntu)

### 1. Clone the repository on your server
```bash
cd /var/www   # or your target directory
git clone <YOUR_GITHUB_REPO_URL> iso_fastapi
cd iso_fastapi
```

### 2. Create and activate Python virtual environment (`venv`)
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip

python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
nano .env # Adjust QDRANT_HOST, PORT, or EMBEDDING_MODEL if needed
```

### 5. Start Qdrant Vector Database
If Qdrant is not already running on your server, start it using Docker:
```bash
docker compose -f deploy/docker-compose.yml up -d
```
*(Qdrant REST API will be available on `http://localhost:6333`)*

---

## ⚙️ Step 3: Run the FastAPI Service

### Option A: Development Mode (Interactive)
```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs: `http://<SERVER_IP>:8000/docs`

### Option B: Production Daemon with `systemd`
1. Copy the service unit file:
   ```bash
   sudo cp deploy/iso-fastapi.service /etc/systemd/system/
   # Edit paths in /etc/systemd/system/iso-fastapi.service if your directory is different:
   sudo nano /etc/systemd/system/iso-fastapi.service
   ```
2. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable iso-fastapi
   sudo systemctl start iso-fastapi
   sudo systemctl status iso-fastapi
   ```

---

## 📡 API Endpoints & Usage

### 1. Health Check
```bash
curl -X GET http://localhost:14251/health
```

### 2. Upload & Ingest ISO PDF (PyMuPDF + Gemma3:12b + Qdrant)
```bash
curl -X POST http://localhost:14251/api/v1/iso/upload/pdf \
  -F "file=@/path/to/ISO_9001_Quality_Manual.pdf" \
  -F "default_iso_standard=ISO 9001:2015" \
  -F "department_or_owner=Quality Assurance"
```

### 3. Dry-Run / Preview PDF Extraction (Without Qdrant Saving)
```bash
curl -X POST http://localhost:14251/api/v1/iso/parse/pdf-preview \
  -F "file=@/path/to/ISO_9001_Quality_Manual.pdf" \
  -F "default_iso_standard=ISO 9001:2015"
```

### 4. Bulk Ingest Pre-Structured ISO Facts (JSON)
```bash
curl -X POST http://localhost:14251/api/v1/iso/embed/bulk \
  -H "Content-Type: application/json" \
  -d @data/sample_iso_facts.json
```

### 3. Ingest a Single Atomic ISO Fact
```bash
curl -X POST http://localhost:8000/api/v1/iso/embed \
  -H "Content-Type: application/json" \
  -d '{
    "fact_id": "iso9001-7-1-5-fact-01",
    "fact_summary": "Measuring equipment must be calibrated against traceable national standards.",
    "fact_content": "Measuring equipment shall be calibrated or verified at specified intervals against measurement standards traceable to international or national measurement standards. In Odoo, this is managed via maintenance equipment calibration stages.",
    "metadata": {
      "iso_standard": "ISO 9001:2015",
      "clause_number": "7.1.5.2",
      "clause_title": "Measurement Traceability",
      "sub_clause": "7.1.5.2 (a)",
      "topic": "Calibration & Quality Control",
      "odoo_module": "maintenance",
      "odoo_model": "maintenance.equipment",
      "odoo_process": "Equipment calibration cycle management",
      "tags": ["calibration", "equipment", "audit-mandatory"],
      "extra_attributes": {
        "audit_frequency": "Annual"
      }
    }
  }'
```

### 4. Semantic Search with Metadata Filters
```bash
curl -X POST http://localhost:8000/api/v1/iso/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do we prove calibration compliance in Odoo for equipment?",
    "top_k": 3,
    "score_threshold": 0.35,
    "filters": {
      "iso_standard": "ISO 9001:2015",
      "odoo_module": "maintenance"
    }
  }'
```

### 5. View Qdrant Collection Statistics
```bash
curl -X GET http://localhost:8000/api/v1/iso/collection/stats
```
