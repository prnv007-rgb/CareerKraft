#  CareerKraft

AI-driven résumé evaluator and career recommender.

## Features

* **Resume Classification**: BERT model → Good/Average/Bad
* **Skill Extraction**: BERT‑NER + whitelist + Llama filtering
* **Improvement Tips**: Llama suggestions via Groq API
* **Career Recommendation**: BERT-based role prediction (6 categories)

## Quick Start

1. **Clone & install**

   ```bash
   git clone https://github.com/<org>/careerkraft.git
   cd careerkraft/backend
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Place models**

   * `./bert_resume_final_model/`
   * `career_recommendation_model.pt`
3. **Run backend**

   ```bash
   flask run --port 5000
   ```
4. **Serve frontend**

   * Open `frontend/index.html` or run Vite setup

## API Endpoints

* `POST /resume_feedback` (file)
* `POST /check_suitability` (JSON)
* `POST /generate_feedback` (JSON)
* `POST /career_recommendation` (JSON)

