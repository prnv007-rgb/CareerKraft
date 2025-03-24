import json
import os
import re
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification,
    pipeline,
    BertForSequenceClassification
)
import tensorflow as tf
import fitz  # PyMuPDF

# PyTorch libraries for the career recommendation model
import torch
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

@app.route("/health")
def health():
    return "OK", 200

# -------------------------------
# Resume Classification Setup
# -------------------------------

# Load a TensorFlow BERT model for resume classification.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
resume_clf_model = TFBertForSequenceClassification.from_pretrained("./bert_resume_final_model")

# Load a Named Entity Recognition (NER) pipeline.
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Groq API Key and URL for chat completions.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Corrected URL with '/openai' segment.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# -------------------------------
# Career Recommendation Model Setup
# -------------------------------

# Load the career recommendation model.
career_rec_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6
)
state_dict = torch.load("career_recommendation_model.pt", map_location=torch.device('cpu'))
career_rec_model.load_state_dict(state_dict, strict=False)
career_rec_model.eval()

def rec_preprocess(text):
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return encoding

# Map class indices to career recommendations.
career_mapping = {
    0: "Data Scientist",
    1: "Software Engineer",
    2: "Business Analyst",
    3: "Product Manager",
    4: "UX/UI Designer",
    5: "Marketing Specialist"
}

# -------------------------------
# Helper Functions for Resume Processing
# -------------------------------

def extract_text_from_pdf(file):
    try:
        file_bytes = file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = " ".join(page.get_text("text") for page in doc)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def is_resume(text):
    resume_keywords = [
        "Objective", "Experience", "Education", "Skills", "Certifications", 
        "Projects", "Summary", "Achievements", "Work Experience", "Professional Summary"
    ]
    match_count = sum(1 for keyword in resume_keywords if keyword.lower() in text.lower())
    return match_count >= 2

def predict_resume_category(text):
    try:
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = resume_clf_model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits)
        predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]
        labels = {0: "Good", 1: "Average", 2: "Bad"}
        return labels.get(predicted_class, "Unknown")
    except Exception as e:
        print(f"Error in resume category prediction: {e}")
        return f"Error: {e}"

# Known skills dictionary (for keyword matching)
KNOWN_SKILLS = {
    # Programming Languages
    "python", "java", "c", "c++", "javascript", "typescript", "ruby", "go", "php", "swift", "kotlin", "r", "scala", "perl", "rust",
    # Web Development & Front-End
    "html", "css", "react", "react.js", "angular", "vue", "node.js", "express.js", "django", "flask", "spring", "laravel", ".net", "asp.net", "ruby on rails",
    # Databases & Query Languages
    "sql", "postgresql", "mysql", "oracle", "sqlite", "mongodb", "redis", "cassandra", "elasticsearch", "graphql",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", "ci/cd", "devops", "git", "github", "bitbucket",
    # Data Science & Machine Learning
    "tensorflow", "pytorch", "machine learning", "deep learning", "nlp", "data analysis", "data visualization", "statistical modeling", "pandas", "numpy", "scikit-learn", "spark", "hadoop", "big data", "matplotlib", "seaborn",
    # Tools & Technologies
    "excel", "powerpoint", "streamlit", "tableau", "powerbi", "jira", "confluence",
    # Specialized / Domain-Specific
    "llama", "ibm watson studio", "chatbot development", "predictive modeling", "microservices", "rest", "soap", "design patterns", "unit testing", "agile", "scrum"
}

def merge_wordpieces(entities):
    merged = []
    buffer = ""
    for entity in entities:
        token = entity.get('word', '')
        # If token starts with "##", append it to the current buffer.
        if token.startswith("##"):
            buffer += token[2:]
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(token)
    if buffer:
        merged.append(buffer)
    return merged

def extract_skills_from_resume(text):
    try:
        # 1. Use the NER pipeline to get candidate tokens.
        ner_results = ner_pipeline(text)
        
        # Merge wordpieces from the NER output.
        merged_tokens = merge_wordpieces(ner_results)
        
        # Clean and normalize the tokens.
        ner_skills = set()
        for token in merged_tokens:
            cleaned = re.sub(r"[^a-zA-Z0-9\+\#\.]", "", token).lower().strip()
            if len(cleaned) > 1:  # Skip very short tokens.
                ner_skills.add(cleaned)
        
        # 2. Use keyword matching with the known skills.
        text_lower = text.lower()
        dict_skills = set()
        for skill in KNOWN_SKILLS:
            # Use word boundaries for accurate matching.
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                dict_skills.add(skill)
        
        # 3. Combine both sets.
        all_skills = ner_skills.union(dict_skills)
        
        # 4. Optional: Further filtering to remove common or irrelevant words.
        common_non_skills = {"and", "the", "for", "with", "job", "experience", "project", "development", "studio"}
        filtered_skills = [skill for skill in all_skills if skill not in common_non_skills]
        
        return filtered_skills if filtered_skills else ["No skills detected"]
    except Exception as e:
        print(f"Error extracting skills: {str(e)}")
        return []


# -------------------------------
# Connect Endpoint using LLM for LinkedIn Profiles
# -------------------------------

@app.route('/connect', methods=['POST'])
def connect():
    """
    Endpoint that takes a JSON payload with a "skills" key (list of skills),
    uses an LLM (via Groq API) to generate 5 LinkedIn profile recommendations in JSON format.
    """
    try:
        data = request.get_json()
        skills = data.get("skills", [])
        if not skills:
            return jsonify({"error": "No skills provided"}), 400

        skills_str = ", ".join(skills)
        prompt = (
           "You are a career advisor. Based on the following skills: "
    f"{skills_str}, provide a JSON array of exactly 5 LinkedIn profile recommendations that match these skills.Search on google or linkedin and return the results appropriately "
    "Each profile must include 'name', 'position', and 'linkedin_url'. "
    "Only include profiles with valid and verifiable LinkedIn URLs that are relevant to the given skills as users will click and be redirected to linkedin juz search the skill in google and find relavent results in linkedin with the skill"
    "Return only the JSON output without any extra text."
        )

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
        )

        if response.status_code != 200:
            print("Groq API error:", response.text)
            return jsonify({"error": response.text}), 500

        # Get the text content from the LLM response.
        result_text = response.json()["choices"][0]["message"]["content"].strip()
        print("Raw LLM response:", result_text)

        try:
            parsed = json.loads(result_text)
            # If the parsed JSON is a dict with a key "profiles", extract it.
            if isinstance(parsed, dict) and "profiles" in parsed:
                profiles = parsed["profiles"]
            else:
                profiles = parsed
        except Exception as parse_error:
            print("Error parsing LLM output:", parse_error)
            return jsonify({"error": f"Error parsing LLM output: {parse_error}. Raw output: {result_text}"}), 500

        return jsonify({"profiles": profiles})
    except Exception as e:
        print(f"Error in /connect: {e}")
        return jsonify({"error": str(e)}), 500



# -------------------------------
# Other Endpoints
# -------------------------------

@app.route('/predict', methods=['POST'])
def predict_resume():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        text = extract_text_from_pdf(file)
        if not text:
            return jsonify({"error": "No text found in PDF"}), 400
        if not is_resume(text):
            return jsonify({"error": "Uploaded PDF is not a valid resume."}), 400

        resume_category = predict_resume_category(text)
        extracted_skills = extract_skills_from_resume(text)
        return jsonify({
            "category": resume_category,
            "skills": extracted_skills
        })
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_suitability', methods=['POST'])
def check_suitability():
    try:
        from groq import Groq  # Import Groq client library
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        data = request.get_json()
        job_description = data.get("job_description", "").strip()
        resume_skills = data.get("resume_skills", [])
        if not job_description:
            return jsonify({"error": "Job description cannot be empty"}), 400
        if not resume_skills:
            return jsonify({"error": "No skills provided for suitability check"}), 400

        skills_str = ", ".join(resume_skills)
        prompt = (
            "You are an expert career consultant. Based on the following job description and resume skills, ignore irrelevant skills if given as input, "
            "evaluate the candidate's resume suitability for the job. Provide a concise answer as 'Suitable', 'Not Suitable', or 'Partially Suitable' only.\n\n"
            f"Job Description: {job_description}\n"
            f"Resume Skills: {skills_str}\n\n"
            "Evaluation:"
        )
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        result = chat_completion.choices[0].message.content.strip()
        if not result:
            result = "No evaluation generated"
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error in /check_suitability: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_feedback', methods=['POST'])
def generate_feedback():
    try:
        from groq import Groq  # Import Groq client library
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        data = request.get_json()
        job_description = data.get("job_description", "").strip()
        resume_skills = data.get("resume_skills", [])
        if not job_description:
            return jsonify({"error": "Job description cannot be empty"}), 400
        if not resume_skills:
            return jsonify({"error": "No skills provided for feedback"}), 400

        skills_str = ", ".join(resume_skills)
        prompt = (
            "You are an expert career consultant. Based on the job description and resume skills provided, ignore irrelevant skills if given as input, "
            "generate concise suggestions (3-5 sentences) for improving the candidate's resume so that it better aligns with the job requirements. "
            "Do not repeat the input in your answer. Only output the suggestions.\n\n"
            f"Job Description: {job_description}\n"
            f"Resume Skills: {skills_str}\n\n"
            "Suggestions:\n-"
        )
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        feedback = chat_completion.choices[0].message.content.strip()
        if not feedback:
            feedback = "No feedback generated"
        return jsonify({"feedback": feedback})
    except Exception as e:
        print(f"Error in /generate_feedback: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/career_recommendation', methods=['POST'])
def career_recommendation():
    try:
        data = request.get_json()
        candidate_profile = data.get("profile", "").strip()
        if not candidate_profile:
            return jsonify({"error": "No candidate profile provided"}), 400
        
        encoding = rec_preprocess(candidate_profile)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        with torch.no_grad():
            output = career_rec_model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = output.logits.squeeze()  # Shape: [6]
        probabilities = F.softmax(logits, dim=0)
        predicted_class = torch.argmax(probabilities).item()
        recommendation_label = career_mapping.get(predicted_class, "Unknown")
        prob_dict = {career_mapping[i]: float(probabilities[i]) for i in range(len(probabilities))}
        
        return jsonify({
            "recommendation": recommendation_label,
            "probabilities": prob_dict
        })
    except Exception as e:
        print(f"Error in /career_recommendation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
