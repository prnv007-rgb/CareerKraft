from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline
import tensorflow as tf
import fitz  # PyMuPDF
import re
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/")
def home():
    return "Hello World!"
# -------------------------------
# Resume Classification Setup
# -------------------------------

# Load pre-trained tokenizer and model for resume classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained("C:/Users/prana/OneDrive/Desktop/nlp/bert_resume_final_model")

# Load Named Entity Recognition (NER) model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Groq API Key (Ensure it's set as an environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"

# -------------------------------
# Helper Functions
# -------------------------------

def extract_text_from_pdf(file):
    try:
        file_bytes = file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = " ".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def is_resume(text):
    resume_keywords = [
        "Objective", "Experience", "Education", "Skills", "Certifications", "Projects",
        "Summary", "Achievements", "Work Experience", "Professional Summary"
    ]
    match_count = sum(1 for keyword in resume_keywords if keyword.lower() in text.lower())
    return match_count >= 2

def predict_resume_category(text):
    try:
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits)
        predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]
        labels = {0: "Good", 1: "Average", 2: "Bad"}
        return labels.get(predicted_class, "Unknown")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return f"Error: {str(e)}"

def extract_skills_from_resume(text):
    try:
        entities = ner_pipeline(text)
        raw_skills = list(set([
            re.sub(r"[^a-zA-Z\s]", "", entity['word']).strip()
            for entity in entities if entity.get('word')
        ]))
        return raw_skills if raw_skills else ["No skills detected"]
    except Exception as e:
        print(f"Error extracting skills: {str(e)}")
        return []

# -------------------------------
# Flask Endpoints
# -------------------------------

@app.route('/predict', methods=['POST'])
def predict():
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
        print(f"Error in /predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_suitability', methods=['POST'])
def check_suitability():
    try:
        from groq import Groq  # Import Groq client library
        import os
        
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
            "You are an expert career consultant. Based on the following job description and resume skills, "
            "evaluate the candidate's resume suitability for the job. Provide a concise answer as 'Suitable', 'Not Suitable', "
            "or 'Partially Suitable' only.\n\n"
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
        print(f"Error in /check_suitability: {str(e)}")
        return jsonify({"error": str(e)}), 500



@app.route('/generate_feedback', methods=['POST'])
def generate_feedback():
    try:
        from groq import Groq  # Import Groq client library
        import os
        
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
            "You are an expert career consultant. Based on the job description and resume skills provided, "
            "generate three concise, bullet-point suggestions for improving the candidate's resume so that it better aligns with the job requirements. "
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
        print(f"Error in /generate_feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run()