from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline, T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf
import fitz  # PyMuPDF
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# -------------------------------
# Resume Classification Setup
# -------------------------------

# Load pre-trained tokenizer and model for resume classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained("C:/Users/prana/OneDrive/Desktop/nlp/bert_resume_final_model")

# Load Named Entity Recognition (NER) model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Predefined Skills Database
SKILLS_DB = [
    "Python", "Java", "C++", "JavaScript", "React", "Angular", "Django", "Flask",
    "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "Data Science",
    "SQL", "MongoDB", "AWS", "Azure", "Docker", "Kubernetes", "NLP", "BERT",
    "Linux", "Git", "PostgreSQL", "CSS", "HTML", "REST API", "GraphQL", "Android",
    "iOS", "Swift", "Kotlin", "Cybersecurity", "Penetration Testing", "Blockchain",
    "Cloud Computing", "Tableau", "Power BI", "Natural Language Processing",
    "Computer Vision", "Agile", "Scrum", "JIRA", "CI/CD", "DevOps"
]

# -------------------------------
# T5 Feedback Generation Setup
# -------------------------------

# Use T5-small (you can try larger variants if resources allow)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

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
        text_lower = text.lower()
        entities = ner_pipeline(text)
        raw_skills = list(set([
            re.sub(r"[^a-zA-Z\s]", "", entity['word']).strip()
            for entity in entities if entity.get('word')
        ]))
        extracted_skills = [skill for skill in SKILLS_DB if skill.lower() in text_lower]
        final_skills = list(set(raw_skills + extracted_skills))
        return final_skills if final_skills else ["No skills detected"]
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
        data = request.get_json()
        print(f"Received data: {data}")
        job_description = data.get("job_description", "").strip().lower()
        resume_skills = data.get("resume_skills", [])
        if not job_description:
            return jsonify({"error": "Job description cannot be empty"}), 400
        if not resume_skills:
            return jsonify({"error": "No skills provided"}), 400

        resume_skills_lower = {skill.lower() for skill in resume_skills}
        job_words = set(job_description.split())
        matched_skills = resume_skills_lower.intersection(job_words)
        if matched_skills:
            return jsonify({"result": "Suitable for Job", "matched_skills": list(matched_skills)})
        else:
            return jsonify({"result": "Not Suitable for Job", "matched_skills": []})
    except Exception as e:
        print(f"Error in /check_suitability: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_feedback', methods=['POST'])
def generate_feedback():
    try:
        data = request.get_json()
        job_description = data.get("job_description", "").strip()
        resume_skills = data.get("resume_skills", [])
        if not job_description:
            return jsonify({"error": "Job description cannot be empty"}), 400
        if not resume_skills:
            return jsonify({"error": "No skills provided for feedback"}), 400

        skills_str = ", ".join(resume_skills)
        # Refined prompt: explicitly instruct T5 to output three bullet points only
        prompt = (
            "You are an expert career consultant. Based on the job description and resume skills provided, "
            "generate three concise, bullet-point suggestions for improving the candidate's resume so that it better aligns with the job requirements. "
            "Do not repeat the input in your answer. Only output the suggestions.\n\n"
            "Job Description: " + job_description + "\n"
            "Resume Skills: " + skills_str + "\n\n"
            "Suggestions:\n-"
        )
        
        # Encode the prompt
        inputs = t5_tokenizer.encode(prompt, return_tensors="tf", max_length=512, truncation=True)
        # Generate feedback using beam search for more controlled output
        outputs = t5_model.generate(
            inputs,
            max_length=200,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        feedback = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("T5 Feedback:", feedback)
        return jsonify({"feedback": feedback})
    except Exception as e:
        print(f"Error in /generate_feedback: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
