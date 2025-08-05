import json
import os
import re
import requests

from flask import Flask, request, jsonify, redirect
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
    0: "Software Engineer",
    1: "Data Scientist",
    2: "AI Engineer",
    3: "Cybersecurity Engineer",
    4: "Full Stack Developer",
    5: "DevOps Engineer"
}

career_skills = {
    "Software Engineer": ["C++", "Java", "Python", "Git", "OOP", "Agile"],
    "Data Scientist": ["Python", "R", "SQL", "Machine Learning", "Statistics", "TensorFlow"],
    "AI Engineer": ["Python", "Deep Learning", "PyTorch", "TensorFlow", "NLP", "Computer Vision"],
    "Cybersecurity Engineer": ["Network Security", "Ethical Hacking", "Cryptography", "Penetration Testing", "Linux", "IDS/IPS"],
    "Full Stack Developer": ["JavaScript", "React", "Node.js", "HTML", "CSS", "MongoDB"],
    "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "CI/CD", "Bash", "Terraform"]
}

# -------------------------------
# Known Skills (Whitelist)
# -------------------------------
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
        "objective", "experience", "education", "skills", "certifications", 
        "projects", "summary", "achievements", "work experience", "professional summary"
    ]
    match_count = sum(1 for keyword in resume_keywords if keyword in text.lower())
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

def merge_wordpieces(entities):
    merged = []
    buffer = ""
    for entity in entities:
        token = entity.get('word', '')
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
        # 1. Use NER to extract candidate tokens.
        ner_results = ner_pipeline(text)
        print("NER Results:", ner_results)
        
        merged_tokens = merge_wordpieces(ner_results)
        print("Merged Tokens:", merged_tokens)
        
        # Clean tokens: remove non-alphanumeric (allow +, #, .) and lower-case.
        ner_candidates = {re.sub(r"[^a-z0-9\+\#\.]", "", token.lower()).strip() 
                          for token in merged_tokens if len(token) > 1}
        print("NER Candidates:", ner_candidates)
        
        # 2. Use keyword matching with known skills.
        text_lower = text.lower()
        keyword_candidates = {skill for skill in KNOWN_SKILLS if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)}
        print("Keyword Candidates:", keyword_candidates)
        
        # 3. Combine candidates and then intersect with the whitelist.
        all_candidates = ner_candidates.union(keyword_candidates)
        print("All Candidates:", all_candidates)
        valid_candidates = {skill for skill in all_candidates if skill in {s.lower() for s in KNOWN_SKILLS}}
        print("Valid Candidates (After Intersection with Whitelist):", valid_candidates)
        
        candidate_list = list(valid_candidates)
        
        # 4. Now filter using LLM to get exactly 5 dominant skills.
        filtered_skills = filter_skills_with_llm(candidate_list)
        print("Final Filtered Skills:", filtered_skills)
        return filtered_skills if filtered_skills else ["No skills detected"]
    except Exception as e:
        print(f"Error extracting skills: {str(e)}")
        return []

def filter_skills_with_llm(skills):
    try:
        # First iteration: build the prompt using the candidate list.
        skills_str = ", ".join(skills)
        prompt = (
            f"You are a professional career consultant. Given the following list of skills extracted from a resume: {skills_str}, "
            "remove any skills that are irrelevant, overly generic, or not directly related to technical or domain-specific expertise. "
            "Only include skills that are recognized as valid technical or specialized skills (such as programming languages, frameworks, tools, or methodologies). "
            "Ensure that each skill is at least 3 characters long and not a fragment. "
            "Also, only include skills that appear in the following whitelist: "
            f"{list(KNOWN_SKILLS)}. "
            "Return exactly a JSON array containing exactly 5 dominant skills. For example: [\"python\", \"node.js\", \"react\", \"java\", \"mongodb\"]."
        )
        print("LLM Prompt (Iteration 1):", prompt)
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-r1-distill-llama-70b",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
        )
        print("LLM Response Status (Iteration 1):", response.status_code)
        if response.status_code == 200:
            result_text = response.json()["choices"][0]["message"]["content"].strip()
            print("Raw LLM Output (Iteration 1):", result_text)
            try:
                filtered_skills = json.loads(result_text)
            except Exception as e:
                print(f"JSON parsing error (Iteration 1): {e}")
                filtered_skills = [s.strip() for s in re.split(r'[,\n]+', result_text) if s.strip()]
            if isinstance(filtered_skills, list):
                exclusions = {"and", "the", "for", "with", "job", "experience", "project",
                              "development", "studio", "of", "an", "ll", "mu", "re", "nl", "ma", "godb", "cs", "model", "language", "thoot"}
                first_final = [skill.strip() for skill in filtered_skills
                               if len(skill.strip()) >= 3 and skill.lower() not in exclusions]
                print("Local Filtered Skills (Iteration 1):", first_final)
                if len(first_final) == 5:
                    return first_final
                else:
                    new_skills_str = ", ".join(first_final)
                    new_prompt = (
                        f"You are a professional career consultant. Given the following list of skills extracted from a resume: {new_skills_str}, "
                        "remove any skills that are irrelevant, overly generic, or not directly related to technical or domain-specific expertise. "
                        "Only include skills that are recognized as valid technical or specialized skills (such as programming languages, frameworks, tools, or methodologies). "
                        "Ensure that each skill is at least 3 characters long and not a fragment. "
                        "Return exactly a JSON array containing exactly 5 dominant skills. For example: [\"python\", \"node.js\", \"react\", \"java\", \"mongodb\"]."
                    )
                    print("LLM Prompt (Iteration 2):", new_prompt)
                    second_response = requests.post(
                        GROQ_API_URL,
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "deepseek-r1-distill-llama-70b",
                            "messages": [{"role": "user", "content": new_prompt}],
                            "response_format": {"type": "json_object"}
                        }
                    )
                    print("LLM Response Status (Iteration 2):", second_response.status_code)
                    if second_response.status_code == 200:
                        second_result_text = second_response.json()["choices"][0]["message"]["content"].strip()
                        print("Raw LLM Output (Iteration 2):", second_result_text)
                        try:
                            second_filtered_skills = json.loads(second_result_text)
                        except Exception as e:
                            print(f"JSON parsing error (Iteration 2): {e}")
                            second_filtered_skills = [s.strip() for s in re.split(r'[,\n]+', second_result_text) if s.strip()]
                        if isinstance(second_filtered_skills, list):
                            second_final = [skill.strip() for skill in second_filtered_skills
                                            if len(skill.strip()) >= 3 and skill.lower() not in exclusions]
                            print("Local Filtered Skills (Iteration 2):", second_final)
                            if len(second_final) == 5:
                                return second_final
                            else:
                                return second_final[:5]
                    return first_final[:5]
        print("LLM filtering did not return a valid list. Returning first 5 candidate skills.")
        return skills[:5]
    except Exception as e:
        print(f"Error filtering skills with LLM: {e}")
        return skills[:5]

# -------------------------------
# Endpoints
# -------------------------------

@app.route('/resume_feedback', methods=['POST'])
def resume_feedback():
    try:
        # Validate file input.
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Extract resume text.
        resume_text = extract_text_from_pdf(file)
        if not resume_text:
            return jsonify({"error": "No text found in PDF"}), 400
        if not is_resume(resume_text):
            return jsonify({"error": "Uploaded PDF is not a valid resume."}), 400

        # Classify resume using BERT.
        resume_category = predict_resume_category(resume_text)

        # Extract skills from resume.
        extracted_skills = extract_skills_from_resume(resume_text)

        # If resume is classified as "Good", skip Llama feedback.
        if resume_category.lower() not in ["bad", "average"]:
            return jsonify({
                "category": resume_category,
                "feedback": "Your resume is already categorized as 'Good'. No additional feedback is needed.",
                "skills": extracted_skills
            }), 200

        # For "Bad" or "Average" resumes, build the Llama feedback prompt.
        # Note: We instruct the model to include the word "json" for format confirmation, but we remove it later.
        trimmed_resume = resume_text[:1000]  # Optionally trim for brevity if needed.
        prompt = (
            f"You are a professional career consultant. The resume provided below has been categorized as {resume_category}. "
            "Please provide constructive, friendly, and professional feedback in 3-5 concise sentences on how to improve the resume. "
            "Focus on actionable recommendations in terms of formatting, content, and skills presentation. "
            "Also, include the word 'json' somewhere in your response to confirm the format. "
            "\n\nResume:\n" + trimmed_resume + "\n\nFeedback:"
        )

        # Call the Llama API via GROQ.
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-r1-distill-llama-70b",
                "messages": [{"role": "user", "content": prompt}],
                # We request plain text instead of JSON object.
                "response_format": {"type": "text"}
            }
        )
        
        if response.status_code == 200:
            # Get the feedback text.
            feedback = response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Llama API call failed with status code: {response.status_code}")
            print("Response content:", response.text)
            feedback = "Failed to generate feedback."
        
        # Remove the word "json" from the feedback before sending it to the frontend.
        cleaned_feedback = re.sub(r'\bjson\b', '', feedback, flags=re.IGNORECASE).strip()
        
        return jsonify({
            "category": resume_category,
            "feedback": cleaned_feedback,
            "skills": extracted_skills
        })
    except Exception as e:
        print(f"Error in /resume_feedback: {e}")
        return jsonify({"error": str(e)}), 500

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
            "skills": extracted_skills,
            "category": resume_category
        })
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_suitability', methods=['POST'])
def check_suitability():
    try:
        from groq import Groq
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
        from groq import Groq
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

# -------------------------------
# Note on LinkedIn Redirection
# -------------------------------
# The original "/connect" endpoint that used the LLM to generate LinkedIn profile recommendations has been removed.
# Instead, your frontend button can simply redirect the user to LinkedIn.
# For example:
#   window.open("https://www.linkedin.com/", "_blank");
# This ensures users are taken directly to LinkedIn without backend intervention.

if __name__ == '__main__':
    app.run()
