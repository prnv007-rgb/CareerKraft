from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline
import tensorflow as tf
import fitz  # PyMuPDF
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the tokenizer and model for resume classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained("C:/Users/prana/OneDrive/Desktop/nlp/bert_resume_final_model")

# Initialize NER pipeline using Hugging Face (using a generic pre-trained model for NER)
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(file):
    try:
        file_bytes = file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

# Function to detect if the PDF is a resume
def is_resume(text):
    keywords = ["Objective", "Experience", "Education", "Skills", "Certifications", "Projects", "Summary", "Achievements", "Profile"]
    matches = [word for word in keywords if word.lower() in text.lower()]
    return len(matches) >= 2

# Function to make predictions with softmax for resume classification
def predict_resume_category(text):
    try:
        # Tokenize the input resume text
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
        outputs = model(inputs)
        logits = outputs.logits

        # Apply softmax to convert logits into probabilities
        probabilities = tf.nn.softmax(logits)

        # Get the predicted class (Good, Average, or Bad)
        predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]

        labels = {0: "Good", 1: "Average", 2: "Bad"}
        return labels.get(predicted_class, "Unknown")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return f"Error: {str(e)}"

# Function to extract skills using NER model
def extract_skills_from_resume(text):
    try:
        # Apply NER pipeline to extract entities
        entities = ner_pipeline(text)
        
        # Print entities to see what is being detected (for debugging purposes)
        print("Entities detected:", entities)
        
        # Extract skills based on NER entity labels (if found)
        skills = [entity['word'] for entity in entities if 'SKILL' in entity['entity'].upper()]  # Adjust filtering if needed
        
        # Return skills as a formatted string
        return ", ".join(skills)  # Join skills into a comma-separated string
    except Exception as e:
        print(f"Error in extracting skills: {str(e)}")
        return ""

# Flask route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Check if the uploaded file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(file)

        if not text:
            return jsonify({"error": "No text found in PDF"}), 400

        # Check if the PDF is a resume
        if not is_resume(text):
            return jsonify({"error": "Uploaded PDF is not a valid resume. Please upload a resume in PDF format."}), 400

        # Make the prediction for resume category (Good, Average, Bad)
        resume_category = predict_resume_category(text)

        # Extract skills from the resume using NER
        extracted_skills = extract_skills_from_resume(text)

        # Return the results: category of the resume and extracted skills
        return jsonify({
            "category": resume_category,
            "skills": extracted_skills
        })

    except Exception as e:
        print(f"Error in /predict route: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
