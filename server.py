from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import fitz  # PyMuPDF
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(file):
    try:
        # Read the file as a byte stream
        file_bytes = file.read()
        
        # Open the PDF from the byte stream
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
    return len(matches) >= 2  # Consider it a resume if at least 2 keywords are found

# Function to make predictions
def predict_resume_category(text):
    try:
        # Tokenize the input resume text
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
        
        # Get logits from the model
        outputs = model(inputs)
        logits = outputs.logits

        # Debugging: Print logits and predictions
        print("Logits:", logits)

        # Get the predicted class (Good, Average, or Bad)
        prediction = tf.argmax(logits, axis=1).numpy()[0]
        
        # Map class index to label
        labels = {0: "Bad", 1: "Average", 2: "Good"}
        return labels.get(prediction, "Unknown")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # Debugging: Print error
        return f"Error: {str(e)}"

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
            return jsonify({"error": "Uploaded PDF is not a resume"}), 400

        # Make the prediction
        result = predict_resume_category(text)

        # Debugging: Check the result before returning
        print("Prediction result:", result)

        return jsonify({"category": result})

    except Exception as e:
        print(f"Error in /predict route: {str(e)}")  # Debugging: Print error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
