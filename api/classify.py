import PyPDF2
from transformers import pipeline
from io import BytesIO
import json

# Initialize the text classification model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Expanded CV categories
CV_CATEGORIES = {
    "POSITIVE": [
        "Java Developer", "Testing Engineer", "Python Developer", "DevOps Engineer",
        "Web Designer", "Web Developer", "HR Manager", "Blockchain Developer",
        "ETL Developer", "Operations Manager", "Data Science Manager", "Sales Manager",
        "Mechanical Engineer", "Arts Developer", "Electrical Engineering",
        "Health and Fitness Coach", "Business Analyst", "DotNet Developer",
        "Automation Testing Engineer", "Network Security Engineer", "SAP Developer",
        "Civil Engineer", "Advocate"
    ],
    "NEGATIVE": [
        "Software Engineer", "Data Scientist", "Product Manager", "Systems Analyst",
        "Cloud Architect", "Mobile Developer", "UI/UX Designer", "Cybersecurity Analyst"
    ]
}

def handler(request):
    try:
        if request.method != 'POST':
            return {'error': 'Only POST requests are supported'}, 405

        # Get the uploaded file
        file = request.files.get('file')
        if not file or not file.filename.endswith('.pdf'):
            return {'error': 'Only PDF files are supported'}, 400

        # Read PDF content
        content = file.read()
        pdf_file = BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if not text.strip():
            return {'error': 'No text could be extracted from the PDF'}, 400

        # Classify the extracted text
        result = classifier(text, truncation=True, max_length=512)
        label = result[0]["label"]
        confidence = result[0]["score"]
        categories = CV_CATEGORIES.get(label, ["Unknown Category"])
        category = categories[0] if categories else "Unknown Category"

        return {
            'category': category,
            'confidence': confidence,
            'extracted_text': text[:500] + "..." if len(text) > 500 else text
        }, 200

    except Exception as e:
        return {'error': f'Error processing CV: {str(e)}'}, 500