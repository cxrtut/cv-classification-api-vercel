from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import PyPDF2
from transformers import pipeline
from io import BytesIO
import uvicorn

app = FastAPI(title="CV Classification API")

# Initialize the text classification model (BERT-based)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Expanded CV categories
CV_CATEGORIES = {
    "POSITIVE": [
        "Java Developer",
        "Testing Engineer",
        "Python Developer",
        "DevOps Engineer",
        "Web Designer",
        "Web Developer",
        "HR Manager",
        "Blockchain Developer",
        "ETL Developer",
        "Operations Manager",
        "Data Science Manager",
        "Sales Manager",
        "Mechanical Engineer",
        "Arts Developer",
        "Electrical Engineering",
        "Health and Fitness Coach",
        "Business Analyst",
        "DotNet Developer",
        "Automation Testing Engineer",
        "Network Security Engineer",
        "SAP Developer",
        "Civil Engineer",
        "Advocate"
    ],
    "NEGATIVE": [
        "Software Engineer",
        "Data Scientist",
        "Product Manager",
        "Systems Analyst",
        "Cloud Architect",
        "Mobile Developer",
        "UI/UX Designer",
        "Cybersecurity Analyst"
    ]
}

@app.post("/classify")
async def classify_cv(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read PDF file
        content = await file.read()
        pdf_file = BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        # Classify the extracted text
        result = classifier(text, truncation=True, max_length=512)
        label = result[0]["label"]
        confidence = result[0]["score"]

        # Map model output to CV category
        categories = CV_CATEGORIES.get(label, ["Unknown Category"])
        category = categories[0] if categories else "Unknown Category"

        return JSONResponse({
            "category": category,
            "confidence": confidence,
            "extracted_text": text[:500] + "..." if len(text) > 500 else text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)