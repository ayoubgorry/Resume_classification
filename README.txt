RESUME CLASSIFICATION AI
=======================

Project Overview
----------------
Resume Classification AI is an intelligent Streamlit application that uses machine learning 
to automatically categorize resumes into professional domains. By leveraging advanced 
natural language processing and TF-IDF vectorization, this tool helps job seekers and 
recruiters quickly understand the potential career path of a candidate.

Features
--------
* Multi-Format Support: Upload resumes in PDF, DOCX, TXT, and image formats
* AI-Powered Classification: Predict resume category with high accuracy
* User-Friendly Interface: Clean, modern Streamlit design
* Comprehensive Career Insights: Provides recommendations based on predicted category

Technologies Used
-----------------
- Python
- Streamlit
- Scikit-learn
- NLTK
- OpenCV
- Tesseract OCR
- PyPDF2
- python-docx

Installation
------------
1. Clone the repository:
   git clone https://github.com/AyoubGorry/resume-classification-ai.git
   cd resume-classification-ai

2. Install dependencies:
   pip install -r requirements.txt

3. Install Tesseract OCR:
   - Windows: Download from Tesseract GitHub
   - macOS: brew install tesseract
   - Linux: sudo apt-get install tesseract-ocr

Usage
------
Run the Streamlit app:
streamlit run app.py

Supported Input Methods:
- Upload resume file (PDF, DOCX, TXT, PNG, JPG)
- Paste resume text directly

Supported Job Categories
-----------------------
The AI can classify resumes into 43 different professional categories, including:
- Software Development
- Engineering
- Business Analysis
- Design
- Sales
- Marketing
- And many more!

How It Works
------------
1. Text Preprocessing
   - Lowercase conversion
   - Special character removal
   - Stopword elimination

2. Feature Extraction
   - TF-IDF Vectorization

3. Machine Learning Classification
   - Trained on a diverse dataset of professional resumes

Contributing
------------
Contributions are welcome! Please feel free to submit a Pull Request.

License
--------
This project is open-source and available under the MIT License.

Author
-------
Ayoub Gorry
- GitHub: https://github.com/AyoubGorry
- LinkedIn: https://www.linkedin.com/in/ayoub-gorry-8772a0236

Acknowledgments
---------------
- Inspired by the challenge of automating resume screening
- Thanks to the open-source community for incredible tools and libraries