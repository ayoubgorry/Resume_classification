# Resume Classification AI

An intelligent Streamlit application that categorizes resumes into professional domains using advanced machine learning techniques. This tool aids job seekers and recruiters in quickly identifying career paths by leveraging natural language processing and TF-IDF vectorization.

---

## Features

- **Multi-Format Support**: Upload resumes in PDF, DOCX, TXT, and image formats.
- **AI-Powered Classification**: Predict resume categories with high accuracy.
- **User-Friendly Interface**: Clean, modern Streamlit design for seamless interaction.
- **Comprehensive Career Insights**: Provides tailored career recommendations based on predicted categories.

---

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Framework for building interactive web apps.
- **Scikit-learn**: Machine learning library for classification models.
- **NLTK**: Toolkit for natural language processing.
- **OpenCV & Tesseract OCR**: Extract text from images.
- **PyPDF2**: Process PDF files.
- **python-docx**: Read and parse DOCX files.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AyoubGorry/resume-classification-ai.git
   cd resume-classification-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   - **Windows**: Download and install from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract)
   - **macOS**: Run `brew install tesseract`
   - **Linux**: Run `sudo apt-get install tesseract-ocr`

---

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

### Supported Input Methods:
- Upload resume file (PDF, DOCX, TXT, PNG, JPG)
- Paste resume text directly into the interface

### Live Demo:
- [Resume Classification AI](https://agresumeclassification.streamlit.app/)

---

## Supported Job Categories

The AI can classify resumes into 43 distinct professional categories, including:
- Software Development
- Engineering
- Business Analysis
- Design
- Sales
- Marketing
- And many more!

---

## How It Works

1. **Text Preprocessing:**
   - Converts text to lowercase.
   - Removes special characters and stopwords.

2. **Feature Extraction:**
   - Utilizes TF-IDF Vectorization to encode textual data.

3. **Machine Learning Classification:**
   - Trained on a diverse dataset of professional resumes to ensure robust predictions.

---

## Contributing

Contributions are welcome! If you’d like to improve the project, please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request for review.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Author

**Ayoub Gorry**

- **GitHub**: [https://github.com/AyoubGorry](https://github.com/AyoubGorry)
- **LinkedIn**: [https://www.linkedin.com/in/ayoub-gorry-8772a0236](https://www.linkedin.com/in/ayoub-gorry-8772a0236)

---

## Acknowledgments

- Inspired by the challenge of automating resume screening.
- Thanks to the open-source community for providing incredible tools and libraries that made this project possible.

---

