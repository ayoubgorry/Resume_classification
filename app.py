import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pytesseract
from PIL import Image
import cv2
import docx
import PyPDF2

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update based on the environment



# Custom CSS for enhanced styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Download NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Category mapping
category_map = {0: 'Accountant', 1: 'Advocate', 2: 'Agricultural', 3: 'Apparel', 4: 'Architects', 5: 'Arts', 6: 'Automobile', 7: 'Aviation', 8: 'BPO', 9: 'Banking', 10: 'Blockchain', 11: 'Building _Construction', 12: 'Business Analyst', 13: 'Civil Engineer', 14: 'Consultant', 15: 'Database', 16: 'Designing', 17: 'DevOps Engineer', 18: 'Digital Media', 19: 'DotNet Developer', 20: 'ETL Developer', 21: 'Education', 22: 'Electrical Engineering', 23: 'Finance', 24: 'Food_Beverages', 25: 'HR', 26: 'Health_Fitness', 27: 'Information Technology', 28: 'Java Developer', 29: 'Managment', 30: 'Mechanical Engineer', 31: 'Network Security Engineer', 32: 'Operations Manager', 33: 'PMO', 34: 'Public Relations', 35: 'Python Developer', 36: 'React Developer', 37: 'SAP Developer', 38: 'SQL Developer', 39: 'Sales', 40: 'Testing', 41: 'data science', 42: 'web designing'}

# Load pre-trained model and TF-IDF vectorizer
model = pickle.load(open('./models/model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('./models/tfidf.pkl', 'rb'))

def preprocess_text(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')

    text = re.sub(url_pattern, '', text)
    text = re.sub(email_pattern, '', text)
    text = re.sub('[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)

    text = text.lower()

    return text

def extract_text_from_file(uploaded_file):
    """Extract text from various file types"""
    
    # Image file processing
    if uploaded_file.type.startswith('image/'):
        # Read the image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Resize for consistency
        img_array = cv2.resize(img_array, (1024, 1024))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply Otsu's thresholding
        final = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(final)
        return text

    # PDF file processing
    elif uploaded_file.type == 'application/pdf':
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    # DOCX file processing
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(uploaded_file)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    # Text file processing
    elif uploaded_file.type == 'text/plain':
        return uploaded_file.getvalue().decode('utf-8')
    
    else:
        st.error(f"Unsupported file type: {uploaded_file.type}")
        return None


def predict_category(resume_text):
    # Preprocess the resume text
    processed_text = preprocess_text(resume_text)
    
    # Vectorize the text
    text_vectorized = tfidf_vectorizer.transform([processed_text])
    
    # Predict category
    prediction = model.predict(text_vectorized)
    
    # Map numeric prediction to category name
    return category_map.get(prediction[0], f"Unknown Category (Code: {prediction[0]})")

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Resume Classification AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Add custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 36px;
        text-align: center;
        margin-bottom: 30px;
    }
    .stRadio > label {
        font-size: 18px;
        color: #34495e;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
    }
    .stSuccess {
        background-color: #2ecc71;
        color: white;
        padding: 15px;
        border-radius: 10px;
    }
    .stError {
        background-color: #e74c3c;
        color: white;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title with icon
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📄 CVGENiUS 🤖</h1>", unsafe_allow_html=True)
    
    # Subtitle
    st.markdown("<h3 style='text-align: center; color: #7f8c8d;'>Predict Your Career Path with AI</h3>", unsafe_allow_html=True)

    # Input method selection
    input_method = st.radio('Choose Input Method', ['Upload File', 'Enter Text'])

    resume_text = None
    
    if input_method == 'Upload File':
        # File upload section with custom styling
        st.markdown("### 📤 Upload Your Resume")
        uploaded_file = st.file_uploader(
            'Choose a resume file', 
            type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help='Upload PDF, DOCX, TXT, or image files'
        )
        
        if uploaded_file is not None:
            # Extract text from uploaded file
            resume_text = extract_text_from_file(uploaded_file)
            
            if resume_text:
                st.text_area('Extracted Text', resume_text, height=200)
    
    else:
        # Text input section
        st.markdown("### ✍️ Paste Resume Text")
        resume_text = st.text_area('Paste Resume Text Here', height=300)

    # Prediction button with custom styling
    if st.button('🔮 Predict Category'):
        if resume_text:
            try:
                prediction = predict_category(resume_text)
                st.success(f'🎯 Predicted Category: {prediction}')
                
                # Add some additional information or advice
                st.markdown(f"""
                ### 💡 Career Insights
                Based on our AI analysis, your resume suggests a strong fit for a **{prediction}** role.
                
                **Recommended Next Steps:**
                - Update your resume to highlight relevant skills
                - Research job opportunities in {prediction} domain
                - Consider additional certifications or training
                """)
            
            except Exception as e:
                st.error(f'❌ Error in prediction: {e}')
        else:
            st.warning('⚠️ Please enter or upload resume text')

    # Footer
    st.markdown("""
    ---
    * Made by Ayoub Gorry | [GitHub](https://github.com/AyoubGorry) | [LinkedIn](https://www.linkedin.com/in/ayoub-gorry-8772a0236) 
    """)

if __name__ == '__main__':
    main()
