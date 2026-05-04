import os
import sys
import torch
import pandas as pd
import numpy as np
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import json

# Import Google Gemini AI
# Import Google Gemini (New SDK)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("WARNING: google-genai not installed. Run: pip install google-genai")
    GEMINI_AVAILABLE = False

# Try importing CNN module properly
try:
    import CNN
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import CNN
    except ModuleNotFoundError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from AgriDoc import CNN
        except ModuleNotFoundError:
            print("ERROR: Could not import CNN module. Please check file structure.")
            class CNN:
                def __init__(self, num_classes):
                    self.num_classes = num_classes
                def __call__(self, x):
                    return np.zeros((1, self.num_classes))
                def eval(self):
                    pass
                def load_state_dict(self, state_dict):
                    pass

# Define directory for storing model file
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, 'datasetofdisease.pt')

# Define alternative download method for Render deployment
def download_model_directly():
    try:
        import requests
        import re
        from urllib.parse import unquote
        
        print(f"Attempting direct download of model file to {model_path}")
        
        # Your Google Drive file ID
        file_id = '1En73N317hTlvJpZDa-FqsMsIMskzU70h'
        
        # First, get the download token
        URL = f"https://drive.google.com/uc?id={file_id}&export=download"
        session = requests.Session()
        
        print(f"Getting Google Drive download token from {URL}")
        response = session.get(URL, stream=True)
        
        # Try to find the confirm token in the response
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
                
        if token:
            print(f"Found token: {token}")
            params = {'id': file_id, 'confirm': token, 'export': 'download'}
            response = session.get("https://drive.google.com/uc", params=params, stream=True)
        else:
            print("No token needed, attempting direct download")
            
        # Save the file
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Model downloaded successfully to {model_path}")
            return True
        else:
            print(f"Failed to download model: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

# Try to load the model
def load_model_safely():
    # Check if model file exists at the expected location
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        
        # Try direct download method first (more reliable on Render)
        if download_model_directly():
            print("Direct download successful")
        else:
            # Fall back to gdown if direct download fails
            try:
                import gdown
                file_id = '1En73N317hTlvJpZDa-FqsMsIMskzU70h'
                url = f'https://drive.google.com/uc?id={file_id}'
                print(f"Attempting to download with gdown from {url}")
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                print(f"gdown download failed: {str(e)}")
                print("WARNING: Could not download model. App will run with limited functionality.")
                return CNN.CNN(39)  # Return dummy model
    
    try:
        # Load model using absolute path
        print(f"Loading model from {model_path}")
        # model = CNN.CNN(39)
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.eval()
        model = CNN.CNN(39)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("WARNING: Running with limited functionality")
        # Return a dummy model that will return zeros
        return CNN.CNN(39)

# Load disease and supplement info with error handling
def load_data_safely():
    try:
        # Use absolute paths for CSV files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        disease_csv = os.path.join(base_dir, 'disease_info.csv')
        supplement_csv = os.path.join(base_dir, 'supplement_info.csv')
        
        disease_info = pd.read_csv(disease_csv, encoding='cp1252')
        supplement_info = pd.read_csv(supplement_csv, encoding='cp1252')
        return disease_info, supplement_info
    except Exception as e:
        print(f"Error loading CSV data: {str(e)}")
        # Return empty DataFrames with expected columns as fallback
        disease_cols = ['disease_name', 'description', 'Possible Steps', 'image_url']
        supplement_cols = ['supplement name', 'supplement image', 'buy link']
        
        return pd.DataFrame(columns=disease_cols), pd.DataFrame(columns=supplement_cols)

# Load model and data
model = load_model_safely()
disease_info, supplement_info = load_data_safely()

# Initialize Gemini AI
# GEMINI_API_KEY = os.environ.get('AIzaSyD6h9kqy5Dvm9oca_o-VPHiQu7gGA0bVV8', 'AIzaSyD6h9kqy5Dvm9oca_o-VPHiQu7gGA0bVV8')
# model = genai.GenerativeModel("gemini-pro")
# response = model.generate_content("Hello, are you working?")
# print(response.text)
# ================= GEMINI CONFIG =================

# ================= GEMINI CONFIG =================

# ================= GEMINI CONFIG =================

# ================= GEMINI CONFIG =================

if GEMINI_AVAILABLE:

    GEMINI_API_KEY = "AIzaSyBPFwWdv4tL_O9rcb80_Hw0TgbcaGVUdQQ"  # Regenerate later

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Use best vision model for your project
        GEMINI_MODEL = "gemini-2.5-flash"

        print("✅ Gemini Initialized with", GEMINI_MODEL)

    except Exception as e:
        print("❌ Gemini Error:", e)
        GEMINI_AVAILABLE = False
        client = None

# ================================================

# ================================================
# ================================================

# ================================================
# if GEMINI_AVAILABLE:
#     try:
#         genai.configure(api_key=GEMINI_API_KEY)
#         # Use gemini-2.5-flash which is the latest stable vision model
#         gemini_model = genai.GenerativeModel('gemini-2.5-flash')
#         print("Gemini AI initialized successfully with model: gemini-2.5-flash")
#     except Exception as e:
#         print(f"Error initializing Gemini AI: {str(e)}")
#         try:
#             # Fallback to gemini-2.0-flash if 2.5 is not available
#             gemini_model = genai.GenerativeModel('gemini-2.0-flash')
#             print("Gemini AI initialized with fallback model: gemini-2.0-flash")
#         except Exception as e2:
#             print(f"Error initializing Gemini AI with fallback: {str(e2)}")
#             GEMINI_AVAILABLE = False
#             gemini_model = None
else:
    client = None

# Gemini AI analysis function for plant disease detection
def analyze_with_gemini(image_path):
    """
    Analyze plant disease image using Google Gemini AI.
    Returns a dictionary with disease name, description, treatment, and confidence.
    """

    if not GEMINI_AVAILABLE or client is None:
        print("Gemini AI not available - GEMINI_AVAILABLE:", GEMINI_AVAILABLE)
        return None

    try:
        print(f"Starting Gemini AI analysis for image: {image_path}")

        # ================= YOUR ORIGINAL PROMPT (UNCHANGED) =================
        prompt = """You are an expert plant pathologist. Analyze this plant leaf image carefully and provide a detailed diagnosis.

IMPORTANT: Analyze the ACTUAL image provided, not generic information. Look at the specific visual symptoms, colors, patterns, and characteristics visible in THIS particular image.

Please provide your analysis in the following JSON format (ONLY JSON, no additional text):
{
    "disease_name": "Specific disease name based on what you see in the image, or 'Healthy' if the plant appears healthy",
    "plant_type": "Type of plant visible in the image (e.g., Apple, Tomato, Corn, Grape, etc.)",
    "description": "Detailed description of what you actually see in this image - describe the symptoms, spots, colors, patterns, and any visible disease characteristics",
    "severity": "Mild, Moderate, or Severe - based on the visible symptoms in this image",
    "treatment": "Specific treatment and prevention steps for the disease identified in this image",
    "confidence": "High, Medium, or Low - based on how clear the symptoms are in this image"
}

Analyze the image carefully:
1. What visual symptoms do you see? (spots, discoloration, wilting, lesions, etc.)
2. What colors and patterns are visible?
3. What disease matches these symptoms?
4. How severe does it appear?
5. What treatment is recommended?

Return ONLY the JSON object, no markdown formatting, no explanations before or after."""
        # =====================================================================

        # ✅ Open image using PIL (Required for new Gemini SDK)
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        print("Calling Gemini API...")

        # ✅ Call Gemini with PIL image
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                img
            ]
        )

        # Parse the response
        response_text = response.text.strip()
        print(f"Gemini response received: {response_text[:200]}...")

        # Try to extract JSON from the response
        try:
            cleaned_text = response_text

            # Remove markdown blocks if present
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_text:
                parts = cleaned_text.split("```")
                if len(parts) > 1:
                    cleaned_text = parts[1].strip()
                    if cleaned_text.startswith("json"):
                        cleaned_text = cleaned_text[4:].strip()

            # Extract JSON object
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                cleaned_text = cleaned_text[start_idx:end_idx]

            # Parse JSON
            gemini_result = json.loads(cleaned_text)

            # Build result
            result = {
                "disease_name": gemini_result.get("disease_name", "Unknown Disease"),
                "plant_type": gemini_result.get("plant_type", "Unknown Plant"),
                "description": gemini_result.get("description", "No description available"),
                "severity": gemini_result.get("severity", "Unknown"),
                "treatment": gemini_result.get("treatment", "No treatment information available"),
                "confidence": gemini_result.get("confidence", "Medium"),
                "full_analysis": response_text
            }

            print(f"Gemini AI analysis successful: {result['disease_name']} on {result['plant_type']}")

            return result

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Response text: {response_text}")

            # Fallback text analysis
            disease_name = "AI Analysis Available"

            if "healthy" in response_text.lower():
                disease_name = "Healthy Plant"
            elif "scab" in response_text.lower():
                disease_name = "Scab"
            elif "rot" in response_text.lower():
                disease_name = "Rot"
            elif "blight" in response_text.lower():
                disease_name = "Blight"

            return {
                "disease_name": disease_name,
                "plant_type": "Unknown",
                "description": response_text,
                "severity": "Unknown",
                "treatment": "Please consult the description for treatment information",
                "confidence": "Medium",
                "full_analysis": response_text
            }

    except Exception as e:
        import traceback
        print(f"Error in Gemini AI analysis: {str(e)}")
        print(traceback.format_exc())
        return None

# Prediction function with error handling
def prediction(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        output = model(input_data)
        
        # Handle both tensor and numpy outputs
        if isinstance(output, torch.Tensor):
            output = output.detach().numpy()
        
        index = np.argmax(output)
        return index
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 0  # Return default index in case of error

# Flask app
app = Flask(__name__)

# Ensure uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            image = request.files['image']
            filename = image.filename
            file_path = os.path.join(UPLOAD_DIR, filename)
            image.save(file_path)
            
            # Primary: Use Gemini AI for analysis (MANDATORY if available)
            print(f"\n{'='*60}")
            print(f"Processing uploaded image: {filename}")
            print(f"Image saved to: {file_path}")
            print(f"{'='*60}")
            
            gemini_analysis = analyze_with_gemini(file_path)
            
            # Secondary: Use CNN model as fallback/confirmation (only if needed)
            pred = 0
            try:
                pred = prediction(file_path)
                print(f"CNN prediction index: {pred}")
            except Exception as e:
                print(f"CNN prediction error (this is OK): {str(e)}")
                pred = 0
            
            # Determine which analysis to use - ALWAYS prioritize Gemini AI if available
            if gemini_analysis:
                print(f"\n{'='*60}")
                print("✓ USING GEMINI AI ANALYSIS RESULTS")
                print(f"{'='*60}")
                # Use Gemini AI analysis as primary - USE DIRECTLY, don't match to CSV
                title = gemini_analysis.get('disease_name', 'Unknown Disease')
                description = gemini_analysis.get('description', 'No description available')
                prevent = gemini_analysis.get('treatment', 'No treatment information available')
                plant_type = gemini_analysis.get('plant_type', 'Unknown Plant')
                severity = gemini_analysis.get('severity', 'Unknown')
                confidence = gemini_analysis.get('confidence', 'Medium')
                full_analysis = gemini_analysis.get('full_analysis', '')
                
                # ALWAYS use the uploaded image, not CSV image
                image_url = '/static/uploads/' + filename
                
                # Try to find supplement from CSV only if we can match disease name
                # Otherwise use generic supplement
                supplement_name = "General Plant Supplement"
                supplement_image_url = ""
                supplement_buy_link = ""
                
                # Try to match disease name to find supplement
                if len(supplement_info) > 0:
                    title_lower = title.lower()
                    for idx, disease_name in enumerate(disease_info['disease_name']):
                        disease_name_lower = str(disease_name).lower()
                        # Check if there's any match
                        if (title_lower in disease_name_lower or 
                            disease_name_lower in title_lower or
                            any(word in disease_name_lower for word in title_lower.split() if len(word) > 3)):
                            if idx < len(supplement_info):
                                supplement_name = supplement_info['supplement name'][idx]
                                supplement_image_url = supplement_info['supplement image'][idx]
                                supplement_buy_link = supplement_info['buy link'][idx]
                            break
                
                # Set pred for template compatibility (check if healthy)
                is_healthy = 'healthy' in title.lower()
                if is_healthy:
                    pred = 3  # Apple healthy (default healthy index)
                else:
                    pred = 0  # Use 0 as default for diseased plants
                
                use_gemini = True
                print(f"Gemini results - Title: {title}, Plant: {plant_type}, Severity: {severity}")
            else:
                # Fallback to CNN model prediction (Gemini AI failed or not available)
                print(f"\n{'='*60}")
                print("WARNING: GEMINI AI NOT AVAILABLE - Using CNN/CSV fallback")
                print(f"{'='*60}")
                print("This means you're seeing hardcoded results from the CSV file.")
                print("Please check:")
                print("1. Gemini AI API key is correct")
                print("2. Internet connection is available")
                print("3. Gemini API is accessible")
                print(f"{'='*60}\n")
                
                if pred >= len(disease_info):
                    pred = 0
                
                title = disease_info['disease_name'][pred]
                description = disease_info['description'][pred]
                prevent = disease_info['Possible Steps'][pred]
                image_url = disease_info['image_url'][pred]
                plant_type = "Unknown"
                severity = "Unknown"
                confidence = "Medium"
                full_analysis = ""
                
                if pred >= len(supplement_info):
                    pred = 0
                
                supplement_name = supplement_info['supplement name'][pred]
                supplement_image_url = supplement_info['supplement image'][pred]
                supplement_buy_link = supplement_info['buy link'][pred]
                
                use_gemini = False
            
            return render_template('submit.html', 
                                   title=title, 
                                   desc=description, 
                                   prevent=prevent,
                                   image_url=image_url, 
                                   pred=pred, 
                                   sname=supplement_name,
                                   simage=supplement_image_url, 
                                   buy_link=supplement_buy_link,
                                   plant_type=plant_type,
                                   severity=severity,
                                   confidence=confidence,
                                   full_analysis=full_analysis,
                                   use_gemini=use_gemini,
                                   uploaded_image=filename)
        except Exception as e:
            import traceback
            print(f"Error in submit route: {str(e)}")
            print(traceback.format_exc())
            return render_template('error.html', error=str(e))

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # In a real application, you would process the form data here
        # For example, send an email or save to database
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        subject = request.form.get('subject', '')
        message = request.form.get('message', '')
        
        # You could add email sending functionality here
        # For now, we'll just render the contact page with a success message
        return render_template('contact.html', success=True, 
                             message='Thank you for your message! We will get back to you soon.')
    
    return render_template('contact.html')

# Add a simple health check endpoint
@app.route('/health')
def health_check():
    return {'status': 'ok'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)