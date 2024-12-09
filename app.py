import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import logging
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SightAssist:
    def __init__(self):
        """Initialize SightAssist with necessary configurations"""
        self.setup_tesseract()
        self.setup_api()
        self.setup_tts()
        self.setup_streamlit()

    def setup_tesseract(self):
        """Configure Tesseract OCR path"""
        try:
            tesseract_path = os.getenv("TESSERACT_PATH")
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info("Tesseract OCR path configured")
            else:
                logger.warning("TESSERACT_PATH not found in environment variables")
                # Set a default path for Tesseract (if needed)
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows default path
        except Exception as e:
            logger.error(f"Failed to configure Tesseract OCR: {e}")
            st.error("Failed to initialize Tesseract OCR. Please check the installation.")

    def setup_api(self):
        """Setup API configurations and validations"""
        try:
            # Use the provided API key directly
            self.gemini_api_key = "Your Google API"
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
            self.llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=self.gemini_api_key)
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("API configured successfully")
        except Exception as e:
            logger.error(f"API setup failed: {e}")
            raise

    def setup_tts(self):
        """Initialize Text-to-Speech engine"""
        try:
            if not hasattr(st.session_state, 'tts_engine'):
                st.session_state.tts_engine = pyttsx3.init()
                st.session_state.tts_engine.setProperty('rate', 150)
                st.session_state.tts_engine.setProperty('volume', 0.9)
            self.engine = st.session_state.tts_engine
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            st.error("Text-to-Speech engine initialization failed")

    def setup_streamlit(self):
        """Configure Streamlit UI settings with background and text color"""
        st.set_page_config(page_title="🖥️ AI Powered Solution for Assisting Visually Impaired Individuals", layout="wide")
        
        # Set custom CSS for background, text color, and buttons
        st.markdown(
            """
            <style>
                body {
                    background: linear-gradient(to right, #ff9a9e, #fad0c4, #fbc2eb, #a6c1ee);
                    color: #333333;  /* Dark text */
                }
                .stButton>button {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    padding: 15px 30px;  /* Consistent padding */
                    cursor: pointer;
                    border-radius: 8px;  /* Consistent rounded corners */
                    width: 100%;  /* Make buttons fill the column width */
                    margin: 5px 0;  /* Consistent gap between buttons */
                }
                .stButton>button:hover {
                    background-color: #45a049;
                }
                .stButton {
                    margin: 10px;  /* Space between buttons */
                }
                .stButton>button:active {
                    transform: scale(0.98);  /* Button click animation */
                }
            </style>
            """, unsafe_allow_html=True
        )

        st.title("🖥️ AI Powered Solution for Assisting Visually Impaired Individuals")
        
        # Sidebar configuration
        st.sidebar.title("🔧 Features")
        st.sidebar.markdown("""   
        - 🔍 Scene Understanding
        - 📝 Text Extraction (OCR)
        - 🔊 Text-to-Speech
        - 🚨 Obstacle Detection
        """)
        
        # Add version info and credits
        st.sidebar.markdown("---")

    def extract_text_from_image(self, image: Image) -> str:
        """
        Extract text from image using OCR
        Args:
            image (Image): PIL Image object
        Returns:
            str: Extracted text
        """
        try:
            text = pytesseract.image_to_string(image)
            if not text.strip():
                logger.info("No text detected in image")
                return ""
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            st.error("Text extraction failed. Please check if Tesseract is properly installed.")
            return ""

    def text_to_speech(self, text: str) -> bool:
        """
        Convert text to speech
        Args:
            text (str): Text to convert to speech
        Returns:
            bool: Success status
        """
        try:
            # Stop any existing speech
            self.stop_speech()
            # Create new speech
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

    def stop_speech(self) -> bool:
        """
        Stop the current text-to-speech output
        Returns:
            bool: Success status
        """
        try:
            if hasattr(self.engine, '_inLoop') and self.engine._inLoop:
                self.engine.endLoop()
            return True
        except Exception as e:
            logger.error(f"Failed to stop speech: {e}")
            return False

    def generate_scene_description(self, image_data: List[Dict]) -> str:
        """
        Generate scene description using Gemini AI
        Args:
            image_data (List[Dict]): Image data in required format
        Returns:
            str: Generated description
        """
        input_prompt = """
        As an AI assistant for visually impaired individuals, please provide:
        1. A clear, concise list of main objects and people in the scene
        2. Spatial relationships between objects (left, right, center, etc.)
        3. Any potential hazards or obstacles
        4. Important text or signage visible in the image
        5. Relevant context about the environment (indoor/outdoor, lighting, etc.)
        
        Format the response in a clear, easy-to-understand way.
        """
        try:
            response = self.model.generate_content([input_prompt, image_data[0]])
            return response.text
        except Exception as e:
            logger.error(f"Scene description generation failed: {e}")
            return "Failed to generate scene description. Please try again."

    def process_image(self, uploaded_file) -> Optional[List[Dict]]:
        """
        Process uploaded image file
        Args:
            uploaded_file: Streamlit uploaded file
        Returns:
            Optional[List[Dict]]: Processed image data
        """
        if uploaded_file is not None:
            try:
                bytes_data = uploaded_file.getvalue()
                return [{
                    "mime_type": uploaded_file.type,
                    "data": bytes_data
                }]
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                return None
        return None

    def run(self):
        """Main application loop"""
        uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Create columns for buttons
            col1, col2, col3, col4, col5 = st.columns(5)

            # Process based on user interaction
            image_data = self.process_image(uploaded_file)

            if col1.button("🔍 Describe Scene"):
                if image_data:
                    with st.spinner("Analyzing the scene..."):
                        description = self.generate_scene_description(image_data)
                        st.subheader("Scene Description")
                        st.write(description)
                        # Do not read aloud during scene description

            if col2.button("📝 Extract Text"):
                with st.spinner("Extracting text..."):
                    text = self.extract_text_from_image(image)
                    st.subheader("Extracted Text")
                    if text:
                        st.write(text)
                    else:
                        st.warning("No text detected in the image.")

            if col3.button("🔊 Read Text"):
                with st.spinner("Converting to speech..."):
                    text = self.extract_text_from_image(image)
                    if text:
                        success = self.text_to_speech(text)
                        if success:
                            st.success("Text read successfully!")
                        else:
                            st.error("Failed to read text.")
                    else:
                        st.warning("No text to read.")

            # New button for Describe Scene in Voice functionality
            if col4.button("🔊 Describe Scene in Voice"):
                if image_data:
                    with st.spinner("Analyzing the scene..."):
                        description = self.generate_scene_description(image_data)
                        success = self.text_to_speech(description)
                        if success:
                            st.success("Scene description read aloud!")
                        else:
                            st.error("Failed to read the scene description aloud.")

            if col5.button("❌ Stop Speech"):
                if self.stop_speech():
                    st.success("Speech stopped.")
                else:
                    st.error("Failed to stop speech.")

if __name__ == "__main__":
    try:
        app = SightAssist()
        app.run()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        st.error(f"Failed to start the application: {e}") 
