import cv2
import numpy as np
import pytesseract
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os
from .config import TESSERACT_PATH, ENABLE_OCR

class FrameAnalyzer:
    def __init__(self):
        # Initialize OCR and NLP tools
        self.ocr_config = r'--oem 3 --psm 6'
        self._ocr_debug_saved = 0
        self._ocr_debug_max = 10
        self._ocr_debug_dir = os.path.join("output", "frames")
        
        # Configure Tesseract path if available
        if ENABLE_OCR:
            if os.path.exists(TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
                self.ocr_enabled = True
            else:
                print(f"Warning: Tesseract not found at {TESSERACT_PATH}. OCR will be disabled.")
                self.ocr_enabled = False
        else:
            self.ocr_enabled = False
            
        # Download necessary NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def analyze_frame(self, frame):
        # Main function to analyze a video frame
        processed_data = self.extract_data(frame)
        preprocessed_content = self.preprocess_content(processed_data)
        return preprocessed_content

    def extract_data(self, frame):
        # Extract various types of data from the frame
        data = {}
        
        # Extract text using OCR
        data['text'] = self.extract_text(frame)
        
        # Extract objects in the frame
        data['objects'] = self.detect_objects(frame)
        
        # Extract any other relevant information
        data['frame_info'] = self.extract_frame_info(frame)
        
        return data

    def extract_text(self, frame):
        """Extract text from the frame using OCR"""
        if not self.ocr_enabled:
            return ""
            
        try:
            # Detect moving/shifted captions by finding text-like regions.
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            regions = self._find_text_regions(gray_full)

            if not regions:
                # Fallback to full frame if no text regions found.
                regions = [(0, 0, frame.shape[1], frame.shape[0])]

            words = []
            for x, y, w, h in regions:
                roi = frame[y:y + h, x:x + w]
                thresh, raw_gray = self._preprocess_for_ocr(roi)

                # Save a few debug crops to inspect OCR input
                if self._ocr_debug_saved < self._ocr_debug_max:
                    os.makedirs(self._ocr_debug_dir, exist_ok=True)
                    raw_path = os.path.join(
                        self._ocr_debug_dir,
                        f"ocr_debug_raw_{self._ocr_debug_saved:03d}.png"
                    )
                    debug_path = os.path.join(
                        self._ocr_debug_dir,
                        f"ocr_debug_{self._ocr_debug_saved:03d}.png"
                    )
                    cv2.imwrite(raw_path, raw_gray)
                    cv2.imwrite(debug_path, thresh)
                    if self._ocr_debug_saved == 0:
                        print(f"OCR debug images saved to: {os.path.abspath(self._ocr_debug_dir)}")
                    self._ocr_debug_saved += 1

                # Convert the OpenCV image to PIL format for pytesseract
                pil_img = Image.fromarray(thresh)

                # Extract text with confidence scores
                data = pytesseract.image_to_data(
                    pil_img, config=self.ocr_config, output_type=pytesseract.Output.DICT
                )

                for text, conf in zip(data.get("text", []), data.get("conf", [])):
                    try:
                        conf_val = float(conf)
                    except Exception:
                        conf_val = -1

                    cleaned = self._clean_ocr_token(text)
                    if not cleaned:
                        continue
                    if conf_val < 60:
                        continue
                    words.append(cleaned)

            return " ".join(words).strip()
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return ""

    def detect_objects(self, frame):
        """Detect objects in the frame
        Note: For production, use pre-trained models like YOLO or SSD
        """
        # Placeholder for object detection
        objects = []
        
        # Simple object detection using contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'type': 'unknown',
                    'position': (x, y, w, h)
                })
        
        return objects

    def extract_frame_info(self, frame):
        """Extract general information about the frame"""
        height, width, channels = frame.shape
        brightness = np.mean(frame)
        
        # Detect if the frame likely contains a slide/presentation
        is_slide = brightness > 200
        
        return {
            'dimensions': (width, height),
            'brightness': brightness,
            'is_slide': is_slide,
        }

    def preprocess_content(self, data):
        """Preprocess the extracted data to make it suitable for Gemini"""
        processed_data = {}
        
        # Process extracted text
        if 'text' in data and data['text']:
            processed_data['text'] = self.preprocess_text(data['text'])
        
        # Process detected objects
        if 'objects' in data and data['objects']:
            processed_data['objects'] = data['objects']
            processed_data['object_summary'] = f"Detected {len(data['objects'])} objects in frame"
        
        # Include frame information
        if 'frame_info' in data:
            processed_data['frame_info'] = data['frame_info']
            
        return processed_data
    
    def preprocess_text(self, text):
        """Clean and preprocess the extracted text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Reconstruct text
        processed_text = ' '.join(filtered_tokens)
        
        return processed_text

    def _clean_ocr_token(self, token):
        """Clean OCR tokens and drop obvious garbage."""
        if not token:
            return ""
        token = token.strip()
        if len(token) < 2:
            return ""
        # Remove tokens with too many non-alphanumerics
        non_alnum = sum(1 for c in token if not c.isalnum())
        if non_alnum / max(len(token), 1) > 0.3:
            return ""
        # Remove tokens with repeated single character patterns
        if len(set(token.lower())) <= 2 and len(token) >= 4:
            return ""
        # Drop tokens that are mostly digits
        digit_ratio = sum(1 for c in token if c.isdigit()) / max(len(token), 1)
        if digit_ratio > 0.5:
            return ""
        return token

    def _preprocess_for_ocr(self, image):
        """Return (thresh, gray) images preprocessed for OCR."""
        # Upscale 2x to make text larger for OCR
        image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.equalizeHist(gray)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        thresh = cv2.filter2D(thresh, -1, kernel)

        return thresh, gray

    def _find_text_regions(self, gray):
        """Find text-like regions to handle moving captions."""
        # Compute gradient to emphasize text edges
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)

        # Blur and threshold
        blurred = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        # Close gaps between text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Remove small blobs
        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        h, w = gray.shape[:2]
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < 800:
                continue
            if cw < 40 or ch < 12:
                continue
            if cw / max(ch, 1) < 2.0:
                continue
            # Keep regions in the middle band to reduce false positives
            if y < int(h * 0.15) or y > int(h * 0.85):
                continue
            regions.append((x, y, cw, ch))

        # Sort by y, then x
        regions.sort(key=lambda r: (r[1], r[0]))
        return regions
