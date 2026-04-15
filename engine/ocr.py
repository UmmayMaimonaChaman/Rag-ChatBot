import pytesseract
from PIL import Image
import pdf2image
import os
import io
import shutil
import logging

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, tesseract_path=None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Verify dependencies
        self.tesseract_available = shutil.which("tesseract") is not None
        self.poppler_available = shutil.which("pdftoppm") is not None
        
        if not self.tesseract_available:
            logger.error("Tesseract-OCR not found in system path.")
        if not self.poppler_available:
            logger.warning("Poppler (pdftoppm) not found. PDF OCR fallback will fail.")

    def extract_text_from_image(self, image_path, lang='eng+ben'):
        """Extract text from an image file."""
        if not self.tesseract_available:
            return "Error: Tesseract-OCR is not installed on the server."
            
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip()
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            return f"Error during Image OCR: {str(e)}"

    def extract_text_from_pdf(self, pdf_path, lang='eng+ben'):
        """Extract text from a PDF file using pypdf (direct) with OCR fallback."""
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            direct_text = ""
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        direct_text += text + "\n\n"
                except Exception as e:
                    logger.warning(f"Could not extract text directly from page {i}: {str(e)}")
            
            # If direct extraction found significant text, return it
            if len(direct_text.strip()) > 50:
                logger.info(f"Successfully extracted {len(direct_text)} chars directly from PDF.")
                return direct_text.strip()
            
            logger.info("Direct extraction failed or text too short. Attempting OCR fallback...")
            
            if not self.poppler_available:
                return "Error: Direct extraction failed and Poppler (pdftoppm) is missing for OCR fallback."
            if not self.tesseract_available:
                return "Error: Direct extraction failed and Tesseract-OCR is missing for OCR fallback."

            # Convert PDF pages to images for OCR
            images = pdf2image.convert_from_path(pdf_path)
            full_text = []
            for i, image in enumerate(images):
                logger.debug(f"OCR processing page {i+1}...")
                text = pytesseract.image_to_string(image, lang=lang)
                full_text.append(text.strip())
            
            final_text = "\n\n".join(full_text)
            if not final_text.strip():
                return "Error: Document appears to be empty or unreadable even with OCR."
                
            return final_text
        except Exception as e:
            logger.exception("PDF processing crashed")
            return f"Error during PDF processing: {str(e)}"

    def extract_text_from_bytes(self, content_bytes, filename, lang='eng+ben'):
        """Extract text from file bytes (PDF or Image)."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.pdf']:
            temp_path = f"temp_{filename}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(content_bytes)
                text = self.extract_text_from_pdf(temp_path, lang=lang)
                return text
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            try:
                img = Image.open(io.BytesIO(content_bytes))
                if not self.tesseract_available:
                    return "Error: Tesseract-OCR is not installed on the server."
                text = pytesseract.image_to_string(img, lang=lang)
                return text.strip()
            except Exception as e:
                return f"Error processing image: {str(e)}"
        else:
            return f"Error: Unsupported file format '{ext}'."
