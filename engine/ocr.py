import pytesseract
from PIL import Image
import pdf2image
import os
import io

class OCRProcessor:
    def __init__(self, tesseract_path=None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
    def extract_text_from_image(self, image_path, lang='eng+ben'):
        """Extract text from an image file."""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip()
        except Exception as e:
            return f"Error during Image OCR: {str(e)}"

    def extract_text_from_pdf(self, pdf_path, lang='eng+ben'):
        """Extract text from a PDF file using pypdf (direct) with OCR fallback."""
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            direct_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    direct_text += text + "\n\n"
            
            # If direct extraction found significant text, return it
            if len(direct_text.strip()) > 50:
                print(f"INFO: Successfully extracted {len(direct_text)} chars directly from PDF.")
                return direct_text.strip()
            
            print("INFO: Direct extraction failed or too short. Falling back to OCR...")
            # Convert PDF pages to images for OCR
            images = pdf2image.convert_from_path(pdf_path)
            full_text = []
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang=lang)
                full_text.append(text.strip())
            return "\n\n".join(full_text)
        except Exception as e:
            return f"Error during PDF processing: {str(e)}"

    def extract_text_from_bytes(self, content_bytes, filename, lang='eng+ben'):
        """Extract text from file bytes (PDF or Image)."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.pdf']:
            # Save bytes to temp file because pdf2image needs a path
            temp_path = f"temp_{filename}"
            with open(temp_path, "wb") as f:
                f.write(content_bytes)
            text = self.extract_text_from_pdf(temp_path, lang=lang)
            os.remove(temp_path)
            return text
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            img = Image.open(io.BytesIO(content_bytes))
            text = pytesseract.image_to_string(img, lang=lang)
            return text.strip()
        else:
            return "Unsupported file format for OCR."
