import pdfplumber
import re

class DocumentParser:
    def __init__(self):
        self.header_patterns = {
            "experience": r"\b(experience|work|employment|history)\b",
            "projects": r"\b(projects|technical projects)\b",
            "skills": r"\b(skills|technologies|stack|competencies)\b",
            "education": r"\b(education|academic|certification)\b"
        }

    def extract_text_with_layout(self, pdf_path):
        """Extracts text while maintaining horizontal alignment for columns."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # layout=True helps preserve column structure
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Layout Extraction Error: {e}")
        return text

    def segment_by_headers(self, text):
        sections = {k: [] for k in self.header_patterns.keys()}
        sections["general"] = []
        lines = text.split('\n')
        current = "general"

        for line in lines:
            clean = line.strip().lower()
            if len(clean) < 40:
                for sec, pattern in self.header_patterns.items():
                    if re.search(pattern, clean):
                        current = sec
                        break
            sections[current].append(line)
        return {k: " ".join(v) for k, v in sections.items() if v}