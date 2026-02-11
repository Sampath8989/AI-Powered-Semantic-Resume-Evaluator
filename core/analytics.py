import numpy as np
import spacy
from config import SIGMOID_K, SIGMOID_X0

nlp = spacy.load("en_core_web_sm")

class AnalyticsEngine:
    @staticmethod
    def calibrate_score(sim):
        """Logistic Calibration (Sigmoid). Better than power scaling."""
        # Standardizes raw similarity into a 0-100 probability curve
        calibrated = 1 / (1 + np.exp(-SIGMOID_K * (sim - SIGMOID_X0)))
        return round(calibrated * 100, 2)

    @staticmethod
    def extract_skills_with_normalization(text):
        doc = nlp(text.lower())
        # Broaden the filter to include more Nouns and Proper Nouns
        skills = {token.lemma_ for token in doc if 
                  token.pos_ in ["NOUN", "PROPN"] and 
                  not token.is_stop and 
                  len(token.text) > 2}
        return skills

    @staticmethod
    def calculate_confidence(chunk_similarities):
        """Standardized Confidence: Normalized Z-Score of the best fit."""
        if len(chunk_similarities) < 2: return 0.5
        mu = np.mean(chunk_similarities)
        sigma = np.std(chunk_similarities) + 1e-6
        z = (max(chunk_similarities) - mu) / sigma
        return round(1 / (1 + np.exp(-z)), 3)