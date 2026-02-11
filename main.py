import os
import json
import numpy as np
from core.parser import DocumentParser
from core.ai_engine import AIEngine
from core.analytics import AnalyticsEngine
from config import SECTION_WEIGHTS

def run_production_pipeline(resume_path, jd_folder):
    parser = DocumentParser()
    ai = AIEngine()
    analytics = AnalyticsEngine()
    
    # 1. Process & Cache Resume Chunks
    raw_resume = parser.extract_text_with_layout(resume_path)
    sections = parser.segment_by_headers(raw_resume)
    resume_id = os.path.basename(resume_path)
    resume_skills = analytics.extract_skills_with_normalization(raw_resume)

    res_chunk_embs = {}
    for sec, content in sections.items():
        chunks = ai.get_token_chunks(content)
        res_chunk_embs[sec] = [
            ai.get_embedding(c, f"{resume_id}_{sec}_{i}") 
            for i, c in enumerate(chunks)
        ]

    final_results = []
    for jd_file in os.listdir(jd_folder):
        with open(os.path.join(jd_folder, jd_file), 'r', encoding='utf-8') as f:
            jd_text = f.read()
        
        jd_emb = ai.get_embedding(jd_text, jd_file)
        jd_skills = analytics.extract_skills_with_normalization(jd_text)

        weighted_score_sum = 0
        total_weight = 0
        all_calibrated_sims = []

        for sec, embs in res_chunk_embs.items():
            if not embs: continue
            raw_sims = [ai.compute_similarity(e, jd_emb) for e in embs]
            
            # Top-K Average inside section
            top_3_raw = sorted(raw_sims, reverse=True)[:3]
            sec_raw_avg = np.mean(top_3_raw)
            
            calibrated_sec_score = analytics.calibrate_score(sec_raw_avg)
            
            weight = SECTION_WEIGHTS.get(sec, 1.0)
            weighted_score_sum += (calibrated_sec_score * weight)
            total_weight += weight
            all_calibrated_sims.extend(raw_sims)

        # Output Construction
        final_score = weighted_score_sum / total_weight if total_weight > 0 else 0
        match, miss = resume_skills.intersection(jd_skills), jd_skills - resume_skills

        final_results.append({
            "job": jd_file,
            "match_score": round(final_score, 2),
            "confidence": analytics.calculate_confidence(all_calibrated_sims),
            "matching_skills": list(match)[:6],
            "missing_skills": list(miss)[:6]
        })

    final_results.sort(key=lambda x: x['match_score'], reverse=True)
    print(json.dumps(final_results, indent=2))

if __name__ == "__main__":
    # Ensure folders exist
    for d in ["data/resumes", "data/jds", "data/cache"]: os.makedirs(d, exist_ok=True)
    run_production_pipeline("data/resumes/resume.pdf", "data/jds/")