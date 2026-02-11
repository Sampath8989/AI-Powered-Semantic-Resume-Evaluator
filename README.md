Semantic Resume Evaluator and Job Fit Predictor
This project is a production-grade natural language processing system designed to perform deep semantic analysis between candidate resumes and job descriptions. Unlike traditional keyword-matching systems, this tool utilizes sentence embeddings and logistic calibration to determine the contextual alignment of a candidate's background with specific role requirements.

Core Methodology
The system operates through a modular pipeline designed for high precision and computational efficiency:

Layout-Aware Parsing: Uses specialized PDF extraction to maintain horizontal text alignment, ensuring multi-column resumes are processed accurately without merging unrelated text blocks.

Token-Aware Semantic Chunking: To respect the 256-token limit of the underlying transformer model, the engine recursively splits sections into overlapping chunks to preserve local context.

Dual-Sided Vector Caching: Both job descriptions and resume chunks are stored as persistent NumPy arrays to minimize redundant computations.

Logistic Score Calibration: Raw cosine similarity values are transformed using a Sigmoid function to map abstract vector distances into a human-readable match probability (0-100%).

Domain-Specific Skill Filtering: Utilizes part-of-speech tagging and a technical ontology to differentiate between high-signal technical skills and generic structural words.

Technical Architecture
NLP Model: sentence-transformers (all-MiniLM-L6-v2)

Linguistic Analysis: spaCy (en_core_web_sm)

Document Processing: pdfplumber

Statistical Framework: NumPy and scikit-learn

Project Structure
Installation and Usage
1. Environment Setup
Create a virtual environment and install dependencies:

2. Data Preparation
Place your files in the following local directories (these are ignored by Git for privacy):

Resumes: data/resumes/ (Text-selectable PDF)

Job Descriptions: data/jds/ (Plain text .txt)

3. Execution
Run the analysis via the terminal:


Performance Validation

The system includes a validation framework to measure the correlation between AI-generated match scores and human ground-truth assessments using Pearson Correlation and Mean Squared Error. This ensures the model provides a reliable signal for recruitment decision support.

