# Surname & Demographic Bias Analysis

This project analyzes how racial and ethnic demographic distributions are associated
with surnames using cosine similarity and vector-based representations. The system processes 600+ surnames across six racial and ethnic categories to examine similarity structure and potential dataset bias. 
## Methods
- Constructed a surname × demographic matrix from census-style data
- Normalized demographic vectors and computed cosine similarity
- Implemented query modes for:
  - Finding demographically similar surnames
  - Identifying common surnames within a demographic group
- Visualized similarity structure using PCA and t-SNE
- Performed sentiment analysis on selected surnames to explore downstream effects

## Source Code Structure
- `setup.py`: Data preprocessing and vector construction
- `matrix.py`: Matrix representation and similarity computation
- `names.py`: Query interface for similarity and popularity
- `visualize.py`: PCA and t-SNE visualizations
- `sentiment.py`: Sentiment analysis on selected surname sets

## Data Availability
The original surname dataset was provided internally by Swarthmore College and
is not publicly distributable. As a result, this repository contains code only
and cannot be executed end-to-end.

## Collaboration
This project was completed in collaboration with Cindy Wang as part of the
Computing with Text course at Swarthmore College.




### *This project is intended for research and educational purposes only.*
