# 📑 Resume Matcher App  
An interactive **Streamlit web application** that matches a **resume with a job description** to calculate the similarity score. This project leverages **BERT embeddings** for natural language processing and displays insights with colorful visualizations.  

---

## 🎯 Features  
- **PDF Uploads**: Upload job descriptions and resumes in **PDF format**.
- **BERT Model Integration**: Uses **BERT embeddings** to calculate similarity.
- **Progress Bars & Metrics**: Get **percentage scores** and actionable feedback.
- **Data Visualization**: See similarity distribution with charts.
- **Animation**: Balloons animation for high similarity scores.

---
## 🔍 How It Works  
1. **Upload two PDFs**:  
   - **Job Description**
   - **Resume**  

2. The app uses **BERT embeddings** to generate text vectors for both PDFs.  

3. It calculates the **cosine similarity** between the vectors and shows:
   - **Percentage match**
   - **Progress bar**
   - **Charts with similarity metrics**

4. The app provides actionable **feedback** based on the similarity score.
