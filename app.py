import streamlit as st
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
import docx2txt
import PyPDF2
import io
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources (uncomment first time)
nltk.download('punkt_tab')
nltk.download('stopwords')

# ========== Constants ==========
SKILL_MATCH_THRESHOLD = 80  # Fuzzy matching threshold
WEIGHT_SKILL = 0.7          # Weight for direct skill matching
WEIGHT_TFIDF = 0.3          # Weight for TF-IDF similarity
TOP_N_MATCHES = 5           # Number of top job matches to display

# ========== Skill Synonyms Dictionary ==========
# Expanded dictionary with more synonyms
skill_synonyms = {
    "node.js": ["nodejs", "node js", "node.js", "node"],
    "express.js": ["express", "expressjs", "express.js", "express framework"],
    "postgresql": ["postgres", "psql", "pg", "postgresql database"],
    "git": ["github", "git version control", "gitlab", "bitbucket", "version control"],
    "docker": ["container", "containerization", "docker container", "docker compose", "kubernetes"],
    "mongodb": ["mongo", "mongo db", "nosql", "document database"],
    "fastapi": ["fast api", "fastapi framework", "python api framework"],
    "reactjs": ["react", "react.js", "react framework", "react library"],
    "html/css": ["html", "css", "frontend", "web development", "markup", "stylesheet"],
    "flask": ["python flask", "flask framework", "flask api"],
    "javascript": ["js", "ecmascript", "typescript", "frontend development"],
    "python": ["py", "python programming", "python development", "python3"],
    "aws": ["amazon web services", "amazon cloud", "ec2", "s3", "lambda"],
    "azure": ["microsoft azure", "azure cloud", "ms cloud"],
    "machine learning": ["ml", "ai", "artificial intelligence", "deep learning"],
    "ci/cd": ["continuous integration", "continuous deployment", "devops", "jenkins", "github actions"],
    "sql": ["structured query language", "mysql", "sqlite", "database query", "relational database"],
    "restful api": ["rest api", "rest", "api development", "web services"],
    "java": ["java programming", "java development", "spring", "j2ee"],
    "agile": ["scrum", "kanban", "agile methodology", "sprint planning"],
}

# ========== Utility Functions ==========

def normalize_skill(skill):
    """Normalize a skill by removing special characters and spaces"""
    return re.sub(r'[^a-zA-Z0-9]', '', skill.lower())

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and normalizing"""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [normalize_skill(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file based on file type"""
    if uploaded_file.name.endswith('.docx'):
        return docx2txt.process(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    else:
        return ""
        
def extract_text_from_pdf(uploaded_file):
    """Extract text content from PDF files"""
    try:
        # Create a file-like object from the uploaded file
        pdf_file = io.BytesIO(uploaded_file.getvalue())
        
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from each page
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
            
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def load_roles():
    """Load job roles from CSV file"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "job_roles.csv")
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error loading job roles: {e}")
        # Provide fallback sample data if file can't be loaded
        return pd.DataFrame({
            'title': ['Software Engineer', 'Data Scientist', 'Product Manager'],
            'required_skills': [
                'python, javascript, git, sql, aws',
                'python, machine learning, sql, statistics, data visualization',
                'agile, product management, user research, communication, analytics'
            ]
        })

# ========== Skill Matching Functions ==========

def get_skill_match_details(resume_text, role_skills, threshold=SKILL_MATCH_THRESHOLD):
    """Enhanced skill matching with detailed information about how skills match"""
    resume_text_lower = resume_text.lower()
    missing_skills = []
    matched_skills = []
    match_details = {}
    
    for skill in role_skills:
        skill = skill.strip()
        skill_lower = skill.lower()
        normalized_skill = normalize_skill(skill)
        
        # Check if skill directly mentioned
        direct_match = skill_lower in resume_text_lower
        
        # Check partial match using fuzzy matching
        partial_match_score = fuzz.partial_ratio(skill_lower, resume_text_lower)
        partial_match = partial_match_score >= threshold
        
        # Check synonyms
        synonym_matches = []
        if skill_lower in skill_synonyms:
            for synonym in skill_synonyms[skill_lower]:
                sym_score = fuzz.partial_ratio(synonym.lower(), resume_text_lower)
                if sym_score >= threshold:
                    synonym_matches.append((synonym, sym_score))
        
        # Determine if skill is found
        is_found = direct_match or partial_match or len(synonym_matches) > 0
        
        # Store match details
        match_details[skill] = {
            'direct_match': direct_match,
            'partial_match': partial_match,
            'partial_match_score': partial_match_score,
            'synonym_matches': synonym_matches,
            'is_found': is_found
        }
        
        # Add to appropriate list
        if is_found:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)
            
    return {
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'match_details': match_details
    }

def get_top_job_matches(resume_text, df, top_n=TOP_N_MATCHES):
    """Get top job matches with detailed skill analysis"""
    # Clean and preprocess text
    processed_resume = preprocess_text(resume_text)
    df['processed_skills'] = df['required_skills'].apply(preprocess_text)
    
    # Calculate TF-IDF similarity
    all_texts = df['processed_skills'].tolist() + [processed_resume]
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform(all_texts)
        tfidf_similarities = cosine_similarity(vectors[-1:], vectors[:-1]).flatten()
    except:
        # Fallback if TF-IDF fails
        tfidf_similarities = [0] * len(df)
    
    results = []
    for idx, row in df.iterrows():
        role = row['title']
        required_skills = [s.strip() for s in row['required_skills'].split(',')]
        
        # Get detailed skill matching information
        skill_match_info = get_skill_match_details(resume_text, required_skills)
        matched = len(skill_match_info['matched_skills'])
        total = len(required_skills)
        
        skill_match_score = matched / total if total > 0 else 0
        tfidf_score = tfidf_similarities[idx]
        
        # Calculate combined score with weightings
        combined_score = (skill_match_score * WEIGHT_SKILL) + (tfidf_score * WEIGHT_TFIDF)
        
        results.append({
            'title': role,
            'required_skills': row['required_skills'],
            'matched_skills': skill_match_info['matched_skills'],
            'missing_skills': skill_match_info['missing_skills'],
            'match_details': skill_match_info['match_details'],
            'skill_match_score': round(skill_match_score, 2),
            'tfidf_score': round(tfidf_score, 2),
            'combined_score': round(combined_score, 2),
            'matched_count': matched,
            'total_count': total
        })
    
    result_df = pd.DataFrame(results)
    return result_df.sort_values(by='combined_score', ascending=False).head(top_n)

# ========== Streamlit UI ==========

def main():
    st.set_page_config(page_title="AI Resume Matcher", layout="wide")
    st.title("🧠 AI-Powered Career & Resume Assistant")
    
    with st.expander("About this tool", expanded=False):
        st.markdown("""
        This tool analyzes your resume and matches it against various job roles to:
        1. Identify the best job matches based on your skills
        2. Show which skills you already have that match the role
        3. Highlight skills you're missing but would be valuable to learn
        
        **How it works:** The tool uses natural language processing and fuzzy matching to identify skills in your resume, 
        even when they're mentioned in different ways. It then compares these to the requirements for different roles.
        """)
    
    # File upload or text input
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("📄 Upload your resume (.docx, .txt, or .pdf)", type=["docx", "txt", "pdf"])
    with col2:
        manual_input = st.text_area("✍️ Or paste your resume content below", height=150)
    
    # Process the resume if provided
    if uploaded_file or manual_input:
        with st.spinner("🔍 Analyzing your resume..."):
            source_text = manual_input if manual_input else extract_text_from_file(uploaded_file)
            
            # Show the extracted resume text in an expandable section
            with st.expander("📜 View Extracted Resume Text", expanded=False):
                st.text_area("Resume Content", value=source_text, height=200, max_chars=None)
            
            # Load job roles and get top matches
            roles_df = load_roles()
            top_matches = get_top_job_matches(source_text, roles_df)
            
            if len(top_matches) == 0:
                st.error("No matches found. Please check your resume content and try again.")
                return
            
            # Display top matching roles summary
            st.markdown("## 🏆 Top Matching Job Roles")
            role_scores = []
            for i, (_, row) in enumerate(top_matches.iterrows(), 1):
                score = row['combined_score'] * 100  # Convert to percentage
                role_scores.append((row['title'], score))
                
            # Show scores as horizontal bars
            for title, score in role_scores:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{title}**")
                with col2:
                    st.progress(score/100)
                    st.markdown(f"<small>Match Score: {score:.1f}%</small>", unsafe_allow_html=True)
            
            # Allow user to select a role to see detailed analysis
            selected_role = st.selectbox(
                "👇 Select a role to see detailed skill analysis:",
                options=[row['title'] for _, row in top_matches.iterrows()]
            )
            
            # Get the selected role details
            selected_row = top_matches[top_matches['title'] == selected_role].iloc[0]
            
            # Detailed analysis of the selected role
            st.markdown(f"## 🔍 Detailed Analysis for {selected_role}")
            
            # Skill match summary
            st.markdown("### 📊 Skill Match Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Match", f"{selected_row['combined_score'] * 100:.1f}%")
            with col2:
                st.metric("Skills Match", f"{selected_row['matched_count']}/{selected_row['total_count']} skills")
            with col3:
                st.metric("Skills Score", f"{selected_row['skill_match_score'] * 100:.1f}%")
            
            # Display matched and missing skills
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Skills You Have")
                if selected_row['matched_skills']:
                    for skill in selected_row['matched_skills']:
                        match_info = selected_row['match_details'][skill]
                        
                        # Determine how the match was found
                        if match_info['direct_match']:
                            match_type = "Direct match"
                        elif match_info['synonym_matches']:
                            syns = [s[0] for s in match_info['synonym_matches']]
                            match_type = f"Matched via synonyms: {', '.join(syns)}"
                        else:
                            match_type = f"Partial match ({match_info['partial_match_score']}%)"
                        
                        st.markdown(f"""
                        <div style="padding: 10px; border-left: 4px solid green; margin-bottom: 10px; background-color: rgba(0,128,0,0.1);">
                            <span style="font-weight: bold; color: green;">{skill}</span><br>
                            <small style="color: #666;">{match_type}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("*None of the required skills were found in your resume.*")
            
            with col2:
                st.markdown("### ⛔ Skills You're Missing")
                if selected_row['missing_skills']:
                    for skill in selected_row['missing_skills']:
                        # Show related skills they might have
                        related_skills = []
                        if skill.lower() in skill_synonyms:
                            related_skills = skill_synonyms[skill.lower()]
                        
                        related_text = ""
                        if related_skills:
                            related_text = f"<small>Related skills you might already have: {', '.join(related_skills)}</small>"
                        
                        st.markdown(f"""
                        <div style="padding: 10px; border-left: 4px solid red; margin-bottom: 10px; background-color: rgba(255,0,0,0.1);">
                            <span style="font-weight: bold; color: red;">{skill}</span><br>
                            {related_text}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("✨ *Great job! You have all the required skills for this role.*")
            
            # Recommendations section
            st.markdown("### 🚀 Recommendations to Improve Your Resume")
            if selected_row['missing_skills']:
                st.markdown("""
                Consider adding these missing skills to your resume if you have experience with them.
                If not, these would be valuable skills to learn to increase your job match score.
                """)
                
                suggestions = []
                for skill in selected_row['missing_skills']:
                    if skill.lower() in skill_synonyms:
                        synonyms = skill_synonyms[skill.lower()]
                        related_in_resume = False
                        for syn in synonyms:
                            if fuzz.partial_ratio(syn.lower(), source_text.lower()) >= SKILL_MATCH_THRESHOLD:
                                related_in_resume = True
                                suggestions.append(f"You mention '{syn}' in your resume. Consider explicitly mentioning '{skill}' as well.")
                                break
                
                if suggestions:
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
                
                # Wording suggestions
                st.markdown("""
                **Resume Wording Tips:**
                - Be explicit about your skills - don't assume recruiters will make connections
                - Use industry-standard terminology for skills
                - Include both the spelled-out and acronym versions of technical terms
                """)
            else:
                st.markdown("Your resume already covers all the required skills for this role. Great job!")

if __name__ == "__main__":
    main()