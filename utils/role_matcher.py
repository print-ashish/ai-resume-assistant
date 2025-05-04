import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# def load_roles():
#     print("loading data csv")
#     csv_path = r"job_roles.csv"
#     return pd.read_csv(csv_path)


import os


def load_roles():
    print("loading data csv")
    base_dir = os.path.dirname(os.path.abspath(__file__))  # directory of role_matcher.py
    csv_path = os.path.join(base_dir, "job_roles.csv")
    return pd.read_csv(csv_path)

# def get_top_matches(resume_text, df, top_n=3):
#     all_texts = df['required_skills'].tolist() + [resume_text]
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(all_texts)
#     similarity = cosine_similarity(vectors[-1:], vectors[:-1])
#     scores = similarity.flatten()
    
#     df['score'] = scores
#     return df.sort_values(by='score', ascending=False).head(top_n)


def get_top_matches(resume_text, df, top_n=3):
    all_texts = df['required_skills'].tolist() + [resume_text]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)
    similarity = cosine_similarity(vectors[-1:], vectors[:-1])
    scores = similarity.flatten()

    df['score'] = scores
    matches = df.sort_values(by='score', ascending=False).head(top_n).copy()

    resume_skills = set(resume_text.lower().split(","))
    for idx, row in matches.iterrows():
        required = set(map(str.strip, row['required_skills'].lower().split(",")))
        matched = required.intersection(resume_skills)
        missing = required - matched
        matches.at[idx, 'matched_skills'] = ", ".join(matched)
        matches.at[idx, 'missing_skills'] = ", ".join(missing)

    return matches


# roles_df = load_roles()
# print(roles_df.head(5))