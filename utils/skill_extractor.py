def extract_skills_from_text(text, skill_set):
    text = text.lower()
    found_skills = [skill for skill in skill_set if skill in text]
    return list(set(found_skills))
