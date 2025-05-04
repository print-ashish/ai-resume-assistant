from groq import Groq
import os

client = Groq(api_key="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")  # Use env variable

def generate_bullets(resume_text, job_title):
    prompt = f"""
You're a resume writing assistant. Based on the following user experience and the job title "{job_title}", write 3 concise, action-driven resume bullet points that highlight the candidate's skills relevant to the role.

Resume:
{resume_text}

Format:
- Bullet 1
- Bullet 2
- Bullet 3
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",  # Official Groq model name
    )

    return chat_completion.choices[0].message.content.strip()
