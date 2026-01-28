REWRITE_QUERY_PROMPT = """
You are a professional medical assistant. Your task is to create one high quality query of the given image and content. 
Given the following image and content, generate a concise and clear medical query that effectively captures the essential information.
You may choose one of the following tasks to create the query:
1. Questions about specific entities, brands, or basic facts (e.g., "Where were the microcalcifications located in the mammography image?").
2. Questions requiring logical inference or analysis (e.g., "Based on the MRI image and the provided clinical notes, what is the most likely diagnosis for the patient?").

Content:
{content}

Return your response in the following JSON format:
{{
    "question": "your refined medical query here",
    "key_words": "list of key medical terms extracted from the content"
}}

Refined Medical Query (JSON):
"""

REWRITE_CONTENT_PROMPT = """
You are an expert medical professional. Your task is to analyze the provided question and the associated medical images to generate a detailed, descriptive summary of the case. 
This summary will be used as the searchable content in a medical vector database. Focus on capturing:
1. Primary visual findings and abnormalities in the images.
2. The clinical scenario and specific question being asked.
3. Relevant medical terminology that would help in future retrieval of this case.

Question:
{question}

Return your response in the following JSON format:
{{
    "content": "comprehensive medical case description here",
    "key_words": "list of key medical terms extracted from the question and case"
}}

Comprehensive Medical Case Description (JSON):
"""

REWRITE_CONTENT_QUESTION_PROMPT ="""
You are a medical expert. Your task is to refine and rewrite the medical question based on the provided content context.
Analyze the content and generate an improved version of the original question that is:
1. More precise and clinically relevant
2. Better aligned with the medical context provided
3. Clear and professional in medical terminology

Content:
{content}


Return your response in the following JSON format:
{{
    "question": "your refined medical question here",
    "content": "refine content to form report of the case here, including details such as image findings, clinical context, and relevant medical terms",
    "key_words": "list of key medical terms extracted from the question"
}}

Refined Question (JSON):
"""
