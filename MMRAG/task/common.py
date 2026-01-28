REWRITE_QUERY_PROMPT = """
You are a professional medical assistant. Your task is to create one high quality query of the given image and content. 
Given the following image and content, generate a concise and clear medical query that effectively captures the essential information.
You may choose one of the following tasks to create the query:
1. Questions about specific entities, brands, or basic facts (e.g., "Where were the microcalcifications located in the mammography image?").
2. Questions requiring logical inference or analysis (e.g., "Based on the MRI image and the provided clinical notes, what is the most likely diagnosis for the patient?").

Content:
{content}

Refined Medical Query:
"""

REWRITE_CONTENT_PROMPT = """
You are an expert medical professional. Your task is to analyze the provided question and the associated medical images to generate a detailed, descriptive summary of the case. 
This summary will be used as the searchable content in a medical vector database. Focus on capturing:
1. Primary visual findings and abnormalities in the images.
2. The clinical scenario and specific question being asked.
3. Relevant medical terminology that would help in future retrieval of this case.

Question:
{question}

Comprehensive Medical Case Description:
"""