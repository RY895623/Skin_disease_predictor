from groq import Groq
import json
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_explanation(condition: str, confidence: float):
    prompt = f"""
    You are an experienced dermatologist AI assistant.

    The deep learning model predicted: **{condition}** with **{confidence:.1f}%** confidence.

    Provide response in valid JSON format with these exact keys:
    - explanation
    - symptoms (list)
    - precautions (list)
    - when_to_see_doctor
    - disclaimer

    Use simple and clear language.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {
            "explanation": "Unable to generate detailed explanation right now.",
            "symptoms": ["Please consult a dermatologist for accurate diagnosis."],
            "precautions": ["Keep the area clean and dry"],
            "when_to_see_doctor": "Consult a doctor soon",
            "disclaimer": "This is AI generated information. Not a substitute for professional medical advice."
        }
      
