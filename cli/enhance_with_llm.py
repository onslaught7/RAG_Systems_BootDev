import os
from dotenv import load_dotenv
from google import genai


class Gemini:
    @staticmethod
    def enhance_query(query: str) -> str:
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        client = genai.Client(api_key=api_key)

        prompt = f"""Fix any spelling errors in this movie search query.
        Only correct obvious typos. Don't change correctly spelled words.
        Query: "{query}"
        If no errors, return the original query.
        Corrected:"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text.strip().replace("Corrected:", "").strip()