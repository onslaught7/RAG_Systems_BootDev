import os
import time
from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

client = genai.Client(api_key=api_key)


class Gemini:
    @staticmethod
    def enhance_search(query: str, choice: str) -> str:
        match choice:
            case "spell": 
                prompt = f"""Fix any spelling errors in this movie search query.
                Only correct obvious typos. Don't change correctly spelled words.
                Query: "{query}"
                If no errors, return the original query.
                Corrected:"""
            case "rewrite":
                prompt = f"""Rewrite this movie search query to be more specific and searchable.

                Original: "{query}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                Rewritten query:"""
            case "expand":
                prompt = f"""Expand this movie search query with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                This will be appended to the original query.

                Examples:

                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                Query: "{query}"
                """
            case "":
                pass
            case _:
                return query

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text.strip()
    
    def enhanced_score(query: str, docs: list[dict]) -> dict:
        for i, doc in enumerate(docs):
            prompt = f"""Rate how well this movie matches the search query.

                        Query: "{query}"
                        Movie: {doc.get("title", "")} - {doc.get("document", "")}

                        Consider:
                        - Direct relevance to query
                        - User intent (what they're looking for)
                        - Content appropriateness

                        Scoring rules:
                        - 0 = completely unrelated
                        - 10 = perfect match
                        - Use integers or decimals (e.g., 7 or 7.5)
                        - Do NOT explain your reasoning
                        - Do NOT include any words
                        - Output ONLY the numeric score
                        - Give me ONLY the number in your response, no other text or explanation.

                        Score:"""
            score = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            docs[i]["rerank_score"] = float(score.text.strip())

            time.sleep(13)

        results = sorted(
            docs,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return results