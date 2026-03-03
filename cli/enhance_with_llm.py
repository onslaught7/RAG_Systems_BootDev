import os
import time
import json
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder


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
    
    def enhanced_score(query: str, docs: list[dict], choice: str, limit: int, k: int) -> dict:
        match choice:
            case "individual":
                print(f"Reranking top {limit} results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
                print()

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
            case "batch":
                print(f"Reranking top {limit} results using batch method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
                print()

                prompt = f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {docs}

                Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                [75, 12, 34, 2, 1]
                """
                score_list = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                raw_text = score_list.text.strip()

                # Remove markdown code blocks if present
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()

                # Extract only the JSON list part
                start = raw_text.find("[")
                end = raw_text.rfind("]") + 1

                if start == -1 or end == -1:
                    raise ValueError(f"Invalid LLM JSON output:\n{raw_text}")

                json_str = raw_text[start:end]

                ranked_ids = json.loads(json_str)
                doc_lookup = {doc["id"]: doc for doc in docs}
                results = []

                for rank, doc_id in enumerate(ranked_ids, start=1):
                    if doc_id in doc_lookup:
                        doc = doc_lookup[doc_id]
                        doc["rerank_rank"] = rank
                        results.append(doc)

                return results
            case "cross_encoder":
                print(f"Reranking top {limit} results using cross_encoder method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
                print()

                pairs = []
                for doc in docs:
                    pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
                
                # Loading the model everytime someone hits the cross_encoder is quite expensive
                # and not the best practice
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)

                results = []

                for i, score in enumerate(scores):
                    doc = docs[i]
                    doc["cross_encoder_score"] = score
                    results.append(doc)

                results = sorted(
                    results,
                    key=lambda x: x["cross_encoder_score"],
                    reverse=True
                )

                return results