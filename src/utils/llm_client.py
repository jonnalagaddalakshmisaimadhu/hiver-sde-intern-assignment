"""
Unified LLM client interface for Groq and Gemini with structured JSON extraction,
exponential backoff, and robust error handling.
"""
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Provider-agnostic LLM caller with JSON output enforcement."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq").lower()
        self.temperature = temperature

        if self.provider == "groq":
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
            self.client = Groq(api_key=api_key)
            self.model_name = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
            genai.configure(api_key=api_key)
            self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-1.5-flash")
            self.client = genai.GenerativeModel(self.model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        json_mode: bool = True,
        max_retries: int = 3,
    ) -> str:
        """Execute LLM generation with automatic JSON parsing retries."""
        for attempt in range(max_retries):
            try:
                if self.provider == "groq":
                    messages = []
                    if system_instruction:
                        messages.append({"role": "system", "content": system_instruction})
                    messages.append({"role": "user", "content": prompt})

                    kwargs = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                    }
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}

                    response = self.client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content

                elif self.provider == "gemini":
                    full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                    response = self.client.generate_content(full_prompt)
                    return response.text

            except Exception as e:
                wait_time = (2 ** attempt) * 1.5
                print(f"[WARN] LLM call failed on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError("Exceeded maximum retries for LLM generation.")

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate and parse structured JSON response."""
        for attempt in range(max_retries):
            raw_text = self.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                json_mode=True,
                max_retries=1,
            )
            # Clean possible markdown block wrappers ```json ... ```
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as jde:
                print(f"[WARN] JSON parsing error on attempt {attempt + 1}: {jde}. Raw text: {raw_text[:200]}")
                if attempt == max_retries - 1:
                    raise
        raise RuntimeError("Failed to obtain valid JSON from LLM.")
