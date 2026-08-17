# domains/bfsi_fraud/tools/image_input.py
# Ported from fraud_detection/tools/image_input.py — logic unchanged.

import base64
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from domains.bfsi_fraud.settings import OPENAI_API_KEY


class ImageInputTool:

    def __init__(self):
        # gpt-4o-mini supports vision
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0
        )

    def encode_image(self, image_path: str) -> str:
        """Convert image file path to base64 string — used by main.py"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_bytes(self, image_bytes: bytes) -> str:
        """Convert raw image bytes to base64 string — used by Streamlit uploader"""
        return base64.b64encode(image_bytes).decode("utf-8")

    def _extract(self, base64_image: str, mime_type: str = "image/jpeg") -> dict:
        """
        Core extraction logic — shared by both extract_transaction() and extract_from_bytes().
        Sends the base64 image to GPT-4o-mini vision and returns a transaction dict.
        """
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Extract transaction details from this cheque or receipt image.
                    Return ONLY a JSON object with these exact keys:
                    {
                        "Amount": <number>,
                        "hour": 12,
                        "Time": 50000,
                        "source": "image"
                    }
                    If you cannot read the amount clearly, use 0.
                    Return ONLY the JSON, nothing else."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        )

        response = self.llm.invoke([message])
        raw = response.content.strip()

        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    def extract_transaction(self, image_path: str) -> dict:
        """
        Extract transaction from a local file path.
        Used by main.py image input demo.
        """
        base64_image = self.encode_image(image_path)

        # Detect mime type from extension
        mime_type = "image/jpeg"
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_path.lower().endswith(".webp"):
            mime_type = "image/webp"

        return self._extract(base64_image, mime_type)

    def extract_from_bytes(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
        """
        Extract transaction from raw bytes.
        Used by Streamlit file uploader — no temp file needed.
        """
        base64_image = self.encode_bytes(image_bytes)
        return self._extract(base64_image, mime_type)
