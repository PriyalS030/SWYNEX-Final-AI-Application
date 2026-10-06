"""
services/ai_api.py
==================
AI API Integration Layer.

This module integrates an external AI API to generate a natural-language
explanation of the ML model's prediction.

It reads credentials securely from the environment and gracefully handles
any API failures so the primary ML prediction is never disrupted.
"""

import os
import json
import urllib.request
from urllib.error import URLError, HTTPError


def get_api_credentials():
    """
    Safely extract the API key and URL from environment variables.
    Handles potential misconfigurations where the URL and Key might be swapped.
    
    Returns:
        tuple: (api_key, api_url)
    """
    key_val = os.environ.get('AI_API_KEY', '').strip()
    url_val = os.environ.get('AI_API_URL', '').strip()
    
    api_key = None
    # Identify the likely Groq API key by its prefix
    if key_val.startswith('gsk_'):
        api_key = key_val
    elif url_val.startswith('gsk_'):
        api_key = url_val
    else:
        api_key = key_val  # Fallback

    # Default to Groq chat completions endpoint if not provided or valid
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    if url_val.startswith('http'):
        api_url = url_val
    elif key_val.startswith('http'):
        api_url = key_val

    return api_key, api_url


def is_api_configured():
    """
    Check whether the external AI API is configured.

    Returns:
        bool: True if a valid API key is present, False otherwise.
    """
    api_key, _ = get_api_credentials()
    # Assume configured if we have any non-empty key and it isn't a placeholder
    return bool(api_key and "groq api" not in api_key.lower())


def get_ai_explanation(features: dict, prediction: str, model_name: str) -> str | None:
    """
    Get an AI-generated explanation for the prediction.

    Sends the prediction context to the external API and returns a neutral,
    student-friendly explanation. If the API fails or is not configured, 
    returns a clear fallback message.

    Args:
        features: Dictionary of input feature names and their values.
        prediction: The model's prediction ("At Risk" or "Low Risk").
        model_name: Name of the ML model that made the prediction.

    Returns:
        str: AI-generated explanation text, or a fallback message.
    """
    if not is_api_configured():
        return "The AI explanation feature is currently not configured."

    api_key, api_url = get_api_credentials()

    # Construct the prompt with strict constraints
    prompt = (
        f"You are providing an explanation for an educational early-support ML prototype.\n\n"
        f"Context:\n"
        f"- ML Prediction: {prediction}\n"
        f"- Model Used: {model_name}\n"
        f"- Student Inputs: {json.dumps(features)}\n\n"
        "Instructions:\n"
        "Provide a short, neutral, student-friendly explanation of this prediction. "
        "Focus gently on the factors (like study habits, attendance, or support) that usually influence such predictions. "
        "CRITICAL RULES: "
        "1. Do NOT claim certainty about the student's future or guarantee outcomes. "
        "2. Do NOT diagnose the student. "
        "3. Do NOT give medical or financial advice. "
        "4. Do NOT invent information that was not provided. "
        "5. Do NOT replace or dismiss the ML prediction. "
        "6. Keep the response to 2 or 3 short sentences."
    )

    # Use the OpenAI-compatible payload format for Groq
    data = {
        "model": "qwen/qwen3.8-27b",  # Supported model based on API listing
        "messages": [
            {"role": "system", "content": "You are a helpful, neutral educational assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.3
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(data).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            explanation = result['choices'][0]['message']['content'].strip()
            # Clean up the response just in case the model wraps it in quotes
            if explanation.startswith('"') and explanation.endswith('"'):
                explanation = explanation[1:-1]
            return explanation
    except (URLError, HTTPError, Exception) as e:
        print(f"[Warning] AI API call failed: {e}")
        return "The AI explanation is temporarily unavailable. Please rely on the primary ML prediction above."
