#!/usr/bin/env python3
"""Banana Claude -- Direct API Fallback: Image Editing

Edit images via Gemini REST API when MCP is unavailable.
Uses only Python stdlib (no pip dependencies).

Usage:
    edit.py --image path/to/image.png --prompt "remove the background"
            [--model MODEL] [--api-key KEY]
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
OUTPUT_DIR = Path.home() / "Documents" / "nanobanana_generated"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".webp": "image/webp", ".gif": "image/gif"}


def edit_image(image_paths, prompt, model, api_key):
    """Call Gemini API to edit an image, optionally with additional reference images.

    image_paths: list of paths. The first is the primary image being edited;
    any additional paths are supplementary reference images (e.g. a specific
    piece of furniture to incorporate, or a layout for scale context).
    """
    resolved_paths = [Path(p).resolve() for p in image_paths]
    for p in resolved_paths:
        if not p.exists():
            print(json.dumps({"error": True, "message": f"Image not found: {p}"}))
            sys.exit(1)

    image_parts = []
    for p in resolved_paths:
        with open(p, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        mime_type = MIME_TYPES.get(p.suffix.lower(), "image/png")
        image_parts.append({"inlineData": {"mimeType": mime_type, "data": image_b64}})

    image_path = resolved_paths[0]

    url = f"{API_BASE}/{model}:generateContent?key={api_key}"

    body = {
        "contents": [
            {
                "parts": [{"text": prompt}] + image_parts,
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    max_retries = 3
    result = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break  # Success
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(json.dumps({"retry": True, "attempt": attempt + 1, "wait_seconds": wait, "reason": "rate_limited"}), file=sys.stderr)
                time.sleep(wait)
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
                continue
            if e.code == 400 and "FAILED_PRECONDITION" in error_body:
                print(json.dumps({"error": True, "status": 400, "message": "Billing not enabled. Enable billing at https://aistudio.google.com/apikey"}))
                sys.exit(1)
            print(json.dumps({"error": True, "status": e.code, "message": error_body}))
            sys.exit(1)
        except urllib.error.URLError as e:
            print(json.dumps({"error": True, "message": str(e.reason)}))
            sys.exit(1)

    if result is None:
        print(json.dumps({"error": True, "message": "Max retries exceeded"}))
        sys.exit(1)

    # Extract image from response
    candidates = result.get("candidates", [])
    if not candidates:
        finish_reason = result.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
        print(json.dumps({"error": True, "message": f"No candidates returned. Reason: {finish_reason}"}))
        sys.exit(1)

    parts = candidates[0].get("content", {}).get("parts", [])
    image_data = None
    text_response = ""

    for part in parts:
        if "inlineData" in part:
            image_data = part["inlineData"]["data"]
        elif "text" in part:
            text_response = part["text"]

    if not image_data:
        finish_reason = candidates[0].get("finishReason", "UNKNOWN")
        print(json.dumps({"error": True, "message": f"No image in response. finishReason: {finish_reason}"}))
        sys.exit(1)

    # Save image
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"banana_edit_{timestamp}.png"
    output_path = (OUTPUT_DIR / filename).resolve()

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(image_data))

    return {
        "path": str(output_path),
        "model": model,
        "source": str(image_path),
        "reference_images": [str(p) for p in resolved_paths[1:]],
        "text": text_response,
    }


def main():
    parser = argparse.ArgumentParser(description="Edit images via Gemini REST API")
    parser.add_argument("--image", required=True, action="append",
                         help="Path to an input image. Repeat to pass multiple: "
                              "first is the primary image being edited, additional "
                              "ones are reference images (e.g. a specific furniture "
                              "piece or a layout for scale context).")
    parser.add_argument("--prompt", required=True, help="Edit instruction")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default=None, help="Google AI API key (or set GOOGLE_AI_API_KEY env)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(json.dumps({"error": True, "message": "No API key. Set GOOGLE_AI_API_KEY env or pass --api-key"}))
        sys.exit(1)

    result = edit_image(
        image_paths=args.image,
        prompt=args.prompt,
        model=args.model,
        api_key=api_key,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
