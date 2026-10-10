#!/usr/bin/env python3
"""Send one minimal request to a Microsoft Foundry project endpoint."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()
    api_key = os.environ.get("FOUNDRY_API_KEY")
    if not api_key:
        raise RuntimeError("FOUNDRY_API_KEY is required")

    url = args.endpoint.rstrip("/") + "/openai/v1/responses"
    payload = json.dumps(
        {
            "model": args.model,
            "input": "Reply with exactly: READY",
            "temperature": 0,
            "max_output_tokens": 16,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Foundry returned HTTP {error.code}: {detail}") from None

    output_text = ""
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    output_text += content.get("text", "")
    print(f"status=ok model={body.get('model', args.model)} output={output_text.strip()}")


if __name__ == "__main__":
    main()
