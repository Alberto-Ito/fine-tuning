from __future__ import annotations

from common import load_environment, normalize_foundry_base_url, require_environment
from openai import OpenAI


def main() -> None:
    load_environment()
    env = require_environment(
        "FOUNDRY_BASE_URL", "FOUNDRY_MODEL_DEPLOYMENT", "FOUNDRY_API_KEY"
    )
    client = OpenAI(
        base_url=normalize_foundry_base_url(env["FOUNDRY_BASE_URL"]) + "/",
        api_key=env["FOUNDRY_API_KEY"],
    )
    response = client.chat.completions.create(
        model=env["FOUNDRY_MODEL_DEPLOYMENT"],
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: FOUNDRY_CONNECTION_OK",
            }
        ],
        temperature=0,
        max_tokens=32,
    )
    content = response.choices[0].message.content or ""
    print(content.strip())
    if "FOUNDRY_CONNECTION_OK" not in content:
        raise SystemExit(
            "Foundry responded, but the smoke-test marker was not returned."
        )


if __name__ == "__main__":
    main()
