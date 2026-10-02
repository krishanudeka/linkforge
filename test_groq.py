import json
from groq import Groq
from backend.config import settings


def main() -> None:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    print("Model:", settings.groq_model)

    client = Groq(
        api_key=settings.groq_api_key,
        timeout=settings.llm_timeout_sec,
        max_retries=0,
    )

    response = client.chat.completions.create(
        model=settings.groq_model,
        temperature=0.1,
        max_tokens=1800,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Return valid JSON only."
            },
            {
                "role": "user",
                "content": 'Return {"ok": true, "items": [1, 2, 3]}.'
            },
        ],
    )

    choice = response.choices[0]

    print("Finish reason:", choice.finish_reason)
    print("Usage:", response.usage)
    print("Raw content:", repr(choice.message.content))

    if not choice.message.content:
        raise RuntimeError("Groq returned empty content.")

    parsed = json.loads(choice.message.content)
    print("Parsed JSON:", parsed)


if __name__ == "__main__":
    main()