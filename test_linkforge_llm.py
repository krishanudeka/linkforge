import logging

from ai_engine.extraction.llm_extractor import (
    LLMClient,
    LLMExtractor,
    parse_json_payload,
)

logging.basicConfig(level=logging.INFO)


def main() -> None:
    print("=== LLMClient JSON test ===")

    client = LLMClient()

    raw = client.chat(
        [
            {
                "role": "user",
                "content": 'Reply only with JSON: {"hello":"world"}'
            }
        ],
        json_mode=True,
        temperature=0.1,
        max_tokens=1800,
    )

    print("Raw:", repr(raw))
    print("Parsed:", parse_json_payload(raw))

    print("\n=== Triplet extraction test ===")

    text = (
        "Cocoa flavanols reduced TNF-alpha and IL-6 release in "
        "microglial cells by inhibiting NF-kB signalling. "
        "Neuroinflammation contributes to Alzheimer's disease progression."
    )

    extractor = LLMExtractor()

    triplets = extractor.extract_triplets(
        text,
        title="LinkForge LLM Diagnostic"
    )

    print("Triplets:")
    for triplet in triplets:
        print(triplet)


if __name__ == "__main__":
    main()