import csv
import os
import time
import requests

API_KEY = os.environ.get("OPENAI_API_KEY")

MODEL = "gpt-4.1-mini-2025-04-14"

PROMPT_TEMPLATE = """Translate the following English sentence into idiomatic Swedish.

Rules:
- Translate all English words into Swedish, both inside and outside brackets.
- Never translate the labels. Keep these labels in English exactly as written.
- Keep the bracket format exactly: [Label translated text]
- Only the text after the label should be translated.
- Do not remove any brackets.
- Do not add any new labels.
- The Target label must stay exactly as Target.
- Return only the translated annotated sentence.

Sentence:
{sentence}
"""

INPUT_FILE = "data/input_sentences_635.csv"
OUTPUT_FILE = "data/chatgpt_outputs.csv"


def call_openai(prompt):
    url = "https://api.openai.com/v1/responses"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": prompt,
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    try:
        return result["output"][0]["content"][0]["text"]
    except Exception:
        return "ERROR"


def main():
    with open(INPUT_FILE, newline="", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["frame", "english", "chatgpt_translation"])

        for i, row in enumerate(reader, start=1):
            sentence = row["english"]
            prompt = PROMPT_TEMPLATE.format(sentence=sentence)

            output = call_openai(prompt)

            writer.writerow([row["frame"], sentence, output])

            print(f"{i}: done")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
