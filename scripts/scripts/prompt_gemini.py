import csv
import os
import time
import requests

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

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
OUTPUT_FILE = "data/gemini_outputs.csv"


def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(url, json=payload)
    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "ERROR"


def main():
    with open(INPUT_FILE, newline="", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["frame", "english", "gemini_swedish"])

        for i, row in enumerate(reader, start=1):
            sentence = row["english"]
            prompt = PROMPT_TEMPLATE.format(sentence=sentence)

            output = call_gemini(prompt)

            writer.writerow([row["frame"], sentence, output])

            print(f"{i}: done")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
