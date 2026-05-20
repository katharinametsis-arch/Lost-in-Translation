import csv
import re
import sacrebleu


GOLD_FILE = "data/gold_standard.csv"
CHATGPT_FILE = "data/chatgpt_outputs.csv"
GEMINI_FILE = "data/gemini_outputs.csv"
OUTPUT_FILE = "results/sacrebleu_results.csv"


def remove_annotations(text):
    """Remove FrameNet-style annotation brackets while keeping the translated text."""
    return re.sub(r"\[([A-Za-z_]+) ([^\]]+)\]", r"\2", text or "")


def load_by_english(filename, output_column):
    data = {}

    with open(filename, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["english"]] = row[output_column]

    return data


def compute_bleu(gold_rows, model_outputs, model_name):
    refs = []
    hyps = []

    for row in gold_rows:
        english = row["english"]
        gold = row["gold standard"]
        output = model_outputs.get(english, "")

        if output and "ERROR" not in output:
            refs.append(remove_annotations(gold))
            hyps.append(remove_annotations(output))

    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    return {
        "model": model_name,
        "rows_evaluated": len(hyps),
        "sacrebleu": round(bleu.score, 2),
        "details": str(bleu),
    }


def main():
    with open(GOLD_FILE, newline="", encoding="utf-8-sig") as f:
        gold_rows = list(csv.DictReader(f))

    chatgpt_outputs = load_by_english(CHATGPT_FILE, "chatgpt_translation")
    gemini_outputs = load_by_english(GEMINI_FILE, "gemini_swedish")

    results = [
        compute_bleu(gold_rows, chatgpt_outputs, "GPT-4.1 Mini"),
        compute_bleu(gold_rows, gemini_outputs, "Gemini 2.5 Flash"),
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "rows_evaluated", "sacrebleu", "details"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(row)


if __name__ == "__main__":
    main()
