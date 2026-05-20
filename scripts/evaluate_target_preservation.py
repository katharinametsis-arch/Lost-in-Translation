import csv
import re


INPUT_FILE = "data/gold_standard.csv"
CHATGPT_FILE = "data/chatgpt_outputs.csv"
GEMINI_FILE = "data/gemini_outputs.csv"
OUTPUT_FILE = "results/target_preservation_results.csv"


def extract_target(text):
    match = re.search(r"\[Target ([^\]]+)\]", text or "")
    return match.group(1).strip() if match else None


def load_by_english(filename, output_column):
    data = {}

    with open(filename, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["english"]] = row[output_column]

    return data


def evaluate_model(gold_rows, model_outputs, model_name):
    total_sentences = len(gold_rows)
    successful_outputs = 0
    valid_targets = 0
    exact_matches = 0

    for row in gold_rows:
        english = row["english"]
        gold = row["gold standard"]
        output = model_outputs.get(english, "")

        if output and "ERROR" not in output:
            successful_outputs += 1

        gold_target = extract_target(gold)
        model_target = extract_target(output)

        if gold_target and model_target:
            valid_targets += 1
            if gold_target == model_target:
                exact_matches += 1

    coverage = valid_targets / total_sentences if total_sentences else 0
    accuracy = exact_matches / valid_targets if valid_targets else 0
    overall_success = exact_matches / total_sentences if total_sentences else 0
    response_rate = successful_outputs / total_sentences if total_sentences else 0

    return {
        "model": model_name,
        "total_sentences": total_sentences,
        "successful_outputs": successful_outputs,
        "response_rate": round(response_rate, 3),
        "valid_targets": valid_targets,
        "coverage": round(coverage, 3),
        "exact_matches": exact_matches,
        "accuracy": round(accuracy, 3),
        "overall_success": round(overall_success, 3),
    }


def main():
    with open(INPUT_FILE, newline="", encoding="utf-8-sig") as f:
        gold_rows = list(csv.DictReader(f))

    chatgpt_outputs = load_by_english(CHATGPT_FILE, "chatgpt_translation")
    gemini_outputs = load_by_english(GEMINI_FILE, "gemini_swedish")

    results = [
        evaluate_model(gold_rows, chatgpt_outputs, "GPT-4.1 Mini"),
        evaluate_model(gold_rows, gemini_outputs, "Gemini 2.5 Flash"),
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model",
            "total_sentences",
            "successful_outputs",
            "response_rate",
            "valid_targets",
            "coverage",
            "exact_matches",
            "accuracy",
            "overall_success",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(row)


if __name__ == "__main__":
    main()
