python3 - <<'PY'
import csv
import re
import sacrebleu

def remove_annotations(text):
    return re.sub(r"\[([A-Za-z_]+) ([^\]]+)\]", r"\2", text or "")

# ---------- CHATGPT ----------

refs = []
hyps = []

with open("gold_standard.csv", newline="", encoding="utf-8-sig") as f:
    gold = list(csv.DictReader(f))

chat = {}
with open("chatgpt_outputs.csv", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        chat[row["english"]] = row["chatgpt_translation"]

for row in gold:
    out = chat.get(row["english"], "")

    if out and "ERROR" not in out:
        refs.append(remove_annotations(row["gold standard"]))
        hyps.append(remove_annotations(out))

chrf = sacrebleu.corpus_chrf(hyps, [refs])

print("CHATGPT")
print("Rows evaluated:", len(hyps))
print("chrF:", round(chrf.score, 2))
print()

# ---------- GEMINI ----------

refs = []
hyps = []

gem = {}
with open("gemini_outputs.csv", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        gem[row["english"]] = row["gemini_swedish"]

for row in gold:
    out = gem.get(row["english"], "")

    if out and "ERROR" not in out:
        refs.append(remove_annotations(row["gold standard"]))
        hyps.append(remove_annotations(out))

chrf = sacrebleu.corpus_chrf(hyps, [refs])

print("GEMINI")
print("Rows evaluated:", len(hyps))
print("chrF:", round(chrf.score, 2))
PY
