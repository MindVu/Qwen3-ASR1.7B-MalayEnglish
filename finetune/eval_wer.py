import os
import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from prepare_data import normalize_text


def compute_wer_builtin(predictions, references):
    """
    Standard dynamic programming Word Error Rate (WER) computation.
    """
    total_words = 0
    total_edits = 0

    for pred, ref in zip(predictions, references):
        ref_words = ref.strip().split()
        pred_words = pred.strip().split()

        r_len = len(ref_words)
        p_len = len(pred_words)
        total_words += r_len

        # DP table: dp[i][j] = min edits between ref_words[:i] and pred_words[:j]
        dp = [[0] * (p_len + 1) for _ in range(r_len + 1)]

        for i in range(r_len + 1):
            dp[i][0] = i
        for j in range(p_len + 1):
            dp[0][j] = j

        for i in range(1, r_len + 1):
            for j in range(1, p_len + 1):
                if ref_words[i - 1] == pred_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1],  # Substitution
                    )

        total_edits += dp[r_len][p_len]

    if total_words == 0:
        return 0.0
    return total_edits / total_words


def load_items(file_path):
    """
    Loads text references or predictions from a JSON or JSONL file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    items = []
    is_jsonl = False
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)
        if first_line.startswith("{") and first_line.endswith("}"):
            lines = [l.strip() for l in f if l.strip()]
            if len(lines) > 1 and all(l.startswith("{") and l.endswith("}") for l in lines[:5]):
                is_jsonl = True
                for l in lines:
                    items.append(json.loads(l))

    if not is_jsonl:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in data.items()]

    return items


def extract_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ["prediction", "pred", "hypothesis", "normalized_text", "text", "reference", "transcript"]:
            if key in item and item[key] is not None:
                return str(item[key])
    return str(item)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Word Error Rate (WER) with normalized text")
    parser.add_argument("--predictions", type=str, required=True, help="Path to JSON or JSONL file with predictions")
    parser.add_argument("--references", type=str, required=True, help="Path to JSON or JSONL file with references")
    parser.add_argument("--no_normalize", action="store_true", help="Disable text normalization")
    args = parser.parse_args()

    pred_items = load_items(args.predictions)
    ref_items = load_items(args.references)

    has_pred_ids = any(isinstance(x, dict) and "id" in x for x in pred_items)
    has_ref_ids = any(isinstance(x, dict) and "id" in x for x in ref_items)

    preds = []
    refs = []

    if has_pred_ids and has_ref_ids:
        pred_dict = {x["id"]: extract_text(x) for x in pred_items if isinstance(x, dict) and "id" in x}
        ref_dict = {x["id"]: extract_text(x) for x in ref_items if isinstance(x, dict) and "id" in x}

        common_ids = sorted(set(pred_dict.keys()) & set(ref_dict.keys()))
        print(f"Aligned {len(common_ids)} samples by ID.")
        for cid in common_ids:
            preds.append(pred_dict[cid])
            refs.append(ref_dict[cid])
    else:
        min_len = min(len(pred_items), len(ref_items))
        print(f"Pairing first {min_len} samples sequentially.")
        preds = [extract_text(x) for x in pred_items[:min_len]]
        refs = [extract_text(x) for x in ref_items[:min_len]]

    if not args.no_normalize:
        preds = [normalize_text(p) for p in preds]
        refs = [normalize_text(r) for r in refs]

    # Calculate WER using evaluate if available, else builtin
    try:
        import evaluate
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=preds, references=refs)
    except Exception:
        try:
            import jiwer
            wer = jiwer.wer(reference=refs, hypothesis=preds)
        except Exception:
            wer = compute_wer_builtin(predictions=preds, references=refs)

    print(f"Evaluated {len(preds)} samples.")
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")


if __name__ == "__main__":
    main()
