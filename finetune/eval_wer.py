import os
import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from prepare_data import normalize_text, ASR_TEXT_TAG


def strip_language_tag(text):
    """
    Removes the "language {Lang}<asr_text>" prefix the fine-tuned model is
    trained to emit, leaving just the transcript. Safe to call on text that
    never had the tag (e.g. scoring the base/non-fine-tuned model, or an
    already-clean reference) -- returned unchanged in that case rather than
    mangled.
    """
    if text is None:
        return ""
    text = str(text)
    if ASR_TEXT_TAG in text:
        return text.split(ASR_TEXT_TAG, 1)[1].strip()
    return text.strip()


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
        # "transcript" (the clean, untagged field prepare_data.py writes)
        # and "reference" are checked before "text", since "text" is now
        # the tagged training target ("language X<asr_text>...") and would
        # corrupt scoring if picked up here. "prediction"/"pred"/"hypothesis"
        # still come first since that's unambiguously the model's output.
        for key in ["prediction", "pred", "hypothesis", "transcript", "reference", "normalized_text", "text"]:
            if key in item and item[key] is not None:
                return str(item[key])
    return str(item)


def load_infer_output(path):
    """
    Reads the combined output of infer.py (one JSONL file where each record
    already carries both the clean ground-truth "transcript" and the
    model's raw "prediction" for the same sample). No separate alignment
    pass needed -- they're inherently paired per line.

    Strips the language tag from predictions (and defensively from
    references too, in case someone points this at a record where
    "transcript" is missing and it falls back to the tagged "text").
    """
    items = load_items(path)
    preds, refs, ids = [], [], []
    skipped_no_pred = 0
    skipped_no_ref = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        if "prediction" not in item or item["prediction"] is None:
            skipped_no_pred += 1
            continue

        ref_raw = item.get("transcript", item.get("text"))
        if ref_raw is None:
            skipped_no_ref += 1
            continue

        preds.append(strip_language_tag(item["prediction"]))
        refs.append(strip_language_tag(ref_raw))
        ids.append(item.get("id"))

    if skipped_no_pred:
        print(f"WARNING: skipped {skipped_no_pred} item(s) with no 'prediction' field.")
    if skipped_no_ref:
        print(f"WARNING: skipped {skipped_no_ref} item(s) with no 'transcript'/'text' field to score against.")

    print(f"Loaded {len(preds)} paired (prediction, transcript) samples from {path}.")
    return preds, refs


def main():
    parser = argparse.ArgumentParser(description="Evaluate Word Error Rate (WER) with normalized text")
    parser.add_argument("--infer_output", type=str, default=None,
                         help="Path to the combined JSONL written by infer.py (each record has both "
                              "'transcript' (clean reference) and 'prediction' (model output) already "
                              "paired). Preferred mode -- no separate --predictions/--references needed, "
                              "and the language tag is stripped from 'prediction' automatically.")
    parser.add_argument("--predictions", type=str, default=None,
                         help="(Legacy two-file mode) Path to JSON or JSONL file with predictions.")
    parser.add_argument("--references", type=str, default=None,
                         help="(Legacy two-file mode) Path to JSON or JSONL file with references.")
    parser.add_argument("--no_normalize", action="store_true", help="Disable text normalization")
    parser.add_argument(
        "--min_aligned_fraction", type=float, default=0.9,
        help="(Legacy two-file mode only) Minimum fraction of the smaller file's items that must be "
             "aligned by ID before proceeding, instead of silently reporting WER over a near-empty "
             "or mismatched sample set.",
    )
    args = parser.parse_args()

    if args.infer_output:
        if args.predictions or args.references:
            raise ValueError("Pass either --infer_output, or --predictions/--references, not both.")
        preds, refs = load_infer_output(args.infer_output)

    elif args.predictions and args.references:
        pred_items = load_items(args.predictions)
        ref_items = load_items(args.references)

        n_pred_with_id = sum(1 for x in pred_items if isinstance(x, dict) and "id" in x)
        n_ref_with_id = sum(1 for x in ref_items if isinstance(x, dict) and "id" in x)
        has_pred_ids = n_pred_with_id > 0
        has_ref_ids = n_ref_with_id > 0

        preds = []
        refs = []

        if has_pred_ids and has_ref_ids:
            if n_pred_with_id < len(pred_items):
                print(f"WARNING: {len(pred_items) - n_pred_with_id} prediction items have no 'id' "
                      f"and will be silently dropped from ID-based alignment.")
            if n_ref_with_id < len(ref_items):
                print(f"WARNING: {len(ref_items) - n_ref_with_id} reference items have no 'id' "
                      f"and will be silently dropped from ID-based alignment.")

            # Normalize id type to string on both sides so int-vs-str id mismatches
            # don't silently zero out the intersection.
            pred_dict = {str(x["id"]): strip_language_tag(extract_text(x)) for x in pred_items if isinstance(x, dict) and "id" in x}
            ref_dict = {str(x["id"]): strip_language_tag(extract_text(x)) for x in ref_items if isinstance(x, dict) and "id" in x}

            common_ids = sorted(set(pred_dict.keys()) & set(ref_dict.keys()))
            smaller_n = min(len(pred_dict), len(ref_dict))
            aligned_fraction = (len(common_ids) / smaller_n) if smaller_n > 0 else 0.0

            print(f"Aligned {len(common_ids)} samples by ID "
                  f"({len(pred_dict)} pred ids, {len(ref_dict)} ref ids, "
                  f"{aligned_fraction*100:.1f}% of the smaller set).")

            if aligned_fraction < args.min_aligned_fraction:
                sample_pred_ids = list(pred_dict.keys())[:5]
                sample_ref_ids = list(ref_dict.keys())[:5]
                raise RuntimeError(
                    f"Only {len(common_ids)}/{smaller_n} ids matched between predictions and "
                    f"references ({aligned_fraction*100:.1f}%), below --min_aligned_fraction="
                    f"{args.min_aligned_fraction}. This usually means an id-format mismatch "
                    f"between the two files, not a real WER result -- refusing to report a "
                    f"number computed over a near-empty or misaligned set.\n"
                    f"Sample prediction ids: {sample_pred_ids}\n"
                    f"Sample reference ids:  {sample_ref_ids}\n"
                    f"Pass --min_aligned_fraction 0 to override if this is expected."
                )

            for cid in common_ids:
                preds.append(pred_dict[cid])
                refs.append(ref_dict[cid])
        else:
            min_len = min(len(pred_items), len(ref_items))
            print(f"No usable ids on one or both sides -- pairing first {min_len} samples sequentially. "
                  f"Double check this ordering is actually correct before trusting the result.")
            preds = [strip_language_tag(extract_text(x)) for x in pred_items[:min_len]]
            refs = [strip_language_tag(extract_text(x)) for x in ref_items[:min_len]]
    else:
        raise ValueError("Pass either --infer_output, or both --predictions and --references.")

    if len(preds) == 0:
        raise RuntimeError("No samples to evaluate -- refusing to report WER=0.00% "
                            "for an empty comparison.")

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