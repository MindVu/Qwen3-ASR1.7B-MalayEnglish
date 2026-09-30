import evaluate
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to JSON file with predictions")
    parser.add_argument("--references", type=str, required=True, help="Path to JSON file with references")
    args = parser.parse_args()

    wer_metric = evaluate.load("wer")
    
    # Placeholder for loading logic
    # with open(args.predictions) as f:
    #     preds = json.load(f)
    # with open(args.references) as f:
    #     refs = json.load(f)
        
    preds = ["this is a test", "another test"]
    refs = ["this is a test", "another text"]
    
    wer = wer_metric.compute(predictions=preds, references=refs)
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")

if __name__ == "__main__":
    main()
