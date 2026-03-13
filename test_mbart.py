"""
test_mbart.py — Testing script for the fine-tuned MBart Akkadian→English translation model.

Usage:
    # Evaluate on a CSV file (transliteration + translation columns):
    python test_mbart.py --checkpoint save_path/E<N> --data processed/cleaned_train_complete.csv

    # Translate a single Akkadian string:
    python test_mbart.py --checkpoint save_path/E<N> --text "um-ma i-ku-pì-a-ma ..."

    # Run full evaluation with BLEU + ChrF and save results to CSV:
    python test_mbart.py --checkpoint save_path/E<N> --data processed/cleaned_train_complete.csv --save_results results.csv
"""

import argparse
import os
import sys

import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# ── Optional metric libraries 
try:
    from sacrebleu.metrics import BLEU, CHRF
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("[WARN] sacrebleu not found. Install with: pip install sacrebleu")

# Constants (must match training notebook) 
# same as MAX_LEN used in training
AKKADIAN_LANG_TOKEN = "akk_XX"
TARGET_LANG_TOKEN   = "en_XX"
MAX_LEN             = 748   
DEFAULT_NUM_BEAMS   = 4


# Model loading
def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """Load fine-tuned MBart model and its paired tokenizer from a checkpoint."""
    print(f"[INFO] Loading model from: {checkpoint_path}")

    if not os.path.isdir(checkpoint_path):
        sys.exit(f"[ERROR] Checkpoint directory not found: {checkpoint_path}")

    tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint_path)
    tokenizer.src_lang = AKKADIAN_LANG_TOKEN
    tokenizer.tgt_lang = TARGET_LANG_TOKEN

    # Ensure the Akkadian language token is registered in lang_code_to_id
    if AKKADIAN_LANG_TOKEN not in tokenizer.lang_code_to_id:
        akk_id = tokenizer.convert_tokens_to_ids(AKKADIAN_LANG_TOKEN)
        tokenizer.lang_code_to_id[AKKADIAN_LANG_TOKEN] = akk_id
        print(f"[INFO] Registered {AKKADIAN_LANG_TOKEN} → id {akk_id}")

    model = MBartForConditionalGeneration.from_pretrained(checkpoint_path).to(device)
    model.eval()

    print(f"[INFO] Vocab size : {len(tokenizer)}")
    print(f"[INFO] Device     : {device}")
    return model, tokenizer


# Translation
def translate(
    texts: list[str],
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    device: torch.device,
    max_len: int = MAX_LEN,
    num_beams: int = DEFAULT_NUM_BEAMS,
    length_penalty: float = 0.6,
    batch_size: int = 8,
) -> list[str]:
    """
    Translate a list of Akkadian transliteration strings to English.
    Processes inputs in mini-batches for efficiency.
    """
    all_translations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[TARGET_LANG_TOKEN],
                max_length=max_len,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_translations.extend(decoded)

        print(f"  Translated {min(i + batch_size, len(texts))}/{len(texts)} samples...", end="\r")

    print()  # newline after progress
    return all_translations


# Metrics
def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    """Compute BLEU and ChrF scores using sacrebleu."""
    if not HAS_SACREBLEU:
        print("[WARN] Skipping metrics — sacrebleu not installed.")
        return {}

    # sacrebleu expects references as a list of lists
    refs = [references]

    bleu  = BLEU(effective_order=True)
    chrf  = CHRF()

    bleu_score = bleu.corpus_score(hypotheses, refs)
    chrf_score = chrf.corpus_score(hypotheses, refs)

    return {
        "BLEU"  : round(bleu_score.score, 4),
        "ChrF"  : round(chrf_score.score, 4),
    }


# Evaluation on a CSV dataset
def evaluate_dataset(
    csv_path: str,
    model,
    tokenizer,
    device,
    args,
):
    print(f"\n[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"transliteration", "translation"}
    if not required_cols.issubset(df.columns):
        sys.exit(
            f"[ERROR] CSV must contain columns: {required_cols}. "
            f"Found: {set(df.columns)}"
        )

    # Optionally limit to N samples for a quick sanity check
    if args.num_samples:
        df = df.sample(n=min(args.num_samples, len(df)), random_state=42).reset_index(drop=True)
        print(f"[INFO] Evaluating on {len(df)} randomly sampled rows")
    else:
        print(f"[INFO] Evaluating on all {len(df)} rows")

    sources    = df["transliteration"].tolist()
    references = df["translation"].tolist()

    # ── Translate ────────────────────────────────────────────────────────────
    print("\n[INFO] Generating translations...")
    predictions = translate(
        sources, model, tokenizer, device,
        max_len=args.max_len,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
    )

    # ── Print a few examples ──────────────────────────────────────────────────
    n_show = min(args.show_examples, len(sources))
    print(f"  Sample translations (first {n_show})")
    for idx in range(n_show):
        print(f"\n[{idx+1}] SOURCE   : {sources[idx][:120]}{'...' if len(sources[idx])>120 else ''}")
        print(f"    REFERENCE: {references[idx][:120]}{'...' if len(references[idx])>120 else ''}")
        print(f"    PREDICTED: {predictions[idx][:120]}{'...' if len(predictions[idx])>120 else ''}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("  Corpus Metrics")

    metrics = compute_metrics(predictions, references)
    if metrics:
        for name, value in metrics.items():
            print(f"  {name:<10}: {value}")
    else:
        print("  (Install sacrebleu to see BLEU / ChrF scores)")

    # ── Optionally save results CSV ───────────────────────────────────────────
    if args.save_results:
        df["predicted_translation"] = predictions
        df.to_csv(args.save_results, index=False)
        print(f"\n[INFO] Results saved to: {args.save_results}")

    return metrics


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the fine-tuned MBart Akkadian→English model."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the saved model checkpoint directory (e.g. save_path/E12)."
    )
    # Mutually exclusive: evaluate CSV vs. translate single text
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data",
        help="Path to a CSV with 'transliteration' and 'translation' columns."
    )
    group.add_argument(
        "--text",
        help="Single Akkadian transliteration string to translate."
    )

    parser.add_argument("--max_len",      type=int,   default=MAX_LEN,          help=f"Max token length (default: {MAX_LEN})")
    parser.add_argument("--num_beams",    type=int,   default=DEFAULT_NUM_BEAMS, help=f"Beam search width (default: {DEFAULT_NUM_BEAMS})")
    parser.add_argument("--batch_size",   type=int,   default=8,                help="Batch size for dataset evaluation (default: 8)")
    parser.add_argument("--num_samples",  type=int,   default=None,             help="Limit evaluation to N random samples (default: all)")
    parser.add_argument("--show_examples",type=int,   default=5,                help="Number of example translations to print (default: 5)")
    parser.add_argument("--save_results", type=str,   default=None,             help="If set, save predictions + references to this CSV path")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)

    # ── Single-string mode ────────────────────────────────────────────────────
    if args.text:
        print(f"\n[INFO] Translating single input...")
        result = translate(
            [args.text], model, tokenizer, device,
            max_len=args.max_len,
            num_beams=args.num_beams,
        )
        print(f"\n  SOURCE   : {args.text}")
        print(f"  PREDICTED: {result[0]}")

    # ── Dataset evaluation mode ───────────────────────────────────────────────
    else:
        evaluate_dataset(args.data, model, tokenizer, device, args)


if __name__ == "__main__":
    main()
