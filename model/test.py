from tqdm.auto import tqdm
import random
import model.decoder as decoder
import torch
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
import pandas as pd
bleu_metric = BLEU()
chrf_metric = CHRF(word_order=2)  # word_order=2 = chrF++

def test_model(model, test_loader, device, n_samples=5, model_type="l", eval_mode="beam", teacher_forcing=False): 
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=decoder.vocab["<pad>"],
        label_smoothing=0.1
    )

    #eval_mode: "beam" | "mbr" | "raw"
    model.eval()
    test_loss = 0
    test_tokens = 0
    test_correct = 0

    all_preds = []
    all_refs  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc="Testing", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            if tgt.size(1) < 2:
                continue

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_input  = torch.clamp(tgt_input,  0, len(decoder.vocab)-1)
            tgt_output = torch.clamp(tgt_output, 0, len(decoder.vocab)-1)

            if teacher_forcing:
                # Teacher forcing: feed the ground-truth token at every step
                output = model(src, tgt_input)
            else:
                # No teacher forcing: autoregressively feed the model's own
                # predictions as the next input token at each step
                batch_size = src.size(0)
                max_len    = tgt_output.size(1)
 
                # Start with the <sos> token for every item in the batch
                decoder_input = tgt_input[:, :1]
 
                output_steps = []
                for t in range(max_len):
                    step_out = model(src, decoder_input)   # (B, t+1, vocab)
                    last_step = step_out[:, -1:, :]        # (B, 1, vocab)
                    output_steps.append(last_step)
 
                    # Greedy next token
                    next_token = last_step.argmax(-1)      # (B, 1)
                    next_token = torch.clamp(next_token, 0, len(decoder.vocab)-1)
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
 
                output = torch.cat(output_steps, dim=1)    # (B, max_len, vocab)

            # loss
            output_flat = output.reshape(-1, output.shape[-1])
            tgt_output_flat = tgt_output.reshape(-1)
            loss = criterion(output_flat, tgt_output_flat)
            test_loss += loss.item()

            # token accuracy (ignoring pad)
            preds = output.argmax(-1)
            mask  = tgt_output != decoder.vocab["<pad>"]
            test_correct += (preds == tgt_output).masked_select(mask).sum().item()
            test_tokens  += mask.sum().item()

            # collect for BLEU/chrF++ scoring
            for i in range(src.size(0)):
                pred_ids = preds[i].cpu().tolist()
                ref_ids  = tgt_output[i].cpu().tolist()
                all_preds.append(decoder.decode_tokens(pred_ids))
                all_refs.append(decoder.decode_tokens(ref_ids))

    # aggregate metrics
    avg_loss = test_loss / len(test_loader)
    avg_acc  = test_correct / max(1, test_tokens)

    all_preds = [decoder.detokenize(p) for p in all_preds]
    all_refs  = [decoder.detokenize(r) for r in all_refs]

    # corpus-level BLEU and chrF++
    bleu_score = bleu_metric.corpus_score(all_preds, [all_refs]).score
    chrf_score = chrf_metric.corpus_score(all_preds, [all_refs]).score
    geomean    = np.sqrt((bleu_score / 100) * (chrf_score / 100)) * 100

    print(f"METRICS")
    print(f"Test Loss     : {avg_loss:.4f}")
    print(f"Test Accuracy : {avg_acc:.4f}")
    print(f"BLEU          : {bleu_score:.4f}")
    print(f"chrF++        : {chrf_score:.4f}")
    print(f"Geometric Mean: {geomean:.4f}")

    # MBR spot-check on n random samples from the test set
    pred_texts = []
    if eval_mode == "raw":
        pred_texts = all_preds
        for i in range(len(all_preds)):
            print(f"PRED : {all_preds[i]}")
            print(f"REF  : {all_refs[i]}")
    else:
        pred_texts = search_eval_decode(model, test_loader, n_samples, device, model_type, eval_mode)

    # save predictions to CSV in sample_submission format
    epoch_num  = model.checkpoint_epoch
    submission_name = f"submission.csv"

    submission_df = pd.DataFrame({
        "id":          range(len(pred_texts)),
        "translation": pred_texts
    })
    submission_df.to_csv(submission_name, index=False)
    print(f"\nSubmission saved to {submission_name}")

    return avg_loss, avg_acc, bleu_score, chrf_score, geomean


def search_eval_decode(model, test_loader, n_samples, device, model_type, eval_mode):
    pred_texts = []
    print(f"\n--- {eval_mode.upper()} SAMPLES (n={n_samples}) ---")
    model.eval()
    sample_indices = random.sample(range(len(test_loader.dataset)), min(n_samples, len(test_loader.dataset)))


    for idx in sample_indices:
        src, tgt = test_loader.dataset[idx]
        src = src.unsqueeze(0).to(device)
        
        if eval_mode == "mbr":
            pred_ids = decoder.mbr_decode(
                model,
                src,
                max_len=decoder.MAX_LEN,
                sos_id=decoder.vocab["<sos>"],
                eos_id=decoder.vocab["<eos>"],
                model_type=model_type,
                samples=8
            )
        elif eval_mode == "beam":
            pred_ids = decoder.beam_search(
                model,
                src,
                max_len=decoder.MAX_LEN,
                sos_id=decoder.vocab["<sos>"],
                eos_id=decoder.vocab["<eos>"],
                is_transformer=model_type=="t",
                beam_width=5
            )

        ref_ids = tgt.tolist()

        pred_text = decoder.detokenize(decoder.decode_tokens(pred_ids))
        ref_text  = decoder.detokenize(decoder.decode_tokens(ref_ids))
        pred_texts.append(pred_texts)
        print(f"PRED : {pred_text}")
        print(f"REF  : {ref_text}")
    
    return pred_texts