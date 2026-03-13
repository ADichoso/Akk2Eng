import torch
import utils.bpe as bpe
from sacrebleu.metrics import BLEU, CHRF
import numpy as np

bleu = BLEU(effective_order=True)
chrf = CHRF(word_order=2) 

#Import BPE vocab and tokens
BPE = bpe.BytePairEncoder()
BPE.load("processed/akk2eng.json")
MAX_LEN=918

vocab = {}

#Get all tokens in the BPE
for i, token in enumerate(sorted(BPE.tokens.keys())):
    vocab[token] = i

#Ensure the sos and eos tokens exist 
#<pad> token for padding in transformer because of different sizes of input, <unk> token for unknown predictions by model, - for Akkadian
specials = ['<pad>', '<unk>', '<sos>', '-', '<eos>']

for tok in specials:
    if tok not in vocab:
        vocab[tok] = len(vocab)

#Dictionary for translating IDs back to tokens.
inv_vocab = {v: k for k, v in vocab.items()}

def decode_tokens(token_ids):
    tokens = []
    #Translate through the dictionary
    for idx in token_ids:
        token = inv_vocab.get(idx, "<unk>")
        if token in ["<pad>", "<sos>", "<eos>"]: #Skip when you see these characters.
            continue
        tokens.append(token)
    return " ".join(tokens)

def truncate_at_eos(tokens, eos_id):
    if eos_id in tokens:
        return tokens[:tokens.index(eos_id)]
    return tokens

def detokenize(tokens_str):
    # join tokens and strip the BPE end-of-word marker. Needed as SacreBLEU requires plain text as input.
    return ''.join(tokens_str.replace(' ', '').replace('_', ' '))

#Function used to convert tokens to token ids
def tokens_to_ids(tokens):
    return [vocab.get(t, vocab['<unk>']) for t in tokens]

def sample_decode_batched_l(model, src, max_len, sos_id, eos_id, samples=8):
    model.eval()
    with torch.no_grad():
        # Duplicate src set [sample] number of times for batch decoding (samples, src_len).
        src_expanded = src.repeat(samples, 1)
        
        #Encoder pass for sample batch
        encoder_outputs, hidden, cell = model.encoder(src_expanded)
        
        #Initialized decoded text array (starts with <sos>)
        generated = torch.full(
            (samples, 1), sos_id, dtype=torch.long, device=src.device
        )

        #Monitor finished sequences to stop decoding early
        finished = torch.zeros(samples, dtype=torch.bool, device=src.device)

        #Predict each next token
        for _ in range(max_len):
            logits, hidden, cell = model.decoder(generated, hidden, cell, encoder_outputs)
            
            # Sample from the last token's distribution
            probs = logits[:, -1].softmax(-1)
            next_tokens = probs.multinomial(1)  # (samples, 1)
            next_tokens[finished] = eos_id

            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Mark newly finished samples
            finished |= (next_tokens.squeeze(1) == eos_id)

            if finished.all():
                break

    return generated[:, 1:]  # remove <sos>

def sample_decode_batched_t(model, src, max_len, sos_id, eos_id, samples=8):
    model.eval()
    with torch.no_grad():
        # Duplicate src for batch decoding (samples, src_len)
        src_expanded = src.repeat(samples, 1)

        # Initialized decoded text array (starts with <sos>)
        generated = torch.full(
            (samples, 1), sos_id, dtype=torch.long, device=src.device
        )

        # Monitor finished sequences to stop decoding early
        finished = torch.zeros(samples, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            # transformer takes full src and full generated sequence so far
            logits = model(src_expanded, generated)

            # sample from the last token's distribution
            probs = logits[:, -1].softmax(-1)
            next_tokens = probs.multinomial(1)  # (samples, 1)
            next_tokens[finished.unsqueeze(1)] = eos_id

            generated = torch.cat([generated, next_tokens], dim=1)

            # mark newly finished samples
            finished |= (next_tokens.squeeze(1) == eos_id)

            if finished.all():
                break

    return generated[:, 1:]  # remove <sos>

def similarity(hypothesis_tokens, reference_tokens):
    hyp = detokenize(decode_tokens(hypothesis_tokens))
    ref = detokenize(decode_tokens(reference_tokens))
    bleu_score = bleu.sentence_score(hyp, [ref]).score / 100
    chrf_score = chrf.sentence_score(hyp, [ref]).score / 100
    geo_mean = np.sqrt(bleu_score * chrf_score)

    # penalise candidates that are much longer than the reference
    len_ratio = min(len(hyp.split()), len(ref.split())) / max(len(hyp.split()), len(ref.split()), 1)
    return geo_mean * len_ratio

def mbr_decode(model, src, max_len, sos_id, eos_id, model_type="l", samples=8):
    #Get candidate translations from the model

    if model_type == "l":
        all_samples = sample_decode_batched_l(model, src, max_len, sos_id, eos_id, samples)
    elif model_type == "t":
        all_samples = sample_decode_batched_t(model, src, max_len, sos_id, eos_id, samples)
    
    candidates = all_samples.cpu().tolist()

    # Truncate each candidate at <eos> before scoring
    candidates = [truncate_at_eos(c, eos_id) for c in candidates]
    scores = []
    for i, c1 in enumerate(candidates):
        score = sum(similarity(c1, c2) for j, c2 in enumerate(candidates) if i != j)
        scores.append(score)

    return candidates[scores.index(max(scores))]

def beam_search(model, src, max_len, sos_id, eos_id, is_transformer, beam_width=5):
    model.eval()

    with torch.no_grad():
        if is_transformer:
            # pre-compute encoder output once
            src_positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
            memory = model.transformer.encoder(
                model.token_embedding(src) + model.pos_embedding(src_positions),
                src_key_padding_mask=model.make_src_mask(src)
            )
        else:
            encoder_outputs, hidden, cell = model.encoder(src)

        # each beam: (score, tokens, hidden, cell) — hidden/cell unused for transformer
        beams = [(0.0, [sos_id], None if is_transformer else hidden,
                                 None if is_transformer else cell)]
        completed = []

        for _ in range(max_len):
            candidates = []

            for score, tokens, hid, cel in beams:
                if tokens[-1] == eos_id:
                    completed.append((score, tokens))
                    continue

                if is_transformer:
                    tgt = torch.tensor([tokens], dtype=torch.long, device=src.device)
                    tgt_positions = torch.arange(0, tgt.size(1), device=src.device).unsqueeze(0)
                    tgt_emb = model.token_embedding(tgt) + model.pos_embedding(tgt_positions)
                    tgt_mask = model.make_tgt_mask(tgt)
                    out = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                    logits = model.fc_out(out)
                    log_probs = torch.log_softmax(logits[:, -1], dim=-1).squeeze(0)
                else:
                    inp = torch.tensor([[tokens[-1]]], dtype=torch.long, device=src.device)
                    logits, hid, cel = model.decoder(inp, hid, cel, encoder_outputs)
                    log_probs = torch.log_softmax(logits[:, -1], dim=-1).squeeze(0)

                length_penalty = ((5 + len(tokens)) / 6) ** 0.6
                topk_probs, topk_ids = log_probs.topk(beam_width)

                for prob, tok in zip(topk_probs, topk_ids):
                    candidates.append((
                        (score + prob.item()) / length_penalty,
                        tokens + [tok.item()],
                        hid, cel
                    ))

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            if len(completed) >= beam_width:
                break

        completed += [(s, t) for s, t, _, _ in beams]
        best = max(completed, key=lambda x: x[0])

    return truncate_at_eos(best[1][1:], eos_id)