import torch
import torch.nn as nn
import model.decoder as decoder
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, pad_id, max_len=decoder.MAX_LEN):
        super().__init__()

        self.d_model = 128
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=decoder.vocab['<pad>'])
        self.pos_embedding = nn.Embedding(max_len, self.d_model)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )

        self.fc_out = nn.Linear(self.d_model, vocab_size)

    def make_src_mask(self, src):
        return (src == self.pad_id)

    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(tgt.device)

    def forward(self, src, tgt):
        # src, tgt are token IDs
        src_positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
        tgt_positions = torch.arange(0, tgt.size(1), device=tgt.device).unsqueeze(0)

        # Token + Positional embeddings
        src_emb = self.token_embedding(src) + self.pos_embedding(src_positions)
        tgt_emb = self.token_embedding(tgt) + self.pos_embedding(tgt_positions)

        # Masks
        src_key_padding_mask = (src == self.pad_id)           # True for pad tokens
        tgt_key_padding_mask = (tgt == self.pad_id)           # True for pad tokens
        tgt_mask = self.make_tgt_mask(tgt)                   # causal mask for decoder

        #Forward through transformer
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.fc_out(out)

def load_transformer_checkpoint(checkpoint_path, device):
    pad_id = decoder.vocab["<pad>"]
    vocab_size = len(decoder.vocab)

    model = Seq2SeqTransformer(vocab_size, pad_id).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.checkpoint_epoch = checkpoint['epoch']
    model.eval()

    print(f"Loaded epoch {checkpoint['epoch']} | "
          f"Val Acc: {checkpoint['accuracy']:.4f} | "
          f"Val Loss: {checkpoint['loss']:.4f}")

    return model