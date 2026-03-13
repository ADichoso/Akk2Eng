import torch
import torch.nn as nn
import model.decoder as DECODER

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_id, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)

        # Hidden: (num_layers*2, batch, hidden_dim) due to bidirectional
        # Concatenate forward and backward for each layer
        # so decoder gets (num_layers, batch, hidden_dim*2)
        hidden = torch.cat(
            [torch.cat((hidden[2*i], hidden[2*i+1]), dim=1).unsqueeze(0)
             for i in range(self.num_layers)],
            dim=0
        )
        cell = torch.cat(
            [torch.cat((cell[2*i], cell[2*i+1]), dim=1).unsqueeze(0)
             for i in range(self.num_layers)],
            dim=0
        )

        # Now shape becomes:
        # (1, batch, hidden_dim * 2)

        return outputs, hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_id, num_layers=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim * 2,
            hidden_dim * 2,  # IMPORTANT
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(tgt))
        batch_size, tgt_len, _ = embedded.shape
        src_len = encoder_outputs.size(1)

        outputs = []

        for t in range(tgt_len):

            # Current decoder hidden state
            hidden_last = hidden[-1]  # (batch, hidden*2)

            # ATTENTION LAYER
            attn_scores = torch.bmm(
                encoder_outputs,
                hidden_last.unsqueeze(2)
            ).squeeze(2)  # (batch, src_len)

            attn_weights = torch.softmax(attn_scores, dim=1)

            context = torch.bmm(
                attn_weights.unsqueeze(1),
                encoder_outputs
            )  # (batch, 1, hidden*2)

            # --------------------------------

            lstm_input = torch.cat(
                (embedded[:, t:t+1, :], context),
                dim=2
            )

            output, (hidden, cell) = self.lstm(
                lstm_input,
                (hidden, cell)
            )

            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        outputs = self.dropout(outputs)
        
        logits = self.fc_out(outputs)
        return logits, hidden, cell

class LSTMSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_outputs, hidden, cell = self.encoder(src)
        logits, hidden, cell = self.decoder(tgt, hidden, cell, encoder_outputs)
        return logits

def load_lstm_checkpoint(checkpoint_path, device):
    pad_id = DECODER.vocab["<pad>"]
    vocab_size = len(DECODER.vocab)

    # rebuild the model architecture
    encoder = LSTMEncoder(vocab_size, embed_dim=128, hidden_dim=128, pad_id=pad_id, num_layers=1)
    decoder = LSTMDecoder(vocab_size, embed_dim=128, hidden_dim=128, pad_id=pad_id, num_layers=1)
    model   = LSTMSeq2Seq(encoder, decoder).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.checkpoint_epoch = checkpoint["epoch"]
    model.eval()

    print(f"Loaded epoch {checkpoint['epoch']} | "
          f"Val Acc: {checkpoint['accuracy']:.4f} | "
          f"Val Loss: {checkpoint['loss']:.4f}")

    return model