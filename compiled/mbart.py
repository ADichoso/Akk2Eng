import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import os
import utils.bpe as bpe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_complete_df = pd.read_csv("processed/processed_train_complete_untokenized.csv")
processed_incomplete_df = pd.read_csv("processed/processed_train_incomplete_untokenized.csv")

processed_complete_df.sample(5)

akkadian_texts = processed_complete_df["transliteration"].tolist()
english_texts  = processed_complete_df["translation"].tolist()

print(f"Total pairs:      {len(processed_complete_df)}")
print(f"Sample Akkadian:  {akkadian_texts[0]}")
print(f"Sample English:   {english_texts[0]}")

# Check sequence lengths to set max_len appropriately
akk_lens = processed_complete_df["transliteration"].str.split().str.len()
eng_lens  = processed_complete_df["translation"].str.split().str.len()
print(f"\nAkkadian lengths — mean: {akk_lens.mean():.1f} | max: {akk_lens.max()}")
print(f"English lengths  — mean: {eng_lens.mean():.1f}  | max: {eng_lens.max()}")

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
model     = MBartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)


BPE = bpe.BytePairEncoder()
BPE.load("processed/akkonly.json")

# Extract learned BPE subwords from your trained BytePairEncoder
def get_bpe_subwords(bpe_encoder, top_n=200):
    subwords = []
    for token, freq in bpe_encoder.tokens.most_common():
        clean = token.replace('_', '').strip()
        if (len(clean) > 1
            and not (clean.startswith('<') and clean.endswith('>'))
            and clean not in subwords
        ):
            subwords.append(clean)
        if len(subwords) >= top_n:
            break
    return subwords

# Akkadian determinative tokens used in transliteration
AKK_SPECIAL_TOKENS = [
    "<god>", "<star>", "<place>", "<person>", "<building>",
    "<city>", "<land>", "<female>", "<male>", "<wood>",
    "<textile>", "<tablet>", "<river>", "<bird>", "<stone>",
    "<hide>", "<plant>",
    "<sos>", "<eos>", "<pad>", "<unk>",  # base special tokens
]

# Subwords obtained from BPE
AKK_SUBWORDS = get_bpe_subwords(BPE, top_n=200)
print(f"Extracted {len(AKK_SUBWORDS)} subwords from BPE model")
print(f"Top 50: {AKK_SUBWORDS[:50]}")

# Add custom language token for Akkadian
AKKADIAN_LANG_TOKEN = "akk_XX"

# Register all new tokens
existing_special_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])
num_added = tokenizer.add_special_tokens({
    "additional_special_tokens": (
        existing_special_tokens
        + AKK_SPECIAL_TOKENS
        + [AKKADIAN_LANG_TOKEN]
    )
})
tokenizer.add_tokens(AKK_SUBWORDS)

# Resize model embeddings to account for new tokens
model.resize_token_embeddings(len(tokenizer))

# Set Akkadian as the source language
tokenizer.src_lang = AKKADIAN_LANG_TOKEN
tokenizer.tgt_lang = "en_XX"

# Register the new lang token ID so model.generate() can use forced_bos correctly
tokenizer.lang_code_to_id[AKKADIAN_LANG_TOKEN] = tokenizer.convert_tokens_to_ids(AKKADIAN_LANG_TOKEN)

print(f"Added {num_added} special tokens + {len(AKK_SUBWORDS)} subwords")
print(f"New vocab size: {len(tokenizer)}")
print(f"Akkadian lang token ID: {tokenizer.lang_code_to_id[AKKADIAN_LANG_TOKEN]}")

class AkkadianDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_len=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):

        model_inputs = self.tokenizer(
            self.src_texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )

        labels = self.tokenizer(
            text_target=self.tgt_texts[idx],    
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )

        label_ids = labels["input_ids"]

        # Ignore padding tokens in loss
        label_ids = [
            (l if l != self.tokenizer.pad_token_id else -100)
            for l in label_ids
        ]

        return {
            "input_ids": torch.tensor(model_inputs["input_ids"]),
            "attention_mask": torch.tensor(model_inputs["attention_mask"]),
            "labels": torch.tensor(label_ids)
        }

MAX_LEN = 748

dataset   = AkkadianDataset(akkadian_texts, english_texts, tokenizer, max_len=MAX_LEN)
total     = len(dataset)
train_len = int(total * 0.8)
val_len   = total - train_len

train_set, val_set = random_split(
    dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=8,  shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=8,  shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain: {train_len} | Val: {val_len}")

def train_mbart(model, train_loader, val_loader, epochs=30, save_path="checkpoints/mbart"):
    os.makedirs(save_path, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * 3,
        num_training_steps=len(train_loader) * epochs
    )

    best_val_loss = float("inf")

    #Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        torch.cuda.empty_cache()

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += output.loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += output.loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(f"{save_path}/E{epoch}")
            tokenizer.save_pretrained(f"{save_path}/E{epoch}")
            print(f"Saved checkpoint (val loss: {avg_val:.4f})")

def translate(text, max_len=128):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    ).to(device)

    with torch.no_grad():
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
            max_length=max_len,
            num_beams=4,
            length_penalty=0.6,   # penalise long outputs
            early_stopping=True
        )

    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def load_checkpoint(save_path="checkpoints/mbart"):
    model     = MBartForConditionalGeneration.from_pretrained(save_path).to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained(save_path)
    tokenizer.src_lang = AKKADIAN_LANG_TOKEN
    tokenizer.tgt_lang = "en_XX"
    return model, tokenizer

train_mbart(model, train_loader, val_loader, epochs=10)

# Test a translation
sample = akkadian_texts[0]
print(f"\nSource:    {sample}")
print(f"Expected:  {english_texts[0]}")
print(f"Predicted: {translate(sample)}")