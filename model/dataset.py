import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import model.decoder as decoder

class TestDataset(Dataset):
    def __init__(self, dataframe, src_row="transliteration", tgt_row="translation", pad_id=None):
        self.df      = dataframe.reset_index(drop=True)
        self.src_row = src_row
        self.tgt_row = tgt_row
        self.pad_id  = pad_id if pad_id is not None else decoder.vocab["<pad>"]

        self.src_ids, self.tgt_ids = self.process_dataset()

    def process_dataset(self):
        # store token ID lists separately instead of writing back into the DataFrame
        src_ids = [decoder.tokens_to_ids(row) for row in self.df[self.src_row]]
        tgt_ids = [decoder.tokens_to_ids(row) for row in self.df[self.tgt_row]]
        return src_ids, tgt_ids

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        src = torch.tensor(self.src_ids[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_ids[idx], dtype=torch.long)
        return src, tgt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=decoder.vocab["<pad>"]
    ).to(device)

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=decoder.vocab["<pad>"]
    ).to(device)

    return src_batch, tgt_batch

def get_test_loader(df):
    dataset = TestDataset(df)
    loader  = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"Test samples: {len(dataset)}")
    return loader