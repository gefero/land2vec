from torch.utils.data import Dataset
import torch
import pandas as pd
import tqdm
from pathlib import Path

from land2vec.tokenizer import Tokenizer


class SequenceDataset(Dataset):
    def __init__(self, sequences: pd.Series, window: int):
        self.window = window

        encoded_sequences: list[torch.Tensor] = []
        for seq_idx, seq in enumerate(tqdm.tqdm(sequences)):
            encoded_sequences.append(torch.tensor(Tokenizer.encode(seq)))

        self.encoded = torch.stack(encoded_sequences)

    def __len__(self):
        return len(self.encoded) * (self.encoded.shape[-1] - self.window)

    def __getitem__(self, idx):
        start = idx % (self.encoded.shape[-1] - self.window)
        row = idx // (self.encoded.shape[-1] - self.window)
        x = self.encoded[row, start : start + self.window]
        y = self.encoded[row, start + 1 : start + self.window + 1]
        return x, y


class SequenceDatasetNonWindow(Dataset):
    def __init__(self, sequences: pd.Series):
        self.encoded: list[torch.Tensor] = []
        for seq in tqdm.tqdm(sequences):
            encoded = torch.tensor(Tokenizer.encode(seq), dtype=torch.long)
            if len(encoded) < 2:
                continue
            self.encoded.append(encoded)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = self.encoded[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y


def load_data(
    *,
    file_path: Path | None = None,
    data_column: str = "seqs",
    window: int | None = None,
):
    if file_path is None:
        file_path = Path("data") / "id_seqs_text_2000_2022_chaco_santiago_frontier.zip"
    df = pd.read_csv(file_path)
    if window is not None:
        return SequenceDataset(df[data_column], window=window)
    return SequenceDatasetNonWindow(df[data_column])


def main():
    dataset = load_data(file_path=Path("data") / "seqs_short.csv")
    print(len(dataset))
    i = 0
    while True:
        try:
            print(i, dataset[i])
        except:
            break
        i += 1


if __name__ == "__main__":
    main()
