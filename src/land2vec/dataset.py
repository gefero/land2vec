from torch.utils.data import Dataset
import torch
import pandas as pd
import tqdm

from land2vec.tokenizer import Tokenizer


class SequenceDataset(Dataset):
    def __init__(self, sequences: pd.Series, window: int):
        self.window = window
        self.samples: list[tuple[list[int], list[int]]] = []

        for seq in tqdm.tqdm(sequences):
            seq_encoded = Tokenizer.encode(seq)
            if len(seq_encoded) <= window:
                continue
            for start in range(len(seq_encoded) - window):
                x = seq_encoded[start : start + window]
                y = seq_encoded[start + 1 : start + window + 1]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))


def load_data(
    *,
    file_path: str | None = None,
    data_column: str = "seqs",
    window: int = 5,
):
    if file_path is None:
        file_path = "data/id_seqs_text_2000_2022_chaco_santiago_frontier.zip"
    df = pd.read_csv(file_path)
    return SequenceDataset(df[data_column], window=window)


def main():
    dataset = load_data()
    print(len(dataset))


if __name__ == "__main__":
    main()
