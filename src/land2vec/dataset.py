from torch.utils.data import Dataset
import torch
import pandas as pd
import tqdm

from land2vec.tokenizer import Tokenizer


class SequenceDataset(Dataset):
    def __init__(self, sequences: pd.Series, window: int):
        self.window = window

        encoded_sequences: list[torch.Tensor] = []
        for seq_idx, seq in enumerate(tqdm.tqdm(sequences)):
            encoded_sequences.append(torch.tensor(Tokenizer.encode(seq)))
        
        self.encoded = torch.stack(encoded_sequences)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        start = idx % (self.encoded.shape[-1] - self.window)
        row = idx // (self.encoded.shape[-1] - self.window)
        x = self.encoded[row, start:start + self.window]
        y = self.encoded[row, start:start + self.window + 1]
        return x, y

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
    dataset = load_data(file_path="data/seqs_short.csv")
    print(len(dataset))
    for i in range(20):
        print(i, dataset[i])


if __name__ == "__main__":
    main()