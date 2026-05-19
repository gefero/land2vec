from torch.utils.data import Dataset
import torch
import pandas as pd
import tqdm

from land2vec.tokenizer import Tokenizer


class SequenceDataset(Dataset):
    def __init__(self, sequences: pd.Series, window: int):
        self.window = window
        self.encoded_sequences: list[torch.Tensor] = []
        self.index_map: list[tuple[int, int]] = []

        for seq_idx, seq in enumerate(tqdm.tqdm(sequences)):
            encoded = Tokenizer.encode(seq)
            if len(encoded) <= window:
                continue
            encoded_tensor = torch.tensor(encoded, dtype=torch.long)
            self.encoded_sequences.append(encoded_tensor)
            n_windows = len(encoded) - window
            for start in range(n_windows):
                self.index_map.append((len(self.encoded_sequences) - 1, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        seq_idx, start = self.index_map[idx]
        seq = self.encoded_sequences[seq_idx]
        x = seq[start : start + self.window]
        y = seq[start + 1 : start + self.window + 1]
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
    dataset = load_data()
    print(len(dataset))
    [print(dataset[i]) for i in range(10, 18)]


if __name__ == "__main__":
    main()
