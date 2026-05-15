from torch.utils.data import Dataset
import torch
import pandas as pd
from tokenizer import Tokenizer
import tqdm


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


def main():
    pass
    # data = [
    #     [1, 2, 3, 4, 5],
    #     [0, 1, 2, 3, 4],
    #     [10, 11, 12, 13, 14],
    # ]
    # data = torch.Tensor(data)

    # sequences = SequenceDataset(data, 3)
    # i = 0
    # try:
    #     while True:
    #         x, y = sequences[i]
    #         print(f"x: {x}")
    #         print(f"y: {y}")
    #         print()
    #         i += 1
    # except:
    #     print(f"fin a {i}")


if __name__ == "__main__":
    main()
