from torch.utils.data import Dataset
import torch


class SequenceDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, window: int):
        self.window = window
        self.samples = []
        for seq in sequences:
            if len(seq) <= window:
                continue
            for start in range(len(seq) - window):
                x = seq[start : start + window]
                y = seq[start + 1 : start + window + 1]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    data = [
        [1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4],
        [10, 11, 12, 13, 14],
    ]
    data = torch.Tensor(data)

    seqences = SequenceDataset(data, 3)
    i = 0
    try:
        while True:
            x, y = seqences[i]
            print(f"x: {x}")
            print(f"y: {y}")
            print()
            i += 1
    except:
        print(f"fin a {i}")
