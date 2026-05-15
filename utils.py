import pandas as pd
from dataset import SequenceDataset


def load_data(
    *, file_path: str | None = None, data_column: str = "seqs", window: int = 5
):
    if file_path is None:
        file_path = "data/id_seqs_text_2000_2022_chaco_santiago_frontier.csv"
    df = pd.read_csv(file_path)
    return SequenceDataset(df[data_column], window=window)


def main():
    dataset = load_data()
    print(len(dataset))


if __name__ == "__main__":
    main()
