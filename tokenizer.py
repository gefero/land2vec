VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "A": 6,
    "F": 7,
    "G": 8,
    "Wt": 9,
    "U": 10,
    "Sh": 11,
    "Sp": 12,
    "B": 13,
    "Wa": 14,
    "Nd": 15,
    "-": 16,
}
REVERSE_VOCAB = {v: k for k, v in VOCAB.items()}


class Tokenizer:
    @staticmethod
    def encode(seq: str):
        states = seq.split("-")
        return [VOCAB[state] for state in states]

    @staticmethod
    def decode(states: list[int]):
        return "-".join([REVERSE_VOCAB[state] for state in states])

if __name__ == "__main__":
    data = [
        "A-A-A-F-F-F",
        "A-A-B-U-U-U",
        "U-U-Sh-Sh-Sp-Wa",
    ]
    
    encoded = []
    print("Encoding")
    for i, seq in enumerate(data):
        encoded.append(Tokenizer.encode(seq))
        print(f"{encoded[i]}   len: {len(encoded[i])}")
    print("Decoding")
    for seq in encoded:
        print(Tokenizer.decode(seq))