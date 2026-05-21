import torch

class Tokenizer:
    VOCAB = {
        "[UNK]":  1,
        "A":  2,
        "F":  3,
        "G":  4,
        "Wt": 5,
        "U":  6,
        "Sh": 7,
        "Sp": 8,
        "B":  9,
        "Wa": 10,
        "Nd": 11,
    }
    REVERSE_VOCAB = {v: k for k, v in VOCAB.items()}

    @staticmethod
    def encode(seq: str):
        "Convierte una secuencia en una lista de numeros"
        states = seq.split("-")
        return [Tokenizer.VOCAB.get(state, Tokenizer.VOCAB["[UNK]"]) for state in states]

    @staticmethod
    def decode(states: torch.Tensor):
        "Convierte Tensor plano (1D) a una secuencia de estados"
        return "-".join([Tokenizer.REVERSE_VOCAB[state] for state in states.tolist()])


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
