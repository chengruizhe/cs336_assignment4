from typing import Any
import pathlib
import re
import fasttext
from nltk.tokenize import word_tokenize


def identify_language(input: str) -> tuple[str, float]:
    model_path = pathlib.Path(__file__).parent.parent / "data/lid.176.bin"
    assert model_path.exists(), model_path
    model = fasttext.load_model(str(model_path))
    lan, score = model.predict(input.replace("\n", ""), k=1)
    return lan[0].replace("__label__", ""), score[0]


def identify_language_file(
    file_path: pathlib.Path,
    k: int = 100,
) -> None:
    i = 0
    with open(file_path, "r") as fp:
        for _ in range(k):
            line = fp.readline()
            if not line:
                continue
            lan, prob = identify_language(line)
            print(line)
            print(f"Language: {lan}, prob: {prob}")


def mask_emails(input: str) -> tuple[str, int]:
    mask = "|||EMAIL_ADDRESS|||"
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.subn(pattern, mask, input)


def mask_phone_numbers(input: str) -> tuple[str, int]:
    mask = "|||PHONE_NUMBER|||"
    pattern = (
        r"(?:\+?1[-.\s]*)?(?:\(\s*[2-9]\d{2}\s*\)|[2-9]\d{2})[-.\s]*\d{3}[-.\s]*\d{4}"
    )
    return re.subn(pattern, mask, input)


def mask_ip(input: str) -> tuple[str, int]:
    mask = "|||IP_ADDRESS|||"
    pattern = (
        r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
    )
    return re.subn(pattern, mask, input)


def classify_nsfw(input: str) -> tuple[Any, float]:
    model_path = (
        pathlib.Path(__file__).parent.parent
        / "data/jigsaw_fasttext_bigrams_nsfw_final.bin"
    )
    assert model_path.exists(), model_path
    model = fasttext.load_model(str(model_path))
    label, score = model.predict(input.replace("\n", ""), k=1)
    return label[0].replace("__label__", ""), score[0]


def classify_toxic_speech(input: str) -> tuple[Any, float]:
    model_path = (
        pathlib.Path(__file__).parent.parent
        / "data/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    )
    assert model_path.exists(), model_path
    model = fasttext.load_model(str(model_path))
    label, score = model.predict(input.replace("\n", ""), k=1)
    return label[0].replace("__label__", ""), score[0]


def gopher_quality_filter(input: str) -> bool:
    # Tokenize the input
    words = word_tokenize(input)
    word_count = len(words)

    # Check word count (50-100,000)
    if word_count < 50 or word_count > 100_000:
        return False

    # Check mean word length (3-10 characters)
    mean_length = sum(len(word) for word in words) / word_count
    if mean_length < 3 or mean_length > 10:
        return False

    # Check ellipsis percentage (max 30%)
    lines = input.split("\n")
    ellipsis_count = sum(1 for line in lines if line.rstrip().endswith("..."))
    ellipsis_percentage = ellipsis_count / len(lines) if lines else 0
    if ellipsis_percentage > 0.3:
        return False

    # Check alphabetic character percentage (min 80%)
    words_with_alpha = sum(1 for word in words if any(c.isalpha() for c in word))
    alpha_percentage = words_with_alpha / word_count if word_count > 0 else 0
    if alpha_percentage < 0.8:
        return False

    return True


def low_quality_filter(input: str) -> tuple[Any, float]:
    model_path = pathlib.Path(__file__).parent.parent / "data/quality_classifier.bin"
    model = fasttext.load_model(str(model_path))
    cat, score = model.predict(input.replace("\n", ""))
    return cat[0].replace("__label__", ""), score[0]
