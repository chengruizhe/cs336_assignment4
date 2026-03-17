import sys
import re
import shutil
import itertools
import functools
import unicodedata
import pathlib
import xxhash

_WHITESPACE_RE = re.compile(r"\s+")

_PUNCT_TRANSLATION_TABLE = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


def hash_text(text: str, seed: int = 42) -> int:
    return xxhash.xxh64(text, seed=seed).intdigest()


def get_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = text.split(" ")
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def get_min_hash(ngrams: set[tuple[str, ...]], seed: int) -> int:
    return min(hash_text(f",".join(gram), seed=seed) for gram in ngrams)


def exact_line_deduplication(
    input_paths: list[pathlib.Path],
    output_folder: pathlib.Path,
) -> None:
    seen_hash: dict[int, int] = {}

    for input_path in input_paths:
        with open(input_path, "rb") as fi:
            for line in fi:
                h = hash_text(line)
                if h not in seen_hash:
                    seen_hash[h] = 1
                else:
                    seen_hash[h] += 1

    output_folder.mkdir(exist_ok=True)
    for input_path in input_paths:
        output_path = output_folder / input_path.name
        with (
            open(input_path, "rb") as fi,
            open(output_path, "wb") as fo,
        ):
            for line in fi:
                h = hash_text(line)
                if seen_hash[h] == 1:
                    fo.write(line)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = text.translate(_PUNCT_TRANSLATION_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def minhash_deduplication(
    input_paths: list[pathlib.Path],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_dir: pathlib.Path,
) -> None:
    assert num_bands <= num_hashes and num_hashes % num_bands == 0
    r = num_hashes // num_bands
    output_dir.mkdir(exist_ok=True)

    # Use union-find for clustering, keyed with file_id
    parents: dict[int, int] = {}

    def find(x: int) -> int:
        if x not in parents:
            parents[x] = x
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(x: int, y: int) -> None:
        parents[find(x)] = parents[find(y)]

    @functools.cache
    def normalized_ngrams(file_idx: int) -> set[tuple[str, ...]]:
        with open(input_paths[file_idx], "r") as f:
            return get_ngrams(normalize_text(f.read()), ngrams)

    @functools.cache
    def jaccard(f1: int, f2: int) -> float:
        grams1 = normalized_ngrams(f1)
        grams2 = normalized_ngrams(f2)
        return len(grams1.intersection(grams2)) / len(grams1.union(grams2))

    # Mapping of sub-signature to file_idx for each band
    signatures: list[dict[tuple[int, ...], list[int]]] = [{} for _ in range(num_bands)]
    # For each file produce a n-gram signature of shape [num_hash].
    # For each band, we cluster into candidate duplicates per band
    # We do this by grouping pair-wise ngram jaccard >= threshold
    # for matching sub-signatures.
    salt = 615
    for file_idx, input_path in enumerate(input_paths):
        with open(input_path, "r") as f:
            text = normalize_text(f.read())
            grams = get_ngrams(text, n=ngrams)
            min_hashes = [get_min_hash(grams, s + salt) for s in range(num_hashes)]
            for band in range(num_bands):
                start_idx = band * r
                sub_sig = tuple(min_hashes[start_idx : start_idx + r])
                signatures[band].setdefault(sub_sig, []).append(file_idx)

    # Merge duplicates across bands for global candidate duplicates.
    for files_map in signatures:
        for f_ids in files_map.values():
            if len(f_ids) <= 1:
                continue
            for f1, f2 in itertools.combinations(f_ids, 2):
                if f1 != f2 and jaccard(f1, f2) >= jaccard_threshold:
                    union(f1, f2)

    for idx, input_path in enumerate(input_paths):
        if find(idx) == idx:
            shutil.copy(input_path, output_dir / input_path.name)
