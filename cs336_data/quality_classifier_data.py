import gzip
import pathlib
import random

# cd url_chunks 
# split -l 1250 ../subsampled_positive_urls.txt urls_chunk_
# for f in urls_chunk_*; do
#   wget \
#     --input-file="$f" \
#     --warc-file="${f}" \
#     --output-file="${f}.log" \
#     --timeout=5 \
#     --tries=2 \
#     --max-redirect=3 \
#     -O /dev/null &
# done
# wait

def extract_k_urls(k: int, seed: int | None = None) -> str:
    if k < 0:
        raise ValueError("k must be non-negative")

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    input_path = data_dir / "enwiki-20240420-extracted_urls.txt.gz"
    output_path = data_dir / "subsampled_positive_urls.txt"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rng = random.Random(seed)
    sampled_urls: list[str] = []
    num_urls_seen = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as in_file:
        for line in in_file:
            url = line.strip()
            if not url:
                continue

            num_urls_seen += 1
            if len(sampled_urls) < k:
                sampled_urls.append(url)
                continue

            replace_idx = rng.randrange(num_urls_seen)
            if replace_idx < k:
                sampled_urls[replace_idx] = url

    if num_urls_seen < k:
        raise ValueError(
            f"Requested {k} URLs, but only found {num_urls_seen} non-empty URLs in {input_path}."
        )

    with open(output_path, "w", encoding="utf-8") as out_file:
        for url in sampled_urls:
            out_file.write(url + "\n")

    return str(output_path)
