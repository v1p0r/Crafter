"""Static registry of common ML datasets with preprocessing details.

Used by the gap agent to resolve dataset gaps without API calls.
"""

import re

DATASET_REGISTRY: dict[str, dict] = {
    # --- Vision (classification) ---
    "imagenet": {
        "full_name": "ImageNet (ILSVRC2012)",
        "source": "torchvision.datasets.ImageNet or huggingface:imagenet-1k",
        "train_size": 1281167, "val_size": 50000, "num_classes": 1000,
        "standard_transforms": "RandomResizedCrop(224), RandomHorizontalFlip, Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        "eval_transforms": "Resize(256), CenterCrop(224), Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
    },
    "cifar10": {
        "full_name": "CIFAR-10",
        "source": "torchvision.datasets.CIFAR10",
        "train_size": 50000, "val_size": 10000, "num_classes": 10,
        "standard_transforms": "RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])",
        "eval_transforms": "Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])",
    },
    "cifar100": {
        "full_name": "CIFAR-100",
        "source": "torchvision.datasets.CIFAR100",
        "train_size": 50000, "val_size": 10000, "num_classes": 100,
        "standard_transforms": "RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize(mean=[0.5071,0.4867,0.4408], std=[0.2675,0.2565,0.2761])",
        "eval_transforms": "Normalize(mean=[0.5071,0.4867,0.4408], std=[0.2675,0.2565,0.2761])",
    },
    "mnist": {
        "full_name": "MNIST",
        "source": "torchvision.datasets.MNIST",
        "train_size": 60000, "val_size": 10000, "num_classes": 10,
        "standard_transforms": "ToTensor(), Normalize(mean=[0.1307], std=[0.3081])",
        "eval_transforms": "ToTensor(), Normalize(mean=[0.1307], std=[0.3081])",
    },
    "fashion_mnist": {
        "full_name": "Fashion-MNIST",
        "source": "torchvision.datasets.FashionMNIST",
        "train_size": 60000, "val_size": 10000, "num_classes": 10,
        "standard_transforms": "ToTensor(), Normalize(mean=[0.2860], std=[0.3530])",
        "eval_transforms": "ToTensor(), Normalize(mean=[0.2860], std=[0.3530])",
    },
    "svhn": {
        "full_name": "Street View House Numbers",
        "source": "torchvision.datasets.SVHN",
        "train_size": 73257, "val_size": 26032, "num_classes": 10,
        "standard_transforms": "Normalize(mean=[0.4377,0.4438,0.4728], std=[0.1980,0.2010,0.1970])",
        "eval_transforms": "Normalize(mean=[0.4377,0.4438,0.4728], std=[0.1980,0.2010,0.1970])",
    },
    "stl10": {
        "full_name": "STL-10",
        "source": "torchvision.datasets.STL10",
        "train_size": 5000, "val_size": 8000, "unlabeled_size": 100000, "num_classes": 10,
        "standard_transforms": "RandomResizedCrop(96), RandomHorizontalFlip, Normalize(mean=[0.4467,0.4398,0.4066], std=[0.2242,0.2215,0.2239])",
        "eval_transforms": "Resize(96), Normalize(mean=[0.4467,0.4398,0.4066], std=[0.2242,0.2215,0.2239])",
    },
    # --- Vision (detection/segmentation) ---
    "coco": {
        "full_name": "MS COCO (2017)",
        "source": "torchvision.datasets.CocoDetection or pycocotools",
        "train_size": 118287, "val_size": 5000, "num_classes": 80,
        "standard_transforms": "Resize, RandomHorizontalFlip, Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        "eval_transforms": "Resize, Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
    },
    "voc": {
        "full_name": "PASCAL VOC 2012",
        "source": "torchvision.datasets.VOCSegmentation or VOCDetection",
        "train_size": 5717, "val_size": 5823, "num_classes": 21,
        "standard_transforms": "Resize, RandomHorizontalFlip, Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        "eval_transforms": "Resize, Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
    },
    # --- Vision (generation) ---
    "celeba": {
        "full_name": "CelebA",
        "source": "torchvision.datasets.CelebA",
        "train_size": 162770, "val_size": 19867, "test_size": 19962,
        "standard_transforms": "CenterCrop(178), Resize(64 or 128 or 256), Normalize to [-1,1]",
        "eval_transforms": "CenterCrop(178), Resize(same), Normalize to [-1,1]",
    },
    "ffhq": {
        "full_name": "Flickr-Faces-HQ",
        "source": "https://github.com/NVlabs/ffhq-dataset or huggingface:flax-community/ffhq_lmdb",
        "train_size": 70000, "num_classes": None,
        "standard_transforms": "Resize(256 or 1024), Normalize to [-1,1]",
        "eval_transforms": "Resize(same), Normalize to [-1,1]",
    },
    "lsun": {
        "full_name": "Large-scale Scene Understanding",
        "source": "torchvision.datasets.LSUN",
        "train_size": "varies by category (bedroom: ~3M)", "num_classes": 10,
        "standard_transforms": "Resize(256), CenterCrop(256), Normalize to [-1,1]",
        "eval_transforms": "Resize(256), CenterCrop(256), Normalize to [-1,1]",
    },
    # --- NLP (classification/NLU) ---
    "sst2": {
        "full_name": "Stanford Sentiment Treebank v2",
        "source": "huggingface:glue/sst2",
        "train_size": 67349, "val_size": 872, "num_classes": 2,
        "standard_transforms": "Tokenize with model tokenizer, max_length=128, padding=max_length, truncation=True",
        "eval_transforms": "Same as train",
    },
    "mnli": {
        "full_name": "Multi-Genre Natural Language Inference",
        "source": "huggingface:glue/mnli",
        "train_size": 392702, "val_matched_size": 9815, "val_mismatched_size": 9832, "num_classes": 3,
        "standard_transforms": "Tokenize premise+hypothesis, max_length=128",
        "eval_transforms": "Same as train",
    },
    "qqp": {
        "full_name": "Quora Question Pairs",
        "source": "huggingface:glue/qqp",
        "train_size": 363846, "val_size": 40430, "num_classes": 2,
        "standard_transforms": "Tokenize question1+question2, max_length=128",
        "eval_transforms": "Same as train",
    },
    "mrpc": {
        "full_name": "Microsoft Research Paraphrase Corpus",
        "source": "huggingface:glue/mrpc",
        "train_size": 3668, "val_size": 408, "num_classes": 2,
        "standard_transforms": "Tokenize sentence1+sentence2, max_length=128",
        "eval_transforms": "Same as train",
    },
    # --- NLP (QA) ---
    "squad": {
        "full_name": "Stanford Question Answering Dataset v1.1",
        "source": "huggingface:squad",
        "train_size": 87599, "val_size": 10570,
        "standard_transforms": "Tokenize question+context, max_length=384, doc_stride=128, return_offsets_mapping=True",
        "eval_transforms": "Same as train",
    },
    "squad_v2": {
        "full_name": "SQuAD v2.0 (with unanswerable questions)",
        "source": "huggingface:squad_v2",
        "train_size": 130319, "val_size": 11873,
        "standard_transforms": "Same as SQuAD v1 with is_impossible handling",
        "eval_transforms": "Same as train",
    },
    # --- NLP (generation/instruction) ---
    "alpaca": {
        "full_name": "Stanford Alpaca (instruction tuning)",
        "source": "huggingface:tatsu-lab/alpaca",
        "train_size": 52002,
        "standard_transforms": "Format as instruction/input/output template, tokenize with model tokenizer",
        "eval_transforms": "Same format, generate with model",
    },
    "sharegpt": {
        "full_name": "ShareGPT (multi-turn conversations)",
        "source": "huggingface:anon8231489123/ShareGPT_Vicuna_unfiltered",
        "train_size": "~70k conversations",
        "standard_transforms": "Format as multi-turn chat, tokenize with chat template",
        "eval_transforms": "Same format",
    },
    # --- NLP (translation) ---
    "wmt14_en_de": {
        "full_name": "WMT 2014 English-German",
        "source": "huggingface:wmt14 (en-de)",
        "train_size": 4508785,
        "standard_transforms": "Tokenize source+target with shared tokenizer, max_length=128",
        "eval_transforms": "Same, decode with beam search (beam_size=4-5)",
    },
    "wmt14_en_fr": {
        "full_name": "WMT 2014 English-French",
        "source": "huggingface:wmt14 (en-fr)",
        "train_size": 35762532,
        "standard_transforms": "Tokenize source+target with shared tokenizer, max_length=128",
        "eval_transforms": "Same, decode with beam search",
    },
    # --- LLM benchmarks ---
    "mmlu": {
        "full_name": "Massive Multitask Language Understanding",
        "source": "huggingface:cais/mmlu",
        "test_size": 14042, "num_subjects": 57,
        "standard_transforms": "Format as 4-choice QA with few-shot examples (5-shot typical)",
        "eval_transforms": "Measure accuracy on answer letter (A/B/C/D)",
    },
    "hellaswag": {
        "full_name": "HellaSwag (commonsense NLI)",
        "source": "huggingface:Rowan/hellaswag",
        "val_size": 10042,
        "standard_transforms": "Format as 4-choice completion, 0-shot or 10-shot",
        "eval_transforms": "Normalized accuracy (length-normalized log-probabilities)",
    },
    "arc": {
        "full_name": "AI2 Reasoning Challenge",
        "source": "huggingface:allenai/ai2_arc (ARC-Easy and ARC-Challenge)",
        "test_easy_size": 2376, "test_challenge_size": 1172,
        "standard_transforms": "Format as multiple-choice QA, 25-shot typical",
        "eval_transforms": "Accuracy on correct answer",
    },
    "winogrande": {
        "full_name": "WinoGrande (coreference resolution)",
        "source": "huggingface:allenai/winogrande",
        "val_size": 1267,
        "standard_transforms": "Format as binary choice fill-in-blank, 5-shot typical",
        "eval_transforms": "Accuracy",
    },
    "truthfulqa": {
        "full_name": "TruthfulQA (truthfulness benchmark)",
        "source": "huggingface:truthful_qa",
        "test_size": 817,
        "standard_transforms": "Format as multiple-choice or generative, 0-shot",
        "eval_transforms": "MC accuracy or GPT-judge for generative",
    },
    "gsm8k": {
        "full_name": "GSM8K (grade school math)",
        "source": "huggingface:gsm8k",
        "train_size": 7473, "test_size": 1319,
        "standard_transforms": "Format as chain-of-thought QA, 8-shot typical",
        "eval_transforms": "Extract final numerical answer, exact match",
    },
    "humaneval": {
        "full_name": "HumanEval (code generation)",
        "source": "huggingface:openai_humaneval",
        "test_size": 164,
        "standard_transforms": "Provide function signature + docstring, generate completion",
        "eval_transforms": "pass@k with k=1,10,100; execute unit tests",
    },
}

# Common paper spellings that are not represented by the registry key or full
# display name. Keep this list explicit: automatic substring expansion is what
# caused unrelated names such as HotpotQA and EmbedLLM to resolve to SQuAD.
_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "arc": ("ARC-Challenge", "ARC-Easy"),
    "fashion_mnist": ("Fashion MNIST",),
    "squad": ("SQuAD v1.1", "SQuAD 1.1"),
    "squad_v2": ("SQuAD v2", "SQuAD 2.0"),
    "wmt14_en_de": ("WMT14 En-De", "WMT 2014 En-De"),
    "wmt14_en_fr": ("WMT14 En-Fr", "WMT 2014 En-Fr"),
}


def _normalize_name(name: str) -> str:
    """Normalize a *dataset name*, not arbitrary prose, for exact lookup."""
    return "".join(re.findall(r"[a-z0-9]+", name.casefold()))


def _aliases_for(key: str) -> tuple[str, ...]:
    info = DATASET_REGISTRY[key]
    aliases = {
        key,
        key.replace("_", "-"),
        key.replace("_", ""),
        info["full_name"],
        *_EXTRA_ALIASES.get(key, ()),
    }
    return tuple(sorted(aliases, key=lambda alias: (_normalize_name(alias), alias)))


# Exact normalized lookup for callers that already have a candidate name.
_LOOKUP: dict[str, str] = {}
for _key in DATASET_REGISTRY:
    for _alias in _aliases_for(_key):
        _normalized = _normalize_name(_alias)
        if _normalized:
            _LOOKUP[_normalized] = _key


def _alias_pattern(alias: str) -> re.Pattern[str] | None:
    """Compile a bounded alias pattern with flexible punctuation separators."""
    parts = re.findall(r"[a-z0-9]+", alias.casefold())
    if not parts:
        return None
    body = r"[\s_.-]*".join(re.escape(part) for part in parts)
    # Do not accept a registered name as one component of an unregistered
    # compound (MMLU-Pro, ImageNet-R, CIFAR-10-C). Legitimate compounds such as
    # ARC-Challenge must be listed explicitly above and consumed in full.
    return re.compile(
        rf"(?<![A-Za-z0-9])(?<![A-Za-z0-9][_.-])"
        rf"{body}(?!(?:[A-Za-z0-9]|[_.-][A-Za-z0-9]))",
        re.IGNORECASE,
    )


# Longest aliases run first so, for example, "SQuAD v2" is not also accepted
# as the v1 dataset through its shorter "SQuAD" alias.
_TEXT_MATCHERS: list[tuple[str, str, re.Pattern[str]]] = []
for _key in DATASET_REGISTRY:
    for _alias in _aliases_for(_key):
        _pattern = _alias_pattern(_alias)
        if _pattern is not None:
            _TEXT_MATCHERS.append((_key, _alias, _pattern))
_TEXT_MATCHERS.sort(key=lambda item: len(_normalize_name(item[1])), reverse=True)

# These short aliases are ordinary words or names when written in lowercase.
# Requiring their conventional spelling avoids resolving prose like "an arc"
# or a person named "Coco" as a benchmark.
_UPPERCASE_TEXT_ALIASES = {"arc", "voc", "coco"}


def find_dataset(name: str) -> dict | None:
    """Look up one candidate dataset name using exact normalized aliases.

    The function intentionally does not perform substring matching. Use
    :func:`find_datasets_in_text` when scanning prose.
    """
    key = _LOOKUP.get(_normalize_name(name))
    return DATASET_REGISTRY[key] if key else None


def find_datasets_in_text(text: str) -> list[dict]:
    """Return datasets explicitly named in prose using bounded alias matches.

    Each result includes its registry key and match evidence. All accepted
    matches have high confidence because an entire registered alias, with token
    boundaries, must be present. Overlapping shorter aliases are discarded.
    """
    candidates: list[tuple[int, int, int, str, str, str]] = []
    for key, alias, pattern in _TEXT_MATCHERS:
        normalized_alias = _normalize_name(alias)
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            if normalized_alias in _UPPERCASE_TEXT_ALIASES and not matched_text.isupper():
                continue
            candidates.append(
                (match.start(), match.end(), -len(normalized_alias), key, alias, matched_text)
            )

    accepted_spans: list[tuple[int, int]] = []
    found_keys: set[str] = set()
    results: list[dict] = []
    ordered = sorted(candidates, key=lambda item: (item[0], item[2], item[1], item[3]))
    for start, end, _negative_length, key, alias, matched_text in ordered:
        if key in found_keys:
            continue
        if any(start < accepted_end and end > accepted_start
               for accepted_start, accepted_end in accepted_spans):
            continue
        accepted_spans.append((start, end))
        found_keys.add(key)
        results.append({
            "dataset_key": key,
            "matched_alias": alias,
            "matched_text": matched_text,
            "match_type": "bounded_alias",
            "match_confidence": "high",
            **DATASET_REGISTRY[key],
        })
    return results
