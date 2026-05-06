import os
from argparse import ArgumentParser

from huggingface_hub import snapshot_download

# Use argparse to handle command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Dataset name to download",
    default="HuggingFaceFW/fineweb",
    choices=[
        "gair-prox/FineWeb-pro",
        "gair-prox/open-web-math-pro",
        "gair-prox/c4-pro",
        "gair-prox/RedPajama-pro",
        "HuggingFaceFW/fineweb",
        "allenai/c4",
        "EleutherAI/proof-pile-2",
    ],
)
parser.add_argument(
    "--allow_patterns",
    type=str,
    help="Allow patterns to download",
    default=None,
    choices=[
        "sample/10BT/*",  # smallest FineWeb random sample (~10B tokens)
        "sample/350BT/*",  # for downloading fineweb
        "en/*",  # for downloading c4
    ],
)
args = parser.parse_args()
_script_dir = os.path.dirname(os.path.abspath(__file__))
_default_raw = os.path.normpath(os.path.join(_script_dir, "..", "..", "data", "raw"))
raw_data_dir = os.environ.get("RAW_DATA_DIR", _default_raw)

snapshot_download(
    repo_id=args.dataset_name,
    allow_patterns=args.allow_patterns,
    repo_type="dataset",
    local_dir=f"{raw_data_dir}/{args.dataset_name}",
    local_dir_use_symlinks=False,
    force_download=True,
)
