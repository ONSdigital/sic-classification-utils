"""Build and reload a persisted SAYT artifact with the built-in retrievers."""

# pylint: disable=duplicate-code

# %%
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTBuilder,
    SAYTSuggester,
    SemanticRetrieverSpec,
)

# %%
############# toy example to verify SAYT artifact build/load works #############
small_corpus = [
    ("Car wash", "Car Wash"),
    ("Car wash", "CAR WASH (duplicate)"),
    ("Car waxing", "Car Waxing"),
    ("Waxing car", "Car Waxing"),
    ("Carpentry services", "Carpentry services"),
    ("Dog grooming", "Dog grooming"),
    ("Cat grooming", "Cat grooming"),
    ("USed car sales", "Used car sales"),
    ("Car rental", "Car rental"),
    ("Car repair", "Car repair"),
    ("Car servicing", "Car servicing"),
]

retrievers = [
    PrefixRetrieverSpec(),
    NgramRetrieverSpec(max_df=0.8),
    SemanticRetrieverSpec(),
]

# Keep the temporary directory alive across notebook cells.
# pylint: disable-next=consider-using-with
temp_dir = TemporaryDirectory(prefix="sayt_artifact_demo_")
artifact_dir = Path(temp_dir.name) / "car_services_sayt"
print("artifact will be written to:", artifact_dir)

# %%
# Semantic artifact builds may take longer the first time if the model cache
# needs to be created locally.
artifact_path = SAYTBuilder(
    small_corpus,
    retrievers=retrievers,
    min_chars=3,
    max_suggestions=5,
).build_artifact(artifact_dir, overwrite=True)

print("artifact saved to:", artifact_path)
print("artifact files:")
for path in sorted(artifact_path.rglob("*")):
    if path.is_file():
        print("-", path.relative_to(artifact_path))

# %%
manifest = json.loads((artifact_path / "manifest.json").read_text(encoding="utf-8"))
print(json.dumps(manifest, indent=2))

# %%
live_suggester = SAYTSuggester(
    small_corpus,
    retrievers=retrievers,
    min_chars=3,
    max_suggestions=5,
)
loaded_suggester = SAYTSuggester.from_artifact(artifact_path)

for query in ["car", "cars", "waxi", "grom", "wash", "duplicate", "auto"]:
    live_suggestions = live_suggester.suggest(query, 5)
    loaded_suggestions = loaded_suggester.suggest(query, 5)

    print("searching for:", query)
    print("live", "->", live_suggestions)
    print("loaded", "->", loaded_suggestions)
    print("loaded_scores", "->", loaded_suggester.suggest_with_scores(query, 5))
    if live_suggestions != loaded_suggestions:
        raise RuntimeError("Loaded suggester results did not match live build")
    print()

# %%
# Run `temp_dir.cleanup()` when you are finished exploring the saved files.
print("artifact ready for inspection:", artifact_path)

# %%
temp_dir.cleanup()

# %%
