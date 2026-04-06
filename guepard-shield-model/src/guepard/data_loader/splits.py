import random
from dataclasses import dataclass

from .lidds_corpus import LiddsCorpus, LiddsRecordingMeta


@dataclass
class LiddsSplits:
    train: list[LiddsRecordingMeta]
    val: list[LiddsRecordingMeta]
    test: list[LiddsRecordingMeta]


def make_supervised_splits(
    corpus: LiddsCorpus,
    seed: int = 42,
    attack_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> LiddsSplits:
    """Split LID-DS corpus into train/val/test with both normal and attack in each.

    Strategy:
    - training/ (normal only)           → train_normal
    - validation/ (normal only)         → val_normal
    - test/normal/                      → test_normal
    - test/normal_and_attack/ exploit=True  → shuffle, split by attack_ratios
    - test/normal_and_attack/ exploit=False → appended to test_normal
    """
    train_r, val_r, _ = attack_ratios

    train_normal = corpus.get_split("training")
    val_normal = corpus.get_split("validation")
    test_normal = corpus.get_split("test_normal")
    test_mixed = corpus.get_split("test_attack")

    attack_recs = [m for m in test_mixed if m.label == 1]
    normal_from_mixed = [m for m in test_mixed if m.label == 0]

    rng = random.Random(seed)
    rng.shuffle(attack_recs)
    n = len(attack_recs)
    train_attack = attack_recs[: int(train_r * n)]
    val_attack = attack_recs[int(train_r * n) : int((train_r + val_r) * n)]
    test_attack = attack_recs[int((train_r + val_r) * n) :]

    train = train_normal + train_attack
    val = val_normal + val_attack
    test = test_normal + normal_from_mixed + test_attack

    for m in train:
        m.seq_class = "p2_train"
    for m in val:
        m.seq_class = "p2_val"
    for m in test:
        m.seq_class = "p2_test"

    corpus.metadata = train + val + test
    return LiddsSplits(train=train, val=val, test=test)
