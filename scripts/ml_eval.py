"""HRR-only sequential recommendation eval on MovieLens-100K.

Rebuilds the lost eval: no ResidualFSQ, no neural net — item vectors are HRR
phase atoms (id + genre atoms bundled), the transition signal is a bucketed
hetero-associative bank, and recall is unbind + phase-cosine cleanup. Metrics
are leave-one-out Recall/MRR/NDCG@10 against a most-popular baseline.

The HRR primitives here are byte-for-byte the same as lib/holo/core/hrr.ex; the
script asserts parity against test/fixtures/hrr_golden.json before running, so
these numbers reflect the Elixir library's algebra, not a numpy approximation.

    python scripts/ml_eval.py            # dim=4096, B=128 buckets

Reads scratch_ml/ml-100k/{u.data,u.item}. numpy only, no pandas/pyarrow.
"""

import hashlib
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

TWO_PI = 2.0 * math.pi
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML = os.path.join(ROOT, "scratch_ml", "ml-100k")

DIM = int(os.environ.get("HOLO_DIM", 4096))
N_BUCKETS = int(os.environ.get("HOLO_BUCKETS", 2048))  # ~1/source: collision-free banks
K = 10                   # top-K cutoff for all metrics
N_NEG = 100              # sampled negatives per user (target ranked among 1+100)
SEED = 20260713          # deterministic negative sampling
W_CONTENT = 0.4          # combined-signal weights (mirror Holo.Core.Memory)
W_TRANSITION = 0.6


# ── HRR primitives (parity with lib/holo/core/hrr.ex) ───────────────────────
def encode_atom(word, dim=DIM):
    blocks = (dim + 15) // 16
    buf = b"".join(hashlib.sha256(f"{word}:{i}".encode()).digest() for i in range(blocks))
    vals = np.frombuffer(buf, dtype="<u2")[:dim].astype(np.float64)
    return vals * (TWO_PI / 65536.0)


def bind(a, b):
    return np.mod(a + b, TWO_PI)


def unbind(m, k):
    return np.mod(m - k, TWO_PI)


def bundle(vs):  # vs: (k, dim) -> (dim,) circular mean, wrapped to [0, 2π)
    return np.mod(np.arctan2(np.sin(vs).sum(0), np.cos(vs).sum(0)), TWO_PI)


def similarity(a, b):
    return float(np.cos(a - b).mean())


def assert_golden():
    path = os.path.join(ROOT, "test", "fixtures", "hrr_golden.json")
    g = json.load(open(path))
    d = g["dim"]
    alice, bob = encode_atom("alice", d), encode_atom("bob", d)
    role = encode_atom("__holo_role_q0__", d)
    checks = {
        "atom_alice": alice,
        "bind_alice_bob": bind(alice, bob),
        "unbind_bind_alice_bob__bob": unbind(bind(alice, bob), bob),
        "bundle_alice_bob_role": bundle(np.stack([alice, bob, role])),
    }
    for key, got in checks.items():
        exp = np.array(g[key])
        assert np.allclose(got, exp, atol=1e-9), f"golden mismatch: {key}"
    assert abs(similarity(alice, bob) - g["similarity_alice_bob"]) < 1e-9
    print("HRR primitives verified against test/fixtures/hrr_golden.json")


# ── data ────────────────────────────────────────────────────────────────────
STOPWORDS = {"the", "a", "an", "of", "and", "in", "to", "part"}


def load_items():
    """movie_id (1-based) -> content atom words: genres, year, title tokens.

    Textual features (RecGPT-style) rather than an opaque id alone — this is
    the only source of content generalization in a pure-HRR setup, so richer
    text = a stronger content signal.
    """
    feats = {}
    with open(os.path.join(ML, "u.item"), "rb") as fh:
        for line in fh:
            f = line.decode("latin-1").rstrip("\n").split("|")
            mid = int(f[0])
            words = [f"genre:{k}" for k, flag in enumerate(f[5:24]) if flag == "1"]
            title = f[1]
            if title.endswith(")") and "(" in title:  # trailing "(YYYY)"
                head, year = title.rsplit("(", 1)
                year = year.rstrip(")")
                if year[:4].isdigit():
                    words.append(f"year:{year[:4]}")
                    words.append(f"decade:{year[:3]}0s")
                title = head
            for tok in title.lower().replace(",", " ").replace(":", " ").split():
                tok = tok.strip(".!?;\"'()[]{}")
                if len(tok) > 1 and tok not in STOPWORDS:
                    words.append(f"tok:{tok}")
            feats[mid] = words
    return feats


def load_sequences():
    """user_id -> item sequence ordered by (timestamp, item)."""
    by_user = defaultdict(list)
    with open(os.path.join(ML, "u.data")) as fh:
        for line in fh:
            u, i, _r, ts = line.split("\t")
            by_user[int(u)].append((int(ts), int(i)))
    return {u: [i for _ts, i in sorted(rows)] for u, rows in by_user.items()}


def item_vectors(feats, n_items):
    """(n_items, DIM): bundle of the id atom + one atom per content feature word."""
    V = np.empty((n_items, DIM), dtype=np.float64)
    for mid in range(1, n_items + 1):
        atoms = [encode_atom(f"item:{mid}")]
        atoms += [encode_atom(w) for w in feats.get(mid, [])]
        V[mid - 1] = bundle(np.stack(atoms))
    return V


def bucket_of(mid):
    return int(hashlib.sha256(f"item:{mid}".encode()).hexdigest(), 16) % N_BUCKETS


# ── metrics ──────────────────────────────────────────────────────────────────
def sample_candidates(histories, target_idx, n_items):
    """Per user: [target, 100 negatives ∉ history]. Deterministic (SEED)."""
    rng = np.random.default_rng(SEED)
    cands = []
    for h, t in zip(histories, target_idx):
        blocked = set(h.tolist()) | {int(t)}
        negs = []
        while len(negs) < N_NEG:
            draw = rng.integers(0, n_items, size=N_NEG * 2)
            for x in draw:
                if x not in blocked:
                    negs.append(int(x))
                    blocked.add(int(x))
                    if len(negs) == N_NEG:
                        break
        cands.append(np.array([int(t)] + negs))  # index 0 is the positive
    return cands


def rank_of_target(scores_row, cand_idx):
    """1-based rank of the positive (cand_idx[0]) among candidates, else None."""
    sub = scores_row[cand_idx]
    rank = int((sub > sub[0]).sum()) + 1  # items strictly better than the positive
    return rank if rank <= K else None


def score_configs(V, seqs, n_items):
    cosV = np.cos(V).astype(np.float32)
    sinV = np.sin(V).astype(np.float32)

    # test set: one held-out last item per user (len >= 2, always true for ML-100K)
    users = [u for u, s in seqs.items() if len(s) >= 2]
    prev_idx = np.array([seqs[u][-2] - 1 for u in users])
    target_idx = np.array([seqs[u][-1] - 1 for u in users])
    histories = [np.array([i - 1 for i in seqs[u][:-1]]) for u in users]

    # fold every non-held-out consecutive transition into its source's bucket
    src, dst = [], []
    for u in users:
        s = seqs[u]
        for k in range(len(s) - 2):        # drop the final (prev -> target) pair
            src.append(s[k] - 1)
            dst.append(s[k + 1] - 1)
    src, dst = np.array(src), np.array(dst)
    bck = np.array([bucket_of(i + 1) for i in src])
    print(f"transitions folded into bank: {len(src)}")

    bank_sin = np.zeros((N_BUCKETS, DIM), dtype=np.float32)
    bank_cos = np.zeros((N_BUCKETS, DIM), dtype=np.float32)
    for c in range(0, len(src), 5000):
        pv = V[src[c:c + 5000]] + V[dst[c:c + 5000]]
        np.add.at(bank_sin, bck[c:c + 5000], np.sin(pv).astype(np.float32))
        np.add.at(bank_cos, bck[c:c + 5000], np.cos(pv).astype(np.float32))
    bank_phase = np.arctan2(bank_sin, bank_cos)  # (N_BUCKETS, DIM)

    def scores_from_phase(phase):  # phase: (U, DIM) -> (U, n_items)
        qc = np.cos(phase).astype(np.float32)
        qs = np.sin(phase).astype(np.float32)
        return (qc @ cosV.T + qs @ sinV.T) / DIM

    # content probe: bundle of the user's whole history
    content_phase = np.stack([bundle(V[h]) for h in histories])
    content_scores = scores_from_phase(content_phase)

    # transition probe: unbind the previous item from its bucket's bank
    user_bucket = np.array([bucket_of(seqs[u][-2]) for u in users])
    trans_phase = np.mod(bank_phase[user_bucket] - V[prev_idx], TWO_PI)
    trans_scores = scores_from_phase(trans_phase)

    combined = W_CONTENT * content_scores + W_TRANSITION * trans_scores

    # most-popular baseline (training popularity, i.e. everything but held-out targets)
    pop = Counter()
    for u in users:
        for i in seqs[u][:-1]:
            pop[i - 1] += 1
    pop_vec = np.zeros(n_items, dtype=np.float32)
    for i, c in pop.items():
        pop_vec[i] = c
    pop_scores = np.broadcast_to(pop_vec, (len(users), n_items))

    return {
        "most-popular baseline": pop_scores,
        "HRR content-only": content_scores,
        "HRR transition-only": trans_scores,
        "HRR combined (0.4/0.6)": combined,
    }, histories, target_idx


def evaluate(scores, candidates):
    recall = mrr = ndcg = 0.0
    n = len(candidates)
    for u in range(n):
        r = rank_of_target(scores[u], candidates[u])
        if r is not None:
            recall += 1.0
            mrr += 1.0 / r
            ndcg += 1.0 / math.log2(r + 1)
    return 100 * recall / n, 100 * mrr / n, 100 * ndcg / n


def main():
    assert_golden()
    feats = load_items()
    seqs = load_sequences()
    n_items = max(max(s) for s in seqs.values())
    print(f"users: {len(seqs)}   movies: {n_items}   dim: {DIM}   buckets: {N_BUCKETS}\n")

    V = item_vectors(feats, n_items)
    configs, histories, target_idx = score_configs(V, seqs, n_items)
    candidates = sample_candidates(histories, target_idx, n_items)
    print(f"protocol: leave-one-out, target vs {N_NEG} sampled negatives\n")

    print("=" * 64)
    print(f"{'config':<26}{'Recall@'+str(K):<12}{'MRR@'+str(K):<12}{'NDCG@'+str(K)}")
    print("-" * 64)
    for name, scores in configs.items():
        rec, mrr, ndcg = evaluate(np.asarray(scores), candidates)
        print(f"{name:<26}{rec:<12.4f}{mrr:<12.4f}{ndcg:.4f}")


if __name__ == "__main__":
    main()
