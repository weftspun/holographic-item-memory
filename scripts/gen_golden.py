"""Regenerate test/fixtures/hrr_golden.json from the reference Python HRR implementation.

Usage:
    python scripts/gen_golden.py <path-to-reference-holographic-module-dir> [out.json]

The reference is the hermes-agent holographic memory plugin (holographic.py:
Plate-style phase-vector HRR with SHA-256 counter-block atoms). The Elixir port
in lib/holo/hrr.ex must match these values to 1e-9 or better.
"""

import json
import sys

sys.path.insert(0, sys.argv[1])
import holographic as hrr  # noqa: E402

DIM = 16

alice = hrr.encode_atom("alice", DIM)
bob = hrr.encode_atom("bob", DIM)
role = hrr.encode_atom("__holo_role_q0__", DIM)

out = {
    "dim": DIM,
    "atom_alice": alice.tolist(),
    "atom_bob": bob.tolist(),
    "atom_role_q0": role.tolist(),
    "bind_alice_bob": hrr.bind(alice, bob).tolist(),
    "unbind_bind_alice_bob__bob": hrr.unbind(hrr.bind(alice, bob), bob).tolist(),
    "bundle_alice_bob_role": hrr.bundle(alice, bob, role).tolist(),
    "similarity_alice_alice": hrr.similarity(alice, alice),
    "similarity_alice_bob": hrr.similarity(alice, bob),
    "encode_text_hello": hrr.encode_text("Hello, world! (hello)", DIM).tolist(),
    "encode_text_empty": hrr.encode_text("  ", DIM).tolist(),
    "atom_alice_1024_head": hrr.encode_atom("alice", 1024)[:4].tolist(),
    "atom_alice_1024_mean": float(hrr.encode_atom("alice", 1024).mean()),
}

path = sys.argv[2] if len(sys.argv) > 2 else "test/fixtures/hrr_golden.json"
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
