defmodule Holo.MixProject do
  use Mix.Project

  @version "0.2.0"
  @source_url "https://github.com/weftspun/holographic-item-memory"

  def project do
    [
      app: :holographic_item_memory,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      releases: releases(),
      description:
        "Generative next-item recommender (RecGPT / FuXi-Linear linear-attention) over " <>
          "residual FSQ semantic IDs. Trie-constrained beam decode; ID codec certified in " <>
          "Lean via plausible-witness-dag. Ships as a self-contained Burrito binary with an " <>
          "embedded CockroachDB store.",
      package: [
        licenses: ["MIT"],
        links: %{"GitHub" => @source_url}
      ],
      source_url: @source_url
    ]
  end

  def application do
    [
      extra_applications: [:logger, :crypto],
      mod: {Holo.Application, []}
    ]
  end

  # Standalone Burrito binary (`holo`), following the
  # V-Sekai-fire/multiplayer-fabric-taskweft pattern: the wrap step only
  # invokes Burrito when a zig toolchain is present (or HOLO_BURRITO=1 forces
  # it), so a plain `mix release holo` still assembles without the toolchain.
  # The CockroachStep patch step downloads the matching V-Sekai/cockroach
  # single binary per target and lands it in the payload's priv/cockroach/.
  defp releases do
    [
      holo: [
        version: @version,
        applications: [holographic_item_memory: :permanent],
        steps: [:assemble, &Holo.Release.wrap/1],
        burrito: [
          targets: [
            linux_amd64: [os: :linux, cpu: :x86_64],
            macos_arm64: [os: :darwin, cpu: :aarch64],
            windows_amd64: [os: :windows, cpu: :x86_64]
          ],
          extra_steps: [
            patch: [post: [Holo.Release.CockroachStep]]
          ]
        ]
      ]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.11", override: true},
      # RecGPT / FuXi-Linear port: model runtime + checkpoint/data IO. The
      # inference/training stack runs on EXLA (XLA JIT); config sets
      # `:backend_app` to `:exla` and the default backend to `EXLA.Backend`.
      # EXLA downloads a precompiled XLA archive (needs `make` + a C compiler,
      # NOT cmake) — CPU by default; set `XLA_TARGET=cuda12x` for GPU. torchx is
      # left out (its libtorch bindings need `cmake`, absent here).
      {:exla, "~> 0.11"},
      {:axon, "~> 0.7"},
      {:bumblebee, github: "elixir-nx/bumblebee", ref: "main"},
      {:npy, "~> 0.1.2"},
      {:unpickler, "~> 0.1"},
      {:unzip, "~> 0.13"},
      {:nimble_csv, "~> 1.2"},
      {:req, "~> 0.5"},
      {:explorer, "~> 0.11"},
      {:postgrex, "~> 0.19"},
      # Local database / object-storage hosts, extracted to their own repos.
      # CockroachStore / VersityBlobStore still carry their own host lifecycle;
      # delegating provision + start/stop to these is a follow-up.
      {:cockroach_local, github: "weftspun/cockroach-local"},
      {:versitygw_local, github: "weftspun/versitygw-local"},
      {:aria_storage, github: "V-Sekai-fire/aria-storage"},
      {:ex_aws, "~> 2.4"},
      {:ex_aws_s3, "~> 2.4"},
      {:hackney, "~> 1.20"},
      {:burrito, "~> 1.5", runtime: false},
      {:jason, "~> 1.4"},
      {:stream_data, "~> 1.2", only: [:dev, :test]}
    ]
  end
end
