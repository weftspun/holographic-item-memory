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
        "Holographic (HRR phase-vector) item memory over concat-vector ResidualFSQ " <>
          "semantic IDs. Zero-shot next-item recall; phase algebra and ID codec certified " <>
          "in Lean via plausible-witness-dag. Ships as a self-contained Burrito binary " <>
          "with an embedded CockroachDB store.",
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
      {:nx, "~> 0.11"},
      {:explorer, "~> 0.11"},
      {:postgrex, "~> 0.19"},
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
