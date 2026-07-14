defmodule Holo.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/weftspun/holographic-semantic-memory"

  def project do
    [
      app: :holographic_semantic_memory,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "Holographic (HRR phase-vector) session memory over concat-vector ResidualFSQ " <>
          "semantic IDs. Zero-shot next-item recall; codec and phase algebra certified " <>
          "in Lean via plausible-witness-dag.",
      package: [
        licenses: ["MIT"],
        links: %{"GitHub" => @source_url}
      ],
      source_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger, :crypto]]
  end

  defp deps do
    [
      {:nx, "~> 0.11"},
      {:explorer, "~> 0.9"},
      {:jason, "~> 1.4"},
      {:stream_data, "~> 1.2", only: [:dev, :test]}
    ]
  end
end
