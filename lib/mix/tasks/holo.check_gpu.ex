# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Mix.Tasks.Holo.CheckGpu do
  @shortdoc "Report the active Nx backend and run a sample tensor"
  @moduledoc """
  Reports the default Nx backend, starts the configured backend app, and runs a
  sample tensor to confirm the backend works.

  By default the project runs on EXLA (XLA JIT) — `config/config.exs` sets
  `:backend_app` to `:exla` and `Nx.default_backend` to `EXLA.Backend`. EXLA
  downloads a precompiled XLA archive (needs `make` + a C compiler, not cmake).
  For GPU, set `XLA_TARGET=cuda12x` and add a `:cuda` EXLA client.

  Run: `mix holo.check_gpu`
  """
  use Mix.Task

  @impl true
  def run(_args) do
    Application.ensure_all_started(:nx)
    app = Application.get_env(:holographic_item_memory, :backend_app, :nx)

    case Application.ensure_all_started(app) do
      {:ok, _} ->
        do_check(app)

      {:error, {failed, reason}} ->
        IO.puts("Nx default_backend: #{inspect(Nx.default_backend())}")
        IO.puts("Backend app #{inspect(app)} failed to start: #{inspect(failed)} - #{inspect(reason)}")
    end
  end

  defp do_check(app) do
    backend = Nx.default_backend()
    IO.puts("Backend app: #{inspect(app)}")
    IO.puts("Nx default_backend: #{inspect(backend)}")

    t = Nx.tensor([1.0, 2.0, 3.0])
    backend_str = inspect(t)
    IO.puts("Sample tensor: #{backend_str}")

    on_cuda = String.contains?(String.downcase(backend_str), "cuda")

    result =
      if on_cuda,
        do: "backend running; using CUDA.",
        else: "backend running; using CPU."

    IO.puts("")
    IO.puts("Result: #{result}")
  end
end
