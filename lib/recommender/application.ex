# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Recommender.Application do
  @moduledoc """
  OTP application for `:residual_fsq_recommender`.

  As a library dependency (or under `mix test` / `mix run`) this starts an
  empty supervision tree — no behaviour change for consumers. When the app
  boots as the standalone Burrito binary, it starts a single `Task` running
  `Recommender.Adapters.CLI.main/0`, turning the binary into the `rfr` CLI.

  The CLI is only auto-run when the `__BURRITO` runtime marker is set (read
  directly — `Burrito.Util` may not be loaded this early in boot). Set
  `RFR_CLI=0` to suppress it even in the standalone binary.
  """

  use Application

  @impl true
  def start(_type, _args) do
    children =
      if run_cli?() do
        # `:temporary` so a crashing dispatcher can't be spin-restarted;
        # `main/1` is self-contained and halts the VM on completion.
        [Supervisor.child_spec({Task, fn -> Recommender.Adapters.CLI.main() end}, restart: :temporary)]
      else
        []
      end

    Supervisor.start_link(children, strategy: :one_for_one, name: Recommender.Supervisor)
  end

  defp run_cli? do
    System.get_env("RFR_CLI") != "0" and System.get_env("__BURRITO") != nil
  end
end
