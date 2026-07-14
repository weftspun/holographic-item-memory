# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Ports.TestCaseSource do
  @moduledoc """
  Driven port (`*_source`): supplies next-item evaluation test cases to the core.

  `Holo.Core.Eval` computes Hit@k / MRR over a list of `%{"context" => [id], "next_item" => id}`
  maps but must not know how they are stored or parsed. Adapters realize this port against concrete
  backing stores (a JSON file, an in-memory fixture, a database) and normalize to that shape.

  Implemented by: `Holo.Adapters.JsonTestCaseSource`.
  """

  @type test_case :: %{required(String.t()) => term()}

  @doc "Load test cases from `ref` (e.g. a filesystem path)."
  @callback load(ref :: term()) :: {:ok, [test_case()]} | {:error, term()}
end
