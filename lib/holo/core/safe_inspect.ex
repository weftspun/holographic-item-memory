# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

defmodule Holo.Core.SafeInspect do
  @moduledoc """
  Safe stringification that never invokes String.Chars or Inspect on Nx.Tensor
  or expression types, avoiding protocol errors in logs and error messages.

  Ported verbatim from RecGPT.Core.SafeInspect.
  """

  @doc """
  Returns a string representation of `term` without ever printing tensor contents.
  Use in Logger and anywhere term might be or contain an Nx.Tensor.
  """
  def safe_inspect(term, opts \\ [])

  def safe_inspect(%Nx.Tensor{} = _t, _opts), do: "<Nx.Tensor>"

  def safe_inspect(%struct{} = _t, opts) when struct in [Nx.Defn.Expr] do
    if Keyword.get(opts, :struct, true), do: "#<#{inspect(struct)}>", else: "<expr>"
  end

  def safe_inspect(list, opts) when is_list(list) do
    "[#{Enum.map_join(list, ", ", &safe_inspect(&1, opts))}]"
  end

  def safe_inspect(tuple, opts) when is_tuple(tuple) do
    "{#{Enum.map_join(Tuple.to_list(tuple), ", ", &safe_inspect(&1, opts))}}"
  end

  def safe_inspect(map, opts) when is_map(map) and not is_struct(map) do
    pairs =
      Enum.map_join(map, ", ", fn {k, v} ->
        "#{safe_inspect(k, opts)} => #{safe_inspect(v, opts)}"
      end)

    "%{#{pairs}}"
  end

  def safe_inspect(%_{} = struct, _opts) do
    # Struct that might contain tensor: only show module name
    "#{inspect(struct.__struct__)}<>"
  end

  def safe_inspect(x, _opts) when is_binary(x), do: inspect(x)

  def safe_inspect(x, _opts) when is_integer(x) or is_float(x) or is_atom(x) or is_boolean(x),
    do: inspect(x)

  def safe_inspect(_x, _opts), do: "<term>"
end
