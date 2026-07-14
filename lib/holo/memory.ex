defmodule Holo.Memory do
  @moduledoc """
  Holographic session memory: zero-shot next-item recall over semantic IDs.

  The catalog is a map of `item_id => phase vector`, where every vector is
  derived purely from the item's ResidualFSQ semantic ID
  (`Holo.SemanticID.item_vector/2`). Two recall signals combine:

    * **Content** (always available, zero-shot): the session's recent items are
      bundled into one vector; candidates are scored by phase-cosine similarity.
      Items sharing semantic-ID tokens (same coarse codes = similar content
      upstream) score high — allocentric item-to-item recall with no training.

    * **Transitions** (optional, accumulates online): observed `a → b` steps are
      superposed into a single hetero-associative bank
      `T = bundle(bind(vec(a), vec(b)), …)`. Probing `unbind(T, vec(last))`
      yields a noisy `vec(next)`; cleanup against the catalog ranks candidates.
      Capacity is `O(√dim)` transitions (`snr_estimate/1` warns past that).

  Everything is a plain immutable struct — no processes, no storage backend.
  """

  alias Holo.HRR
  alias Holo.SemanticID

  defstruct dim: 1024,
            items: %{},
            tokens: %{},
            trans_sin: nil,
            trans_cos: nil,
            n_transitions: 0

  @type item_id :: term()
  @type t :: %__MODULE__{}

  @recent_window 8
  @content_weight 0.4
  @transition_weight 0.6

  @doc "New empty memory. Options: `:dim` (default #{HRR.default_dim()})."
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{dim: Keyword.get(opts, :dim, HRR.default_dim())}
  end

  @doc "Add one item by semantic ID. Idempotent for the same `{id, tokens}`."
  @spec add_item(t(), item_id(), [non_neg_integer()]) :: t()
  def add_item(%__MODULE__{} = mem, item_id, tokens) do
    vector = SemanticID.item_vector(tokens, mem.dim)

    %{
      mem
      | items: Map.put(mem.items, item_id, vector),
        tokens: Map.put(mem.tokens, item_id, tokens)
    }
  end

  @doc "Add many `{item_id, tokens}` pairs (e.g. from `Holo.SemanticID.load_parquet/1`)."
  @spec add_items(t(), Enumerable.t()) :: t()
  def add_items(%__MODULE__{} = mem, pairs) do
    Enum.reduce(pairs, mem, fn {id, tokens}, acc -> add_item(acc, id, tokens) end)
  end

  @doc "Number of catalog items."
  @spec size(t()) :: non_neg_integer()
  def size(%__MODULE__{items: items}), do: map_size(items)

  @doc """
  Fold a session's consecutive transitions into the hetero-associative bank.

  Unknown item ids are skipped. Returns the updated memory.
  """
  @spec observe(t(), [item_id()]) :: t()
  def observe(%__MODULE__{} = mem, session_item_ids) when is_list(session_item_ids) do
    session_item_ids
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.reduce(mem, fn [a, b], acc -> observe_transition(acc, a, b) end)
  end

  @doc "Superpose one `a → b` transition into the bank."
  @spec observe_transition(t(), item_id(), item_id()) :: t()
  def observe_transition(%__MODULE__{} = mem, a, b) do
    with %Nx.Tensor{} = va <- mem.items[a],
         %Nx.Tensor{} = vb <- mem.items[b] do
      pair = HRR.bind(va, vb)

      %{
        mem
        | trans_sin: add_or_init(mem.trans_sin, Nx.sin(pair)),
          trans_cos: add_or_init(mem.trans_cos, Nx.cos(pair)),
          n_transitions: mem.n_transitions + 1
      }
    else
      _ -> mem
    end
  end

  @doc """
  Recommend `top_k` next items for a session (list of item ids, oldest first).

  Scores every catalog item as
  `w_t · sim(unbind(T, vec(last)), vec(c)) + w_c · sim(session_bundle, vec(c))`,
  dropping the transition term when the bank is empty or the last item is
  unknown. Session items are excluded unless `exclude_seen: false`.

  Options: `:top_k` (default 5), `:exclude_seen` (default true),
  `:content_weight` / `:transition_weight` (defaults #{@content_weight} / #{@transition_weight}).

  Returns `{:ok, [{item_id, score}]}` sorted by score, or
  `{:error, reason}` when the session has no known items.
  """
  @spec recommend(t(), [item_id()], keyword()) ::
          {:ok, [{item_id(), float()}]} | {:error, String.t()}
  def recommend(%__MODULE__{} = mem, session_item_ids, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 5)
    exclude_seen = Keyword.get(opts, :exclude_seen, true)

    known =
      session_item_ids
      |> Enum.filter(&Map.has_key?(mem.items, &1))

    if known == [] do
      {:error, "session contains no known items"}
    else
      content_vec =
        known
        |> Enum.take(-@recent_window)
        |> Enum.map(&mem.items[&1])
        |> HRR.bundle()

      predicted_vec = predict_from_transitions(mem, List.last(known))

      {w_c, w_t} = weights(mem, predicted_vec, opts)

      seen = if exclude_seen, do: MapSet.new(session_item_ids), else: MapSet.new()

      scored =
        mem.items
        |> Enum.reject(fn {id, _} -> MapSet.member?(seen, id) end)
        |> Enum.map(fn {id, vec} ->
          content_sim = HRR.similarity(content_vec, vec)

          transition_sim =
            if predicted_vec, do: HRR.similarity(predicted_vec, vec), else: 0.0

          {id, w_c * content_sim + w_t * transition_sim}
        end)
        |> Enum.sort_by(fn {_id, score} -> score end, :desc)
        |> Enum.take(top_k)

      {:ok, scored}
    end
  end

  @doc """
  SNR estimate for the transition bank: `√(dim / n_transitions)`.
  Below ~2.0, transition recall degrades — content recall is unaffected.
  """
  @spec snr_estimate(t()) :: float() | :infinity
  def snr_estimate(%__MODULE__{dim: dim, n_transitions: n}), do: HRR.snr_estimate(dim, n)

  defp predict_from_transitions(%__MODULE__{trans_sin: nil}, _last), do: nil

  defp predict_from_transitions(%__MODULE__{} = mem, last_id) do
    case mem.items[last_id] do
      nil ->
        nil

      last_vec ->
        bank = wrap_angle(Nx.atan2(mem.trans_sin, mem.trans_cos))
        HRR.unbind(bank, last_vec)
    end
  end

  defp weights(_mem, nil, opts), do: {Keyword.get(opts, :content_weight, 1.0), 0.0}

  defp weights(_mem, _predicted, opts) do
    {Keyword.get(opts, :content_weight, @content_weight),
     Keyword.get(opts, :transition_weight, @transition_weight)}
  end

  defp add_or_init(nil, t), do: t
  defp add_or_init(acc, t), do: Nx.add(acc, t)

  defp wrap_angle(t) do
    two_pi = Nx.tensor(2.0 * :math.pi(), type: {:f, 64})
    t |> Nx.remainder(two_pi) |> Nx.add(two_pi) |> Nx.remainder(two_pi)
  end
end
