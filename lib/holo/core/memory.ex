defmodule Holo.Core.Memory do
  @moduledoc """
  Holographic session memory: zero-shot next-item recall over semantic IDs.

  Items are identified by ResidualFSQ semantic IDs from the
  `multimodal-semantic-ids` pipeline: `#{3}` tokens per item, each an index in
  `0..4095` (`levels = [8,8,8,8]`, `num_quantizers = 3`). The ID **is** the
  item representation — each token becomes a deterministic HRR atom bound to
  its stage role, and the item vector is the bundle of the three bound tokens
  (`item_vector/2`). Because atoms are SHA-256-deterministic, any process can
  reconstruct any item's vector from its ID alone: no training, no stored
  embeddings, and a brand-new asset is recommendable the moment it has an ID.

  Two recall signals combine:

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
  Feed it `{item_id, [t0, t1, t2]}` pairs from wherever you read
  `asset_semantic_id.parquet`.
  """

  alias Holo.Core.HRR

  defstruct dim: 1024,
            items: %{},
            tokens: %{},
            trans_sin: nil,
            trans_cos: nil,
            n_transitions: 0

  @type item_id :: term()
  @type semantic_id :: [non_neg_integer()]
  @type t :: %__MODULE__{}

  # The multimodal-semantic-ids ResidualFSQ contract:
  # levels [8,8,8,8] (8^4 = 4096 codes per stage), num_quantizers 3.
  @tokens_per_item 3
  @codebook_size 4096

  @recent_window 8
  @content_weight 0.4
  @transition_weight 0.6

  @doc "Tokens per semantic ID (= ResidualFSQ quantizer stages)."
  def tokens_per_item, do: @tokens_per_item

  @doc "Codes per token (= `8^4`, the per-stage FSQ codebook)."
  def codebook_size, do: @codebook_size

  @doc "True when `tokens` is a valid semantic ID (3 integers in `0..4095`)."
  @spec valid_id?(term()) :: boolean()
  def valid_id?(tokens) do
    is_list(tokens) and length(tokens) == @tokens_per_item and
      Enum.all?(tokens, &(is_integer(&1) and &1 >= 0 and &1 < @codebook_size))
  end

  @doc """
  Deterministic HRR fact vector for an item.

  The **content key** — the ResidualFSQ semantic ID — is stored as HRR data:
  each token becomes an atom bound to its stage role. Optional text and
  entities then extend the same bundle, using the reference plugin's role
  atoms (`__hrr_role_content__` / `__hrr_role_entity__`) so representations
  stay compatible with the Python holographic memory:

      bundle(bind(atom("sid:q0:t0"), ROLE_Q0),
             bind(atom("sid:q1:t1"), ROLE_Q1),
             bind(atom("sid:q2:t2"), ROLE_Q2),
             bind(encode_text(text), ROLE_CONTENT),   # when text given
             bind(atom(entity), ROLE_ENTITY), ...)    # per entity

  Items sharing coarse (early-stage) tokens land near each other — the
  residual-FSQ coarse-to-fine structure — and items sharing entities or text
  vocabulary are pulled together the same way. Everything is derivable from
  the item's ID + metadata alone: no training, no stored embeddings.

  Options: `:text` (string), `:entities` (list of strings).
  """
  @spec item_vector(semantic_id(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def item_vector(tokens, dim \\ HRR.default_dim(), opts \\ []) do
    unless valid_id?(tokens) do
      raise ArgumentError,
            "expected #{@tokens_per_item} tokens in 0..#{@codebook_size - 1}, got: #{inspect(tokens)}"
    end

    sid_components =
      tokens
      |> Enum.with_index()
      |> Enum.map(fn {token, stage} ->
        HRR.bind(
          HRR.encode_atom("sid:q#{stage}:#{token}", dim),
          HRR.encode_atom("__holo_role_q#{stage}__", dim)
        )
      end)

    HRR.bundle(sid_components ++ meta_components(opts, dim))
  end

  @doc """
  HRR query vector from text and/or entities alone (no semantic ID) — the
  probe side of the fact encoding. Returns `nil` when both are empty.
  """
  @spec query_vector(keyword(), pos_integer()) :: Nx.Tensor.t() | nil
  def query_vector(opts, dim \\ HRR.default_dim()) do
    case meta_components(opts, dim) do
      [] -> nil
      components -> HRR.bundle(components)
    end
  end

  defp meta_components(opts, dim) do
    text = opts[:text]
    entities = opts[:entities] || []

    text_component =
      if is_binary(text) and String.trim(text) != "" do
        [HRR.bind(HRR.encode_text(text, dim), HRR.encode_atom("__hrr_role_content__", dim))]
      else
        []
      end

    entity_components =
      for entity <- entities do
        HRR.bind(
          HRR.encode_atom(String.downcase(entity), dim),
          HRR.encode_atom("__hrr_role_entity__", dim)
        )
      end

    text_component ++ entity_components
  end

  @doc "New empty memory. Options: `:dim` (default #{HRR.default_dim()})."
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    %__MODULE__{dim: Keyword.get(opts, :dim, HRR.default_dim())}
  end

  @doc """
  Add one item by semantic ID (+ optional `:text` / `:entities` metadata).
  Idempotent for the same arguments.
  """
  @spec add_item(t(), item_id(), semantic_id(), keyword()) :: t()
  def add_item(%__MODULE__{} = mem, item_id, tokens, opts \\ []) do
    vector = item_vector(tokens, mem.dim, opts)

    %{
      mem
      | items: Map.put(mem.items, item_id, vector),
        tokens: Map.put(mem.tokens, item_id, tokens)
    }
  end

  @doc """
  Add many items: `{item_id, tokens}` or `{item_id, tokens, meta_opts}`.
  """
  @spec add_items(t(), Enumerable.t()) :: t()
  def add_items(%__MODULE__{} = mem, entries) do
    Enum.reduce(entries, mem, fn
      {id, tokens}, acc -> add_item(acc, id, tokens)
      {id, tokens, opts}, acc -> add_item(acc, id, tokens, opts)
    end)
  end

  @doc """
  Rank catalog items against a text/entity probe (`:text`, `:entities`),
  without any session. Returns `{:ok, [{item_id, score}]}` or an error when
  the probe is empty.
  """
  @spec query(t(), keyword()) :: {:ok, [{item_id(), float()}]} | {:error, String.t()}
  def query(%__MODULE__{} = mem, opts) do
    case query_vector(opts, mem.dim) do
      nil ->
        {:error, "query needs :text and/or :entities"}

      probe ->
        top_k = Keyword.get(opts, :top_k, 5)

        scored =
          mem.items
          |> Enum.map(fn {id, vec} -> {id, HRR.similarity(probe, vec)} end)
          |> Enum.sort_by(fn {_id, score} -> score end, :desc)
          |> Enum.take(top_k)

        {:ok, scored}
    end
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

  @doc """
  Superpose one `a → b` transition into the bank.

  `count` weights the superposition — observing the same transition `n` times
  and observing it once with `count: n` produce the same bank.
  """
  @spec observe_transition(t(), item_id(), item_id(), pos_integer()) :: t()
  def observe_transition(%__MODULE__{} = mem, a, b, count \\ 1)
      when is_integer(count) and count >= 1 do
    with %Nx.Tensor{} = va <- mem.items[a],
         %Nx.Tensor{} = vb <- mem.items[b] do
      pair = HRR.bind(va, vb)
      weight = Nx.tensor(count, type: {:f, 64})

      %{
        mem
        | trans_sin: add_or_init(mem.trans_sin, Nx.multiply(Nx.sin(pair), weight)),
          trans_cos: add_or_init(mem.trans_cos, Nx.multiply(Nx.cos(pair), weight)),
          n_transitions: mem.n_transitions + count
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
