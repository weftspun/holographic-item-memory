# Convert the MovieLens 100K dataset to Parquet using Explorer (hex.pm).
#
#   mix run scripts/ml_to_parquet.exs
#
# Reads the raw `u.*` files from scratch_ml/ml-100k and writes three
# Parquet tables (ratings, items, users) to scratch_ml/parquet.
require Explorer.DataFrame, as: DF

src = Path.join(["scratch_ml", "ml-100k"])
out = Path.join(["scratch_ml", "parquet"])
File.mkdir_p!(out)

rename = fn df, names ->
  df |> DF.rename(Enum.zip(DF.names(df), names) |> Map.new())
end

# The raw files are Latin-1 (ISO-8859-1); transcode to UTF-8 for Polars.
read_utf8 = fn path ->
  path |> File.read!() |> :unicode.characters_to_binary(:latin1)
end

# --- ratings: u.data is tab-separated, no header -----------------------------
genres = ~w(unknown Action Adventure Animation Children Comedy Crime Documentary
            Drama Fantasy Film_Noir Horror Musical Mystery Romance Sci_Fi Thriller
            War Western)

ratings =
  DF.load_csv!(read_utf8.(Path.join(src, "u.data")),
    delimiter: "\t",
    header: false,
    dtypes: [{"column_1", :integer}, {"column_2", :integer}, {"column_3", :integer}, {"column_4", :integer}]
  )
  |> rename.(~w(user_id item_id rating timestamp))

# --- items: u.item is pipe-separated -----------------------------------------
item_cols = ~w(movie_id title release_date video_release_date imdb_url) ++ genres

items =
  DF.load_csv!(read_utf8.(Path.join(src, "u.item")), delimiter: "|", header: false)
  |> rename.(item_cols)

# --- users: u.user is pipe-separated -----------------------------------------
users =
  DF.load_csv!(read_utf8.(Path.join(src, "u.user")), delimiter: "|", header: false)
  |> rename.(~w(user_id age gender occupation zip_code))

for {name, df} <- [ratings: ratings, items: items, users: users] do
  path = Path.join(out, "#{name}.parquet")
  DF.to_parquet!(df, path, compression: {:zstd, 3})
  {rows, cols} = DF.shape(df)
  IO.puts("wrote #{path}  (#{rows} rows x #{cols} cols)")
end
