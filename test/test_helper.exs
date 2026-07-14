# `:training` tags heavy gradient-descent smoke tests (full-model forward+backward
# JIT). Excluded from the default suite to keep it fast; run them with
# `mix test --include training`.
ExUnit.start(exclude: [:training])
