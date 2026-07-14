# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee

import Config

# FuXi-Linear / RecGPT model runtime runs on EXLA (XLA JIT). `holo.check_gpu`
# reads :backend_app to start the right app; Nx uses EXLA.Backend by default.
# EXLA downloads a precompiled XLA archive (needs `make` + a C compiler, not
# cmake). CPU by default; set XLA_TARGET=cuda12x (+ EXLA :cuda client) for GPU.
config :holographic_item_memory, :backend_app, :exla

config :nx, default_backend: EXLA.Backend
config :nx, default_defn_options: [compiler: EXLA]

config :exla,
  clients: [
    host: [platform: :host]
  ],
  default_client: :host
