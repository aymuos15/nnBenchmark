**Every experiment requires a config file. No CLI overrides allowed.**

This means:
- Different fold? New config.
- Different parameter? New config.
- 1000 experiments? 1000 configs.

Why? Reproducibility and traceability in the simplest way.

While this does not allow easy reruns of configs (saying running a different fold), it keeps the maintenance simpler and more well defined as well.
