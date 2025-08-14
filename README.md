# fin
synthetic data in finance papers

### ABIDES: Towards High-Fidelity Multi-Agent Market Simulation (SIGSIM-PADS 2020)

- **Data / Variables:**
  - Nanosecond-timestamped limit order book events (orders, cancels, trades)
  - L1–L2 depth, best bid/ask quotes
  - Agent portfolios and internal states, detailed message logs
  - Pairwise network latency and jitter matrix
  - Optional historical price/volume injection via a “data oracle”

- **Methods:**
  - Interactive **discrete-event** simulation engine
  - Centralized exchange and order book agent with NASDAQ-like ITCH/OUCH messaging
  - Deterministic per-agent random seeds, computation delay, and network latency modeling
  - Library of background agents (e.g., Zero-Intelligence, heuristic/momentum traders)

- **Purpose of Generated Data:**
  - Produce high-frequency **synthetic market microstructure** data (quotes, trades, depth)
  - Enable experiments on co-location effects, market impact, and latency–accuracy trade-offs
  - Test ML-based trading strategies in controlled **counterfactual** scenarios

- **Key Findings / Contributions:**
  - Realistic intraday patterns reproduced by heterogeneous background agents
  - Quantification of **price impact** from large single orders through “what-if” experiments
  - Full visibility into agent identities and intents for explainable analyses
