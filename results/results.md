| arch | size | non-emb params | optimizer | lr | seeds | final val loss | tok/s | optimizer time | Muon budget to match AdamW (tokens / time) |
|---|---|---|---|---|---|---|---|---|---|
| modern | M | 10.6M | AdamW | 0.002 | 3 | 5.1822 ± 0.007 | 15.6k | 3% |  |
| modern | M | 10.6M | Muon | 0.02 | 3 | 4.9353 ± 0.006 | 14.6k | 9% | 75% / 81% |
| base | M | 10.6M | AdamW | 0.001 | 3 | 5.4304 ± 0.011 | 17.1k | 3% |  |
| base | M | 10.6M | Muon | 0.01 | 3 | 5.0637 ± 0.015 | 15.9k | 10% | 70% / 75% |
