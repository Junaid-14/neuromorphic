# Mapping M7 Cross-Modal Architecture to SpiNNaker2

## Architecture Decomposition
- **Visual encoder** (34×34 conv): Sparse input, 25 timesteps → fits on 1 SpiNNaker core
- **Audio encoder** (700-dim): Dense input, 100 timesteps → partition across 4 cores by freq band
- **HGRN cell state**: 512-D recurrent state → stored in DTCM (64KB/core limit)
- **Hopfield weights**: 512×512 dense static matrix → synaptic row compression (int8)

## Open Questions
1. Does 100-step audio create multicast congestion vs. 25-step visual?
2. Can HGRN gates (forget/update) be fused into a single population update cycle?

## Next Steps
- Implement HGRN as PyNN populations
- Benchmark routing overhead on sPyNNaker simulator
