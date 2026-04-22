# Neuromorphic Portfolio

Work-in-progress port of cross-modal memory-augmented SNN architectures to neuromorphic hardware deployment stacks.

## Background
Research focus: hardware-software co-design for spiking neural networks (SNNs).  
Publication: *Modality-Dependent Memory Mechanisms in Cross-Modal Neuromorphic Computing* — IEEE Computer, Special Issue on Neuromorphic Computing (Minor Revision).

## What's here
- `hello_lava.py` — Basic LIF neuron and feedforward network in Lava (Intel Loihi stack)
- `hello_spinnaker.py` — Minimal PyNN/sPyNNaker SNN simulation
- `docs/` — Hardware mapping notes for cross-modal HGRN → SpiNNaker2

## Current work
- [x] Lava process definitions for visual encoder blocks
- [x] sPyNNaker population/projection setup
- [ ] HGRN cell state mapping to DTCM constraints
- [ ] Hopfield weight matrix compression for synaptic rows
- [ ] Cross-modal routing analysis (visual 25-step vs audio 100-step)

## Stack
Lava-nc | sPyNNaker | PyNN | snntorch | PyTorch
