"""hello_spinnaker.py — Minimal PyNN/sPyNNaker SNN"""
import pyNN.spiNNaker as sim
import numpy as np

sim.setup(timestep=1.0)

# Input population: 10 neurons, spike at t=10,20,30
spike_times = [[10, 20, 30]] + [[] for _ in range(9)]
input_pop = sim.Population(10, sim.SpikeSourceArray(spike_times=spike_times))

# LIF population: 5 neurons
lif_pop = sim.Population(5, sim.IF_curr_exp(
    tau_m=20.0, v_rest=-65.0, v_thresh=-50.0,
    tau_syn_E=5.0, tau_syn_I=5.0
))

# Projection
weights = np.random.randn(10, 5) * 0.01
proj = sim.Projection(input_pop, lif_pop,
                      sim.AllToAllConnector(),
                      sim.StaticSynapse(weight=weights, delay=1))

# Record
lif_pop.record(['spikes'])

# Run
sim.run(50)

# Get data
spikes = lif_pop.get_data(['spikes'])
print("✓ sPyNNaker simulation completed successfully")
print(f"Recorded spike trains: {spikes.segments[0].spiketrains}")

sim.end()
