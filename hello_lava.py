"""hello_lava.py — Minimal Lava SNN Proof-of-Concept"""
import numpy as np
from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer
from lava.core.run_conditions import RunSteps
from lava.core.run_configs import Loihi1SimCfg

def main():
    # Random input spikes (5 neurons, 50 timesteps, ~10% sparsity)
    input_spikes = (np.random.rand(5, 50) > 0.9).astype(float)
    
    # Define network
    source = RingBuffer(data=input_spikes)
    dense = Dense(weights=np.random.randn(5, 3) * 0.1)
    lif = LIF(shape=(3,), du=0.1, dv=0.1, vth=0.5)
    
    # Connect
    source.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(lif.in_ports.a_in)
    
    # Run
    source.run(RunSteps(num_steps=50), Loihi1SimCfg())
    print("✓ Lava SNN executed successfully")
    source.stop()

if __name__ == "__main__":
    main()
