# MockSim

MockSim is a lightweight event-driven simulator for experimenting with simple hardware pipelines in Python. It integrates with PyTorch so neural network layers can emit hardware events during a forward pass.

## Overview

A simulation is built from small modules that communicate via timestamped events. The core components are:

- **Engine** – advances the global cycle count and delivers events in order.
- **Routers** – form a 2D mesh to move packets between modules.
- **Processing Elements** – simulate GEMM units that talk to **DRAM**.
- **Control Processor (CP)** – issues commands to PEs or NPUs and waits for completion.
- **Event Logger** – records which module processed which events per cycle and generates an interactive timeline.

## Installation

1. Install PyTorch (CPU‑only is sufficient):
   ```bash
   pip install torch torchvision torchaudio
   ```
2. Clone this repository and run the tests:
   ```bash
   python -m unittest discover tests
   ```
   All tests should pass.

## Running the Example

After installing the dependencies, execute:

```bash
python main.py
```

The script builds a small mesh, registers hardware modules and executes a toy decoder block. When the forward pass runs, the hooks inject GEMM events which the simulator processes. Upon completion a `timeline.html` file is generated showing module activity by cycle.

## NPU Task Flow

The control processor can coordinate NPUs using synchronization flags. Each event may specify `sync_type` and `sync_targets` so later phases wait for earlier ones to finish. Here is a minimal example:

```python
from sim_core.event import Event

# Issue DMA input
cp.send_event(Event(
    src=None,
    dst=cp,
    cycle=1,
    program="prog0",
    event_type="NPU_DMA_IN",
    payload={"program_cycles":3, "dma_in_opcode_cycles":2}
))

# Command waits for DMA_IN on NPU_0
cp.send_event(Event(
    src=None,
    dst=cp,
    cycle=1,
    program="prog0",
    event_type="NPU_CMD",
    payload={"program_cycles":3, "sync_type":0, "sync_targets":["NPU_0"]}
))

engine.run_until_idle()
```

## Additional Example

Run `python -m tests.test_traffic.uniform_traffic` to simulate uniform random traffic on a 16×16 mesh. The script prints the average waiting time of all delivered packets.
