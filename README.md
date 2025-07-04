# MockSim

MockSim is a minimal event-driven hardware simulator implemented in Python. It integrates with PyTorch modules through simple hooks so neural network layers emit simulated hardware events.

## Simulator Architecture

The simulator is composed of a few key building blocks:

* **Engine** (`sim_core/engine.py`)
  * Maintains the global cycle counter and delivers queued `Event` objects in timestamp order.
* **Routers** (`sim_core/router.py`)
  * Form a 2-D mesh created via `sim_core/mesh.py`.
  * Forward events through a 4-stage pipeline using virtual channels.
* **Processing Elements (PEs)** (`sim_hw/pe.py`)
  * Simulate matrix-multiplication units that communicate with DRAM.
* **Control Processor (CP)** (`sim_hw/cp.py`)
  * Coordinates GEMM operations by sending commands to PEs and waits for completion messages.
* **DRAM** (`sim_hw/dram.py`)
  * Handles DMA read/write events emitted by PEs and NPUs.
* **Event Logger** (`sim_core/logger.py`)
  * Records which event types each module handles every cycle and can plot a
    timeline showing pipeline activity across modules.

## Package Layout

* **`sim_core`** – Core simulation utilities: the engine, event class, router mesh and common module base classes.
* **`sim_hw`** – Hardware blocks used by the simulator: control processor, processing elements, an NPU and a DRAM model.
* **`sim_ml`** – Lightweight PyTorch modules and hooks. `llama3_decoder.py` defines a tiny decoder block and `llama3_sim_hook.py` attaches hooks so `nn.Linear` layers trigger GEMM events.

## Running the Example

1. Install PyTorch and related packages (CPU-only is fine):
   ```bash
   pip install torch torchvision torchaudio
   ```
2. Run the sample script:
   ```bash
   python main.py
   ```
   The script builds a simple mesh, registers hardware modules and executes a fake decoder block. During the forward pass the hooks inject GEMM events which the simulator processes. After completion an interactive `timeline.html` is generated using Plotly. This timeline visualizes which module processed which events each cycle and allows hovering over a cycle to inspect all overlapping activity.

## Testing

The repository includes a small unittest suite located in the `tests/` directory. Execute the tests with:
```bash
python -m unittest discover tests
```
Two scenarios are covered:

* **GEMM pipeline** (`tests/test_pipeline.py`) – Validates that a CP can orchestrate GEMM operations across a PE and DRAM, ensuring all DMA and computation events complete.
* **NPU task flow** (`tests/test_npu.py`) – Drives the new CP logic for coordinating NPUs. It issues DMA in, compute and DMA out events that depend on the completion of prior phases.

## NPU Task Example

The control processor exposes dedicated synchronization events so higher level
code can sequence NPU commands. A `*_SYNC` event stalls subsequent CP commands
until the requested phase has completed.  The event payload may include
`sync_targets` listing the NPUs that must report completion before execution
continues.

Below is a minimal example replicating `tests/test_npu.py`:

```python
from sim_core.event import Event

# Schedule the DMA input
cp.send_event(Event(
    src=None, dst=cp, cycle=1, program="prog0", event_type="NPU_DMA_IN",
    payload={"program_cycles":3, "in_size":16, "out_size":16,
            "dma_in_opcode_cycles":2, "dma_out_opcode_cycles":2,
            "cmd_opcode_cycles":3}
))

# Wait for DMA_IN completion of NPU_0
cp.send_event(Event(
    src=None, dst=cp, cycle=1, program="prog0", event_type="NPU_DMA_IN_SYNC",
    payload={"sync_targets": ["NPU_0"]}, priority=-1
))

# Issue the compute phase
cp.send_event(Event(
    src=None, dst=cp, cycle=1, program="prog0", event_type="NPU_CMD",
    payload={"program_cycles":3, "in_size":16, "out_size":16,
            "dma_in_opcode_cycles":2, "dma_out_opcode_cycles":2,
            "cmd_opcode_cycles":3}
))

# Wait for command completion
cp.send_event(Event(
    src=None, dst=cp, cycle=1, program="prog0", event_type="NPU_CMD_SYNC",
    payload={"sync_targets": ["NPU_0"]}, priority=-1
))

# DMA_OUT after compute finishes
cp.send_event(Event(
    src=None, dst=cp, cycle=1, program="prog0", event_type="NPU_DMA_OUT",
    payload={"program_cycles":3, "in_size":16, "out_size":16,
            "dma_in_opcode_cycles":2, "dma_out_opcode_cycles":2,
            "cmd_opcode_cycles":3}
))

# Wait for DMA_OUT completion
cp.send_event(Event(
    src=None, dst=cp, cycle=1, program="prog0", event_type="NPU_DMA_OUT_SYNC",
    payload={"sync_targets": ["NPU_0"]}, priority=-1
))

engine.run_until_idle()
```

After the engine idles you can query `cp.npu_dma_in_opcode_done['prog0']`, `cp.npu_cmd_opcode_done['prog0']` and `cp.npu_dma_out_opcode_done['prog0']` to confirm each phase finished.


## Uniform Traffic Example

Run `python -m tests.test_traffic.uniform_traffic` to simulate uniform random traffic on a 16x16 mesh.
The script reports the average waiting time of delivered packets.

