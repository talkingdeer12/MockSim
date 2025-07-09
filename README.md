# MockSim

MockSim is a minimal event-driven hardware simulator implemented in Python. It integrates with PyTorch modules through simple hooks so neural network layers emit simulated hardware events.

## Simulator Architecture

The simulator is composed of a few key building blocks:

* **Engine** (`sim_core/engine.py`)
  * Maintains the global cycle counter and delivers queued `Event` objects in timestamp order.
* **Routers** (`sim_core/router.py`)
  * Form a 2-D mesh created via `sim_core/mesh.py`.
  * Forward events through a 4-stage pipeline using virtual channels.
* **Neural Processing Units (NPUs)** (`sim_hw/npu.py`)
  * Perform compute operations and issue DMA requests to the IOD.
* **Control Processor (CP)** (`sim_hw/cp.py`)
  * Coordinates DMA and compute phases across one or more NPUs.
* **IOD** (`sim_hw/iod.py`)
  * Models stacked HBM memory and services DMA transactions.
* **Event Logger** (`sim_core/logger.py`)
  * Records which event types each module handles every cycle and can plot a
    timeline showing pipeline activity across modules.

## Package Layout

* **`sim_core`** – Core simulation utilities: the engine, event class, router mesh and common module base classes.
* **`sim_hw`** – Hardware blocks used by the simulator: control processor, NPUs and the IOD memory model.
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
Several scenarios are covered:

* **NPU task flow** (`tests/test_npu.py`) – Drives the CP logic for coordinating NPUs.
* Additional stress tests in `tests/test_npu_extended.py`, `tests/test_tile_pipeline.py` and `tests/test_cp_serialization.py`.

## NPU Task Example

The control processor exposes synchronization flags so higher level code can sequence NPU commands.  Each event to the CP may specify:

* `sync_type` – Which previous phase to wait for (`0` = DMA_IN, `1` = CMD, `2` = DMA_OUT).
* `sync_targets` – Iterable of NPU names that must have reported `_DONE` for the given phase before this event will issue.

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

# Compute waits for DMA_IN completion of NPU_0
cp.send_event(Event(
    src=None, dst=cp, cycle=1, program="prog0", event_type="NPU_CMD",
    payload={"program_cycles":3, "in_size":16, "out_size":16,
            "dma_in_opcode_cycles":2, "dma_out_opcode_cycles":2,
            "cmd_opcode_cycles":3, "sync_type":0, "sync_targets":["NPU_0"]}
))

# DMA_OUT waits for the CMD phase to finish
cp.send_event(Event(
    src=None, dst=cp, cycle=1, program="prog0", event_type="NPU_DMA_OUT",
    payload={"program_cycles":3, "in_size":16, "out_size":16,
            "dma_in_opcode_cycles":2, "dma_out_opcode_cycles":2,
            "cmd_opcode_cycles":3, "sync_type":1, "sync_targets":["NPU_0"]}
))

engine.run_until_idle()
```

After the engine idles you can query `cp.npu_dma_in_opcode_done['prog0']`, `cp.npu_cmd_opcode_done['prog0']` and `cp.npu_dma_out_opcode_done['prog0']` to confirm each phase finished.


## Uniform Traffic Example

Run `python -m tests.test_traffic.uniform_traffic` to simulate uniform random traffic on a 16x16 mesh.
The script reports the average waiting time of delivered packets.

