# MockSim

MockSim is a minimal event-driven hardware simulator implemented in Python. It integrates with PyTorch modules through simple hooks so neural network layers emit simulated hardware events.

## Simulator Architecture

The simulator is composed of a few key building blocks:

* **Engine** (`sim_core/engine.py`)
  * Maintains the global cycle counter and delivers queued `Event` objects in timestamp order.
* **Routers** (`sim_core/router.py`)
  * Form a 2‑D mesh created via `sim_core/mesh.py`.
  * Forward events toward destination coordinates and can host a hardware module.
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
   The script builds a simple mesh, registers hardware modules and executes a fake decoder block. During the forward pass the hooks inject GEMM events which the simulator processes. After completion a `timeline.png` file is generated visualizing which module processed which events each cycle.

## Testing

The repository includes a small unittest suite located in the `tests/` directory. Execute the tests with:
```bash
python -m unittest discover tests
```
Two scenarios are covered:

* **GEMM pipeline** (`tests/test_pipeline.py`) – Validates that a CP can orchestrate GEMM operations across a PE and DRAM, ensuring all DMA and computation events complete.
* **NPU task flow** (`tests/test_npu.py`) – Exercises an NPU performing a simple task requiring DMA transfers in/out of DRAM. The test confirms the CP tracks completion of the task.

