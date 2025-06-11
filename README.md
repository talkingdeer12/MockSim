# MockSim

MockSim is a tiny event-driven hardware simulator written in Python. It integrates with PyTorch models through simple hooks so that neural network layers trigger simulated hardware events.

## Simulator Architecture

The simulator is built around a few core components:

* **Engine** (`sim_core/engine.py`)
  * Manages the global cycle counter and an event queue.
  * Modules push `Event` objects to the queue and the engine delivers them in timestamp order.
* **Routers** (`sim_core/router.py`)
  * Form a 2‑D mesh network created via `sim_core/mesh.py`.
  * Each router forwards events toward destination coordinates and can host a hardware module.
* **Processing Elements (PEs)** (`sim_hw/pe.py`)
  * Simulate matrix multiply units.
  * Receive control messages from the control processor, perform work for a number of cycles and then interact with DRAM.
* **Control Processor (CP)** (`sim_hw/cp.py`)
  * Coordinates GEMM operations by sending commands to all PEs.
  * Waits for completion messages before finishing a task.
* **DRAM** (`sim_hw/dram.py`)
  * Handles DMA read/write events emitted by PEs.

## Package Layout

* **`sim_core`** – Core simulation utilities: the engine, basic event class, a mesh/routers and the common `HardwareModule` base class.
* **`sim_hw`** – Hardware building blocks used by the simulator: the control processor, processing elements and a DRAM model.
* **`sim_ml`** – Lightweight PyTorch modules and hooks. `llama3_decoder.py` defines a small decoder block. `llama3_sim_hook.py` attaches hooks that translate PyTorch `nn.Linear` layers into GEMM events for the simulator.

## Running the Example

1. Install PyTorch in your environment (CPU-only is sufficient):
   ```bash
   pip install torch
   ```
2. Run the sample script:
   ```bash
   python main.py
   ```
   The script builds a simple mesh, registers the hardware modules and executes a fake decoder block. During the forward pass the hooks inject GEMM events which the simulator processes.

