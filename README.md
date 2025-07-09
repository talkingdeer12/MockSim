# MockSim

MockSim is a lightweight event-driven hardware simulator written in Python. It hooks into PyTorch modules so that neural network layers can emit hardware events during their forward pass. These events allow the simulator to approximate the behavior of an NPU, memory subsystem and on-chip network.

## Key Hardware Components

Each module simplifies real computer architecture features:

- **Engine** (`sim_core/engine.py`)
  - Maintains a global cycle counter and processes every `Event` in timestamp order.
  - Abstracts hardware behavior through an event-driven model.
- **Router** (`sim_core/router.py`)
  - Models a 2D mesh NoC router with four pipeline stages (RC → VA → SA → ST) and multiple virtual channels.
  - Includes input buffers, a crossbar and VC allocation logic similar to real NoCs.
- **Neural Processing Unit (NPU)** (`sim_hw/npu.py`)
  - A small compute engine with a command pipeline.
  - Reads and writes data via DMA to the IOD memory subsystem and starts the next command once the pipeline is free.
- **Control Processor (CP)** (`sim_hw/cp.py`)
  - Orchestrates DMA and compute phases across multiple NPUs.
  - Maintains per-program scoreboards and supports overlapping work using a `stream_id` for independent work streams.
  - Roughly models a GPU command processor.
- **IOD** (`sim_hw/iod.py`)
  - Simulates a stacked HBM memory controller.
  - Uses simplified timing parameters (tRP, tRCD, tCL) to calculate access latency.
- **Event Logger** (`sim_core/logger.py`)
  - Records which events each module handles every cycle and generates a Plotly timeline of activity.

## Package Layout

- **`sim_core`** – Engine, event class, router and common module base classes.
- **`sim_hw`** – CP, NPU, IOD and other hardware blocks.
- **`sim_ml`** – PyTorch modules and hooks. `llama3_decoder.py` and `llama3_sim_hook.py` are included as examples.

## Running the Example

1. Install PyTorch (the CPU version is sufficient).
   ```bash
   pip install torch torchvision torchaudio
   ```
2. Run the sample script.
   ```bash
   python main.py
   ```
   The script builds a simple mesh, registers the hardware modules and executes a fake Llama3 decoder block. When it finishes you will find `timeline.html` which visualizes each module's pipeline activity cycle by cycle.

## Adding Events

Create an `Event` and send it to the target module.

```python
from sim_core.event import Event

# Example: schedule a DMA input command for the CP
cp.send_event(Event(
    src=None,
    dst=cp,
    cycle=engine.current_cycle + 1,
    program="prog0",
    event_type="NPU_DMA_IN",
    payload={
        "program_cycles": 3,
        "in_size": 16,
        "out_size": 16,
        "dma_in_opcode_cycles": 2,
        "dma_out_opcode_cycles": 2,
        "cmd_opcode_cycles": 3,
        "stream_id": "A",  # use different IDs to overlap work
        "eaddr": 0,
        "iaddr": 0,
    },
))
```

The `cycle` field specifies when the event should be executed. `send_event` automatically retries later if the destination buffer is full.

## Overlapping Work with `stream_id`

The CP can handle multiple streams within a single program. Events with the same `stream_id` maintain order, while different IDs proceed independently. This enables tile-based execution or layer pipelining. See `tests/test_tile_pipeline.py` for an example where each tile uses its own `stream_id` to overlap DMA, compute and write-back phases.

## Logging and Timeline Generation

Use `EventLogger` to visualize event flow.

```python
engine = SimulatorEngine()
logger = EventLogger()
engine.set_logger(logger)
...
engine.run_until_idle()
logger.save_html("timeline.html")
```

Opening the resulting HTML file lets you interactively explore module activity on every cycle.

## Running Tests

A few unit tests are included.

```bash
python -m unittest discover tests
```

The tests cover NPU task flow, tile pipelining and random traffic. Ensure they pass after adding new functionality.
