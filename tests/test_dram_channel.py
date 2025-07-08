import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_core.module import HardwareModule
from sim_hw.dram import DRAM

class Collector(HardwareModule):
    def __init__(self, engine, name, mesh_info, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.received = []
        self.router = None

    def handle_event(self, event):
        if event.event_type in ("WRITE_REPLY", "DMA_READ_REPLY"):
            self.received.append(event)
        else:
            super().handle_event(event)

class DRAMChannelTest(unittest.TestCase):
    def test_parallel_channels(self):
        engine = SimulatorEngine()
        mesh_info = {
            "mesh_size": (2,1),
            "router_map": None,
            "cp_coords": {"SRC": (0,0)},
            "dram_coords": {"DRAM": (1,0)},
            "pe_coords": {},
            "npu_coords": {},
        }
        mesh = create_mesh(engine, 2, 1, mesh_info, buffer_capacity=4)
        mesh_info["router_map"] = mesh
        src = Collector(engine, "SRC", mesh_info, buffer_capacity=1)
        src.router = mesh[(0,0)]
        mesh[(0,0)].attach_module(src)
        engine.register_module(src)

        dram = DRAM(engine, "DRAM", mesh_info, pipeline_latency=2, num_channels=2, buffer_capacity=1)
        mesh[(1,0)].attach_module(dram)
        engine.register_module(dram)

        for i in range(4):
            evt = Event(
                src=src,
                dst=src.router,
                cycle=1,
                program="p",
                event_type="DMA_WRITE",
                payload={
                    "dst_coords": (1,0),
                    "src_name": src.name,
                    "need_reply": True,
                    "opcode_cycles": 2,
                    "task_id": i,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            src.send_event(evt)

        engine.run_until_idle(max_tick=50)

        self.assertEqual(len(src.received), 4)
        ch_ids = {e.payload.get("channel_id") for e in src.received}
        self.assertTrue(all(cid in (0,1) for cid in ch_ids))

if __name__ == "__main__":
    unittest.main()
