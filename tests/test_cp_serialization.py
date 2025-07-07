import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_hw.cp import ControlProcessor
from sim_hw.npu import NPU
from sim_hw.dram import DRAM


def setup_env():
    engine = SimulatorEngine()
    mesh_info = {
        "mesh_size": (3, 1),
        "router_map": None,
        "npu_coords": {},
        "pe_coords": {},
        "cp_coords": {},
        "dram_coords": {},
    }
    mesh = create_mesh(engine, 3, 1, mesh_info, buffer_capacity=1)
    mesh_info["router_map"] = mesh
    npu = NPU(engine, "NPU_0", mesh_info, buffer_capacity=1)
    mesh_info["npu_coords"]["NPU_0"] = (0, 0)
    mesh[(0, 0)].attach_module(npu)
    engine.register_module(npu)
    dram = DRAM(engine, "DRAM", mesh_info, pipeline_latency=2, num_channels=1, buffer_capacity=1)
    mesh_info["dram_coords"]["DRAM"] = (1, 0)
    mesh[(1, 0)].attach_module(dram)
    engine.register_module(dram)
    cp = ControlProcessor(engine, "CP", mesh_info, [], dram, npus=[npu], buffer_capacity=1)
    mesh_info["cp_coords"]["CP"] = (2, 0)
    mesh[(2, 0)].attach_module(cp)
    engine.register_module(cp)
    return engine, cp


class CPSerializationTest(unittest.TestCase):
    def test_serialized_dma_in(self):
        engine1, cp1 = setup_env()
        cfg = {
            "program_cycles": 3,
            "in_size": 16,
            "out_size": 16,
            "dma_in_opcode_cycles": 2,
            "dma_out_opcode_cycles": 2,
            "cmd_opcode_cycles": 3,
        }
        instrs1 = [
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_SYNC", "payload": {"sync_types": ["dma_in"]}},
        ]
        cp1.load_program("p1", instrs1)
        cp1.send_event(Event(src=None, dst=cp1, cycle=1, program="p1", event_type="RUN_PROGRAM"))
        engine1.run_until_idle(max_tick=200)
        cycles_auto = engine1.current_cycle

        engine2, cp2 = setup_env()
        instrs2 = [
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_SYNC", "payload": {"sync_types": ["dma_in"]}},
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_SYNC", "payload": {"sync_types": ["dma_in"]}},
        ]
        cp2.load_program("p2", instrs2)
        cp2.send_event(Event(src=None, dst=cp2, cycle=1, program="p2", event_type="RUN_PROGRAM"))
        engine2.run_until_idle(max_tick=200)
        cycles_explicit = engine2.current_cycle

        self.assertLess(abs(cycles_auto - cycles_explicit), 2)
        self.assertTrue(cp1.npu_dma_in_opcode_done.get("p1"))
        self.assertTrue(cp2.npu_dma_in_opcode_done.get("p2"))


if __name__ == "__main__":
    unittest.main()
