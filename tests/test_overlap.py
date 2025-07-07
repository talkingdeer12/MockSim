import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_core.logger import EventLogger
from sim_hw.cp import ControlProcessor
from sim_hw.npu import NPU
from sim_hw.dram import DRAM

class OverlapTest(unittest.TestCase):
    def setUp(self):
        self.engine = SimulatorEngine()
        self.logger = EventLogger()
        self.engine.set_logger(self.logger)
        self.mesh_info = {
            "mesh_size": (3,1),
            "router_map": None,
            "npu_coords": {},
            "pe_coords": {},
            "cp_coords": {},
            "dram_coords": {},
        }
        mesh = create_mesh(self.engine, 3, 1, self.mesh_info, buffer_capacity=1)
        self.mesh_info["router_map"] = mesh
        self.npu = NPU(self.engine, "NPU_0", self.mesh_info, buffer_capacity=1)
        self.mesh_info["npu_coords"]["NPU_0"] = (0,0)
        mesh[(0,0)].attach_module(self.npu)
        self.engine.register_module(self.npu)
        self.dram = DRAM(self.engine, "DRAM", self.mesh_info, pipeline_latency=2, buffer_capacity=1)
        self.mesh_info["dram_coords"]["DRAM"] = (1,0)
        mesh[(1,0)].attach_module(self.dram)
        self.engine.register_module(self.dram)
        self.cp = ControlProcessor(self.engine, "CP", self.mesh_info, [], self.dram, npus=[self.npu], buffer_capacity=1)
        self.mesh_info["cp_coords"]["CP"] = (2,0)
        mesh[(2,0)].attach_module(self.cp)
        self.engine.register_module(self.cp)

    def test_overlap_dma_compute(self):
        cfg = {
            "program_cycles": 3,
            "in_size": 16,
            "out_size": 16,
            "dma_in_opcode_cycles": 2,
            "dma_out_opcode_cycles": 2,
            "cmd_opcode_cycles": 3,
        }
        instrs = [
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_CMD", "payload": cfg},
            {"event_type": "NPU_DMA_OUT", "payload": cfg},
        ]
        self.cp.load_program("prog", instrs)
        self.cp.send_event(Event(src=None, dst=self.cp, cycle=1, program="prog", event_type="RUN_PROGRAM"))
        self.engine.run_until_idle(max_tick=500)
        # ensure all phases done
        self.assertTrue(self.cp.npu_dma_in_opcode_done.get("prog"))
        self.assertTrue(self.cp.npu_cmd_opcode_done.get("prog"))
        self.assertTrue(self.cp.npu_dma_out_opcode_done.get("prog"))
        # find overlapping cycles containing dram read, write and npu compute
        cycles = {}
        for entry in self.logger.get_entries():
            cycles.setdefault(entry['cycle'], []).append((entry['module'], str(entry['stage'])))
        overlap_found = False
        for cycle, acts in cycles.items():
            mods = set(m for m, _ in acts)
            if "DRAM" in mods and "NPU_0" in mods:
                overlap_found = True
                break
        self.assertTrue(overlap_found)

if __name__ == "__main__":
    unittest.main()
