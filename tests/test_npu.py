import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_hw.cp import ControlProcessor
from sim_hw.npu import NPU
from sim_hw.dram import DRAM

class NPUTest(unittest.TestCase):
    def test_npu_flow(self):
        engine = SimulatorEngine()
        mesh_info = {
            "mesh_size": (3,1),
            "router_map": None,
            "npu_coords": {},
            "pe_coords": {},
            "cp_coords": {},
            "dram_coords": {},
        }
        mesh = create_mesh(engine, 3, 1, mesh_info, buffer_capacity=1)
        mesh_info["router_map"] = mesh
        npu = NPU(engine, "NPU_0", mesh_info, buffer_capacity=1)
        mesh_info["npu_coords"]["NPU_0"] = (0,0)
        mesh_info["pe_coords"]["NPU_0"] = (0,0)
        mesh[(0,0)].attach_module(npu)
        engine.register_module(npu)
        dram = DRAM(engine, "DRAM", mesh_info, pipeline_latency=2, num_channels=4, buffer_capacity=1)
        mesh_info["dram_coords"]["DRAM"] = (1,0)
        mesh[(1,0)].attach_module(dram)
        engine.register_module(dram)
        cp = ControlProcessor(engine, "CP", mesh_info, [], dram, npus=[npu], buffer_capacity=1)
        mesh_info["cp_coords"]["CP"] = (2,0)
        mesh[(2,0)].attach_module(cp)
        engine.register_module(cp)

        program_cfg = {
            "program_cycles": 3,
            "in_size": 16,
            "out_size": 16,
            "dma_in_opcode_cycles": 2,
            "dma_out_opcode_cycles": 2,
            "cmd_opcode_cycles": 3,
        }

        programs = ["prog0"]
        instrs = [
            {"event_type": "NPU_DMA_IN", "payload": dict(program_cfg, stream_id=0)},
            {"event_type": "NPU_CMD", "payload": dict(program_cfg, stream_id=0)},
            {"event_type": "NPU_DMA_OUT", "payload": dict(program_cfg, stream_id=0)},
        ]
        for idx, prog in enumerate(programs):
            cp.load_program(prog, instrs)
            start_evt = Event(src=None, dst=cp, cycle=idx + 1,
                              program=prog, event_type="RUN_PROGRAM")
            cp.send_event(start_evt)

        engine.run_until_idle(max_tick=500)
        for prog in programs:
            self.assertTrue(cp.npu_dma_in_opcode_done.get(prog))
            self.assertTrue(cp.npu_cmd_opcode_done.get(prog))
            self.assertTrue(cp.npu_dma_out_opcode_done.get(prog))
        self.assertEqual(len(engine.event_queue), 0)
        self.assertFalse(cp.active_npu_programs)

if __name__ == "__main__":
    unittest.main()
