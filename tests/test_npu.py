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
        mesh[(0,0)].attached_module = npu
        engine.register_module(npu)
        dram = DRAM(engine, "DRAM", mesh_info, pipeline_latency=2, buffer_capacity=1)
        mesh_info["dram_coords"]["DRAM"] = (1,0)
        mesh[(1,0)].attached_module = dram
        engine.register_module(dram)
        cp = ControlProcessor(engine, "CP", mesh_info, [], dram, npus=[npu], buffer_capacity=1)
        mesh_info["cp_coords"]["CP"] = (2,0)
        mesh[(2,0)].attached_module = cp
        engine.register_module(cp)

        event = Event(
            src=None,
            dst=cp,
            cycle=1,
            identifier="npu_task",
            event_type="NPU_TASK",
            payload={
                "task_cycles": 3,
                "in_size": 16,
                "out_size": 16,
                "dram_cycles": 2,
            },
        )
        cp.send_event(event)
        engine.run_until_idle(max_tick=500)
        self.assertEqual(len(engine.event_queue), 0)
        self.assertFalse(cp.active_npu_tasks)

if __name__ == "__main__":
    unittest.main()
