import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_hw.cp import ControlProcessor
from sim_hw.pe import PE
from sim_hw.dram import DRAM

class PipelineSimTest(unittest.TestCase):
    def test_gemm_flow(self):
        engine = SimulatorEngine()
        mesh_info = {
            "mesh_size": (3,1),
            "router_map": None,
            "pe_coords": {},
            "cp_coords": {},
            "dram_coords": {},
        }
        mesh = create_mesh(engine, 3, 1, mesh_info, buffer_capacity=1)
        mesh_info["router_map"] = mesh
        pe = PE(engine, "PE_0", mesh_info, buffer_capacity=1)
        mesh_info["pe_coords"]["PE_0"] = (0,0)
        mesh[(0,0)].attach_module(pe)
        engine.register_module(pe)
        dram = DRAM(engine, "DRAM", mesh_info, pipeline_latency=2, buffer_capacity=1)
        mesh_info["dram_coords"]["DRAM"] = (1,0)
        mesh[(1,0)].attach_module(dram)
        engine.register_module(dram)
        cp = ControlProcessor(engine, "CP", mesh_info, [pe], dram, buffer_capacity=1)
        mesh_info["cp_coords"]["CP"] = (2,0)
        mesh[(2,0)].attach_module(cp)
        engine.register_module(cp)

        gemm_shape = (2,2,2)
        event = Event(
            src=None,
            dst=cp,
            cycle=1,
            identifier="test_gemm",
            event_type="GEMM",
            payload={
                "gemm_shape": gemm_shape,
                "weights_size": gemm_shape[1]*gemm_shape[2]*4,
                "act_size": gemm_shape[0]*gemm_shape[2]*4,
            },
        )
        cp.send_event(event)
        engine.run_until_idle(max_tick=500)
        self.assertEqual(len(engine.event_queue), 0)
        self.assertFalse(cp.active_gemms)

if __name__ == "__main__":
    unittest.main()
