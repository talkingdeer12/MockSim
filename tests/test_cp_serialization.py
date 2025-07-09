import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_hw.cp import ControlProcessor
from sim_hw.npu import NPU
from sim_hw.iod import IOD


def setup_env():
    engine = SimulatorEngine()
    mesh_info = {
        "mesh_size": (3, 1),
        "router_map": None,
        "npu_coords": {},
        "cp_coords": {},
        "iod_coords": {},
    }
    mesh = create_mesh(engine, 3, 1, mesh_info, buffer_capacity=1)
    mesh_info["router_map"] = mesh
    npu = NPU(engine, "NPU_0", mesh_info, buffer_capacity=1)
    mesh_info["npu_coords"]["NPU_0"] = (0, 0)
    mesh[(0, 0)].attach_module(npu)
    engine.register_module(npu)
    iod = IOD(engine, "IOD", mesh_info, pipeline_latency=2, channels_per_stack=16, buffer_capacity=1)
    mesh_info["iod_coords"]["IOD"] = (1, 0)
    mesh[(1, 0)].attach_module(iod)
    engine.register_module(iod)
    cp = ControlProcessor(engine, "CP", mesh_info, npus=[npu], buffer_capacity=1)
    mesh_info["cp_coords"]["CP"] = (2, 0)
    mesh[(2, 0)].attach_module(cp)
    engine.register_module(cp)
    return engine, cp


class CPSerializationTest(unittest.TestCase):
    def test_serialized_dma_in(self):
        engine, cp = setup_env()
        cfg = {
            "program_cycles": 3,
            "in_size": 16,
            "out_size": 16,
            "dma_in_opcode_cycles": 2,
            "dma_out_opcode_cycles": 2,
            "cmd_opcode_cycles": 3,
        }
        instrs = [
            {"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id=0, eaddr=0, iaddr=0)},
            {"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id=0, eaddr=64, iaddr=64)},
        ]
        cp.load_program("p", instrs)
        cp.send_event(Event(src=None, dst=cp, cycle=1, program="p", event_type="RUN_PROGRAM"))
        engine.run_until_idle(max_tick=200)
        self.assertTrue(cp.npu_dma_in_opcode_done.get("p"))


if __name__ == "__main__":
    unittest.main()
