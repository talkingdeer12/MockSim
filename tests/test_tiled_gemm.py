import unittest
import math
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_hw.cp import ControlProcessor
from sim_hw.npu import NPU
from sim_hw.dram import DRAM

class TiledGEMMTest(unittest.TestCase):
    def test_tiled_gemm(self):
        engine = SimulatorEngine()
        mesh_info = {
            "mesh_size": (3,1),
            "router_map": None,
            "npu_coords": {},
            "cp_coords": {},
            "dram_coords": {},
        }
        mesh = create_mesh(engine, 3, 1, mesh_info, buffer_capacity=1)
        mesh_info["router_map"] = mesh

        npu = NPU(engine, "NPU_0", mesh_info, buffer_capacity=1)
        mesh_info["npu_coords"]["NPU_0"] = (0,0)
        mesh[(0,0)].attach_module(npu)
        engine.register_module(npu)

        dram = DRAM(engine, "DRAM", mesh_info, pipeline_latency=2, buffer_capacity=1)
        mesh_info["dram_coords"]["DRAM"] = (1,0)
        mesh[(1,0)].attach_module(dram)
        engine.register_module(dram)

        cp = ControlProcessor(engine, "CP", mesh_info, [], dram, npus=[npu], buffer_capacity=1)
        mesh_info["cp_coords"]["CP"] = (2,0)
        mesh[(2,0)].attach_module(cp)
        engine.register_module(cp)

        M = K = N = 255
        tile_x = tile_y = 16
        tiles_m = math.ceil(M / tile_x)
        tiles_k = math.ceil(K / tile_x)
        tiles_n = math.ceil(N / tile_y)

        program_cfg = {
            "program_cycles": 3,
            "in_size": 16,
            "out_size": 16,
            "dma_in_opcode_cycles": 2,
            "dma_out_opcode_cycles": 2,
            "cmd_opcode_cycles": 3,
        }

        max_tiles = 2
        for i in range(min(tiles_m, max_tiles)):
            for j in range(min(tiles_n, max_tiles)):
                for k in range(min(tiles_k, max_tiles)):
                    prog_name = f"tile_{i}_{j}_{k}"
                    dma_evt = Event(src=None, dst=cp, cycle=1,
                                    program=prog_name, event_type="NPU_DMA_IN",
                                    payload=program_cfg)
                    cp.send_event(dma_evt)

                    cmd_evt = Event(src=None, dst=cp, cycle=1,
                                    program=prog_name, event_type="NPU_CMD",
                                    payload={**program_cfg, "sync_type": 0, "sync_targets": ["NPU_0"]})
                    cp.send_event(cmd_evt)

                    out_evt = Event(src=None, dst=cp, cycle=1,
                                    program=prog_name, event_type="NPU_DMA_OUT",
                                    payload={**program_cfg, "sync_type": 1, "sync_targets": ["NPU_0"]})
                    cp.send_event(out_evt)

                    engine.run_until_idle(max_tick=200000)
                    self.assertTrue(cp.npu_dma_out_opcode_done.get(prog_name))

        self.assertEqual(len(engine.event_queue), 0)

if __name__ == "__main__":
    unittest.main()
