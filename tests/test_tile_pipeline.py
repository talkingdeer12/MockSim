import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_hw.cp import ControlProcessor
from sim_hw.npu import NPU
from sim_hw.iod import IOD

class TilePipelineTest(unittest.TestCase):
    def test_tile_overlap(self):
        engine = SimulatorEngine()
        mesh_info = {
            "mesh_size": (3, 1),
            "router_map": None,
            "npu_coords": {},
            "cp_coords": {},
            "iod_coords": {},
        }
        mesh = create_mesh(engine, 3, 1, mesh_info, buffer_capacity=8)
        mesh_info["router_map"] = mesh
        npu = NPU(engine, "NPU_0", mesh_info, buffer_capacity=8)
        mesh_info["npu_coords"]["NPU_0"] = (0, 0)
        mesh[(0, 0)].attach_module(npu)
        engine.register_module(npu)
        iod = IOD(engine, "IOD", mesh_info, pipeline_latency=2, channels_per_stack=16, buffer_capacity=2)
        mesh_info["iod_coords"]["IOD"] = (1, 0)
        mesh[(1, 0)].attach_module(iod)
        engine.register_module(iod)
        cp = ControlProcessor(engine, "CP", mesh_info, npus=[npu], buffer_capacity=2)
        mesh_info["cp_coords"]["CP"] = (2, 0)
        mesh[(2, 0)].attach_module(cp)
        engine.register_module(cp)

        # Matrix dimensions and tiling configuration
        M = 4
        N = 4
        K = 4
        tile_y = 2
        tile_x = 2
        tile_k = 2

        cfg = {
            "program_cycles": 3,
            "in_size": tile_y * K * 4,
            "out_size": tile_y * tile_x * 4,
            "dma_in_opcode_cycles": 2,
            "dma_out_opcode_cycles": 2,
            "cmd_opcode_cycles": 3,
        }

        instrs = []
        task = 0
        tiles_y = M // tile_y
        tiles_x = N // tile_x
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                sid = f"T{task}"
                instrs.append({"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id=sid, eaddr=task*64, iaddr=task*64)})
                for tk in range(K // tile_k):
                    instrs.append({"event_type": "NPU_CMD", "payload": dict(cfg, stream_id=sid)})
                instrs.append({"event_type": "NPU_DMA_OUT", "payload": dict(cfg, stream_id=sid, eaddr=task*64, iaddr=task*64)})
                task += 1
                
        cp.load_program("tile_prog", instrs)
        cp.send_event(Event(src=None, dst=cp, cycle=1, program="tile_prog", event_type="RUN_PROGRAM"))
        engine.run_until_idle(max_tick=2000)
        self.assertTrue(cp.npu_dma_in_opcode_done.get("tile_prog"))
        self.assertTrue(cp.npu_cmd_opcode_done.get("tile_prog"))
        self.assertTrue(cp.npu_dma_out_opcode_done.get("tile_prog"))
        self.assertFalse(cp.active_npu_programs)
        # The pipelined execution should finish in well under 800 cycles
        self.assertLess(engine.current_cycle, 800)

        # # Now build a serialized version of the same workload
        # engine2 = SimulatorEngine()
        # mesh_info2 = {
        #     "mesh_size": (3, 1),
        #     "router_map": None,
        #     "npu_coords": {},
        #     "cp_coords": {},
        #     "iod_coords": {},
        # }
        # mesh2 = create_mesh(engine2, 3, 1, mesh_info2, buffer_capacity=8)
        # mesh_info2["router_map"] = mesh2
        # npu2 = NPU(engine2, "NPU_0", mesh_info2, buffer_capacity=8)
        # mesh_info2["npu_coords"]["NPU_0"] = (0, 0)
        # mesh2[(0, 0)].attach_module(npu2)
        # engine2.register_module(npu2)
        # iod2 = IOD(engine2, "IOD", mesh_info2, pipeline_latency=2, channels_per_stack=16, buffer_capacity=8)
        # mesh_info2["iod_coords"]["IOD"] = (1, 0)
        # mesh2[(1, 0)].attach_module(iod2)
        # engine2.register_module(iod2)
        # cp2 = ControlProcessor(engine2, "CP2", mesh_info2, npus=[npu2], buffer_capacity=8)
        # mesh_info2["cp_coords"]["CP2"] = (2, 0)
        # mesh2[(2, 0)].attach_module(cp2)
        # engine2.register_module(cp2)

        # serial_instrs = []
        # for ty in range(tiles_y):
        #     for tx in range(tiles_x):
        #         sid = "S"
        #         serial_instrs.append({"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id=sid, eaddr=task*64, iaddr=task*64)})
        #         for tk in range(K // tile_k):
        #             serial_instrs.append({"event_type": "NPU_CMD", "payload": dict(cfg, stream_id=sid)})
        #         serial_instrs.append({"event_type": "NPU_DMA_OUT", "payload": dict(cfg, stream_id=sid, eaddr=task*64, iaddr=task*64)})

        # cp2.load_program("serial_prog", serial_instrs)
        # cp2.send_event(Event(src=None, dst=cp2, cycle=1, program="serial_prog", event_type="RUN_PROGRAM"))
        # engine2.run_until_idle(max_tick=2000)
        # serial_cycles = engine2.current_cycle

        # self.assertTrue(cp2.npu_dma_in_opcode_done.get("serial_prog"))
        # self.assertTrue(cp2.npu_cmd_opcode_done.get("serial_prog"))
        # self.assertTrue(cp2.npu_dma_out_opcode_done.get("serial_prog"))
        # self.assertLess(engine.current_cycle, serial_cycles)

if __name__ == "__main__":
    unittest.main()
