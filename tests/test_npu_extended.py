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
    mesh_info["pe_coords"]["NPU_0"] = (0, 0)
    mesh[(0, 0)].attach_module(npu)
    engine.register_module(npu)
    dram = DRAM(engine, "DRAM", mesh_info, pipeline_latency=2, buffer_capacity=1)
    mesh_info["dram_coords"]["DRAM"] = (1, 0)
    mesh[(1, 0)].attach_module(dram)
    engine.register_module(dram)
    cp = ControlProcessor(
        engine, "CP", mesh_info, [], dram, npus=[npu], buffer_capacity=1
    )
    mesh_info["cp_coords"]["CP"] = (2, 0)
    mesh[(2, 0)].attach_module(cp)
    engine.register_module(cp)
    return engine, cp


class NPUExtendedTest(unittest.TestCase):
    def test_multiple_dma_in(self):
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
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_SYNC", "payload": {"sync_types": ["dma_in"]}},
            {"event_type": "NPU_CMD", "payload": cfg},
            {"event_type": "NPU_SYNC", "payload": {"sync_types": ["cmd"]}},
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_SYNC", "payload": {"sync_types": ["dma_in"]}},
            {"event_type": "NPU_CMD", "payload": cfg},
            {"event_type": "NPU_SYNC", "payload": {"sync_types": ["cmd"]}},
        ]
        cp.load_program("prog_multi", instrs)
        cp.send_event(
            Event(
                src=None,
                dst=cp,
                cycle=1,
                program="prog_multi",
                event_type="RUN_PROGRAM",
            )
        )
        engine.run_until_idle(max_tick=500)
        self.assertTrue(cp.npu_dma_in_opcode_done.get("prog_multi"))
        self.assertTrue(cp.npu_cmd_opcode_done.get("prog_multi"))
        self.assertTrue(cp.npu_dma_out_opcode_done.get("prog_multi"))
        self.assertEqual(len(engine.event_queue), 0)

    def test_concurrent_dma_in_out(self):
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
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_DMA_OUT", "payload": cfg},
        ]
        cp.load_program("prog_concurrent", instrs)
        cp.send_event(
            Event(
                src=None,
                dst=cp,
                cycle=1,
                program="prog_concurrent",
                event_type="RUN_PROGRAM",
            )
        )
        engine.run_until_idle(max_tick=500)
        self.assertTrue(cp.npu_dma_in_opcode_done.get("prog_concurrent"))
        self.assertTrue(cp.npu_dma_out_opcode_done.get("prog_concurrent"))
        self.assertTrue(cp.npu_cmd_opcode_done.get("prog_concurrent"))
        self.assertEqual(len(engine.event_queue), 0)
        self.assertFalse(cp.active_npu_programs)

    def test_random_dma_sequence(self):
        engine, cp = setup_env()
        cfg = {
            "program_cycles": 3,
            "in_size": 16,
            "out_size": 16,
            "dma_in_opcode_cycles": 2,
            "dma_out_opcode_cycles": 2,
            "cmd_opcode_cycles": 3,
        }

        import random

        random.seed(0)
        instrs = []
        for _ in range(4):
            if random.random() < 0.5:
                instrs.append({"event_type": "NPU_DMA_OUT", "payload": cfg})
                instrs.append(
                    {"event_type": "NPU_SYNC", "payload": {"sync_types": ["dma_out"]}}
                )
            else:
                instrs.append({"event_type": "NPU_DMA_IN", "payload": cfg})
                instrs.append(
                    {"event_type": "NPU_SYNC", "payload": {"sync_types": ["dma_in"]}}
                )

        cp.load_program("prog_random", instrs)
        cp.send_event(
            Event(
                src=None,
                dst=cp,
                cycle=1,
                program="prog_random",
                event_type="RUN_PROGRAM",
            )
        )
        engine.run_until_idle(max_tick=500)
        self.assertTrue(cp.npu_dma_in_opcode_done.get("prog_random", True))
        self.assertTrue(cp.npu_dma_out_opcode_done.get("prog_random", True))
        self.assertEqual(len(engine.event_queue), 0)
        self.assertFalse(cp.active_npu_programs)

    def test_end_with_sync(self):
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
            {"event_type": "NPU_DMA_IN", "payload": cfg},
            {"event_type": "NPU_DMA_OUT", "payload": cfg},
            {"event_type": "NPU_SYNC", "payload": {"sync_types": ["dma_out"]}},
        ]
        cp.load_program("prog_sync_end", instrs)
        cp.send_event(
            Event(
                src=None,
                dst=cp,
                cycle=1,
                program="prog_sync_end",
                event_type="RUN_PROGRAM",
            )
        )
        engine.run_until_idle(max_tick=500)
        self.assertTrue(cp.npu_dma_in_opcode_done.get("prog_sync_end"))
        self.assertTrue(cp.npu_dma_out_opcode_done.get("prog_sync_end"))
        self.assertEqual(len(engine.event_queue), 0)
        self.assertFalse(cp.active_npu_programs)


if __name__ == "__main__":
    unittest.main()
