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
            {"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id="A", eaddr=0, iaddr=0)},
            {"event_type": "NPU_CMD", "payload": dict(cfg, stream_id="A")},
            {"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id="B", eaddr=64, iaddr=64)},
            {"event_type": "NPU_CMD", "payload": dict(cfg, stream_id="B")},
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
            {"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id="A", eaddr=0, iaddr=0)},
            {"event_type": "NPU_DMA_OUT", "payload": dict(cfg, stream_id="B", eaddr=64, iaddr=64)},
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
        for i in range(4):
            if random.random() < 0.5:
                instrs.append({"event_type": "NPU_DMA_OUT", "payload": dict(cfg, stream_id=f"S{i}", eaddr=i*64, iaddr=i*64)})
            else:
                instrs.append({"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id=f"S{i}", eaddr=i*64, iaddr=i*64)})

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
            {"event_type": "NPU_DMA_IN", "payload": dict(cfg, stream_id="X", eaddr=0, iaddr=0)},
            {"event_type": "NPU_DMA_OUT", "payload": dict(cfg, stream_id="X", eaddr=64, iaddr=64)},
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
