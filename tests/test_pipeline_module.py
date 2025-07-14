import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sim_core.engine import SimulatorEngine
from sim_core.module import PipelineModule, _is_nested_list_empty

class SimplePipeline(PipelineModule):
    def __init__(self, engine, name, mesh_info):
        # 3-stage pipeline, so 4 buffer shapes are needed (input + 2 intermediate + output)
        buffer_shapes = [[2], [2], [2], [2]]
        super().__init__(engine, name, mesh_info, num_stages=3, buffer_shapes=buffer_shapes, buffer_capacity=4)

        self.set_stage_funcs([
            lambda data: (False, data + 10), # Stage 0
            lambda data: (False, data * 2),  # Stage 1
            lambda data: (False, data - 5),  # Stage 2
        ])
        self.output = []

    def handle_pipeline_output(self, data):
        self.output.append(data)

class PipelineModuleTest(unittest.TestCase):
    def test_simple_pipeline_flow(self):
        engine = SimulatorEngine()
        mesh_info = {}
        dut = SimplePipeline(engine, "TestPipe", mesh_info)
        engine.register_module(dut)

        self.assertTrue(dut.add_data(5, indices=[0]))
        self.assertTrue(dut.add_data(10, indices=[0]))

        self.assertEqual(dut.stage_buffers[0][0], [5, 10])
        self.assertTrue(dut.pipeline_scheduled)

        # --- Cycle 1 ---
        engine.tick()
        self.assertEqual(dut.stage_buffers[0][0], [10])
        self.assertEqual(dut.stage_buffers[1][0], [15])
        self.assertTrue(dut.pipeline_scheduled)

        # --- Cycle 2 ---
        engine.tick()
        self.assertEqual(dut.stage_buffers[0][0], [])
        self.assertEqual(dut.stage_buffers[1][0], [20])
        self.assertEqual(dut.stage_buffers[2][0], [30])
        self.assertTrue(dut.pipeline_scheduled)

        # --- Cycle 3 ---
        engine.tick()
        self.assertEqual(dut.stage_buffers[1][0], [])
        self.assertEqual(dut.stage_buffers[2][0], [40])
        self.assertEqual(dut.output, [25])
        self.assertTrue(dut.pipeline_scheduled)

        # --- Cycle 4 ---
        engine.tick()
        self.assertEqual(dut.stage_buffers[2][0], [])
        self.assertEqual(dut.output, [25, 35])
        # After this tick, all data has been processed and moved to the output buffer.
        # The internal pipeline stages (0, 1, 2) are now empty.
        # Therefore, the pipeline should NOT be scheduled for the next cycle.
        self.assertFalse(dut.pipeline_scheduled)

        # --- Final Check ---
        # The event queue should be empty as no new events were scheduled.
        self.assertEqual(len(engine.event_queue), 0)
        self.assertTrue(_is_nested_list_empty(dut.stage_buffers[0]))
        self.assertTrue(_is_nested_list_empty(dut.stage_buffers[1]))
        self.assertTrue(_is_nested_list_empty(dut.stage_buffers[2]))

if __name__ == "__main__":
    unittest.main()