import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.logger import EventLogger
from .traffic_gen import TrafficGenerator

class NoRetryTest(unittest.TestCase):
    def test_no_router_retry(self):
        engine = SimulatorEngine()
        logger = EventLogger()
        engine.set_logger(logger)
        mesh_info = {"mesh_size": (4, 4), "router_map": None}
        mesh = create_mesh(engine, 4, 4, mesh_info)
        mesh_info["router_map"] = mesh

        gens = []
        for x in range(4):
            for y in range(4):
                tg = TrafficGenerator(engine, f"TG_{x}_{y}", mesh_info, (x, y), 3)
                mesh[(x, y)].attach_module(tg)
                engine.register_module(tg)
                tg.start()
                gens.append(tg)

        engine.run_until_idle(max_tick=10000)

        for entry in logger.get_entries():
            if entry["event_type"] == "RETRY_SEND" and entry["module"].startswith("Router"):
                self.fail("Router generated RETRY_SEND event")

if __name__ == "__main__":
    unittest.main()
