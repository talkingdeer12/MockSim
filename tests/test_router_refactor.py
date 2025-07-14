import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.router import Router
from tests.test_traffic.traffic_gen import TrafficGenerator
import copy

class RouterRefactorTest(unittest.TestCase):
    def test_full_mesh_traffic_and_credit_conservation(self):
        engine = SimulatorEngine()
        mesh_info = {"mesh_size": (2, 2), "router_map": None}
        
        # Create mesh with new Router
        mesh = create_mesh(engine, 2, 2, mesh_info, num_vcs=2, buffer_capacity=2)
        mesh_info["router_map"] = mesh

        # Attach TrafficGenerators to each router
        gens = []
        for x in range(2):
            for y in range(2):
                tg = TrafficGenerator(engine, f"TG_{x}_{y}", mesh_info, (x, y), num_packets=2, buffer_capacity=2)
                mesh[(x, y)].attach_module(tg)
                engine.register_module(tg)
                gens.append(tg)

        # Record initial credit counts for all routers
        initial_credit_counts = {}
        for coords, router in mesh.items():
            initial_credit_counts[coords] = copy.deepcopy(router.credit_counts)

        # Start traffic generation
        for tg in gens:
            tg.start()

        # Run simulation
        engine.run_until_idle(max_tick=1000)

        # Verify all packets sent were received
        total_sent = sum(tg.sent for tg in gens)
        total_received = sum(tg.received for tg in gens)
        self.assertEqual(total_sent, total_received, "All packets should be received")

        # Verify credit conservation
        for coords, router in mesh.items():
            self.assertEqual(router.credit_counts, initial_credit_counts[coords],
                             f"Credit counts for Router {coords} should be conserved")

if __name__ == "__main__":
    unittest.main()