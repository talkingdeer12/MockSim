import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.router import Router
from tests.test_traffic.traffic_gen import TrafficGenerator

class TestRouterPipeline(unittest.TestCase):
    def test_packet_delivery_and_credit_management(self):
        x_dim = 4
        y_dim = 4
        packets_per_node = 64
        max_tick = -1
        buffer_capacity = 8 # Increased buffer to reduce stalling

        engine = SimulatorEngine()
        mesh_info = {
            "mesh_size": (x_dim, y_dim),
            "router_map": None,
        }
        # Pass buffer_capacity to create_mesh so routers are initialized with it
        mesh = create_mesh(engine, x_dim, y_dim, mesh_info, buffer_capacity=buffer_capacity)
        mesh_info["router_map"] = mesh

        gens = []
        initial_credit_counts = {} # Store initial credit counts for verification

        for cx in range(x_dim):
            for cy in range(y_dim):
                router = mesh[(cx, cy)]
                name = f"TG_{cx}_{cy}"
                tg = TrafficGenerator(engine, name, mesh_info, (cx, cy), packets_per_node, buffer_capacity)
                router.attach_module(tg)
                engine.register_module(tg)
                tg.start()
                gens.append(tg)

                # Store initial credit counts for this router
                initial_credit_counts[router.name] = [
                    list(vc_credits) for vc_credits in router.credit_counts
                ]

        engine.run_until_idle(max_tick=max_tick)

        # Verify all packets are received
        total_sent = sum(g.sent for g in gens)
        total_received = sum(g.received for g in gens)
        self.assertEqual(total_sent, total_received, "Not all packets were received.")
        self.assertGreater(total_received, 0, "No packets were received.")

        # Verify credit counts are restored to initial state for all routers
        for cx in range(x_dim):
            for cy in range(y_dim):
                router = mesh[(cx, cy)]
                current_credit_counts = router.credit_counts
                expected_credit_counts = initial_credit_counts[router.name]
                
                # Deep comparison of nested lists
                self.assertEqual(
                    current_credit_counts,
                    expected_credit_counts,
                    f"Credit counts for router {router.name} did not return to initial state."
                )

if __name__ == '__main__':
    unittest.main()