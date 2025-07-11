import unittest
import random
from tests.test_traffic.uniform_traffic import run_uniform_traffic_with_mesh
from sim_core.router import Router

class CreditReturnTest(unittest.TestCase):
    def test_credits_restored(self):
        random.seed(1)
        avg, engine, mesh = run_uniform_traffic_with_mesh(x=4, y=4, packets_per_node=20, max_tick=20000)
        for coords, router in mesh.items():
            for out_port in range(router.num_ports):
                dest, _ = router.output_links[out_port]
                if dest is None:
                    continue
                vc_count = router.port_num_vcs[out_port]
                expected = dest.buffer_capacity
                for vc in range(vc_count):
                    credit = router.credit_counts[out_port][vc]
                    self.assertEqual(credit, expected, f"{router.name} p{out_port} vc{vc}")

if __name__ == '__main__':
    unittest.main()
