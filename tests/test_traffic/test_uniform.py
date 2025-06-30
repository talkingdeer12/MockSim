import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .uniform_traffic import run_uniform_traffic

class TrafficTest(unittest.TestCase):
    def test_uniform(self):
        avg = run_uniform_traffic(x=16, y=16, packets_per_node=5, max_tick=10000)
        self.assertGreaterEqual(avg, 0)

if __name__ == '__main__':
    unittest.main()
