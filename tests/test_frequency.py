import unittest
from sim_core.engine import SimulatorEngine
from sim_core.module import HardwareModule
from sim_core.event import Event

class DummyMod(HardwareModule):
    def __init__(self, engine, name, mesh_info, frequency):
        super().__init__(engine, name, mesh_info, frequency=frequency)
        self.arrivals = []
    def handle_event(self, event):
        self.arrivals.append(self.engine.current_cycle)

class FreqTest(unittest.TestCase):
    def test_cross_freq(self):
        eng = SimulatorEngine()
        mesh_info = {}
        a = DummyMod(eng, "A", mesh_info, frequency=1000)
        b = DummyMod(eng, "B", mesh_info, frequency=500)
        eng.register_module(a)
        eng.register_module(b)
        evt = Event(src=a, dst=b, cycle=eng.current_cycle + 3, event_type="PING")
        a.send_event(evt)
        eng.run_until_idle()
        self.assertEqual(b.arrivals[0], 2)

if __name__ == "__main__":
    unittest.main()
