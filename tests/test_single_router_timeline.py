import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_core.logger import EventLogger
from sim_core.module import HardwareModule

class PacketSource(HardwareModule):
    def __init__(self, engine, name, mesh_info, coords, dest_coords, buffer_capacity=1):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.coords = coords
        self.dest = dest_coords

    def start(self):
        evt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1, event_type="GENERATE")
        self.engine.push_event(evt)

    def get_my_router(self):
        return self.mesh_info["router_map"][self.coords]

    def handle_event(self, event):
        if event.event_type == "GENERATE":
            payload = {
                "dst_coords": self.dest,
                "start_cycle": self.engine.current_cycle,
                "input_port": 0,
                "vc": 0,
            }
            pkt = Event(src=self, dst=self.get_my_router(), cycle=self.engine.current_cycle,
                         data_size=1, event_type="PACKET", payload=payload)
            self.send_event(pkt)
        else:
            super().handle_event(event)

class PacketSink(HardwareModule):
    def __init__(self, engine, name, mesh_info, coords, buffer_capacity=1):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.coords = coords
        self.received = 0

    def get_my_router(self):
        return self.mesh_info["router_map"][self.coords]

    def handle_event(self, event):
        if event.event_type == "PACKET":
            self.received += 1
        else:
            super().handle_event(event)

class SingleRouterPipelineTest(unittest.TestCase):
    def test_single_router_timeline(self):
        engine = SimulatorEngine()
        logger = EventLogger()
        engine.set_logger(logger)
        mesh_info = {"mesh_size": (2,1), "router_map": None}
        mesh = create_mesh(engine, 2, 1, mesh_info, buffer_capacity=1)
        mesh_info["router_map"] = mesh

        src = PacketSource(engine, "SRC", mesh_info, (0,0), (1,0), buffer_capacity=1)
        dst = PacketSink(engine, "DST", mesh_info, (1,0), buffer_capacity=1)

        mesh[(0,0)].attach_module(src)
        mesh[(1,0)].attach_module(dst)
        engine.register_module(src)
        engine.register_module(dst)

        src.start()
        engine.run_until_idle(max_tick=50)

        self.assertEqual(dst.received, 1)
        # Ensure router internal modules produced log entries
        entries = [e for e in logger.get_entries() if e["module"].startswith("Router_0_0")]
        self.assertTrue(len(entries) > 0)

if __name__ == "__main__":
    unittest.main()
