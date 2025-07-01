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
    def __init__(self, engine, name, mesh_info, coords, dest_coords, num_packets=4, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.coords = coords
        self.dest = dest_coords
        self.num_packets = num_packets
        self.sent = 0

    def start(self):
        evt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1, event_type="GENERATE")
        self.engine.push_event(evt)

    def get_my_router(self):
        return self.mesh_info["router_map"][self.coords]

    def handle_event(self, event):
        if event.event_type == "GENERATE":
            if self.sent < self.num_packets:
                payload = {
                    "dst_coords": self.dest,
                    "start_cycle": self.engine.current_cycle,
                    "input_port": 0,
                    "vc": 0,
                }
                pkt = Event(src=self, dst=self.get_my_router(), cycle=self.engine.current_cycle,
                             data_size=1, event_type="PACKET", payload=payload)
                self.send_event(pkt)
                self.sent += 1
            if self.sent < self.num_packets:
                nxt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1, event_type="GENERATE")
                self.engine.push_event(nxt)
        else:
            super().handle_event(event)

class PacketSink(HardwareModule):
    def __init__(self, engine, name, mesh_info, coords, buffer_capacity=4):
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

class TimelineTest(unittest.TestCase):
    def test_timeline_logging(self):
        engine = SimulatorEngine()
        logger = EventLogger()
        engine.set_logger(logger)
        mesh_info = {"mesh_size": (4,1), "router_map": None}
        mesh = create_mesh(engine, 4, 1, mesh_info, buffer_capacity=1)
        mesh_info["router_map"] = mesh

        src = PacketSource(engine, "SRC", mesh_info, (0,0), (3,0), num_packets=3, buffer_capacity=1)
        dst = PacketSink(engine, "DST", mesh_info, (3,0), buffer_capacity=1)

        mesh[(0,0)].attach_module(src)
        mesh[(3,0)].attach_module(dst)
        engine.register_module(src)
        engine.register_module(dst)

        src.start()
        engine.run_until_idle(max_tick=200)

        self.assertEqual(dst.received, src.sent)
        self.assertGreater(len(logger.get_entries()), 0)

if __name__ == "__main__":
    unittest.main()
