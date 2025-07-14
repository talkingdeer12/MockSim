from sim_core.module import HardwareModule
from sim_core.event import Event
import random

class TrafficGenerator(HardwareModule):
    """Generates uniform random traffic and records latency."""
    def __init__(self, engine, name, mesh_info, coords, num_packets=10, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.coords = coords
        self.num_packets = num_packets
        self.sent = 0
        self.received = 0
        self.latencies = []

    def start(self):
        evt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1,
                    event_type="GENERATE")
        self.engine.push_event(evt)

    def get_my_router(self):
        return self.mesh_info["router_map"][self.coords]

    def handle_event(self, event):
        if event.event_type == "GENERATE":
            if self.sent < self.num_packets:
                x_max, y_max = self.mesh_info["mesh_size"]
                dst = (random.randrange(x_max), random.randrange(y_max))
                payload = {
                    "dst_coords": dst,
                    "start_cycle": self.engine.current_cycle,
                    "input_port": 0,
                    "vc": 0,
                }
                pkt = Event(src=self, dst=self.get_my_router(),
                             cycle=self.engine.current_cycle,
                             data_size=1, event_type="PACKET",
                             payload=payload)
                pkt.payload["id"] = f"{self.name}_pkt_{self.sent}"
                self.send_event(pkt)
                self.sent += 1
            if self.sent < self.num_packets:
                nxt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1,
                            event_type="GENERATE")
                self.engine.push_event(nxt)
        elif event.event_type == "PACKET":
            latency = self.engine.current_cycle - event.payload.get("start_cycle", 0)
            self.latencies.append(latency)
            self.received += 1
            self.engine.logger.log(f"TrafficGenerator {self.name}: Received packet {event.payload.get('id')}. Total received: {self.received}")
        else:
            super().handle_event(event)
