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
        self.credit_counts = buffer_capacity

    def start(self):
        evt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1,
                    event_type="GENERATE")
        self.engine.push_event(evt)

    def get_my_router(self):
        return self.mesh_info["router_map"][self.coords]

    def handle_event(self, event):
        if event.event_type == "GENERATE":
            if self.sent < self.num_packets and self.credit_counts > 0:
                x_max, y_max = self.mesh_info["mesh_size"]
                dst = (random.randrange(x_max), random.randrange(y_max))
                payload = {
                    "last_hop_src": self,
                    "dst_coords": dst,
                    "start_cycle": self.engine.current_cycle,
                    "input_port": 0,
                    "input_vc": 0,
                }
                pkt = Event(src=self, dst=self.get_my_router(),
                             cycle=self.engine.current_cycle,
                             data_size=1, event_type="PACKET",
                             payload=payload)
                pkt.payload["id"] = f"{self.name}_pkt_{self.sent}"
                self.send_event(pkt)
                self.sent += 1
                self.credit_counts -= 1
            if self.sent < self.num_packets:
                nxt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1,
                            event_type="GENERATE")
                self.engine.push_event(nxt)
        elif event.event_type == "PACKET":
            latency = self.engine.current_cycle - event.payload.get("start_cycle", 0)
            self.latencies.append(latency)
            self.received += 1
            self.engine.logger.log(
                f"TrafficGenerator {self.name}: Received packet {event.payload.get('id')}. Total received: {self.received}"
            )

            prev_out_port = event.payload.get("prev_out_port")
            prev_out_vc = event.payload.get("prev_out_vc")
            last_router = event.src
            cred_evt = Event(
                src=self,
                dst=last_router,
                cycle=self.engine.current_cycle + 1,
                event_type="RECV_CRED",
                payload={"prev_out_port": prev_out_port, "prev_out_vc": prev_out_vc},
            )
            self.engine.push_event(cred_evt)
            self.engine.logger.log(
                f"TrafficGenerator {self.name}: Sent credit to {event.src.name} for port {prev_out_port}, vc {prev_out_vc}"
            )
        elif event.event_type == "RECV_CRED":
            self.credit_counts += 1
            self.engine.logger.log(
                f"TrafficGenerator {self.name}: Received credit. New credit count: {self.credit_counts}"
            )
        else:
            super().handle_event(event)