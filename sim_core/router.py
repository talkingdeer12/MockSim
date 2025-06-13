from .module import HardwareModule
from .event import Event

class Router(HardwareModule):
    def __init__(self, engine, name, mesh_x, mesh_y, mesh_info, bitwidth=256, pipeline_delay=4, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.x = mesh_x
        self.y = mesh_y
        self.bitwidth = bitwidth
        self.pipeline_delay = pipeline_delay
        self.neighbors = {}
        self.attached_module = None

    def set_neighbors(self, neighbor_dict):
        self.neighbors = neighbor_dict

    def handle_event(self, event):
        if event.event_type == "RETRY_SEND":
            super().handle_event(event)
            return

        dst_coords = event.payload.get("dst_coords", None)
        if dst_coords is None:
            raise ValueError(f"[{self.name}] 이벤트 payload에 dst_coords가 없습니다.")
        if (self.x, self.y) == dst_coords:
            if self.attached_module is None:
                raise RuntimeError(f"[{self.name}] attached_module이 없습니다!")
            new_event = Event(
                src=self,
                dst=self.attached_module,
                cycle=self.engine.current_cycle + 1,
                data_size=event.data_size,
                identifier=event.identifier,
                event_type=event.event_type,
                payload=event.payload,
            )
            self.send_event(new_event)
            return
        next_dir = None
        dx, dy = dst_coords[0] - self.x, dst_coords[1] - self.y
        if dx != 0:
            next_dir = 'E' if dx > 0 else 'W'
        elif dy != 0:
            next_dir = 'S' if dy > 0 else 'N'
        else:
            raise Exception("Routing error: 목적지에 도달했으나 좌표가 다름")
        next_router = self.neighbors[next_dir]
        data_bits = 8 * event.data_size
        xfer_latency = (data_bits // self.bitwidth) + self.pipeline_delay
        new_event = Event(
            src=self,
            dst=next_router,
            cycle=self.engine.current_cycle + xfer_latency,
            data_size=event.data_size,
            identifier=event.identifier,
            event_type=event.event_type,
            payload=event.payload
        )
        self.send_event(new_event)
