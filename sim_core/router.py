from .module import HardwareModule
from .event import Event

DIRS = ["LOCAL", "E", "W", "N", "S"]
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N", "LOCAL": "LOCAL"}

class Router(HardwareModule):
    """Simple high-radix router with virtual-channel based 4-stage pipeline."""

    def __init__(self, engine, name, mesh_x, mesh_y, mesh_info,
                 bitwidth=256, pipeline_delay=4,
                 num_ports=5, num_vcs=2, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.x = mesh_x
        self.y = mesh_y
        self.bitwidth = bitwidth
        self.pipeline_delay = pipeline_delay
        self.num_ports = num_ports
        self.num_vcs = num_vcs

        # input buffers per [port][vc]
        self.input_buffers = [[[] for _ in range(num_vcs)]
                              for _ in range(num_ports)]
        self.buffer_counts = [[0 for _ in range(num_vcs)]
                              for _ in range(num_ports)]

        # pipeline state per input port
        self.port_stage = [0 for _ in range(num_ports)]  # 0=RC,1=VA,2=SA,3=ST
        self.port_current_event = [None for _ in range(num_ports)]
        self.port_current_vc = [None for _ in range(num_ports)]
        self.port_scheduled = [False for _ in range(num_ports)]
        self.input_rr = [0 for _ in range(num_ports)]

        # output side resources
        self.output_links = [None for _ in range(num_ports)]  # (dst, dst_port)
        self.output_vc_allocation = [[None for _ in range(num_vcs)]
                                     for _ in range(num_ports)]
        self.vc_rr = [0 for _ in range(num_ports)]
        self.crossbar_busy = [False for _ in range(num_ports)]

        self.neighbors = {}
        self.attached_module = None

    # ------------------------------------------------------------------
    # basic infrastructure overrides
    def _process_event(self, event):
        """Router delays releasing reserved slot until packet leaves."""
        if self.engine.logger:
            stage = event.payload.get('stage_idx', 0) if isinstance(event.payload, dict) else 0
            self.engine.logger.log_event(self.engine.current_cycle, self.name, stage, event.event_type)
        self.handle_event(event)
        # no release here; released when packet is forwarded

    def _reserve_slot(self, event=None):
        port = event.payload.get("input_port", 0) if event else 0
        vc = event.payload.get("vc", 0) if event else 0
        if self.buffer_counts[port][vc] >= self.buffer_capacity:
            return False
        self.buffer_counts[port][vc] += 1
        return True

    def _release_slot(self, event):
        port = event.get("input_port", 0)
        vc = event.get("vc", 0)
        if self.buffer_counts[port][vc] > 0:
            self.buffer_counts[port][vc] -= 1

    def can_accept_event(self, event=None):
        port = event.payload.get("input_port", 0) if event else 0
        vc = event.payload.get("vc", 0) if event else 0
        return self.buffer_counts[port][vc] < self.buffer_capacity

    # ------------------------------------------------------------------
    def set_neighbors(self, neighbor_dict):
        self.neighbors = neighbor_dict
        for d, n in neighbor_dict.items():
            out_port = DIR_INDEX[d]
            in_port = DIR_INDEX[OPPOSITE[d]]
            self.output_links[out_port] = (n, in_port)

    def attach_module(self, mod):
        self.attached_module = mod
        self.output_links[DIR_INDEX["LOCAL"]] = (mod, None)

    # ------------------------------------------------------------------
    def _schedule_port(self, port):
        if not self.port_scheduled[port]:
            evt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1,
                        event_type="PORT_STAGE", payload={"port": port})
            self.engine.push_event(evt)
            self.port_scheduled[port] = True

    def handle_event(self, event):
        if event.event_type == "RETRY_SEND":
            super().handle_event(event)
            return
        if event.event_type == "PORT_STAGE":
            port = event.payload["port"]
            self.port_scheduled[port] = False
            stage = self.port_stage[port]
            if stage == 0:
                self._stage_rc(port)
            elif stage == 1:
                self._stage_va(port)
            elif stage == 2:
                self._stage_sa(port)
            else:
                self._stage_st(port)
            return

        # normal incoming packet
        port = event.payload.get("input_port", 0)
        vc = event.payload.get("vc", 0)
        self.input_buffers[port][vc].append(event)
        self._schedule_port(port)

    # ------------------------------------------------------------------
    # pipeline stage implementations
    def _select_vc(self, port):
        for i in range(self.num_vcs):
            idx = (self.input_rr[port] + i) % self.num_vcs
            if self.input_buffers[port][idx]:
                self.input_rr[port] = (idx + 1) % self.num_vcs
                return idx
        return None

    def _stage_rc(self, port):
        if self.port_current_event[port] is None:
            vc = self._select_vc(port)
            if vc is None:
                return  # nothing to process
            event = self.input_buffers[port][vc][0]
            self.port_current_event[port] = event
            self.port_current_vc[port] = vc
        else:
            event = self.port_current_event[port]
            vc = self.port_current_vc[port]

        dst_coords = event.payload.get("dst_coords")
        if dst_coords is None:
            raise ValueError(f"[{self.name}] dst_coords missing in payload")
        if (self.x, self.y) == tuple(dst_coords):
            direction = "LOCAL"
        else:
            dx = dst_coords[0] - self.x
            dy = dst_coords[1] - self.y
            if dx != 0:
                direction = "E" if dx > 0 else "W"
            elif dy != 0:
                direction = "S" if dy > 0 else "N"
            else:
                direction = "LOCAL"
        out_port = DIR_INDEX[direction]
        event.payload["out_port"] = out_port

        self.port_stage[port] = 1
        self._schedule_port(port)

    def _stage_va(self, port):
        event = self.port_current_event[port]
        if event is None:
            self.port_stage[port] = 0
            return
        out_port = event.payload["out_port"]
        selected_vc = None
        for i in range(self.num_vcs):
            vc_idx = (self.vc_rr[out_port] + i) % self.num_vcs
            if self.output_vc_allocation[out_port][vc_idx] is not None:
                continue
            dest, dest_port = self.output_links[out_port] if out_port < len(self.output_links) else (None, None)
            if isinstance(dest, Router):
                if dest.buffer_counts[dest_port][vc_idx] >= dest.buffer_capacity:
                    continue
            selected_vc = vc_idx
            break
        if selected_vc is None:
            self._schedule_port(port)
            return  # stall
        self.output_vc_allocation[out_port][selected_vc] = port
        self.vc_rr[out_port] = (selected_vc + 1) % self.num_vcs
        event.payload["out_vc"] = selected_vc
        self.port_stage[port] = 2
        self._schedule_port(port)

    def _stage_sa(self, port):
        event = self.port_current_event[port]
        if event is None:
            self.port_stage[port] = 0
            return
        out_port = event.payload["out_port"]
        if self.crossbar_busy[out_port]:
            self._schedule_port(port)
            return  # stall
        self.crossbar_busy[out_port] = True
        self.port_stage[port] = 3
        self._schedule_port(port)

    def _stage_st(self, port):
        event = self.port_current_event[port]
        vc = self.port_current_vc[port]
        out_port = event.payload["out_port"]
        out_vc = event.payload["out_vc"]

        dest, dest_port = self.output_links[out_port]
        # prepare payload for next hop
        event.payload["input_port"] = dest_port if dest_port is not None else 0
        event.payload["vc"] = out_vc

        new_event = Event(src=self, dst=dest,
                          cycle=self.engine.current_cycle + 1,
                          data_size=event.data_size,
                          identifier=event.identifier,
                          event_type=event.event_type,
                          payload=event.payload)
        self.send_event(new_event)

        # remove from buffer and release resources
        self.input_buffers[port][vc].pop(0)
        self._release_slot({"input_port": port, "vc": vc})
        self.output_vc_allocation[out_port][out_vc] = None
        self.crossbar_busy[out_port] = False

        self.port_current_event[port] = None
        self.port_current_vc[port] = None
        self.port_stage[port] = 0
        if any(self.input_buffers[port][i] for i in range(self.num_vcs)):
            self._schedule_port(port)
