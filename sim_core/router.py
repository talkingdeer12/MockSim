from .module import PipelineModule
from .event import Event

DIRS = ["LOCAL", "E", "W", "N", "S"]
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N", "LOCAL": "LOCAL"}


class Port(PipelineModule):
    """Input port pipeline handling RC and VA for a router."""

    RC = 0
    VA = 1

    def __init__(self, router, port_idx, num_vcs, buffer_capacity):
        super().__init__(router.engine, f"{router.name}_P{port_idx}",
                         router.mesh_info, 2, buffer_capacity, router.frequency)
        self.router = router
        self.port_idx = port_idx
        self.num_vcs = num_vcs
        self.vc_queues = [[] for _ in range(num_vcs)]
        self.vc_rr = 0
        funcs = [lambda m, d: m._stage_rc(d), lambda m, d: m._stage_va(d)]
        self.set_stage_funcs(funcs)

    # Receive packet from router
    def recv_packet(self, event):
        vc = event.payload.get("vc", 0)
        self.vc_queues[vc].append(event)
        if not any(self.stage_queues):
            self._try_schedule()

    def _try_schedule(self):
        for i in range(self.num_vcs):
            vc_idx = (self.vc_rr + i) % self.num_vcs
            if self.vc_queues[vc_idx]:
                evt = self.vc_queues[vc_idx].pop(0)
                self.vc_rr = (vc_idx + 1) % self.num_vcs
                self.add_data(evt, stage_idx=self.RC)
                break

    def handle_pipeline_output(self, event):
        self.router._add_sa_candidate(self.port_idx, event)
        self._try_schedule()

    # RC and VA are mostly lifted from the old Router implementation
    def _stage_rc(self, event):
        dst_coords = event.payload.get("dst_coords")
        if dst_coords is None:
            raise ValueError(f"[{self.router.name}] dst_coords missing in payload")
        if (self.router.x, self.router.y) == tuple(dst_coords):
            direction = "LOCAL"
        else:
            dx = dst_coords[0] - self.router.x
            dy = dst_coords[1] - self.router.y
            if dx != 0:
                direction = "E" if dx > 0 else "W"
            elif dy != 0:
                direction = "S" if dy > 0 else "N"
            else:
                direction = "LOCAL"
        out_port = DIR_INDEX[direction]
        event.payload["out_port"] = out_port
        return event, self.VA, False

    def _stage_va(self, event):
        out_port = event.payload["out_port"]
        selected_vc = None
        for i in range(self.num_vcs):
            vc_idx = (self.router.vc_rr[out_port] + i) % self.num_vcs
            if self.router.output_vc_allocation[out_port][vc_idx] is not None:
                continue
            credit = self.router.credit_counts[out_port][vc_idx]
            if credit is not None and credit <= 0:
                continue
            selected_vc = vc_idx
            break
        if selected_vc is None:
            return event, self.VA, True
        self.router.output_vc_allocation[out_port][selected_vc] = event
        self.router.vc_rr[out_port] = (selected_vc + 1) % self.num_vcs
        event.payload["out_vc"] = selected_vc
        credit = self.router.credit_counts[out_port][selected_vc]
        if credit is not None:
            self.router.credit_counts[out_port][selected_vc] -= 1
        return event, self.VA + 1, False


class Router(PipelineModule):
    """Simple high-radix router with virtual-channel based 4-stage pipeline."""

    RC = 0
    VA = 1
    SA = 2
    ST = 3
    STAGE_NAMES = {RC: "RC", VA: "VA", SA: "SA", ST: "ST"}

    def __init__(self, engine, name, mesh_x, mesh_y, mesh_info,
                 bitwidth=256, pipeline_delay=4,
                 num_ports=5, num_vcs=2, buffer_capacity=4, frequency=1000):
        super().__init__(engine, name, mesh_info, 4, buffer_capacity, frequency)
        self.x = mesh_x
        self.y = mesh_y
        self.bitwidth = bitwidth
        self.pipeline_delay = pipeline_delay
        self.num_ports = num_ports
        self.num_vcs = num_vcs

        # buffer occupancy per [port][vc]
        self.buffer_counts = [[0 for _ in range(num_vcs)]
                              for _ in range(num_ports)]

        # output side resources
        self.output_links = [None for _ in range(num_ports)]  # (dst, dst_port)
        self.output_vc_allocation = [[None for _ in range(num_vcs)]
                                     for _ in range(num_ports)]
        self.vc_rr = [0 for _ in range(num_ports)]
        self.crossbar_busy = [False for _ in range(num_ports)]
        self.sa_last_grant = -1
        self.credit_counts = [[None for _ in range(num_vcs)]
                              for _ in range(num_ports)]

        self.neighbors = {}
        self.attached_module = None

        # Per-port pipelines
        self.ports = [Port(self, i, num_vcs, buffer_capacity)
                      for i in range(num_ports)]
        self.sa_candidates = [[] for _ in range(num_ports)]
        self.sa_rr_port = 0

        funcs = [
            lambda m, d: (d, self.SA, False),  # unused RC
            lambda m, d: (d, self.SA, False),  # unused VA
            lambda m, d: Router._stage_sa(m, d),
            lambda m, d: Router._stage_st(m, d),
        ]
        self.set_stage_funcs(funcs)

        self.on_stage_funcs = [lambda m: None for _ in range(self.num_stages)]

    # ------------------------------------------------------------------
    # basic infrastructure overrides
    def _process_event(self, event):
        """Router delays releasing reserved slot until packet leaves."""
        if self.engine.logger:
            if isinstance(event.payload, dict):
                stage_idx = event.payload.get('stage_idx', 0)
                if stage_idx == self.RC:
                    port = event.payload.get('input_port', 0)
                else:
                    port = event.payload.get('out_port', 0)
                stage_name = f"P{port}_{self.STAGE_NAMES.get(stage_idx, stage_idx)}"
            else:
                stage_name = '0'
            self.engine.logger.log_event(
                self.engine.current_cycle, self.name, stage_name, event.event_type
            )
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
            if isinstance(n, Router):
                self.credit_counts[out_port] = [n.buffer_capacity for _ in range(self.num_vcs)]
            else:
                self.credit_counts[out_port] = [None for _ in range(self.num_vcs)]

    def attach_module(self, mod):
        self.attached_module = mod
        self.output_links[DIR_INDEX["LOCAL"]] = (mod, None)
        self.credit_counts[DIR_INDEX["LOCAL"]] = [None for _ in range(self.num_vcs)]

    def _add_sa_candidate(self, port_idx, event):
        self.sa_candidates[port_idx].append(event)
        if not self.stage_queues[self.SA]:
            self.stage_queues[self.SA].append(None)
        self._schedule_stage(self.SA)

    # ------------------------------------------------------------------
    def handle_event(self, event):
        if event.event_type == "RECV_CRED":
            port = event.payload.get("port", 0)
            vc = event.payload.get("vc", 0)
            if self.credit_counts[port][vc] is not None:
                self.credit_counts[port][vc] += 1
            return
        if event.event_type in ("RETRY_SEND", "PIPE_STAGE"):
            super().handle_event(event)
            return

        # incoming packet is queued to the appropriate port
        port_idx = event.payload.get("input_port", 0)
        self.ports[port_idx].recv_packet(event)

    # Override to avoid buffer checks for internal stage events
    def _schedule_stage(self, idx):
        if not self.stage_scheduled[idx]:
            evt = Event(src=self,
                        dst=self,
                        cycle=self.engine.current_cycle + 1,
                        event_type="PIPE_STAGE",
                        payload={"stage_idx": idx},
                        priority=-idx)
            self.engine.push_event(evt)
            self.stage_scheduled[idx] = True

    def _on_stage_execute(self, idx):
        func = self.on_stage_funcs[idx]
        if func is not None:
            func(self)

    def _stage_sa(self, _):
        made_progress = False
        used_out = set()
        start = self.sa_rr_port
        for i in range(self.num_ports):
            port_idx = (start + i) % self.num_ports
            if not self.sa_candidates[port_idx]:
                continue
            evt = self.sa_candidates[port_idx][0]
            out_port = evt.payload["out_port"]
            if self.crossbar_busy[out_port] or out_port in used_out:
                continue
            self.sa_candidates[port_idx].pop(0)
            self.crossbar_busy[out_port] = True
            used_out.add(out_port)
            self.stage_queues[self.ST].append(evt)
            self._schedule_stage(self.ST)
            made_progress = True
        self.sa_rr_port = (self.sa_rr_port + 1) % self.num_ports
        more = any(q for q in self.sa_candidates)
        if more:
            return None, self.SA, True
        return None, self.ST + 1, False

    def _stage_st(self, event):
        in_port = event.payload.get("input_port", 0)
        in_vc = event.payload.get("vc", 0)
        out_port = event.payload["out_port"]
        out_vc = event.payload["out_vc"]

        dest, dest_port = self.output_links[out_port]
        event.payload["input_port"] = dest_port if dest_port is not None else 0
        event.payload["vc"] = out_vc

        new_event = Event(src=self, dst=dest,
                          cycle=self.engine.current_cycle + 1,
                          data_size=event.data_size,
                          program=event.program,
                          event_type=event.event_type,
                          payload=event.payload)
        self.send_event(new_event)

        self._release_slot({"input_port": in_port, "vc": in_vc})
        upstream, upstream_port = self.output_links[in_port]
        if isinstance(upstream, Router):
            cred_evt = Event(src=self, dst=upstream,
                             cycle=self.engine.current_cycle,
                             event_type="RECV_CRED",
                             payload={"port": upstream_port, "vc": in_vc})
            upstream._process_event(cred_evt)
        self.output_vc_allocation[out_port][out_vc] = None
        self.crossbar_busy[out_port] = False

        return event, self.ST + 1, False
