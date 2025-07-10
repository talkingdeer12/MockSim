from .module import PipelineModule
from .event import Event
import random

DIRS = ["LOCAL", "E", "W", "N", "S"]
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N", "LOCAL": "LOCAL"}


def select_output_vc(router, out_port):
    """Randomly select an available output VC with credit."""
    vc_count = router.port_num_vcs[out_port]
    choices = [
        vc
        for vc in range(vc_count)
        if router.output_vc_allocation[out_port][vc] is None
        and (
            router.credit_counts[out_port][vc] is None
            or router.credit_counts[out_port][vc] > 0
        )
    ]
    if not choices:
        return None
    return random.choice(choices)


def arbitrate_va(candidates):
    """Randomly pick one VC for each output (port, vc) pair."""
    result = {}
    for pair, vc_list in candidates.items():
        result[pair] = random.choice(vc_list)
    return result


def arbitrate_sa(candidates):
    """Randomly choose one candidate per output port."""
    selected = []
    for out_port, lst in candidates.items():
        selected.append(random.choice(lst))
    return selected


class Buffer(PipelineModule):
    """Single virtual channel buffer handling RC stage."""

    RC = 0

    def __init__(self, port, vc_idx, capacity):
        router = port.router
        name = f"{router.name}_P{port.port_idx}_VC{vc_idx}"
        super().__init__(router.engine, name, router.mesh_info, 1, capacity, router.frequency)
        self.port = port
        self.vc_idx = vc_idx
        self.set_stage_funcs([lambda m, d: m._stage_rc(d)])

    def recv_packet(self, event):
        self.add_data(event, stage_idx=self.RC)

    def handle_pipeline_output(self, _):
        # Output handled directly in _stage_rc
        pass

    def _stage_rc(self, event):
        router = self.port.router
        dst_coords = event.payload.get("dst_coords")
        if dst_coords is None:
            raise ValueError(f"[{router.name}] dst_coords missing in payload")
        if (router.x, router.y) == tuple(dst_coords):
            direction = "LOCAL"
        else:
            dx = dst_coords[0] - router.x
            dy = dst_coords[1] - router.y
            if dx != 0:
                direction = "E" if dx > 0 else "W"
            elif dy != 0:
                direction = "S" if dy > 0 else "N"
            else:
                direction = "LOCAL"
        out_port = DIR_INDEX[direction]
        event.payload["out_port"] = out_port

        q = self.port.va_stage_queues[self.vc_idx]
        if len(q) >= self.port.buffer_capacity:
            return event, self.RC, True
        q.append(event)
        self.port._schedule_va()
        return event, self.RC + 1, False


class Port(PipelineModule):
    """Input port that arbitrates VA across its virtual channels."""

    VA = 0

    def __init__(self, router, port_idx, num_vcs, buffer_capacity):
        super().__init__(router.engine, f"{router.name}_P{port_idx}",
                         router.mesh_info, 1, buffer_capacity, router.frequency)
        self.router = router
        self.port_idx = port_idx
        self.num_vcs = num_vcs
        self.buffer_capacity = buffer_capacity
        self.virtual_channels = [Buffer(self, i, buffer_capacity) for i in range(num_vcs)]
        self.va_stage_queues = [[] for _ in range(num_vcs)]
        self.vc_rr = 0
        self.set_stage_funcs([lambda m, d: m._stage_va(d)])

    def recv_packet(self, event):
        vc = event.payload.get("vc", 0)
        self.virtual_channels[vc].recv_packet(event)

    def _schedule_va(self):
        if not self.stage_queues[self.VA]:
            self.stage_queues[self.VA].append(None)
        self._schedule_stage(self.VA)

    def handle_pipeline_output(self, _):
        pass

    def _stage_va(self, _):
        candidates = {}
        for vc_idx in range(self.num_vcs):
            if not self.va_stage_queues[vc_idx]:
                continue
            pkt = self.va_stage_queues[vc_idx][0]
            out_port = pkt.payload["out_port"]
            out_vc = select_output_vc(self.router, out_port)
            if out_vc is None:
                continue
            pair = (out_port, out_vc)
            candidates.setdefault(pair, []).append(vc_idx)

        chosen = arbitrate_va(candidates)
        progress = False
        for (out_port, out_vc), vc_idx in chosen.items():
            if len(self.router.sa_stage_queues[self.port_idx][vc_idx]) >= self.buffer_capacity:
                continue
            pkt = self.va_stage_queues[vc_idx].pop(0)
            self.router.output_vc_allocation[out_port][out_vc] = pkt
            pkt.payload["out_vc"] = out_vc
            credit = self.router.credit_counts[out_port][out_vc]
            if credit is not None:
                self.router.credit_counts[out_port][out_vc] -= 1
            self.router._add_sa_candidate(self.port_idx, pkt)
            progress = True

        if any(self.va_stage_queues[i] for i in range(self.num_vcs)):
            self._schedule_va()
        return None, self.VA + 1, False


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
        # Per-port VC counts; local port has single VC
        self.port_num_vcs = [1] + [num_vcs] * (num_ports - 1)


        # output side resources
        self.output_links = [None for _ in range(num_ports)]  # (dst, dst_port)
        self.output_vc_allocation = [[None for _ in range(self.port_num_vcs[i])]
                                     for i in range(num_ports)]
        self.vc_rr = [0 for _ in range(num_ports)]
        self.crossbar_busy = [False for _ in range(num_ports)]
        self.credit_counts = [[None for _ in range(self.port_num_vcs[i])]
                              for i in range(num_ports)]

        self.neighbors = {}
        self.attached_module = None

        # Per-port pipelines
        self.ports = [Port(self, i, self.port_num_vcs[i], buffer_capacity)
                      for i in range(num_ports)]
        self.sa_stage_queues = [[[] for _ in range(self.port_num_vcs[i])]
                                for i in range(num_ports)]

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
        """Check downstream VC buffer capacity before accepting packet."""
        if event is None:
            return True
        port = event.payload.get("input_port", 0)
        vc = event.payload.get("vc", 0)
        buf = self.ports[port].virtual_channels[vc]
        count = (
            len(buf.stage_queues[Buffer.RC])
            + len(self.ports[port].va_stage_queues[vc])
            + len(self.sa_stage_queues[port][vc])
        )
        return count < self.buffer_capacity

    def _release_slot(self, payload):
        """Increment credit count when receiving a credit return."""
        port = payload.get("port")
        vc = payload.get("vc", 0)
        if port is None:
            return
        if self.credit_counts[port][vc] is not None:
            self.credit_counts[port][vc] += 1

    # ------------------------------------------------------------------
    def set_neighbors(self, neighbor_dict):
        self.neighbors = neighbor_dict
        for d, n in neighbor_dict.items():
            out_port = DIR_INDEX[d]
            in_port = DIR_INDEX[OPPOSITE[d]]
            self.output_links[out_port] = (n, in_port)
            vc_count = self.port_num_vcs[out_port]
            if isinstance(n, Router):
                self.credit_counts[out_port] = [n.buffer_capacity for _ in range(vc_count)]
            else:
                self.credit_counts[out_port] = [n.buffer_capacity for _ in range(vc_count)]

    def attach_module(self, mod):
        self.attached_module = mod
        self.output_links[DIR_INDEX["LOCAL"]] = (mod, None)
        local_vcs = self.port_num_vcs[DIR_INDEX["LOCAL"]]
        self.credit_counts[DIR_INDEX["LOCAL"]] = [mod.buffer_capacity for _ in range(local_vcs)]

    def _add_sa_candidate(self, port_idx, event):
        vc = event.payload.get("vc", 0)
        self.sa_stage_queues[port_idx][vc].append(event)
        if not self.stage_queues[self.SA]:
            self.stage_queues[self.SA].append(None)
        self._schedule_stage(self.SA)

    # ------------------------------------------------------------------
    def handle_event(self, event):
        if event.event_type == "RECV_CRED":
            self._release_slot(event.payload)
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
        candidates = {}
        for pidx in range(self.num_ports):
            for vc_idx in range(self.port_num_vcs[pidx]):
                if not self.sa_stage_queues[pidx][vc_idx]:
                    continue
                evt = self.sa_stage_queues[pidx][vc_idx][0]
                out_port = evt.payload["out_port"]
                if self.crossbar_busy[out_port]:
                    continue
                candidates.setdefault(out_port, []).append((pidx, vc_idx, evt))

        winners = arbitrate_sa(candidates)
        for pidx, vc_idx, evt in winners:
            out_port = evt.payload["out_port"]
            self.sa_stage_queues[pidx][vc_idx].pop(0)
            self.crossbar_busy[out_port] = True
            self.stage_queues[self.ST].append(evt)
            self._schedule_stage(self.ST)

        more = any(
            self.sa_stage_queues[p][v]
            for p in range(self.num_ports)
            for v in range(self.port_num_vcs[p])
        )
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

        # If sending to local hardware, return credit for this VC immediately
        if not isinstance(dest, Router):
            cred_evt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle,
                event_type="RECV_CRED",
                payload={"port": out_port, "vc": out_vc},
            )
            self._process_event(cred_evt)

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
