from sim_core.module import PipelineModule
from sim_core.event import Event

class Bank:
    def __init__(self, tRP=1, tRCD=2, tCL=1):
        self.active_row = None
        self.tRP = tRP
        self.tRCD = tRCD
        self.tCL = tCL

    def access(self, row, bursts):
        tBurst = bursts
        if self.active_row == row:
            return self.tCL + tBurst
        self.active_row = row
        return self.tRP + self.tRCD + self.tCL + tBurst

class BankGroup:
    def __init__(self, tRP, tRCD, tCL):
        self.banks = [Bank(tRP, tRCD, tCL) for _ in range(4)]

    def access(self, bank, row, bursts):
        return self.banks[bank].access(row, bursts)

class HBMChannel:
    def __init__(self, tRP, tRCD, tCL):
        self.bank_groups = [BankGroup(tRP, tRCD, tCL) for _ in range(4)]

    def access(self, bg, bank, row, bursts):
        return self.bank_groups[bg].access(bank, row, bursts)

class HBMStack:
    def __init__(self, channels, tRP, tRCD, tCL):
        self.channels = [HBMChannel(tRP, tRCD, tCL) for _ in range(channels)]

    def access(self, ch, bg, bank, row, bursts):
        return self.channels[ch].access(bg, bank, row, bursts)


def decode_eaddr(addr):
    return {
        "stack": (addr >> 35) & 0x1,
        "channel": (addr >> 31) & 0xF,
        "bank_group": (addr >> 29) & 0x3,
        "bank": (addr >> 27) & 0x3,
        "row": (addr >> 11) & 0xFFFF,
        "column": (addr >> 3) & 0xFF,
        "byte_offset": addr & 0x7,
    }

class IOD(PipelineModule):
    """Simplified IOD model with HBM stacks and memory controllers."""

    def __init__(
        self,
        engine,
        name,
        mesh_info,
        num_stacks=2,
        channels_per_stack=16,
        pipeline_latency=5,
        buffer_capacity=4,
        tRP=1,
        tRCD=2,
        tCL=1,
        frequency=1000,
    ):
        buffer_shapes = [[], []]
        super().__init__(engine, name, mesh_info, num_stages=1, buffer_shapes=buffer_shapes, buffer_capacity=buffer_capacity, frequency=frequency)
        self.credit_counts = buffer_capacity
        self.pipeline_latency = pipeline_latency
        self.num_stacks = num_stacks
        self.channels_per_stack = channels_per_stack
        self.stacks = [HBMStack(channels_per_stack, tRP, tRCD, tCL) for _ in range(num_stacks)]
        self.mc_queues = [
            [[ ] for _ in range(channels_per_stack)] for _ in range(num_stacks)
        ]
        self.mc_sched = [
            [False for _ in range(channels_per_stack)] for _ in range(num_stacks)
        ]
        self.set_stage_funcs([self._stage_func])
        self.event_handlers = {}
        self._register_default_handlers()

    def _stage_func(self, mod, data):
        data["remaining"] -= 1
        if data["remaining"] > 0:
            return data, 0, True
        return data, 1, False

    def _schedule_mc(self, st, ch):
        if not self.mc_sched[st][ch]:
            payload = {"stack": st, "channel": ch}
            if self.mc_queues[st][ch]:
                payload["op_type"] = self.mc_queues[st][ch][0]["type"]
            evt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                event_type="IOD_MC",
                payload=payload,
            )
            self.engine.push_event(evt)
            self.mc_sched[st][ch] = True

    def register_handler(self, evt_type, fn):
        self.event_handlers[evt_type] = fn

    def _register_default_handlers(self):
        self.register_handler("DMA_WRITE", self._handle_dma_access)
        self.register_handler("DMA_READ", self._handle_dma_access)
        self.register_handler("IOD_MC", self._handle_mc)
        self.register_handler("RECV_CRED", self._handle_recv_credit)
        self.register_handler("RETRY_SEND", self._handle_retry_send)

    def _handle_retry_send(self, event):
        self.send_event(event.payload["event"])

    def handle_event(self, event):
        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            super().handle_event(event)

    def _handle_recv_credit(self, event):
        self.credit_counts += 1
        self.engine.logger.log(
            f"IOD {self.name}: Received credit. New credit count: {self.credit_counts}"
        )

    def _handle_dma_access(self, event):
        self.engine.logger.log(f"IOD {self.name}: Received {event.event_type} from {event.src.name if event.src else 'N/A'}")
        cred_evt = Event(
            src=self,
            dst=event.src,
            cycle=self.engine.current_cycle + 1,
            event_type="RECV_CRED",
            payload={
                "prev_out_port": event.payload.get("prev_out_port"),
                "prev_out_vc": event.payload.get("prev_out_vc"),
            },
        )
        self.engine.push_event(cred_evt)
        size = event.payload.get("data_size", 0)
        addr = event.payload.get("eaddr", 0)
        while size > 0:
            boundary = ((addr // 2048) + 1) * 2048
            chunk = min(size, boundary - addr if addr < boundary else size)
            info = decode_eaddr(addr)
            st = info["stack"] % self.num_stacks
            ch = info["channel"] % self.channels_per_stack
            op = {
                "type": event.event_type,
                "program": event.program,
                "src_name": self.name,
                "stream_id": event.payload.get("stream_id"),
                "dst_name": event.payload.get("src_name") if event.payload.get("need_reply") else None,
                "stack": st,
                "channel": ch,
                "addr": addr,
                "data_size": chunk,
            }
            self.mc_queues[st][ch].append(op)
            self._schedule_mc(st, ch)
            addr += chunk
            size -= chunk

    def _handle_mc(self, event):
        st = event.payload["stack"]
        ch = event.payload["channel"]
        self.engine.logger.log(f"IOD {self.name}: Handling MC for stack {st}, channel {ch}")
        self.mc_sched[st][ch] = False
        if not self.mc_queues[st][ch]:
            return
        op = self.mc_queues[st][ch][0]
        if "remaining" not in op:
            info = decode_eaddr(op["addr"])
            bursts = (op["data_size"] + 7) // 8
            delay = self.stacks[st].access(
                ch,
                info["bank_group"],
                info["bank"],
                info["row"],
                bursts,
            )
            op["remaining"] = self.pipeline_latency + delay
        op["remaining"] -= 1
        if op["remaining"] > 0:
            self._schedule_mc(st, ch)
            return
        self.mc_queues[st][ch].pop(0)
        self.handle_pipeline_output(op)
        if self.mc_queues[st][ch]:
            self._schedule_mc(st, ch)

    def handle_pipeline_output(self, op):
        self.engine.logger.log(f"IOD {self.name}: MC complete, sending reply for {op['type']}")
        if op["type"] == "DMA_WRITE":
            evt_type = "WRITE_REPLY"
        else:
            evt_type = "DMA_READ_REPLY"
        if op.get("dst_name"):
            dst_name = op["dst_name"]
            coords = (
                self.mesh_info.get("pe_coords", {}).get(dst_name)
                or self.mesh_info.get("npu_coords", {}).get(dst_name)
                or self.mesh_info.get("cp_coords", {}).get(dst_name)
            )
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                program=op["program"],
                event_type=evt_type,
                payload={
                    "dst_coords": coords,
                    "input_port": 0,
                    "vc": 0,
                    "stream_id": op.get("stream_id"),
                    "stack": op.get("stack"),
                    "channel": op.get("channel"),
                    "data_size": op.get("data_size"),
                },
            )
            self.send_event(reply_event)

    def send_event(self, event):
        if event.dst is self:
            self.engine.push_event(event)
            return

        if self.credit_counts > 0:
            self.credit_counts -= 1
            self.engine.push_event(event)
        else:
            retry = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                event_type="RETRY_SEND",
                payload={"event": event},
            )
            self.engine.push_event(retry)

    def get_my_router(self):
        coords = self.mesh_info["iod_coords"][self.name]
        return self.mesh_info["router_map"][coords]
