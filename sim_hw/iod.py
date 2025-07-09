from sim_core.module import PipelineModule
from sim_core.event import Event

class Bank:
    def __init__(self):
        self.active_row = None

    def access(self, row):
        delay = 0
        if self.active_row != row:
            if self.active_row is not None:
                delay += 1  # precharge penalty
            self.active_row = row
            delay += 2  # activate penalty
        delay += 1  # column access
        return delay

class BankGroup:
    def __init__(self):
        self.banks = [Bank() for _ in range(4)]

    def access(self, bank, row):
        return self.banks[bank].access(row)

class HBMChannel:
    def __init__(self):
        self.bank_groups = [BankGroup() for _ in range(4)]

    def access(self, bg, bank, row):
        return self.bank_groups[bg].access(bank, row)

class HBMStack:
    def __init__(self, channels):
        self.channels = [HBMChannel() for _ in range(channels)]

    def access(self, ch, bg, bank, row):
        return self.channels[ch].access(bg, bank, row)


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
    ):
        super().__init__(engine, name, mesh_info, 1, buffer_capacity)
        self.pipeline_latency = pipeline_latency
        self.num_stacks = num_stacks
        self.channels_per_stack = channels_per_stack
        self.stacks = [HBMStack(channels_per_stack) for _ in range(num_stacks)]
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
            self.send_event(evt)
            self.mc_sched[st][ch] = True

    def register_handler(self, evt_type, fn):
        self.event_handlers[evt_type] = fn

    def _register_default_handlers(self):
        self.register_handler("DMA_WRITE", self._handle_dma_access)
        self.register_handler("DMA_READ", self._handle_dma_access)
        self.register_handler("IOD_MC", self._handle_mc)

    def handle_event(self, event):
        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            super().handle_event(event)

    def _handle_dma_access(self, event):
        addr = event.payload.get("eaddr", 0)
        info = decode_eaddr(addr)
        st = info["stack"] % self.num_stacks
        ch = info["channel"] % self.channels_per_stack
        delay = self.stacks[st].access(ch, info["bank_group"], info["bank"], info["row"])
        op = {
            "type": event.event_type,
            "program": event.program,
            "src_name": event.payload["src_name"],
            "stream_id": event.payload.get("stream_id"),
            "remaining": self.pipeline_latency + delay,
            "dst_name": event.payload.get("src_name") if event.payload.get("need_reply") else None,
            "stack": st,
            "channel": ch,
        }
        self.mc_queues[st][ch].append(op)
        self._schedule_mc(st, ch)

    def _handle_mc(self, event):
        st = event.payload["stack"]
        ch = event.payload["channel"]
        self.mc_sched[st][ch] = False
        if not self.mc_queues[st][ch]:
            return
        op = self.mc_queues[st][ch][0]
        op["remaining"] -= 1
        if op["remaining"] > 0:
            self._schedule_mc(st, ch)
            return
        self.mc_queues[st][ch].pop(0)
        self.handle_pipeline_output(op)
        if self.mc_queues[st][ch]:
            self._schedule_mc(st, ch)

    def handle_pipeline_output(self, op):
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
                },
            )
            self.send_event(reply_event)

    def get_my_router(self):
        coords = self.mesh_info["iod_coords"][self.name]
        return self.mesh_info["router_map"][coords]
