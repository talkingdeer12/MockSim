from sim_core.module import PipelineModule
from sim_core.event import Event

class DRAM(PipelineModule):
    """Simple DRAM model with a configurable pipeline latency."""

    def __init__(self, engine, name, mesh_info, pipeline_latency=5, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, 1, buffer_capacity)
        self.pipeline_latency = pipeline_latency
        self.set_stage_funcs([self._stage_func])

    def _stage_func(self, mod, data):
        data["remaining"] -= 1
        if data["remaining"] > 0:
            return data, 0, True
        return data, 1, False

    def handle_event(self, event):
        if event.event_type in ("DMA_WRITE", "DMA_READ"):
            task = {
                "type": event.event_type,
                "identifier": event.identifier,
                "src_name": event.payload["src_name"],
                "remaining": event.payload.get("task_cycles", self.pipeline_latency),
            }
            if event.payload.get("need_reply"):
                task["dst_name"] = event.payload["src_name"]
            self.add_data(task)
        else:
            super().handle_event(event)

    def handle_pipeline_output(self, task):
        if task["type"] == "DMA_WRITE":
            evt_type = "WRITE_REPLY"
        else:
            evt_type = "DMA_READ_REPLY"
        if "dst_name" in task:
            dst_name = task["dst_name"]
            coords = (self.mesh_info.get("pe_coords", {}).get(dst_name)
                      or self.mesh_info.get("npu_coords", {}).get(dst_name)
                      or self.mesh_info.get("cp_coords", {}).get(dst_name))
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                identifier=task["identifier"],
                event_type=evt_type,
                payload={
                    "dst_coords": coords,
                },
            )
            self.send_event(reply_event)

    def get_my_router(self):
        coords = self.mesh_info["dram_coords"][self.name]
        return self.mesh_info["router_map"][coords]
