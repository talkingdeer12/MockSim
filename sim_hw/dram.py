from sim_core.module import HardwareModule, PipelineStage
from sim_core.event import Event

class DRAM(HardwareModule):
    def __init__(self, engine, name, mesh_info):
        super().__init__(engine, name, mesh_info)
        mem_stage = PipelineStage(self, "mem", latency=10)
        self.add_stage(mem_stage)
        self.mem_stage = mem_stage

    def handle_event_module(self, event):
        if event.event_type == "DMA_WRITE":
            print(f"[{self.name}] Write 시작: {event.identifier} size={event.data_size} bytes")
            item = {
                "identifier": event.identifier,
                "pe_name": event.payload["pe_name"],
                "cp_name": event.payload["cp_name"],
                "size": event.data_size,
            }
            self.mem_stage.push(item)
        elif event.event_type == "DMA_READ":
            print(f"[{self.name}] Read 완료: {event.identifier} size={event.data_size} bytes")
        else:
            print(f"[{self.name}] 알 수 없는 이벤트: {event.event_type}")

    def get_my_router(self):
        coords = self.mesh_info["dram_coords"][self.name]
        return self.mesh_info["router_map"][coords]

    def on_pipeline_complete(self, stage_name, item):
        if stage_name == "mem":
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                identifier=item["identifier"],
                event_type="WRITE_REPLY",
                payload={
                    "dst_coords": self.mesh_info["pe_coords"][item["pe_name"]],
                    "cp_name": item["cp_name"],
                },
            )
            self.send_event(reply_event)
