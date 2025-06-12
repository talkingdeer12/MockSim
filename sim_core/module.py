class HardwareModule:
    def __init__(self, engine, name, mesh_info):
        self.engine = engine
        self.name = name
        self.mesh_info = mesh_info
    
    def handle_event(self, event):
        pass

    def send_event(self, event):
        self.engine.push_event(event)
