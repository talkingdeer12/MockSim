import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sim_core.engine import SimulatorEngine
from sim_core.mesh import create_mesh
from sim_core.event import Event
from sim_core.logger import EventLogger
from sim_core.module import HardwareModule
from sim_hw.dram import DRAM

class Src(HardwareModule):
    def __init__(self, engine, name, mesh_info, coords):
        super().__init__(engine, name, mesh_info, buffer_capacity=1)
        self.coords = coords
    def start(self):
        # send directly to DRAM so both requests arrive together
        dram = self.mesh_info['dram_module']
        read_evt = Event(src=self, dst=dram, cycle=self.engine.current_cycle+1,
                         data_size=4, event_type='DMA_READ',
                         payload={'src_name': self.name, 'need_reply': True})
        write_evt = Event(src=self, dst=dram, cycle=self.engine.current_cycle+1,
                          data_size=4, event_type='DMA_WRITE',
                          payload={'src_name': self.name, 'need_reply': True})
        self.send_event(read_evt)
        self.send_event(write_evt)
    def handle_event(self, event):
        super().handle_event(event)

class DRAMChannelTest(unittest.TestCase):
    def test_channels_overlap(self):
        engine = SimulatorEngine()
        logger = EventLogger()
        engine.set_logger(logger)
        mesh_info = {"mesh_size": (2,1), "router_map": None, "dram_coords": {}, "cp_coords": {}}
        mesh = create_mesh(engine, 2, 1, mesh_info, buffer_capacity=1)
        mesh_info['router_map'] = mesh
        dram = DRAM(engine, 'DRAM', mesh_info, pipeline_latency=2, buffer_capacity=1)
        mesh_info['dram_coords']['DRAM'] = (1,0)
        mesh_info['dram_module'] = dram
        mesh[(1,0)].attach_module(dram)
        engine.register_module(dram)
        src = Src(engine, 'SRC', mesh_info, (0,0))
        mesh_info['cp_coords']['SRC'] = (0,0)
        mesh[(0,0)].attach_module(src)
        engine.register_module(src)
        src.start()
        engine.run_until_idle(max_tick=50)
        entries = logger.get_entries()
        cycles = {}
        for e in entries:
            if e['module']=='DRAM' and e['event_type']=='PIPE_STAGE':
                cycles.setdefault(e['cycle'], []).append(e['stage'])
        overlap = any('0' in map(str, s) and '1' in map(str, s) for s in cycles.values())
        self.assertTrue(overlap)

if __name__ == '__main__':
    unittest.main()
