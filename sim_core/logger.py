import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

class EventLogger:
    """Collects handled events for plotting."""
    def __init__(self):
        self.entries = []  # list of (cycle, module, stage, event_type)

    def log_event(self, cycle, module, stage, event_type):
        self.entries.append({
            'cycle': cycle,
            'module': module,
            'stage': stage,
            'event_type': event_type,
        })

    def get_entries(self):
        return list(self.entries)

    def plot(self, path='timeline.png'):
        if not self.entries:
            print('No events to plot')
            return
        modules = sorted({e['module'] for e in self.entries})
        # determine stage count per module
        stage_counts = {m: max((e['stage'] for e in self.entries if e['module']==m), default=0) + 1
                        for m in modules}
        offsets = {}
        y_labels = []
        y_positions = []
        offset = 0
        for m in modules:
            offsets[m] = offset
            for s in range(stage_counts[m]):
                y_labels.append(f"{m}[{s}]")
                y_positions.append(offset + s)
            offset += stage_counts[m] + 1  # gap between modules
        cmap = plt.get_cmap('tab10')
        fig, ax = plt.subplots(figsize=(10, max(3, len(y_positions)*0.4)))
        max_cycle = max(e['cycle'] for e in self.entries)
        for idx, m in enumerate(modules):
            color = cmap(idx % 10)
            m_entries = [e for e in self.entries if e['module']==m]
            by_stage = defaultdict(list)
            for e in m_entries:
                by_stage[e['stage']].append(e['cycle'])
            for s, cycles in by_stage.items():
                cycles = sorted(cycles)
                start = cycles[0]
                prev = start
                for c in cycles[1:]:
                    if c == prev + 1:
                        prev = c
                    else:
                        ax.plot([start, prev], [offsets[m]+s]*2,
                                color=color, linewidth=3, solid_capstyle='butt')
                        start = c
                        prev = c
                ax.plot([start, prev], [offsets[m]+s]*2,
                        color=color, linewidth=3, solid_capstyle='butt')
                if start == prev:
                    ax.scatter([start], [offsets[m]+s], marker='s', color=color, s=30)
        ax.set_xlabel('Cycle')
        ax.set_xlim(-0.5, max_cycle+0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.legend(bbox_to_anchor=(1.04,1), loc='upper left')
        ax.grid(axis='x', linestyle='--', linewidth=0.5, color='0.8')
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
