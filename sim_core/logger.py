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
        for idx, m in enumerate(modules):
            color = cmap(idx % 10)
            xs = [e['cycle'] for e in self.entries if e['module']==m]
            ys = [offsets[m]+e['stage'] for e in self.entries if e['module']==m]
            ax.scatter(xs, ys, label=m, color=color, s=10)
        ax.set_xlabel('Cycle')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.legend(bbox_to_anchor=(1.04,1), loc='upper left')
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
