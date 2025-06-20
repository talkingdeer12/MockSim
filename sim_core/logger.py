import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
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
        handles = []
        for idx, m in enumerate(modules):
            color = cmap(idx % 10)
            handles.append(patches.Patch(color=color, label=m))
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
        ax.legend(handles=handles, bbox_to_anchor=(1.04,1), loc='upper left')
        ax.grid(axis='x', linestyle='--', linewidth=0.5, color='0.8')
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_interactive(self, path='timeline.html'):
        """Create an interactive Gantt chart using Plotly."""
        if not self.entries:
            print('No events to plot')
            return
        try:
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
        except Exception as e:
            print('Plotly not available:', e)
            return

        df = pd.DataFrame(self.entries)
        df['task'] = df['module'] + '[' + df['stage'].astype(str) + ']'
        segments = []
        for task, group in df.groupby('task'):
            cycles = sorted(group['cycle'].unique())
            start = cycles[0]
            prev = start
            for c in cycles[1:]:
                if c == prev + 1:
                    prev = c
                else:
                    segments.append({'task': task, 'start': start, 'finish': prev + 1})
                    start = c
                    prev = c
            segments.append({'task': task, 'start': start, 'finish': prev + 1})

        seg_df = pd.DataFrame(segments)
        fig = px.timeline(seg_df, x_start='start', x_end='finish', y='task', color='task')
        fig.update_yaxes(autorange='reversed')

        events_by_cycle = defaultdict(list)
        for e in self.entries:
            events_by_cycle[e['cycle']].append(f"{e['module']}[{e['stage']}] {e['event_type']}")
        scatter_x = []
        scatter_y = []
        scatter_text = []
        for cycle, infos in events_by_cycle.items():
            scatter_x.append(cycle + 0.5)
            scatter_y.append(-1)
            scatter_text.append('<br>'.join(infos))

        fig.add_trace(go.Scatter(x=scatter_x, y=scatter_y, mode='markers',
                                 marker=dict(opacity=0), showlegend=False,
                                 hoverinfo='text', hovertext=scatter_text))

        fig.update_layout(title='Simulation Timeline', xaxis_title='Cycle', yaxis_title='Module[Stage]')
        fig.write_html(path, include_plotlyjs=True)
