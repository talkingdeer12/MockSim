"""Event logging and interactive timeline generation using Plotly."""

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

    def save_html(self, path='timeline.html'):
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

