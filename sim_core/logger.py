import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from collections import defaultdict
import json
import plotly.graph_objects as go
from plotly.io import to_html

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

    def plot_gantt(self, path='timeline.html'):
        """Generate an interactive Gantt chart of events."""
        if not self.entries:
            print('No events to plot')
            return

        # prepare event info
        modules = sorted({e['module'] for e in self.entries})
        stage_counts = {m: max((e['stage'] for e in self.entries if e['module']==m), default=0) + 1
                        for m in modules}

        label_offsets = {}
        labels = []
        offset = 0
        for m in modules:
            label_offsets[m] = offset
            for s in range(stage_counts[m]):
                labels.append(f"{m}[{s}]")
            offset += stage_counts[m]

        # group events by module & stage
        by_ms = defaultdict(list)
        events_by_cycle = defaultdict(list)
        for e in self.entries:
            by_ms[(e['module'], e['stage'])].append(e)
            events_by_cycle[e['cycle']].append(
                f"{e['module']}[{e['stage']}] : {e['event_type']}"
            )

        cmap = plt.get_cmap('tab10')
        fig = go.Figure()

        for idx, (m, s) in enumerate(sorted(by_ms.keys())):
            evs = by_ms[(m, s)]
            cycles = sorted({e['cycle'] for e in evs})
            if not cycles:
                continue
            start = cycles[0]
            prev = start
            for c in cycles[1:]:
                if c == prev + 1:
                    prev = c
                else:
                    fig.add_trace(go.Bar(
                        x=[prev - start + 1],
                        y=[f"{m}[{s}]"],
                        base=start,
                        orientation='h',
                        marker_color=mcolors.to_hex(cmap(idx%10)),
                        name=m,
                        showlegend=(s==0)
                    ))
                    start = c
                    prev = c
            fig.add_trace(go.Bar(
                x=[prev - start + 1],
                y=[f"{m}[{s}]"],
                base=start,
                orientation='h',
                marker_color=mcolors.to_hex(cmap(idx%10)),
                name=m,
                showlegend=(s==0)
            ))

        fig.update_layout(
            barmode='overlay',
            xaxis_title='Cycle',
            yaxis=dict(categoryorder='array', categoryarray=list(reversed(labels))),
            height=max(300, 30*len(labels))
        )

        # embed JS to show events overlapping at clicked cycle
        div_id = 'timeline_plot'
        js = f"""
var events = {json.dumps(events_by_cycle)};
var plot = document.getElementById('{div_id}');
var line = {{type:'line', x0:0, x1:0, y0:0, y1:1, xref:'x', yref:'paper', line:{{color:'red',width:1,dash:'dot'}}}};
plot.on('plotly_click', function(data){{
  var cycle = Math.round(data.points[0].x + (data.points[0].base||0));
  line.x0 = cycle; line.x1 = cycle;
  Plotly.relayout(plot, {{shapes:[line]}});
  var msg = events[cycle] ? events[cycle].join('\n') : 'No events';
  alert('Cycle '+cycle+'\n'+msg);
}});
"""

        html = to_html(fig, include_plotlyjs='cdn', full_html=True, div_id=div_id, post_script=js)
        with open(path, 'w') as f:
            f.write(html)
