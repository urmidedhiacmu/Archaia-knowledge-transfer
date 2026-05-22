import json, os, glob
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────
IMPUTE_FIELDS = [
    'recovered_material', 'recovered_object_type', 'recovered_condition',
    'recovered_period', 'recovered_description'
]
FIELD_LABELS = {
    'recovered_material':    'Material',
    'recovered_object_type': 'Object Type',
    'recovered_condition':   'Condition',
    'recovered_period':      'Period',
    'recovered_description': 'Description',
}
METRICS = {
    'exact_match':      'Exact Match',
    'fuzzy_token_sort': 'Fuzzy Match',
    'semantic_sim':     'Semantic Similarity',
    'top3_match':       'Top-3 Match',
    'bleu':             'BLEU (description only)',
}
COLORS = ['#7D3A10', '#2d6a4f', '#1848A0', '#e9c46a', '#993556']

# ── load all eval jsons ───────────────────────────────────────────────────
def get_eval_files():
    return sorted(glob.glob('*.json') + glob.glob('eval_results*.json'))

def load_eval(path):
    with open(path) as f:
        return json.load(f)

def friendly_name(path):
    n = os.path.basename(path).replace('eval_results','').replace('.json','').strip('_- ')
    return n if n else os.path.basename(path)

eval_files = get_eval_files()
eval_data  = {friendly_name(f): load_eval(f) for f in eval_files}

# ── TAB 1: metrics dashboard ──────────────────────────────────────────────
def make_bar_chart(selected_runs, metric):
    if not selected_runs:
        return go.Figure()
    fig = go.Figure()
    for i, run in enumerate(selected_runs):
        if run not in eval_data: continue
        data   = eval_data[run]['summary']
        fields = list(data.keys())
        vals   = [data[f].get(metric, 0) for f in fields]
        labels = [FIELD_LABELS.get(f, f) for f in fields]
        fig.add_trace(go.Bar(
            name=run, x=labels, y=vals,
            marker_color=COLORS[i % len(COLORS)],
            text=[f'{v:.1%}' for v in vals],
            textposition='outside',
        ))
    fig.update_layout(
        barmode='group',
        yaxis=dict(range=[0,1.15], tickformat='.0%', title='Score', gridcolor='#eee'),
        xaxis_title='Field',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Georgia, serif', size=13),
        legend=dict(orientation='h', y=1.12),
        margin=dict(t=60, b=40, l=40, r=20),
        height=420,
    )
    return fig

def make_radar(selected_runs):
    if not selected_runs:
        return go.Figure()
    cats = ['Exact Match','Fuzzy Match','Semantic Sim','Top-3 Match']
    metric_keys = ['exact_match','fuzzy_token_sort','semantic_sim','top3_match']
    fig = go.Figure()
    for i, run in enumerate(selected_runs):
        if run not in eval_data: continue
        data = eval_data[run]['summary']
        vals = []
        for mk in metric_keys:
            field_vals = [data[f].get(mk, 0) for f in IMPUTE_FIELDS if mk in data.get(f,{})]
            vals.append(np.mean(field_vals) if field_vals else 0)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            name=run,
            line_color=COLORS[i % len(COLORS)],
            fill='toself', fillcolor=COLORS[i % len(COLORS)],
            opacity=0.2,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,1], tickformat='.0%')),
        font=dict(family='Georgia, serif', size=12),
        height=380,
        margin=dict(t=40, b=40),
        paper_bgcolor='white',
    )
    return fig

def make_summary_table(selected_runs):
    if not selected_runs:
        return pd.DataFrame()
    rows = []
    for run in selected_runs:
        if run not in eval_data: continue
        summary = eval_data[run]['summary']
        for field, stats in summary.items():
            row = {'Run': run, 'Field': FIELD_LABELS.get(field, field)}
            for mk, ml in METRICS.items():
                row[ml] = f"{stats.get(mk, 0):.1%}" if mk in stats else '—'
            rows.append(row)
    return pd.DataFrame(rows)

# ── TAB 2: artifact deep dive ─────────────────────────────────────────────
def make_confusion(run, field):
    if not run or run not in eval_data: return go.Figure()
    results = eval_data[run].get('results', {}).get(field, [])
    if not results: return go.Figure()
    gts   = [r['gt'][:35]         for r in results]
    preds = [str(r['pred'])[:35]  for r in results]
    labels = sorted(set(gts) | set(preds))
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((n,n), dtype=int)
    for g, p in zip(gts, preds):
        if g in idx and p in idx:
            mat[idx[g]][idx[p]] += 1
    fig = go.Figure(go.Heatmap(
        z=mat, x=labels, y=labels,
        colorscale='YlOrRd',
        text=mat, texttemplate='%{text}',
    ))
    fig.update_layout(
        xaxis_title='Predicted', yaxis_title='Ground Truth',
        height=max(380, n*28),
        font=dict(family='Georgia, serif', size=11),
        margin=dict(t=20, b=80, l=120, r=20),
        paper_bgcolor='white',
    )
    return fig

def make_scatter(run, field):
    if not run or run not in eval_data: return go.Figure()
    results = eval_data[run].get('results', {}).get(field, [])
    if not results: return go.Figure()
    x     = [r.get('fuzzy_token_sort', 0) for r in results]
    y     = [r.get('semantic_sim', 0)      for r in results]
    em    = [r.get('exact_match', False)   for r in results]
    hover = [f"<b>{r['label']}</b><br>GT: {r['gt'][:50]}<br>PRED: {str(r['pred'])[:50]}" for r in results]
    colors_pt = ['#2d6a4f' if e else '#e76f51' for e in em]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(color=colors_pt, size=9, opacity=0.75, line=dict(width=0.5, color='white')),
        text=hover, hoverinfo='text',
    ))
    fig.add_shape(type='line', x0=0,y0=0,x1=1,y1=1, line=dict(dash='dot', color='#aaa', width=1))
    fig.update_layout(
        xaxis=dict(title='Fuzzy match', range=[0,1.05], gridcolor='#eee'),
        yaxis=dict(title='Semantic similarity', range=[0,1.05], gridcolor='#eee'),
        height=360,
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Georgia, serif', size=12),
        margin=dict(t=20, b=40),
    )
    return fig

def make_error_table(run, field):
    if not run or run not in eval_data: return pd.DataFrame()
    results = eval_data[run].get('results', {}).get(field, [])
    errors  = [r for r in results if not r.get('exact_match', False)]
    rows = []
    for r in errors:
        rows.append({
            'Label':    r['label'],
            'Class':    r.get('item_class',''),
            'Project':  r.get('project','')[:40],
            'GT':       r['gt'][:60],
            'Predicted':str(r['pred'])[:60],
            'Sem Sim':  f"{r.get('semantic_sim',0):.2f}",
            'Fuzzy':    f"{r.get('fuzzy_token_sort',0):.2f}",
        })
    return pd.DataFrame(rows)

# ── TAB 3: per-artifact browser ───────────────────────────────────────────
def get_all_artifacts(run, field, only_errors):
    if not run or run not in eval_data: return [], []
    results = eval_data[run].get('results', {}).get(field, [])
    if only_errors:
        results = [r for r in results if not r.get('exact_match', False)]
    choices = [f"{r['label']} | {r.get('item_class','')} | {r.get('project','')[:30]}" for r in results]
    return choices, results

_artifact_cache = {}

def search_artifacts(run, field, only_errors, query):
    choices, results = get_all_artifacts(run, field, only_errors)
    _artifact_cache['results'] = results
    _artifact_cache['choices'] = choices
    if query:
        filtered = [(c, r) for c, r in zip(choices, results)
                    if query.lower() in c.lower() or query.lower() in r['gt'].lower()]
        choices  = [x[0] for x in filtered]
        _artifact_cache['results'] = [x[1] for x in filtered]
        _artifact_cache['choices'] = choices
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

def show_artifact_card(selection):
    if not selection or 'results' not in _artifact_cache:
        return '<p>Select an artifact above</p>'
    choices = _artifact_cache['choices']
    results = _artifact_cache['results']
    if selection not in choices:
        return '<p>Not found</p>'
    r = results[choices.index(selection)]

    em      = r.get('exact_match', False)
    fuzz    = r.get('fuzzy_token_sort', 0)
    sem     = r.get('semantic_sim', 0)
    top3    = r.get('top3', [])
    bleu    = r.get('bleu', None)
    gt      = r['gt']
    pred    = str(r['pred'])
    field   = list(eval_data[list(eval_data.keys())[0]]['results'].keys())[0]

    status_color = '#2d6a4f' if em else '#e76f51'
    status_text  = 'Exact match' if em else 'No exact match'

    top3_html = ''
    if top3:
        top3_html = '<div style="margin-top:0.5rem"><b>Top-3 candidates:</b> ' + \
            ' · '.join(f'<span style="background:#f5f0e8;padding:2px 6px;border-radius:3px">{c}</span>' for c in top3) + '</div>'

    bleu_html = f'<span style="margin-left:1rem">BLEU: <b>{bleu:.3f}</b></span>' if bleu is not None else ''

    html = f'''
    <div style="font-family: Georgia, serif; padding: 1.2rem; border: 1px solid #ddd; border-radius: 8px; background: white">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem">
        <div>
          <h2 style="margin:0; font-size:1.4rem; color:#18100A">{r["label"]}</h2>
          <p style="margin:0.2rem 0 0; color:#666; font-style:italic">
            {r.get("item_class","")} · {r.get("project","")}
          </p>
        </div>
        <span style="background:{status_color}; color:white; padding:4px 12px; border-radius:4px; font-size:0.85rem">
          {status_text}
        </span>
      </div>

      <table style="width:100%; border-collapse:collapse; font-size:0.92rem">
        <tr style="background:#f5f0e8">
          <th style="padding:0.5rem 1rem; text-align:left; border-bottom:2px solid #ddd; width:120px">Field</th>
          <th style="padding:0.5rem 1rem; text-align:left; border-bottom:2px solid #ddd">Value</th>
        </tr>
        <tr style="border-bottom:1px solid #eee">
          <td style="padding:0.6rem 1rem; color:#7D3A10; font-weight:bold">Ground Truth</td>
          <td style="padding:0.6rem 1rem; color:#1a1a1a">{gt}</td>
        </tr>
        <tr style="background:#fffdf7; border-bottom:1px solid #eee">
          <td style="padding:0.6rem 1rem; color:#2d6a4f; font-weight:bold">Predicted</td>
          <td style="padding:0.6rem 1rem; color:#1a1a1a">{pred}</td>
        </tr>
      </table>

      {top3_html}

      <div style="margin-top:1rem; padding:0.8rem; background:#f9f9f9; border-radius:4px; font-size:0.85rem; color:#444">
        <b>Scores:</b>
        Fuzzy match: <b>{fuzz:.2f}</b>
        <span style="margin-left:1rem">Semantic similarity: <b>{sem:.2f}</b></span>
        {bleu_html}
      </div>
    </div>
    '''
    return html

# ── build app ─────────────────────────────────────────────────────────────
run_names    = list(eval_data.keys())
field_choices = [(FIELD_LABELS[f], f) for f in IMPUTE_FIELDS]

css = '''
.gr-button-primary { background: #7D3A10 !important; }
h1, h2, h3 { font-family: Georgia, serif !important; }
'''

with gr.Blocks(
    title='ArchAIa Imputation Eval',
    theme=gr.themes.Base(font=[gr.themes.GoogleFont('Source Serif 4'), 'Georgia', 'serif']),
    css=css,
) as demo:

    gr.Markdown("""
# ArchAIa — Field Imputation Evaluation Dashboard
**CMU Language Technologies Institute · April 2026**

Evaluation of a multimodal RAG pipeline (DINOv2 + MiniLM + GPT-4o) for filling missing metadata fields
in archaeological artifacts from the OpenContext database.
Compare results across different retrieval settings (top-15 vs top-50 neighbors).
    """)

    with gr.Tabs():

        # ── TAB 1: metrics overview ────────────────────────────────────────
        with gr.Tab('Metrics Overview'):
            with gr.Row():
                run_selector = gr.CheckboxGroup(
                    choices=run_names,
                    value=run_names,
                    label='Select eval runs to compare',
                )
                metric_radio = gr.Radio(
                    choices=list(METRICS.items()),
                    value='exact_match',
                    label='Metric',
                )

            bar_chart = gr.Plot(label='Per-field scores by run')

            with gr.Row():
                radar_chart = gr.Plot(label='Overall radar (mean across fields)')
                with gr.Column():
                    gr.Markdown("### Summary table")
                    summary_table = gr.Dataframe(label='', wrap=True)

            def update_overview(runs, metric):
                return (
                    make_bar_chart(runs, metric),
                    make_radar(runs),
                    make_summary_table(runs),
                )

            run_selector.change(update_overview, [run_selector, metric_radio], [bar_chart, radar_chart, summary_table])
            metric_radio.change(update_overview, [run_selector, metric_radio], [bar_chart, radar_chart, summary_table])

        # ── TAB 2: field deep dive ─────────────────────────────────────────
        with gr.Tab('Field Deep Dive'):
            gr.Markdown("Inspect per-artifact predictions for a specific field and run.")
            with gr.Row():
                dd_run   = gr.Dropdown(choices=run_names, value=run_names[0], label='Eval run')
                dd_field = gr.Dropdown(choices=field_choices, value='recovered_material', label='Field')

            with gr.Row():
                scatter = gr.Plot(label='Fuzzy match vs Semantic similarity (green = exact match)')
                conf_m  = gr.Plot(label='Confusion matrix')

            gr.Markdown("### Errors only")
            error_table = gr.Dataframe(label='Artifacts where exact match failed', wrap=True)

            def update_deepdive(run, field):
                return (
                    make_scatter(run, field),
                    make_confusion(run, field),
                    make_error_table(run, field),
                )

            dd_run.change(update_deepdive,   [dd_run, dd_field], [scatter, conf_m, error_table])
            dd_field.change(update_deepdive, [dd_run, dd_field], [scatter, conf_m, error_table])

        # ── TAB 3: artifact browser ────────────────────────────────────────
        with gr.Tab('Artifact Browser'):
            gr.Markdown("Browse individual artifact predictions. Filter by run, field, and correct/incorrect.")
            with gr.Row():
                ab_run    = gr.Dropdown(choices=run_names, value=run_names[0], label='Eval run')
                ab_field  = gr.Dropdown(choices=field_choices, value='recovered_material', label='Field')
                ab_errors = gr.Checkbox(value=False, label='Show errors only')
                ab_query  = gr.Textbox(label='Search by label or ground truth', placeholder='e.g. Batch 5')

            ab_select = gr.Dropdown(label='Select artifact', choices=[], interactive=True)
            ab_search = gr.Button('Search / Refresh', variant='primary')
            ab_card   = gr.HTML('<p style="color:#aaa">Search for artifacts above</p>')

            ab_search.click(
                search_artifacts,
                inputs=[ab_run, ab_field, ab_errors, ab_query],
                outputs=[ab_select],
            )
            ab_select.change(show_artifact_card, inputs=[ab_select], outputs=[ab_card])

        # ── TAB 4: about ───────────────────────────────────────────────────
        with gr.Tab('About'):
            gr.Markdown("""
## Pipeline Architecture

**Encoding:** Each v4 artifact is encoded as a 1408-dim vector by concatenating:
- DINOv2 ViT-L/14 image embedding (1024-dim) from the artifact's photograph
- all-MiniLM-L6-v2 text embedding (384-dim) from concatenated metadata fields

**Index:** FAISS flat index (IndexFlatIP) built on the 85% train split of v4 (19,215 artifacts).
The remaining 15% (3,392 artifacts) are held out as the eval set.

**Retrieval:** For each eval artifact, the top-N most similar artifacts are retrieved from the index,
filtered to only those that have the target field populated.

**Generation:** GPT-4o receives the artifact image + available fields + up to N retrieved neighbors
as structured JSON context, plus a constrained vocabulary derived from the train split.

## Eval Setup

- 85/15 stratified split of v4 by `(project_label, item_class_label)`
- 100 artifacts sampled per field from the eval split
- Each field evaluated independently — the target field is blanked and predicted
- **Runs compared:** top-15 neighbors vs top-50 neighbors passed to GPT-4o

## Metrics

| Metric | Description |
|---|---|
| Exact Match | Strict case-insensitive string equality |
| Fuzzy Match | Token sort ratio (handles word order variation) |
| Semantic Similarity | Cosine similarity of sentence embeddings |
| Top-3 Match | Ground truth appears in model's top-3 candidates |
| BLEU | N-gram overlap — description field only |

Urmi Dedhia · CMU · April 2026 · ArchAIa Project
            """)

    # load defaults on start
    demo.load(
        lambda: update_overview(run_names, 'exact_match'),
        outputs=[bar_chart, radar_chart, summary_table]
    )
    demo.load(
        lambda: update_deepdive(run_names[0], 'recovered_material'),
        outputs=[scatter, conf_m, error_table]
    )

if __name__ == '__main__':
    demo.launch()
EOF