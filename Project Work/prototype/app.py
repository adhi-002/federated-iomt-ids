"""Read-only demonstration viewer; train with pipeline.py first."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Federated IoMT prototype', layout='wide')
st.title('Federated intrusion detection prototype')
path = Path(st.sidebar.text_input('Results folder', str(Path(__file__).parent/'results'/'synthetic')))
if not (path/'report.json').is_file():
    st.info('Run pipeline.py first, then enter its results folder here.')
    st.stop()
r = json.loads((path/'report.json').read_text(encoding='utf-8'))
if 'SYNTHETIC' in r['source']: st.warning(r['source'])
else: st.info(r['source'])
st.write('Three hospitals train local MLPs. The server combines parameters using FedAvg. A confidence threshold flags unfamiliar traffic.')
st.caption('Local simulation in one process; no real hospitals or network transport. Threshold uses known validation samples only.')
st.subheader('Hospital training data')
st.dataframe(pd.DataFrame([{'Hospital':c['hospital'], **c['classes'], 'Total':c['samples'], 'FedAvg weight':c['aggregation_weight']} for c in r['clients']]), hide_index=True)
st.subheader('Federated training')
st.line_chart(pd.DataFrame(r['history']).set_index('round'))
st.write(f"MLP: {len(r['features'])} inputs → 64 → 32 → {len(r['known_classes'])} known outputs. {r['parameter_count']:,} learned parameters.")
st.subheader('Unknown detection')
st.write('Withheld class:', r['withheld_class'])
st.caption(f"Threshold: {r['threshold']:.4f}. Known test samples: {r['known_test_samples']:,}. Unknown test samples: {r['unknown_test_samples']:,}.")
st.dataframe(pd.DataFrame([{'Measure':k, 'Value':v} for k,v in r['metrics'].items()]), hide_index=True)
pred = pd.read_csv(path/'predictions.csv')
choice = st.selectbox('Show test decisions', ['All', 'True unknown', 'Flagged unknown', 'Missed unknown'])
if choice == 'True unknown': pred = pred[pred.true_label == r['withheld_class']]
if choice == 'Flagged unknown': pred = pred[pred.decision == 'UNKNOWN']
if choice == 'Missed unknown': pred = pred[(pred.true_label == r['withheld_class']) & (pred.decision != 'UNKNOWN')]
st.dataframe(pred.head(100), hide_index=True)
st.subheader('Example update sent by Hospital A')
st.json(r['payload_example'])
with st.expander('Configuration and protocol'):
    st.json({k:v for k,v in r.items() if k not in ['history','payload_example']})
