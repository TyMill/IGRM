"""Deterministic integrity smoke check for recorded benchmark evidence."""
from pathlib import Path
import hashlib
import pandas as pd
root = Path(__file__).parent
p = root/'results/all_paired_7_methods.csv'
d = pd.read_csv(p)
assert len(d) == 9800, len(d)
assert d.groupby(['world','rep','method']).size().eq(1).all()
print('records', len(d), 'worlds', d.world.nunique(), 'methods', d.method.nunique())
print('sha256', hashlib.sha256(p.read_bytes()).hexdigest())
