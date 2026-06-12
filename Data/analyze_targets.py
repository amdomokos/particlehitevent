import numpy as np

with open('training_data/pixel_clusters_d16401.out') as f:
    lines = f.readlines()

cluster_tag = '<cluster>'
cluster_indices = [i for i, line in enumerate(lines) if line.strip() == cluster_tag]
targets = []
for idx in cluster_indices:
    vals = list(map(float, lines[idx+1].strip().split()))
    if len(vals) == 9:
        targets.append(vals)

arr = np.array(targets)
print('Total clusters:', arr.shape[0])
print()
print('idx       min         max        mean         std   always_same')
for i in range(9):
    col = arr[:, i]
    same = bool(np.allclose(col, col[0]))
    print(f'{i:>3}  {col.min():>10.4f}  {col.max():>10.4f}  {col.mean():>10.4f}  {col.std():>10.4f}  {same}')

print()
print('Index 6 unique count:', len(np.unique(arr[:,6])), '  first 5:', np.unique(arr[:,6])[:5])
print()
for a, b in [(5,8),(0,3),(1,3),(5,7),(0,7),(1,8)]:
    r = np.corrcoef(arr[:,a], arr[:,b])[0,1]
    print(f'r(col{a}, col{b}) = {r:.4f}')
