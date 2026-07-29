import json
import os

models = ["mlp", "cnn", "cnn_concat", "gru", "s4", "s4_concat",
          "s4_modulate", "s4_modulate_biased", "s4_modulate_full"]

rows = []
for name in models:
    path = f"results/table2_run_1/{name}/final_report.json"
    if not os.path.exists(path):
        print(f"WARNING: missing {path}")
        continue
    with open(path) as f:
        r = json.load(f)
    tr = r["table_row"]
    rows.append({
        "model": tr["model"],
        "mse": round(tr["mse"], 3),
        "rmse": round(tr["rmse"], 3),
        "mae": round(tr["mae"], 3),
        "r2": round(tr["r2"], 4),
        "nz_mae": round(tr["nz_mae"], 6),
        "ny_mae": round(tr["ny_mae"], 6),
        "params": tr["params"],
    })

print()
print("| Model | MSE | RMSE | MAE | R2 | n_z MAE | n_y MAE | Params |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['model']} | {r['mse']} | {r['rmse']} | {r['mae']} | {r['r2']} | "
          f"{r['nz_mae']} | {r['ny_mae']} | {r['params']:,} |")
print()
