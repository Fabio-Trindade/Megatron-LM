import pandas as pd
import json
df = pd.read_parquet("alpaca.parquet")



output_file = "alpaca.json"

with open(output_file, "w", encoding="utf-8") as f:
    for i, (_, row) in enumerate( df.iterrows()):
        json_data = {"text": row.to_dict()["text"]}
        f.write(json.dumps(json_data) + "\n") 
        # if i > 10:
        #     break

print(f"JSONs salvos em: {output_file}")