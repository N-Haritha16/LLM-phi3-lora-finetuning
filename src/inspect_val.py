from datasets import load_from_disk

# adjust if your val_path is different in config.yaml
path = "data/processed/val"

ds = load_from_disk(path)

print("Num examples:", len(ds))
print("Columns:", ds.column_names)

for i in range(min(5, len(ds))):
    ex = ds[i]
    print(f"\nExample {i}:")
    print("len(input_ids):", len(ex['input_ids']))
    print("len(attention_mask):", len(ex['attention_mask']))
    print("first 20 ids:", ex['input_ids'][:20])
