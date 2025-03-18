

import pyarrow.dataset as ds

dataset = ds.dataset("C:/Users/Anarchy/Documents/Data_Science/CEMA/RCEMA/docling/data/lancedb/docling_tables.lance")
table = dataset.to_table()
df = table.to_pandas()
df[df["metadata.tables.has_table"] == True].head(50)