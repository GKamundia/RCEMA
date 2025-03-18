import lancedb
import sys

# Connect using absolute path
db = lancedb.connect(r"C:\Users\Anarchy\Documents\Data_Science\CEMA\RCEMA\docling\data\lancedb")
table = db.open_table("docling_tables")

# Get search query from command line
query = sys.argv[1] if len(sys.argv) > 1 else "nurse"

# Search with table filtering
results = (table.search(query)
              .where("metadata.tables.has_table = true")
              .limit(5)
              .to_pandas())

# Display results
print(f"Found {len(results)} tables matching '{query}':\n")
for idx, row in results.iterrows():
    print(f"## Table from {row['metadata']['filename']} (Page {row['metadata']['page_numbers'][0]})")
    print(row['text'])
    print("\n---\n")
