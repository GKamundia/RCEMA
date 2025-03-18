import lancedb

db = lancedb.connect("data/lancedb")
table = db.open_table("docling_tables")

print(f"Total chunks: {table.count_rows()}")
print(f"Tables detected: {table.search().where('metadata.tables.has_table = true').count_rows()}")
