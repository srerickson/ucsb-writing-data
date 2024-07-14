import duckdb
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Any

pd.options.display.max_colwidth = 0

df = pd.read_csv(Path("data") / 'reflections.csv') 
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

embeddings_url = "https://dreamlab-public.s3.us-west-2.amazonaws.com/sorapure/mxbai_embeddings.parquet"
embeddings_file = Path("outputs") / 'mxbai_embeddings.parquet'

if not embeddings_file.exists():
    # download embeddings file if the file doesn't exist
    duckdb.execute(f"COPY (SELECT * from read_parquet('{embeddings_url}')) TO '{embeddings_file}' (FORMAT PARQUET);")

def search(q: str) -> List[Any]:
    query = f"Represent this sentence for searching relevant passages: {q}"
    query_embed = model.encode(query)
    sql = f"""
        FROM read_parquet('{embeddings_file}')
        SELECT 
            student_id,
            question_id,
            array_distance(
                CAST(embedding as FLOAT[1024]),
                CAST($embed as FLOAT[1024])
            ) AS distance
        ORDER BY distance ASC
        LIMIT 20;
    """
    return duckdb.execute(sql, {"embed": query_embed}).fetchall()



for row in search("bad at grammar"):
    print(df[ df.perm == row[0] ][row[1]].to_string())