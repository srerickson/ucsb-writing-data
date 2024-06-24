import pathlib
import ollama
import duckdb
from duckdb.typing import VARCHAR
import pandas as pd
from typing import Sequence

model = "mxbai-embed-large"
db_path = str(pathlib.Path(f'outputs/embeddings-{model}.db'))

create_table_sql = """
    CREATE TABLE IF NOT EXISTS embeddings(
        student_id VARCHAR,
        response_id VARCHAR,
        response_text VARCHAR,
        embedding FLOAT[1024],
        UNIQUE(student_id, response_id)
    );         
"""

insert_response_sql = """
    INSERT INTO embeddings (
        student_id,
        response_id,
        response_text
    ) VALUES (?, ?, ?) ON CONFLICT DO NOTHING;
"""

def embedding(text :str) -> Sequence[float] | None:
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except ollama._types.ResponseError as e:
        return None

with duckdb.connect(database=db_path) as conn:
    conn.execute(create_table_sql)
    # conn.create_function("get_embedding", embedding, [VARCHAR], 'FLOAT[1024]')
    df = pd.read_csv(pathlib.Path("data/reflections.csv"))
    
    def upsert_responses(row: pd.core.series.Series):
        for col in ['r1', 'r2', 'r3', 'r4']:
            student_id = row['perm']
            response_text = row[col]
            conn.execute(insert_response_sql, [student_id, col, response_text])
    
    # for col in ['r1', 'r2', 'r3', 'r4']:
    df.apply(upsert_responses, axis=1)





# duckdb.connect()


# df.to_csv("outputs/reflections_embeddings.csv")