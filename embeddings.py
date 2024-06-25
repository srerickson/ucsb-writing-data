import pathlib
import ollama
import duckdb
from duckdb.typing import VARCHAR
from typing import Sequence         

src_path = pathlib.Path('data/reflections.csv')
db_path = pathlib.Path(f'outputs/reflections.duckdb')

create_table_sql = """
    CREATE SEQUENCE IF NOT EXISTS seq_text_responses_id START 1;
    CREATE TABLE IF NOT EXISTS text_responses (
        id INTEGER PRIMARY KEY,
        student_id VARCHAR,
        question_id VARCHAR,
        response_text VARCHAR,
        UNIQUE(student_id, question_id)
    );
"""

import_responses_sql = f"""
    INSERT INTO text_responses (
        id,
        student_id,
        question_id,
        response_text
    ) SELECT
        nextval('seq_text_responses_id'),
        perm as student_id, 
        question_id, 
        response_text
    FROM (
        UNPIVOT '{str(src_path)}' ON r1, r2, r3, r4 INTO NAME 'question_id' VALUE 'response_text'
    ) ON CONFLICT DO NOTHING;  
"""

mxbai_embed_large_sql = """
    CREATE TABLE IF NOT EXISTS mxbai_embed_large (
        id INTEGER PRIMARY KEY,
        embedding FLOAT[1024],
        FOREIGN KEY (id) REFERENCES text_responses (id)
    );
    INSERT INTO mxbai_embed_large (id, embedding)
    SELECT 
        resp_id,
        do_mxbai_embed_large(resp_text)
    FROM (
        SELECT 
            text_responses.id as resp_id,
            text_responses.response_text as resp_text,
            mxbai_embed_large.embedding as embedding
        FROM text_responses
        LEFT OUTER JOIN mxbai_embed_large 
            ON mxbai_embed_large.id = text_responses.id
        WHERE embedding is NULL
    );
"""

def mxbai_embed_large(text :str) -> Sequence[float] | None:
    model = "mxbai-embed-large"
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except ollama._types.ResponseError as e:
        return None

with duckdb.connect(database=str(db_path)) as conn:
    conn.create_function('do_mxbai_embed_large', mxbai_embed_large, [VARCHAR], 'FLOAT[1024]')
    conn.execute(create_table_sql)
    conn.execute(import_responses_sql)
    conn.execute(mxbai_embed_large_sql)