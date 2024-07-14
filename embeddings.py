import pathlib
import duckdb
import ollama
from duckdb.typing import VARCHAR
from typing import Sequence         

src_path = pathlib.Path('data') / 'reflections.csv'
db_path = pathlib.Path('outputs') / 'embeddings.duckdb'
mxbai_path = pathlib.Path('outputs') / 'mxbai_embeddings.parquet'

create_table_sql = """
    CREATE SEQUENCE IF NOT EXISTS seq_texts_id START 1;
    CREATE TABLE IF NOT EXISTS texts (
        id INTEGER PRIMARY KEY,
        student_id VARCHAR,
        question_id VARCHAR,
        text VARCHAR,
        UNIQUE(student_id, question_id)
    );
"""

import_responses_sql = f"""
    INSERT INTO texts (
        id,
        student_id,
        question_id,
        text
    ) SELECT
        nextval('seq_texts_id'),
        perm as student_id, 
        question_id, 
        response_text
    FROM (
        UNPIVOT '{str(src_path)}' ON r1, r2, r3, r4 INTO NAME 'question_id' VALUE 'response_text'
    ) ON CONFLICT DO NOTHING;  
"""

mxbai_embed_large_sql = """
    CREATE TABLE IF NOT EXISTS mxbai_embed_large (
        text_id INTEGER PRIMARY KEY,
        embedding FLOAT[1024],
        FOREIGN KEY (text_id) REFERENCES texts (id)
    );
    INSERT INTO mxbai_embed_large (text_id, embedding)
    SELECT 
        text_id,
        do_mxbai_embed_large(text)
    FROM (
        SELECT 
            texts.id as text_id,
            texts.text as text,
            mxbai_embed_large.embedding as embedding
        FROM texts
        LEFT OUTER JOIN mxbai_embed_large 
            ON mxbai_embed_large.text_id = texts.id
        WHERE embedding is NULL
    );
"""

# export mxbai embeddings to parquet format
mxbai_export_sql = f"""
    COPY (
         SELECT 
            texts.student_id as student_id,
            texts.question_id as question_id,
            mxbai_embed_large.embedding as embedding
        FROM texts
        LEFT JOIN mxbai_embed_large 
            ON mxbai_embed_large.text_id = texts.id
    ) TO '{str(mxbai_path)}' (FORMAT PARQUET);
"""

def mxbai_embed_ollama(text :str) -> Sequence[float] | None:
    model = "mxbai-embed-large"
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except ollama._types.ResponseError as e:
        return None

with duckdb.connect(database=str(db_path)) as conn:
    conn.create_function('do_mxbai_embed_large', mxbai_embed_ollama, [VARCHAR], 'FLOAT[1024]')
    conn.execute(create_table_sql)
    conn.execute(import_responses_sql)
    conn.execute(mxbai_embed_large_sql)
    conn.execute(mxbai_export_sql)

