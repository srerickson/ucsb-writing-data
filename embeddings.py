"""
This script is used to generate embeddings and a duckdb datatabase from raw
survey results. It expects the raw survey results file (`data/reflections.csv`),
an OpenAPI key, and Ollama running locally.
"""

import duckdb
import ollama
from duckdb.typing import VARCHAR
from pathlib import Path 
from typing import Sequence
from openai import OpenAI

src_path = Path('data') / 'reflections.csv'

# generated data:
# duckdb representation of reflections data
db_path = Path('outputs') / 'reflection_embeddings.duckdb'

# exported non-normalized mxbai embeddings
mxbai_nonnorm_path = Path("outputs") / 'mxbai_embeddings_nonnorm.parquet'

# exported normalized mxbai embeddings (added later and not used for study)
# mxbai_norm_file = Path("outputs") / 'mxbai_embeddings_norm.parquet'

# open ai embeddings
openai_path = Path("outputs") / 'openai_3small.parquet'

# requires OPENAI_API_KEY
OAclient = OpenAI()

# SQL to create a table for reflections data: one question/response per row
create_table_sql = f"""
    CREATE SEQUENCE IF NOT EXISTS seq_texts_id START 1;
    CREATE TABLE IF NOT EXISTS texts (
        id INTEGER PRIMARY KEY,
        student_id VARCHAR,
        question_id VARCHAR,
        text VARCHAR,
        UNIQUE(student_id, question_id)
    );
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

# SQL to create mxbai embeddings table and generate embeddings.
# This is written to allow the query to continue where a previous
# run left off. 
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

# returns sql to export mxbai_embed_large table to parquet file
def mxbai_export_sql(parquet_path):
    return f"""
    COPY (
         SELECT 
            texts.student_id as student_id,
            texts.question_id as question_id,
            mxbai_embed_large.embedding as embedding
        FROM texts
        LEFT JOIN mxbai_embed_large 
            ON mxbai_embed_large.text_id = texts.id
    ) TO '{str(parquet_path)}' (FORMAT PARQUET);
    """

# SQL to create openai embeddings table and generate embeddings.
# This is written to allow the query to continue where a previous
# run left off. 
openai_3small_sql = """
    CREATE TABLE IF NOT EXISTS openai_3small (
        text_id INTEGER PRIMARY KEY,
        embedding FLOAT[1536],
        FOREIGN KEY (text_id) REFERENCES texts (id)
    );
    INSERT INTO openai_3small (text_id, embedding)
    SELECT 
        text_id,
        do_openai_3small(text)
    FROM (
        SELECT 
            texts.id as text_id,
            texts.text as text,
            openai_3small.embedding as embedding
        FROM texts
        LEFT OUTER JOIN openai_3small 
            ON openai_3small.text_id = texts.id
        WHERE embedding is NULL
    );
"""

# export openai embeddings to parquet file
def openai_export_sql(openai_path):
    return f"""
    COPY (
         SELECT 
            texts.student_id as student_id,
            texts.question_id as question_id,
            openai_3small.embedding as embedding
        FROM texts
        LEFT JOIN openai_3small 
            ON openai_3small.text_id = texts.id
    ) TO '{str(openai_path)}' (FORMAT PARQUET);
    """

def mxbai_embed_ollama(text :str) -> Sequence[float] | None:
    model = "mxbai-embed-large"
    try:
        # this return a non-normalized embedding!
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
        # for normalized embeddings:
        # response = ollama.embed(model=model, input=text)
        # embeddings = response["embeddings"][0]
        # return embeddings
    except ollama._types.ResponseError as e:
        return None

def openai_3small(text :str) -> Sequence[float] | None:
    response =  OAclient.embeddings.create(input=text,model="text-embedding-3-small")
    return response.data[0].embedding
    
with duckdb.connect(database=str(db_path)) as conn:
    conn.create_function('do_mxbai_embed_large', mxbai_embed_ollama, [VARCHAR], 'FLOAT[1024]')
    conn.create_function('do_openai_3small', openai_3small, [VARCHAR], 'FLOAT[1536]')
    conn.execute(create_table_sql)

    # generate and export (non-normalized) mxbai embeddings
    #conn.execute(mxbai_embed_large_sql)
    #conn.execute(mxbai_export_sql(mxbai_nonnorm_path))
    
    #generate and export openai embeddings
    conn.execute(openai_3small_sql)
    conn.execute(openai_export_sql(openai_path))
