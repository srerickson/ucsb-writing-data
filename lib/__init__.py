import duckdb
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from IPython.display import Markdown as md, display

# load the model for computing embeddings for query string
mxbai = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# embeddings for the text responses in the survey data have been pre-computed
mxbai_url = "https://dreamlab-public.s3.us-west-2.amazonaws.com/sorapure/mxbai_embeddings.parquet"
mxbai_file = Path("outputs") / 'mxbai_embeddings.parquet'

openai_url = "https://dreamlab-public.s3.us-west-2.amazonaws.com/sorapure/openai_3small.parquet"
openai_file = Path("outputs") / 'openai_3small.parquet'

def search_display(q: str, df: pd.DataFrame, limit: int = 25, model: str = "mxbai"):
    result = search_df(q, df, limit=limit, model=model)
    for i, row in result.iterrows():
        text = f"(distance: {str(row['result_distance'])[:4]}, perm: {row['perm']}) {row['result_text']}"
        display(md(text))


def search_df(q: str, df: pd.DataFrame, limit: int = 25, model: str = "openai") -> pd.DataFrame:
    if model == "mxbai":
       return mxbai_search_df(q, df, limit)
    return openai_search_df(q, df, limit=limit)


def openai_search_df(q: str, df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    if not mxbai_file.exists():
        duckdb.execute(f"COPY (SELECT * from read_parquet('{openai_url}')) TO '{openai_file}' (FORMAT PARQUET);")
    client = OpenAI()
    response =  client.embeddings.create(input=q,model="text-embedding-3-small")
    query_embed = response.data[0].embedding
    sql = f"""
        FROM df
        LEFT JOIN read_parquet('{openai_file}') ON (df.perm = student_id)
        SELECT 
            df.*,
            embedding,
            question_id as result_question_id,
            array_distance(
                CAST(embedding as FLOAT[1536]),
                CAST($embed as FLOAT[1536])
            ) AS result_distance
        ORDER BY result_distance ASC
    """
    if limit > 0:
        sql += f" LIMIT {limit};"
    else:
        sql += ";"
    result = duckdb.execute(sql, {"embed": query_embed}).fetch_df()
    result['result_text'] = result.apply(lambda row: row[row['result_question_id']], axis=1)
    return result


def mxbai_search_df(q: str, df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    if not mxbai_file.exists():
        duckdb.execute(f"COPY (SELECT * from read_parquet('{mxbai_url}')) TO '{mxbai_file}' (FORMAT PARQUET);")
    query = f"Represent this sentence for searching relevant passages: {q}"
    query_embed = mxbai.encode(query)
    sql = f"""
        FROM df
        LEFT JOIN read_parquet('{mxbai_file}') ON (df.perm = student_id)
        SELECT 
            df.*,
            embedding,
            question_id as result_question_id,
            array_distance(
                CAST(embedding as FLOAT[1024]),
                CAST($embed as FLOAT[1024])
            ) AS result_distance
        ORDER BY result_distance ASC
    """
    if limit > 0:
        sql += f" LIMIT {limit};"
    else:
        sql += ";"
    result = duckdb.execute(sql, {"embed": query_embed}).fetch_df()
    result['result_text'] = result.apply(lambda row: row[row['result_question_id']], axis=1)
    return result