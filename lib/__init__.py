import duckdb
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from IPython.display import Markdown as md
from IPython import display

# load the model for computing embeddings for query string
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# embeddings for the text responses in the survey data have been pre-computed
embeddings_url = "https://dreamlab-public.s3.us-west-2.amazonaws.com/sorapure/mxbai_embeddings.parquet"
embeddings_file = Path("outputs") / 'mxbai_embeddings.parquet'

# download embeddings file if the file doesn't exist
if not embeddings_file.exists():
    duckdb.execute(f"COPY (SELECT * from read_parquet('{embeddings_url}')) TO '{embeddings_file}' (FORMAT PARQUET);")

def search_df(q: str, df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    query = f"Represent this sentence for searching relevant passages: {q}"
    query_embed = model.encode(query)
    sql = f"""
        FROM df
        LEFT JOIN read_parquet('{embeddings_file}') ON (df.perm = student_id)
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


def search_display(q: str, df: pd.DataFrame, limit: int = 25):
    result = search_df(q, df, limit)
    for i, row in result.iterrows():
        text = f"({str(row['result_distance'])[:6]}) {row['result_text']}"
        display(md(text))