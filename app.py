from fastapi import FastAPI, HTTPException
from schema import PostGet
from typing import List
import os
from catboost import CatBoostClassifier
import pandas as pd
from sqlalchemy import create_engine
import datetime as dt


app = FastAPI()


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("catboost_model_2")
    from_file = CatBoostClassifier(cat_features=['topic'])
    model = from_file.load_model(model_path, format='cbm')
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> pd.DataFrame:
    return batch_load_sql("SELECT * FROM rubtsov_dmitry_lesson_22_info_post_2")

def load_user_df() -> pd.DataFrame:
    return batch_load_sql("SELECT * FROM public.user_data")

def load_liked_posts() -> pd.DataFrame:
    return batch_load_sql("SELECT * FROM rubtsov_dmitry_lesson_22_liked_posts")

model = load_models()
info_posts = load_features().drop('index', axis=1)
user_df = load_user_df()
liked_posts = load_liked_posts().drop('index', axis=1)

def get_db_for_predict(user_id, time):
    us = user_df[user_df['user_id'] == user_id][['user_id', 'gender', 'age', 'exp_group']].head(1)
    users = us.loc[us.index.repeat(info_posts['post_id'].nunique())]
    post_user = pd.concat([users.reset_index(drop=True), info_posts[['post_id', 'topic','city', 'tf_idf']].reset_index(drop=True)], axis=1)
    post_liked = liked_posts[(liked_posts['user_id'] == user_id) & (liked_posts['timestamp'] < time)]['post_id'].values
    posts_rec = post_user[~post_user['post_id'].isin(post_liked)]
    data = pd.DataFrame({'user_id': posts_rec['user_id'], 'gender': posts_rec['gender'], 'age': posts_rec['age'],
                       'city': posts_rec['city'], 'exp_group': posts_rec['exp_group'],
                         'topic': posts_rec['topic'], 'tf_idf': posts_rec['tf_idf'], 'post_id': posts_rec['post_id']})
    return data



@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: dt.datetime, limit: int = 10) -> List[PostGet]:
    result = []
    post_predictions = model.predict_proba(get_db_for_predict(id, time).drop('post_id', axis=1))[:, 1]
    post_predict = pd.DataFrame({'post_id': get_db_for_predict(id, time)['post_id'],
                                  'predictions': post_predictions}).sort_values(by='predictions', ascending=False).head(limit)
    
    all_info_posts = post_predict.merge(info_posts[['post_id', 'text', 'topic']], on='post_id', how='inner').rename(columns={'post_id': 'id'})
    result = all_info_posts.to_dict('records')
    if not result:
        raise HTTPException(404, "posts not found")
    return result
