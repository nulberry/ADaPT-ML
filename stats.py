import pandas as pd


def filter_data():
    query0 = """
    SELECT id, table_name, split, txt_clean_sentences
    FROM climate_tweets
    where array_length(txt_clean_sentences['sentence_0']['tokens'], 1) between 9 and 22 
    and txt_clean_sentences['sentence_1']['tokens'] is null;
    """
    query1 = """
    select id, table_name, split, txt_clean_sentences
    FROM climate_tweets
    where array_length(txt_clean_sentences['sentence_0']['tokens'], 1) between 9 and 22 
    and array_length(txt_clean_sentences['sentence_1']['tokens'], 1) between 6 and 17
    and txt_clean_sentences['sentence_2']['tokens'] is null;"""
    query2 = """
    select id, table_name, split, txt_clean_sentences
    FROM climate_tweets
    where array_length(txt_clean_sentences['sentence_0']['tokens'], 1) between 9 and 22 
    and array_length(txt_clean_sentences['sentence_1']['tokens'], 1) between 6 and 17
    and array_length(txt_clean_sentences['sentence_2']['tokens'], 1) between 5 and 14
    and txt_clean_sentences['sentence_3']['tokens'] is null;"""
    query3 = """
    select id, table_name, split, txt_clean_sentences
    FROM climate_tweets
    where array_length(txt_clean_sentences['sentence_0']['tokens'], 1) between 9 and 22 
    and array_length(txt_clean_sentences['sentence_1']['tokens'], 1) between 6 and 17
    and array_length(txt_clean_sentences['sentence_2']['tokens'], 1) between 5 and 14
    and array_length(txt_clean_sentences['sentence_3']['tokens'], 1) between 4 and 12
    and txt_clean_sentences['sentence_4']['tokens'] is null;"""
    query4 = """
    select id, table_name, split, txt_clean_sentences
    FROM climate_tweets
    where array_length(txt_clean_sentences['sentence_0']['tokens'], 1) between 9 and 22 
    and array_length(txt_clean_sentences['sentence_1']['tokens'], 1) between 6 and 17
    and array_length(txt_clean_sentences['sentence_2']['tokens'], 1) between 5 and 14
    and array_length(txt_clean_sentences['sentence_3']['tokens'], 1) between 4 and 12
    and array_length(txt_clean_sentences['sentence_4']['tokens'], 1) between 3 and 10
    and txt_clean_sentences['sentence_5']['tokens'] is null;"""
    query5 = """
    select id, table_name, split, txt_clean_sentences
    FROM climate_tweets
    where array_length(txt_clean_sentences['sentence_0']['tokens'], 1) between 9 and 22 
    and array_length(txt_clean_sentences['sentence_1']['tokens'], 1) between 6 and 17
    and array_length(txt_clean_sentences['sentence_2']['tokens'], 1) between 5 and 14
    and array_length(txt_clean_sentences['sentence_3']['tokens'], 1) between 4 and 12
    and array_length(txt_clean_sentences['sentence_4']['tokens'], 1) between 3 and 10
    and array_length(txt_clean_sentences['sentence_5']['tokens'], 1) between 3 and 9
    and txt_clean_sentences['sentence_6']['tokens'] is null;"""

    df0 = pd.read_sql(query0, 'crate://localhost:4200')
    df1 = pd.read_sql(query1, 'crate://localhost:4200')
    df2 = pd.read_sql(query2, 'crate://localhost:4200')
    df3 = pd.read_sql(query3, 'crate://localhost:4200')
    df4 = pd.read_sql(query4, 'crate://localhost:4200')
    df5 = pd.read_sql(query5, 'crate://localhost:4200')
    df = pd.concat([df0, df1, df2, df3, df4, df5], ignore_index=True)

    return df
