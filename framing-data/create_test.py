import json

import pandas as pd
import numpy as np
from sqlalchemy.types import DateTime
from crate.client.sqlalchemy.types import Object

crate = 'crate://crate-db:4200'


def get_data():
    tweet_query = "SELECT * FROM climate_tweets;"
    element_query = "SELECT element_id, frame, element_mpnet, element_bertweet FROM frame_elements;"
    tweets_df = pd.read_sql(tweet_query, crate, parse_dates=['created_at_datetime'])
    element_df = pd.read_sql(element_query, crate)
    return element_df, tweets_df


def method_one(tweets_df):
    # method 1: look for tweets that match these keywords and manually annotate the frame
    keyword_dict = {
        'science': ["science", "scientific", "scientifically", "scientist", "scientists"],
        'political_or_ideological_struggle': ["political", "politically", "politics", "policy", "politician",
                                              "politicians", "policymaker", "policymakers", "ideology", "ideological",
                                              "ideologically"],
        'disaster': ["disaster", "catastrophe", "apocalypse", "disastrous", "catastrophic", "catastrophically",
                     "apocalyptic", "extinction", "extinct"],
        'opportunity': ["opportunity", "opportunities", "innovation", "innovations"],
        'economic': ["economic", "economy", "economically", "invest", "divest", "investment", "divestment"],
        'morality_and_ethics': ["moral", "morality", "morally", "ethics", "ethical", "ethically"],
        'role_of_science': ["bias", "misinformation", "propaganda", "media", "biased", "biases"],
        'security': ["security", "secure", "conflict", "conflicts"],
        'health': ["health", "healthy", "death", "deaths"]
    }

    combined_df = pd.DataFrame()
    for frame in keyword_dict:
        found = tweets_df.loc[tweets_df['txt_clean'].str.contains('|'.join(keyword_dict[frame]))].head(10)
        # print(found)
        combined_df = pd.concat([combined_df, found], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['id'], ignore_index=True)
    combined_df['txt_clean_sentences'] = combined_df['txt_clean_sentences'].apply(lambda p: json.dumps(p))
    combined_df['table_name'] = 'test_tweets'
    combined_df.to_sql('test_tweets', crate, if_exists='append', index=False, dtype={'created_at_datetime': DateTime,
                                                                                     'txt_clean_sentences': Object})


def method_two():
    # method 2: look for the top 10 most similar embeddings to the frame elements for each frame
    pass


def main():
    element_df, tweets_df = get_data()
    method_one(tweets_df)


if __name__ == '__main__':
    main()
