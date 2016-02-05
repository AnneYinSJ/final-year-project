"""
A Scalable Classifier for processing Big Data Streams
Authors: Kiran Sudhir, Mayanka Pachaiyappa and Varun Bezzam
Sri Sivasubramaniya Nadar College of Engineering
Kalavakkam, Chennai, Tamil Nadu
"""
import json
import csv

fieldnames = ['created_at', 'description', 'retweeted', 'followers_count', 'retweeted', 'id', 'id_str','in_reply_to_user_id_str', 'in_reply_to_screen_name', 'truncated', 'timestamp_ms', 'contributors', 'coordinates','lang', 'place', 'favorited', 'retweet_count', 'source', 'in_reply_to_status_id', 'geo', 'in_reply_to_status_id_str', 'is_quote_status', 'favorite_count', 'filter_level', 'in_reply_to_user_id', 'possibly_sensitive','extended_entities']

def parse_json(filename):
    data=[]
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    with open('json.text','w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames, extrasaction = 'ignore')
        writer.writeheader()
        for i in range(0,len(data)):
            writer.writerow(data[i])
