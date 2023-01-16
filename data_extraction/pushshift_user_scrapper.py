import datetime
import pandas as pd
import requests
import json
from collections import Counter

df=pd.read_csv("data_extraction/coronavirus_2021q1_all.csv")
df["Publish Date"]=df["Publish Date"].apply(lambda x: datetime.datetime.timestamp(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")))
df=df[df["Publish Date"]>datetime.datetime.timestamp(datetime.datetime(2021, 3, 15, 0))]
# print(f"{len(pd.unique(df['Author']))} Unique users")

from time import sleep, time

from requests import Timeout

next = time()+2
def parse_comment(comment):
    body = comment['body'].replace('\n',' ').replace('\t',' ').replace('\r',' ').replace('\r\n',' ')
    author = comment['author']
    com_id = comment['id']
    score = comment['score']
    created = datetime.datetime.fromtimestamp(comment['created_utc']) 
    permalink = comment['permalink']
    subreddit = comment['subreddit']
    parent_id = comment['parent_id']
    post_id=comment['link_id']
    return (com_id,subreddit,body,author,score,created,permalink,parent_id,post_id)

def search_user_comments(user, i, limit:int, before:int, after:int):
    global next
    user_comments=[]
    sleep(max(0,next-time()))
    next = time()+2
    while True:
        sleep(max(0,next-time()))
        next = time()+2
        url= f"https://api.pushshift.io/reddit/search/comment?limit=1000&before={before}&after={after}{' ' if user is None else '&author='+user}"

        while True:
            try:
                r = requests.get(url, timeout=5)
                data = json.loads(r.text)['data']
                break
            except Timeout as t:
                # print(f"Timed Out on user {user} ({i}): {t}", end='\r')
                sleep(5)
            except Exception as e:
                # print(f"Errored on user {user} ({i}): {e}", end='\r')
                sleep(5)
        
        if len(data)==0 or (data[-1]["created_utc"]-1)==before:
            break
        
        user_comments+=data
        
        if len(user_comments)>=limit:
            user_comments=user_comments[:limit]

            break

        before=user_comments[-1]["created_utc"]-1

    parsed_comments=[]
    for comment in user_comments:
        if comment['subreddit']!="Coronavirus":
            parsed_comments.append(parse_comment(comment))

    assert len(parsed_comments)==len(list(set([comment[0] for comment in parsed_comments])))
    return parsed_comments

before = int(datetime.datetime(2021, 6, 30, 0).timestamp())
after = int(datetime.datetime(2020, 10, 1, 0).timestamp())

comms = []

for i,user in enumerate(pd.unique(df["Author"])):
     usercomms=search_user_comments(user,i,1000,before,after)
     print(f"Gathered {len(usercomms)} comments for user {user} ({i}/{len(pd.unique(df['Author']))}) \t\t ",end='\r')
     comms+=usercomms

import time
import csv
def updateSubs_file():
    upload_count = 0
    print("input filename of submission file, please add .csv")
    filename = input()
    file = filename
    with open(file, 'w', newline='', encoding='utf-8') as file: 
        a = csv.writer(file, delimiter=',')
        headers = ["Post ID","Subreddit","Body","Author","Score","Publish Date","Permalink","Parent_id","Post_id"]
        a.writerow(headers)
        for comm in comms:
            a.writerow(comm)
            upload_count+=1
            
        print(str(upload_count) + " submissions have been uploaded")

updateSubs_file()