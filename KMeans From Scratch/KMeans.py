#Importing the necessary libraries
import sys
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
import string
import re
import math

stop = set(stopwords.words('english'))

def preprocessData(tweet_dict_list):
    for tweet in tweet_dict_list:
        for key in tweet:
            line = tweet[key].rstrip().lstrip()
            result = re.sub(r"http\S+", "", line)
            line1 = nltk.word_tokenize(result)
            line1 = [t.lower() for t in line1 if t not in stop and t not in string.punctuation] 
            for t in  line1 :
                if (t == 'rt')  or (t =='...') or (t == '``') or (t=="''") or (t == 'http'):
                    line1.remove(t)
            tweet[key]=line1
    return (tweet_dict_list)
        

def jaccardDistance(str_tweet,str_centroid):
    term_1  = len(list(set(str_tweet) & set(str_centroid)))
    term_2 = len(list(set(str_tweet) | set(str_centroid)))
    return(round(1-float(term_1/term_2),4))

def getTweetTxts(tweet_data,tweet_ids):
    tweet_txts=[]
    print("Tweet_data in get text",len(tweet_data))
    if len(tweet_ids) != 0:
        for p in tweet_data:
            for key in p:
                if key in tweet_ids:
                    tweet_txts.append(p[key])              
    print(len(tweet_txts))
    return(tweet_txts)

def calculateSimilarity(tweet_txts):
    dist_matrix=[]
    for i in range(len(tweet_txts)):
        sim=[]
        for j in range(len(tweet_txts)):
            sim.append(jaccardDistance(tweet_txts[i],tweet_txts[j]))
        dist_matrix.append(sim)
    
    dis_matrix = [sum(i) for i in dist_matrix]
    return(dis_matrix)
    
    
def recalculateCentroid(centroid_data,tweet_data,cluster_list,cluster):
    new_centroids =[]
    new_centroid_tweets=[]
    print(len(tweet_data))
    for centroid in cluster_list:
        tweet_ids=[]
        for i in cluster:
            for key in i:
                if i[key]==centroid:
                    tweet_ids.append(key)
        print("for cluster",centroid,len(tweet_ids))
        tweet_txts = getTweetTxts(tweet_data,tweet_ids)
        dis_matrix = calculateSimilarity(tweet_txts)
        print(dis_matrix)
        #print([dis_matrix.index(min(dis_matrix))])
        new_centroids.append(tweet_ids[dis_matrix.index(min(dis_matrix))])
        new_centroid_tweets.append(tweet_txts[dis_matrix.index(min(dis_matrix))])
    #print(new_centroid_ids)
    return(new_centroids,new_centroid_tweets)

def createOutputFile(cluster,cluster_list,f):
    for x in cluster_list:
        listing=[]
        for item in cluster:
            for key in item:
                if item[key] ==x:
                    listing.append(key)
        num= str(cluster_list.index(x)+1)
        str_listing = ",".join(str(x) for x in listing)
        f.write(num+" "+str_listing+"\n")

def calculateSSE(cluster_list,cluster,tweet_data,new_centroid_data,f):
    sum=0
    for i in range(len(cluster_list)):
        item = cluster_list[i]
        clust_indices=[]
        tweet=[]
        for c in cluster:
            for key in c:
                if c[key]== item:
                    clust_indices.append(key)
        for m in clust_indices:
            for p in tweet_data:
                if m in p:
                    tweet.append(p[m])         
        for j in range(len(clust_indices)):
            sum+=math.pow(jaccardDistance(tweet[j],new_centroid_data[i]),2) 
    f.write("Sum of Squared Error -> "+str(sum)+"\n" )
    
def kmeans(f,centroid_data,tweet_data,k=25):
    cluster = []
    for item in tweet_data:
        for i in item:
            tweet_id = i
            tweet_txt = item[i]
            tmp=[]
            dis = []
            for n in range(k):
                cent = centroid_data[n]
                for j in cent:
                    cent_id = j
                    cent_txt = cent[j]
                    dis.append([cent_id,jaccardDistance(tweet_txt,cent_txt)])
            for d in dis:
                tmp.append(d[1])
        cluster.append({tweet_id:dis[tmp.index(min(tmp))][0]})
    cluster_list=[]
    for item in cluster:
        for key in item:
            if item[key] not in cluster_list:
                cluster_list.append(item[key])
    print(cluster_list)
    print("The points clustered",len(cluster),len(tweet_data))    
    new_centroids_ids,new_centroid_tweets = recalculateCentroid(centroid_data,tweet_data,cluster_list,cluster)
    print("new_centroids",new_centroids_ids)
    counter= 0
    for i in range(len(new_centroids_ids)):
        if new_centroids_ids[i] == cluster_list[i]:
            cluster_list[i]= new_centroids_ids[i]
            counter +=1
    new_centroid_data=[]
    for i in range(len(cluster_list)):
            new_centroid_data.append({new_centroids_ids[i]:new_centroid_tweets[i]})     

    if counter != k:
        kmeans(f,new_centroid_data,tweet_data,k)
    else:
        createOutputFile(cluster,cluster_list,f)
        calculateSSE(cluster_list,cluster,tweet_data,new_centroid_data,f)
    
def prepareData(tweet_path='Tweets.json'):
    tweets_list =[]
    for line in open(tweet_path,'r'):
        tweets_list.append(json.loads(line))
    tweet_dict_list = []
    for tweet in tweets_list:
        t ={tweet['id'] : tweet['text']}
        tweet_dict_list.append(t)
    return(preprocessData(tweet_dict_list))

def  getCentroidTweets(tweet_dict_list,seed_file='InitialSeeds.json'):
    centroid_list = []
    for line in open(seed_file,'r'):
        centroid_list.append(int(line.split(',')[0]))
    centroid_tweets = []
    for c in centroid_list:
        for tweet in tweet_dict_list:
            if c in tweet:
                centroid_tweets.append({c:tweet[c]})
    return(centroid_tweets)
    
def main():

    if len(sys.argv) < 4:
        print("Not suffiecient Arguments entered ....Exiting with code -1")
        exit(-1)
        
    try:
        if int(sys.argv[1]) > 25:
            print("Running the K-means with k =25")
            k = 25
    except:
        k =25
        seed_file = sys.argv[1]
        tweet_file = sys.argv[2]
        out_file = sys.argv[3]
        
    if len(sys.argv)>=5:
        k = int(sys.argv[1])
        seed_file = sys.argv[2]
        tweet_file = sys.argv[3]
        out_file = sys.argv[4]
    
    f = open(out_file, 'w')

    tweet_data= prepareData(tweet_file)
    print("In main tweet data", len(tweet_data))
    centroid_data = getCentroidTweets(tweet_data,seed_file)
    print("In main centroid data", len(centroid_data))
    kmeans(f,centroid_data,tweet_data,k)
    f.close()
    
if __name__ == "__main__":
    main()
    
    
    
    