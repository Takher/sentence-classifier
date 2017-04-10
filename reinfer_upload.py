import os
import requests
import json
import datetime

# Script uploads data to re:infer. Currently only works with amazon data

# Store the environment variable $TOKEN
TOKEN = os.environ['TOKEN']

batches = []

with open('./data/amazon/amazon_50K_reviews.json', 'r') as file:
    i = 0
    infile = []
    notlabelled = 0
    labelled = 0
    for line in file:
        line = json.loads(line)
        comment = {}
        comment['original_text'] = line.get('reviewText', 'empty') # If there is no reviewText

        comment['timestamp'] = str(datetime.datetime.utcfromtimestamp(line['unixReviewTime']))
        comment['id'] = 'a' + str(i) # Placeholder

        user_properties = {'string:reviewerID': line['reviewerID'],
                           'string:reviewerName': line['reviewerName'],
                           'string:helpful': str(line['helpful']),
                           'string:asin': line['asin'],
                           'string:overall': str(line['overall']),
                           'string:summary': line['summary']}

        comment['user_properties'] = user_properties

        if float(line['overall']) >= 4.0:
            comment['assigned_labels'] = [{"name": line['category'], "sentiment": "positive"}]
            labelled +=1
        elif float(line['overall']) <= 2.0:
            comment['assigned_labels'] = [{"name": line['category'], "sentiment": "negative"}]
            labelled += 1
        else:
            notlabelled +=1

        infile.append(comment)

        if len(infile) == 1000:
            batches.append({'comments':infile})
            infile = []
        i += 1
        # if i == 2500: break
batches.append({'comments':infile}) # catch the last bacth less than 1024 (if it exists)


for index, batch in enumerate(batches):
    index +=1
    data = json.dumps(batch)
    headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}
    if index==3:
        try:
            response = requests.put('https://reinfer.io/api/voc/datasets/tanrajbir/amazon_electronic_reviews/comments', headers=headers, data=data)
            print(index, response.content)
        except:
            print('problem with ', index) # index staring from 1 so we know which batch has an issue!)


print('notlabelled ::', notlabelled)
print('labelled ::', labelled)
