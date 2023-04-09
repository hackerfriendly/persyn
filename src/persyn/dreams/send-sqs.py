#!/usr/bin/env python3
import boto3
import json
import uuid

# Get the service resource
sqs = boto3.resource('sqs', region_name="us-west-2")

# Get the queue. This returns an SQS.Queue instance
queue = sqs.get_queue_by_name(QueueName='anna')

print(queue)

# data = {
#     "event_type": "say",
#     "guid": f"{uuid.uuid4()}",
#     "message": "Hello from send-sqs.py",
#     "service": "discord",
#     "channel": "962806111193428028|962806111742877729",
#     "bot_name": "Anna"
# }

data = {
    "event_type": "image-ready",
    "guid": f"{uuid.uuid4()}",
    "caption": "Attempting to reach feature parity.",
    "service": "mastodon",
    "channel": "https://mas.to/@annathebot",
    "bot_name": "Anna",
    "images": ["https://hackerfriendly.com/pub/anna/52823fce-9af9-4778-bfce-3b4aa42930fc.jpg"]
}

response = queue.send_message(MessageBody=json.dumps(data))

print(response['MessageId'])
