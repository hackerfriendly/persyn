#!/usr/bin/env python3
import boto3
import json
import uuid

# Get the service resource
sqs = boto3.resource('sqs', region_name="us-west-2")

# Get the queue. This returns an SQS.Queue instance
queue = sqs.get_queue_by_name(QueueName='image-ready')

print(queue)

# Process messages by printing out body and optional author name
for message in queue.receive_messages(WaitTimeSeconds=20):
    print(message.body)
    message.delete()

