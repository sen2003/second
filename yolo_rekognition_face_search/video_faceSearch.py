# 將影片與集合中的人臉做比對
import boto3
import json
import sys
import time
import json


def chinese_name(externalImageId):
    name_map = {
        "Huang": "黃士熏",
        "Ke": "柯信汶",
        "Shen": "沈宏勳",
        "Tsou": "鄒博森"
    }
    return name_map.get(externalImageId, "Unknow person")


class VideoDetect:

    jobId = ''
    roleArn = ''
    bucket = ''
    video = ''
    startJobId = ''
    sqsQueueUrl = ''
    snsTopicArn = ''
    processType = ''

    def __init__(self, role, bucket, video, client, rek, sqs, sns):
        self.roleArn = role
        self.bucket = bucket
        self.video = video
        self.client = client
        self.rek = rek
        self.sqs = sqs
        self.sns = sns

    def GetSQSMessageSuccess(self):

        jobFound = False
        succeeded = False

        dotLine = 0
        while jobFound == False:
            sqsResponse = self.sqs.receive_message(QueueUrl=self.sqsQueueUrl, MessageAttributeNames=['ALL'],
                                                   MaxNumberOfMessages=10)

            if sqsResponse:

                if 'Messages' not in sqsResponse:
                    if dotLine < 40:
                        print('.', end='')
                        dotLine = dotLine + 1
                    else:
                        print()
                        dotLine = 0
                    sys.stdout.flush()
                    time.sleep(5)
                    continue

                for message in sqsResponse['Messages']:
                    notification = json.loads(message['Body'])
                    rekMessage = json.loads(notification['Message'])
                    print(rekMessage['JobId'])
                    print(rekMessage['Status'])
                    if rekMessage['JobId'] == self.startJobId:
                        print('Matching Job Found:' + rekMessage['JobId'])
                        jobFound = True
                        if (rekMessage['Status'] == 'SUCCEEDED'):
                            succeeded = True

                        self.sqs.delete_message(QueueUrl=self.sqsQueueUrl,
                                                ReceiptHandle=message['ReceiptHandle'])
                    else:
                        print("Job didn't match:" +
                              str(rekMessage['JobId']) + ' : ' + self.startJobId)
                    # Delete the unknown message. Consider sending to dead letter queue
                    self.sqs.delete_message(QueueUrl=self.sqsQueueUrl,
                                            ReceiptHandle=message['ReceiptHandle'])

        return succeeded

    def StartFaceSearchCollection(self, collection):
        response = self.rek.start_face_search(Video={'S3Object': {'Bucket': self.bucket, 'Name': self.video}},
                                              CollectionId=collection,
                                              NotificationChannel={'RoleArn': self.roleArn,
                                                                   'SNSTopicArn': self.snsTopicArn},
                                              FaceMatchThreshold=50,
                                              # Filtration options, uncomment and add desired labels to filter returned labels
                                              # Features=['GENERAL_LABELS'],
                                              # Settings={
                                              # 'GeneralLabels': {
                                              # 'LabelInclusionFilters': ['Clothing']
                                              # }}
                                              )

        self.startJobId = response['JobId']
        print('Start Job Id: ' + self.startJobId)

    def GetFaceSearchCollectionResults(self):
        maxResults = 100
        paginationToken = ''
        video_info = False
        finished = False
        results = []
        while finished == False:
            response = self.rek.get_face_search(JobId=self.startJobId,
                                                MaxResults=maxResults,
                                                NextToken=paginationToken,
                                                )
            if not video_info:
                print('Codec: ' + response['VideoMetadata']['Codec'])
                print('Duration: ' +
                      str(response['VideoMetadata']['DurationMillis']))
                print('Format: ' + response['VideoMetadata']['Format'])
                print('Frame rate: ' +
                      str(response['VideoMetadata']['FrameRate']))
                print()
                video_info = True

            for personMatch in response['Persons']:
                personFace = personMatch["Person"]["Face"]
                # bounding_box = face['BoundingBox']
                print("Timestamp: " + str(personMatch['Timestamp']))
                if 'FaceMatches' in personMatch and len(personMatch['FaceMatches']) > 0:
                    for faceMatch in personMatch['FaceMatches']:
                        face = faceMatch['Face']

                        # print("BoundingBox: ")
                        # print(f"Width: {face['BoundingBox']['Width']}")
                        # print(f"Heigh: {face['BoundingBox']['Heigh']}")
                        # print(f"Left: {face['BoundingBox']['Left']}")
                        # print(f"Top: {face['BoundingBox']['Top']}")
                        match_data = {
                            "Timestamp": personMatch['Timestamp'],
                            "BoundingBox": personFace['BoundingBox'],
                            "Name": chinese_name(face['ExternalImageId']),
                            "Similarity": faceMatch['Similarity']
                        }
                        results.append(match_data)
                        print("   Face ID: " + face['FaceId'])
                        print("   相似度: " + str(faceMatch['Similarity']))
                        print(
                            f"   姓名: {chinese_name(face['ExternalImageId'])}")
                        print()
                else:
                    no_match_data = {
                        "Timestamp": personMatch['Timestamp'],
                        "BoundingBox": personFace['BoundingBox'],
                        "Name": "Unknow",
                        "Similarity": "0.00%"
                    }
                    results.append(no_match_data)
                    print("   未知人臉")
                    print()
                if 'NextToken' in response:
                    paginationToken = response['NextToken']
                else:
                    finished = True
        return results

    def CreateTopicandQueue(self):

        millis = str(int(round(time.time() * 1000)))

        # Create SNS topic

        snsTopicName = "AmazonRekognitionExample" + millis

        topicResponse = self.sns.create_topic(Name=snsTopicName)
        self.snsTopicArn = topicResponse['TopicArn']

        # create SQS queue
        sqsQueueName = "AmazonRekognitionQueue" + millis
        self.sqs.create_queue(QueueName=sqsQueueName)
        self.sqsQueueUrl = self.sqs.get_queue_url(
            QueueName=sqsQueueName)['QueueUrl']

        attribs = self.sqs.get_queue_attributes(QueueUrl=self.sqsQueueUrl,
                                                AttributeNames=['QueueArn'])['Attributes']

        sqsQueueArn = attribs['QueueArn']

        # Subscribe SQS queue to SNS topic
        self.sns.subscribe(
            TopicArn=self.snsTopicArn,
            Protocol='sqs',
            Endpoint=sqsQueueArn)

        # Authorize SNS to write SQS queue
        policy = """{{
  "Version":"2012-10-17",
  "Statement":[
    {{
      "Sid":"MyPolicy",
      "Effect":"Allow",
      "Principal" : {{"AWS" : "*"}},
      "Action":"SQS:SendMessage",
      "Resource": "{}",
      "Condition":{{
        "ArnEquals":{{
          "aws:SourceArn": "{}"
        }}
      }}
    }}
  ]
}}""".format(sqsQueueArn, self.snsTopicArn)

        response = self.sqs.set_queue_attributes(
            QueueUrl=self.sqsQueueUrl,
            Attributes={
                'Policy': policy
            })

    def DeleteTopicandQueue(self):
        self.sqs.delete_queue(QueueUrl=self.sqsQueueUrl)
        self.sns.delete_topic(TopicArn=self.snsTopicArn)


def main():

    roleArn = 'arn:aws:iam::637423267378:role/LabRole'
    bucket = 'lab-video-search'
    video_s3 = 'C0885v2.MP4'

    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')
    rek = boto3.client('rekognition')
    sqs = boto3.client('sqs')
    sns = boto3.client('sns')

    analyzer = VideoDetect(roleArn, bucket, video_s3, client, rek, sqs, sns)
    analyzer.CreateTopicandQueue()

    collection = 'myCollection1'
    analyzer.StartFaceSearchCollection(collection)
    if analyzer.GetSQSMessageSuccess() == True:
        # analyzer.GetFaceSearchCollectionResults()
        results = analyzer.GetFaceSearchCollectionResults()
        results_json = json.dumps(results, indent=4, ensure_ascii=False)
        with open('serach_results02.json', 'w', encoding='utf-8') as f:
            f.write(results_json)

    analyzer.DeleteTopicandQueue()


if __name__ == "__main__":
    main()
