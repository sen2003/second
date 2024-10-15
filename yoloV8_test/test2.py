import boto3
import json
import sys
import time
import torch
import cv2
from ultralytics import YOLO


# 定義中文名稱映射
def chinese_name(externalImageId):
    name_map = {
        "Huang": "黃士熏",
        "Ke": "柯信汶",
        "Shen": "沈宏勳",
        "Tsou": "鄒博森"
    }
    return name_map.get(externalImageId, "Unknow person")


# AWS Rekognition 的 VideoDetect 類
class VideoDetect:

    def __init__(self, role, bucket, video, client, rek, sqs, sns):
        self.roleArn = role
        self.bucket = bucket
        self.video = video
        self.client = client
        self.rek = rek
        self.sqs = sqs
        self.sns = sns
        self.jobId = ''
        self.sqsQueueUrl = ''
        self.snsTopicArn = ''
        self.startJobId = ''

    def GetSQSMessageSuccess(self):

        jobFound = False
        succeeded = False

        dotLine = 0
        while not jobFound:
            sqsResponse = self.sqs.receive_message(QueueUrl=self.sqsQueueUrl, MessageAttributeNames=['ALL'],
                                                   MaxNumberOfMessages=10)

            if sqsResponse:

                if 'Messages' not in sqsResponse:
                    if dotLine < 40:
                        print('.', end='')
                        dotLine += 1
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
                        if rekMessage['Status'] == 'SUCCEEDED':
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
                                              FaceMatchThreshold=0)

        self.startJobId = response['JobId']
        print('Start Job Id: ' + self.startJobId)

    def GetFaceSearchCollectionResults(self):
        maxResults = 10
        paginationToken = ''
        video_info = False
        finished = False
        results = []
        while not finished:
            response = self.rek.get_face_search(JobId=self.startJobId,
                                                MaxResults=maxResults,
                                                NextToken=paginationToken)
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
                if 'FaceMatches' in personMatch and len(personMatch['FaceMatches']) > 0:
                    for faceMatch in personMatch['FaceMatches']:
                        face = faceMatch['Face']
                        match_data = {
                            "Timestamp": personMatch['Timestamp'],
                            "BoundingBox": personFace['BoundingBox'],
                            "Name": chinese_name(face['ExternalImageId']),
                            "Similarity": faceMatch['Similarity']
                        }
                        results.append(match_data)

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
    # AWS Rekognition
    roleArn = 'arn:aws:iam::637423267378:role/LabRole'
    bucket = 'lab-video-search'
    video = 'C0885v2.MP4'

    session = boto3.Session(profile_name='default')
    client = session.client('rekognition')
    rek = boto3.client('rekognition')
    sqs = boto3.client('sqs')
    sns = boto3.client('sns')

    analyzer = VideoDetect(roleArn, bucket, video, client, rek, sqs, sns)
    analyzer.CreateTopicandQueue()

    collection = 'myCollection1'
    analyzer.StartFaceSearchCollection(collection)
    if analyzer.GetSQSMessageSuccess():
        results = analyzer.GetFaceSearchCollectionResults()
        with open('rekognition_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    analyzer.DeleteTopicandQueue()

    # YOLOv8
    # tmp_filename = './input.mp4'
    # s3_client = boto3.resource('s3')
    # s3_client.meta.client.download_file(bucket, video, tmp_filename)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # model = YOLO("../yoloV8_test/yolov8m-face.pt").to(device)

    # cap = cv2.VideoCapture(tmp_filename)
    # if not cap.isOpened():
    #     raise Exception("影片無法開啟")

    with open('rekognition_results.json', 'r', encoding='utf-8') as f:
        rekognition_results = json.load(f)

    # def check_overlap(yolo_box, rek_box):
    #     x1, y1, x2, y2 = yolo_box
    #     rx1 = rek_box["Left"]
    #     ry1 = rek_box["Top"]
    #     rx2 = rek_box["Left"] + rek_box["Width"]
    #     ry2 = rek_box["Top"] + rek_box["Height"]

    #     overlap = not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)
    #     return overlap

    # frame_count = 0

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     time_start = time.time()
    #     results = model(frame)

    # # 繪製YOLO檢測到的框
    #     for result in results:
    #         for box in result.boxes:
    #             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    #             confidence = box.conf[0].item()
    #             label = f"{confidence:.2f}"

    #         # 檢查是否有Rekognition結果重疊
    #             matched = False
    #             for rek_result in rekognition_results:
    #                 timestamp = rek_result["Timestamp"]
    #                 if frame_count == int(timestamp / 33.33):  # assuming 30 fps
    #                     rek_box = rek_result["BoundingBox"]
    #                     name = rek_result["Name"]
    #                     if name != "Unknow" and check_overlap((x1, y1, x2, y2), rek_box):
    #                         similarity = float(rek_result["Similarity"])
    #                         label = f"{name}: {similarity:.2f}%"
    #                         cv2.rectangle(frame, (x1, y1),
    #                                       (x2, y2), (0, 0, 255), 2)
    #                         cv2.putText(frame, label, (x1, y1 - 10),
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #                         matched = True
    #                         break

    #         # 如果沒有匹配到Rekognition結果，使用綠色框
    #             if not matched:
    #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                 cv2.putText(frame, label, (x1, y1 - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #     frame_count += 1

    #     frame_resized = cv2.resize(frame, (1920, 1080))
    #     cv2.imshow("YOLO Detection", frame_resized)

    #     print("success", time.time() - time_start)

    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
