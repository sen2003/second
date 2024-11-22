import boto3

def empty_bucket(bucket_name):
    s3 = boto3.client('s3')
    count = 0
    
    try:
        objects = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in objects:
            items = [{'Key': obj['Key']} for obj in objects['Contents']]
            s3.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': items}
            )
            count = len(items)
            print(f"已清空 bucket: {bucket_name}")
            print(f"共刪除 {count} 張圖片")
        else:
            print(f"Bucket {bucket_name} 是空的")
            
    except Exception as e:
        print(f"發生錯誤: {e}")

# 使用方式
bucket_name = "img-face-rekognition"
empty_bucket(bucket_name)