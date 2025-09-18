import os
import boto3
from dotenv import load_dotenv


load_dotenv()

bucket_name = "game-data"

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
    region_name=os.getenv('AWS_REGION')
)

response = s3.list_objects_v2(Bucket=bucket_name)

for obj in response.get('Contents', []):
    key = obj['Key']
    if key.endswith('.tar'):
        tar_path = os.path.join("tmp_tar", key)
        s3.download_fileobj(bucket_name, key, tar_path)
        print(f"Downloaded {key} to {tar_path}")
        break

