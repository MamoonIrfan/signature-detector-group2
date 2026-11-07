import boto3
import os

s3 = boto3.client(
    's3',
    endpoint_url = 'https://<account_id>.r2.cloudflarestorage.com',
    aws_access_key_id = os.getenv("R2_ACCESS_KEY"),
    aws_secret_access_key = os.getenv("R2_SECRET_KEY"),
    region_name = "auto"
)
BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

def upload_to_r2(file_path, filename):
    s3.upload_file(file_path, BUCKET_NAME, filename)
    return f"https://pub-<id>.r2.dev/{filename}"
