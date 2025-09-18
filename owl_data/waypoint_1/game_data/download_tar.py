import boto3
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
load_dotenv()


bucket_name = "game-data"
after_date = datetime(2025, 8, 28, tzinfo=timezone.utc)

import shutil
import tarfile
import random

class TarDownloader:
    def __init__(self, max_size_mb=100, min_size_mb=5, earliest_date=datetime(2025, 8, 28, tzinfo=timezone.utc)):
        self.min_size_mb = min_size_mb
        self.earliest_date = earliest_date
        self.max_size_mb = max_size_mb

        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
            region_name=os.getenv('AWS_REGION')
        )
        self.bucket_name = bucket_name
        self.tmp_dir = "tmp_tar"
        self._cleanup_tmp_dir()

    def _cleanup_tmp_dir(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def __del__(self):
        self._cleanup_tmp_dir()

    def _get_eligible_tars(self):
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)
        eligible = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            last_modified = obj['LastModified']
            size = obj['Size']
            size_mb = size / (1024 * 1024)
            if (
                key.endswith('.tar')
                and last_modified > self.earliest_date
                and self.min_size_mb <= size_mb <= self.max_size_mb
            ):
                eligible.append((key, size))
        return eligible

    def temp_download(self):
        # Clean up and recreate tmp_tar directory
        self._cleanup_tmp_dir()
        eligible_tars = self._get_eligible_tars()
        if not eligible_tars:
            raise RuntimeError("No eligible tar files found.")
        tar_key, _ = random.choice(eligible_tars)
        tar_path = os.path.join(self.tmp_dir, "archive.tar")
        # Download the tar file
        with open(tar_path, "wb") as f:
            self.s3.download_fileobj(self.bucket_name, tar_key, f)
        # Extract tar contents into tmp_tar
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(self.tmp_dir)
        # Optionally, remove the tar file itself after extraction
        os.remove(tar_path)
        return self.tmp_dir

if __name__ == "__main__":
    downloader = TarDownloader()
    tmp_dir = downloader.temp_download()
    print(tmp_dir)
    print(os.listdir(tmp_dir))