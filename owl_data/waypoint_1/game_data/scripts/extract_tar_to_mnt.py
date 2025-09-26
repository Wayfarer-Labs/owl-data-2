import io, logging, os, sys, shutil, pathlib, tarfile, functools, boto3
from dotenv import load_dotenv

load_dotenv()

@functools.cache
def s3_client() -> boto3.client:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
        region_name=os.getenv('AWS_REGION')
    )
    return s3_client

def _is_kbm(metadata: dict) -> bool:
    pass

def _is_controller(metadata: dict) -> bool:
    pass

def _is_first_person_shooter(metadata: dict) -> bool:
    pass

def _is_third_person(metadata: dict) -> bool:
    pass

TASK_LIST_PATH = pathlib.Path('task_list.txt')
MNT_DST_PATH = pathlib.Path('/mnt/data/datasets/waypoint_1/')

def download_tar(task_id: str, bucket_name: str = 'game-data') -> pathlib.Path:
    """
    Downloads a tar file from the bucket.
    """
    response = s3_client().get_object(Bucket=bucket_name, Key=task_id)
    tar_bytes = response['Body'].read()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as tar:
        member_names = [m.name for m in tar.getmembers()]
        pass


def extract_tar(tar_path: pathlib.Path) -> None:
    """
    Takes a tar file and extracts it into a directory with the same name as the tar file.

    For example: uuid.tar -> uuid/
    """
    pass


if __name__ == '__main__':
    download_tar('001bdcb568244377.tar')