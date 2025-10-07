from __future__ import annotations
from collections import defaultdict
import os, io, pathlib, boto3, logging
from dataclasses import dataclass
from owl_data.waypoint_1.game_data.constants import MAX_FILE_SIZE_BYTES


S3ClientType = type(boto3.client('s3'))
_s3_client: S3ClientType = boto3.client('s3')
_raw_data_bucket: str = 'todo'
_extracted_data_bucket: str = 'todo'


@dataclass 
class DownsampledTAR_Data:
    s3_key: str
    downsampled_video_bytes: list[bytes]
    controls_csv_str: str
    session_metadata: dict
    in_video_metadata: dict
    out_video_metadata: list[dict]


class ExtractedData:
    @staticmethod
    def set_s3_client(s3_client: S3ClientType) -> None:
        global _s3_client
        _s3_client = s3_client

    @staticmethod
    def set_raw_data_bucket(src_bucket: str) -> None:
        global _download_bucket
        _download_bucket = src_bucket

    @staticmethod
    def set_extracted_data_bucket(dst_bucket: str) -> None:
        global _extracted_data_bucket
        _extracted_data_bucket = dst_bucket

    @staticmethod
    def downsample_raw_data_to_tmp(
        raw_tar_bytes: io.BytesIO
    ) -> pathlib.Path:
        # takes a src tar, iterates over data from it, saves extracted data to dst_tar_path
        # extracts tar into tmp dir
        """
        Extracts the input TAR to a temp dir, finds the mp4, runs downsampling,
        packages only the downsampled mp4 chunks into a new TAR on disk, cleans temps,
        and returns the local path to the new TAR.
        """
        import tempfile, tarfile, shutil
        raw_tar_bytes.seek(0)

        work_dir = pathlib.Path(tempfile.mkdtemp(prefix="owl_downsample_"))
        extracted_dir = work_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        tmp_tar_path: pathlib.Path | None = None
        downsampled_paths: list[pathlib.Path] = []
        csv_path = mp4_path = json_path = None

        try:
            # 1) Extract original tar to temp (safe)
            with tarfile.open(fileobj=raw_tar_bytes, mode='r') as tar:
                tar.extractall(path=extracted_dir, filter='data')

            # 2) Find mp4, csv, json paths:
            tar_file_map: defaultdict[str, list[pathlib.Path]] = defaultdict(list)

            for root, _, files in os.walk(extracted_dir):
                for fn in files:
                    tar_file_map[pathlib.Path(fn).suffix].append(pathlib.Path(root) / fn)

            required = ['.csv', '.mp4', '.json']
            if not all(ext in tar_file_map for ext in required):
                missing = [ext for ext in required if ext not in tar_file_map]
                logging.error(f"Missing required files in TAR: {missing}")
                raise FileNotFoundError(f"Missing required files: {missing}")

            # Validation: exactly one of each required file and it exists
            for extension in required:
                files = tar_file_map.get(extension, [])
                if len(files) != 1 or not files[0].exists():
                    logging.error(f"Multiple or no {extension} found: count={len(files)}")
                    raise FileNotFoundError(f"Multiple or no {extension} found: count={len(files)}")

            csv_path, mp4_path, json_path = (
                tar_file_map['.csv'][0],
                tar_file_map['.mp4'][0],
                tar_file_map['.json'][0]
            )

            # 3) Downsample to chunks (outputs under /tmp/<stem>/...)
            import ffmpeg, json
            in_video_metadata: dict = ffmpeg.probe(mp4_path)
            in_video_metadata_path = extracted_dir / 'in_video_metadata.json'
            # create & write
            with open(in_video_metadata_path, 'w+') as f:
                json.dump(in_video_metadata, f)

            from owl_data.waypoint_1.game_data.utils.mp4_utils import downsample_video_from_path
            downsampled_paths = downsample_video_from_path(mp4_path)
            all_paths = [csv_path, json_path, in_video_metadata_path, *downsampled_paths]

            # 4) Create a new tar containing ONLY the downsampled mp4 chunks
            tmp_tar = tempfile.NamedTemporaryFile(prefix="downsampled_", suffix=".tar", delete=False)
            tmp_tar_path = pathlib.Path(tmp_tar.name)
            tmp_tar.close()
            with tarfile.open(tmp_tar_path, mode="w") as out_tar:
                for p in all_paths:
                    if p.exists(): out_tar.add(str(p), arcname=p.name)
                    else: logging.warning(f"File missing: {p}")

            logging.info(f"Created downsampled TAR at {tmp_tar_path} with {len(downsampled_paths)} chunks")

            return tmp_tar_path

        finally:
            # Cleanup downsampled files and extracted tree
            try:
                # Remove downsampled files
                for p in downsampled_paths or []:
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception as e:
                        logging.warning(f"Failed to remove downsampled file {p}: {e}")

                # Also remove extracted csv/json if present
                for p in [csv_path, json_path]:
                    try:
                        if p and pathlib.Path(p).exists():
                            pathlib.Path(p).unlink()
                    except Exception as e:
                        logging.warning(f"Failed to remove extracted file {p}: {e}")

                # Try to remove their parent directory if empty
                try:
                    if downsampled_paths:
                        downsampled_paths[0].parent.rmdir()
                except Exception:
                    pass

            finally:
                import shutil as _shutil
                _shutil.rmtree(work_dir, ignore_errors=True)


    @staticmethod
    def upload_extracted_data_to_s3(
        src_tar_path: pathlib.Path,
        s3_key: str,
        *,
        cleanup_tmp: bool = True
    ) -> None:
        global _extracted_data_bucket, _s3_client

        if not src_tar_path or not pathlib.Path(src_tar_path).exists():
            raise FileNotFoundError(f"Source TAR not found: {src_tar_path}")

        bucket = _extracted_data_bucket
        logging.info(f"Uploading downsampled TAR to s3://{bucket}/{s3_key} from {src_tar_path}")
        try:
            _s3_client.upload_file(
                Filename=str(src_tar_path),
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs={"ContentType": "application/x-tar"}
            )
            logging.info(f"Uploaded downsampled TAR to s3://{bucket}/{s3_key}")
        except Exception as e:
            logging.error(f"Failed to upload {src_tar_path} to s3://{bucket}/{s3_key}: {e}", exc_info=True)
            raise
        finally:
            if cleanup_tmp:
                try:
                    os.unlink(src_tar_path)
                    logging.info(f"Removed tmp file {src_tar_path}")
                except Exception as e:
                    logging.warning(f"Failed to remove tmp file {src_tar_path}: {e}")


    @staticmethod
    def download_raw_data_from_s3(
        s3_key: str,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES
    ) -> tuple[io.BytesIO | None, int]:
        global _download_bucket, _s3_client
        meta = _s3_client.head_object(Bucket=_download_bucket, Key=s3_key)
        size = meta['ContentLength']

        if size > max_file_size_bytes or size == 0:
            logging.warning(f"SKIPPING {s3_key} - Invalid size: {size / 1e6:.2f} MB.")
            return None, size

        logging.info(f"Downloading {s3_key} ({size / 1e6:.2f} MB)...")
        response = _s3_client.get_object(Bucket=_download_bucket, Key=s3_key)
        tar_bytes = response['Body'].read()
        return tar_bytes, size


    @staticmethod
    def download_extracted_data_from_s3(
        key: str,
        dst_tar_path: pathlib.Path,
    ) -> tuple[ExtractedData, pathlib.Path]:
        global _extracted_data_bucket, _s3_client

        dst_tar_path = pathlib.Path(dst_tar_path)
        dst_tar_path.parent.mkdir(parents=True, exist_ok=True)
        bucket = _extracted_data_bucket
        logging.info(f"Downloading downsampled TAR s3://{bucket}/{key} to {dst_tar_path}")
        try:
            response = _s3_client.get_object(Bucket=bucket, Key=key)
            body = response['Body']
            with open(dst_tar_path, 'wb') as f:
                while True:
                    chunk = body.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            # Minimal placeholder; contents can be populated by a later load/parsing step if needed
            data = ExtractedData(
                s3_key=key,
                downsampled_video_bytes=[],
                controls_csv_str="",
                session_metadata={},
                in_video_metadata={},
                out_video_metadata=[]
            )
            logging.info(f"Downloaded to {dst_tar_path}")
            return data, dst_tar_path
        except Exception as e:
            logging.error(f"Failed to download s3://{bucket}/{key}: {e}", exc_info=True)
            raise

    @staticmethod
    def get_tar_mismatches_in_buckets() -> list[str]:
        pass

    @staticmethod
    def read_downsampled_tar_bytes(tar: io.BytesIO, s3_key: str) -> DownsampledTAR_Data:
        import tarfile, json, tempfile, os
        tar.seek(0)

        downsampled_video_bytes: list[bytes] = []
        controls_csv_str: str = ""
        session_metadata: dict = {}
        in_video_metadata: dict = {}
        out_video_metadata: list[dict] = []

        with tarfile.open(fileobj=tar, mode='r') as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name_lower = member.name.lower()
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                data = fobj.read()

                if name_lower.endswith(".mp4"):
                    downsampled_video_bytes.append(data)
                    # Try to probe chunk metadata via a temp file for out_video_metadata
                    try:
                        import ffmpeg
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_mp4:
                            tmp_mp4.write(data)
                            tmp_mp4.flush()
                            meta = ffmpeg.probe(tmp_mp4.name)
                        try:
                            os.unlink(tmp_mp4.name)
                        except Exception:
                            pass
                        out_video_metadata.append(meta)
                    except Exception as e:
                        logging.warning(f"Failed probing downsampled chunk '{member.name}': {e}")

                elif name_lower.endswith(".csv"):
                    try:
                        controls_csv_str = data.decode('utf-8', errors='replace')
                    except Exception:
                        controls_csv_str = ""

                elif name_lower.endswith("in_video_metadata.json"):
                    try:
                        in_video_metadata = json.loads(data)
                    except Exception as e:
                        logging.warning(f"Failed to parse in_video_metadata.json: {e}")

                elif name_lower.endswith(".json"):
                    try:
                        session_metadata = json.loads(data)
                    except Exception as e:
                        logging.warning(f"Failed to parse session metadata json '{member.name}': {e}")

        return DownsampledTAR_Data(
            s3_key=s3_key,
            downsampled_video_bytes=downsampled_video_bytes,
            controls_csv_str=controls_csv_str,
            session_metadata=session_metadata,
            in_video_metadata=in_video_metadata,
            out_video_metadata=out_video_metadata
        )

    @staticmethod
    def read_downsampled_tar_path(tar_path: pathlib.Path, s3_key: str) -> DownsampledTAR_Data:
        with open(tar_path, 'rb') as f:
            buf = io.BytesIO(f.read())
        return ExtractedData.read_downsampled_tar_bytes(buf, s3_key)


if __name__ == "__main__":
    raw_tar_name = '003392b7230e411a.tar'
    