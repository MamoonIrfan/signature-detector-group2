"""Private R2 objects via the S3 API. Secret values remain in process memory."""
from __future__ import annotations
import json, mimetypes, os, re
from pathlib import Path
from common import sha256_file, safe_relative


class R2UploadError(RuntimeError):
    pass


class R2Store:
    def __init__(self, bucket, prefix="signature-detection", client=None):
        if not bucket or "/" in bucket:
            raise ValueError("A valid R2 bucket name is required")
        self.bucket = bucket
        self.prefix = str(safe_relative(prefix.strip("/")))
        if client is None:
            import boto3
            from botocore.config import Config
            account = os.environ.get("R2_ACCOUNT_ID", "").strip()
            access = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
            secret = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
            if not re.fullmatch(r"[0-9a-fA-F]{32}", account) or not access or not secret:
                raise ValueError("Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY in Colab Secrets or environment variables")
            client = boto3.client("s3", endpoint_url="https://"+account+".r2.cloudflarestorage.com",
                aws_access_key_id=access, aws_secret_access_key=secret, region_name="auto",
                config=Config(signature_version="s3v4", retries={"mode":"standard","max_attempts":5},
                    connect_timeout=15, read_timeout=90,
                    request_checksum_calculation="when_required", response_checksum_validation="when_required"))
        self.client = client

    @classmethod
    def from_env(cls, prefix="signature-detection"):
        return cls(os.environ.get("R2_BUCKET", "").strip(), prefix)

    def key(self, relative):
        return self.prefix + "/" + str(safe_relative(relative))

    @staticmethod
    def _error(exc, operation):
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", type(exc).__name__))
        return R2UploadError(operation + " failed (" + code + "). Check bucket permissions and credentials; local files were retained.")

    def check_connection(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception as exc:
            raise self._error(exc, "R2 connection check") from None

    def exists(self, relative):
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(relative))
            return True
        except Exception as exc:
            response = getattr(exc, 'response', {})
            code = str(response.get('Error', {}).get('Code', ''))
            status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            if code in ('NoSuchKey', 'NotFound', '404') or status == 404:
                return False
            raise self._error(exc, 'R2 existing-run check') from None

    def put_file(self, source, relative, content_type=None):
        source = Path(source)
        digest, key = sha256_file(source), self.key(relative)
        content_type = content_type or mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        try:
            self.client.upload_file(str(source), self.bucket, key,
                ExtraArgs={"ContentType":content_type,"Metadata":{"sha256":digest}})
            actual = self.client.head_object(Bucket=self.bucket, Key=key)
            if actual["ContentLength"] != source.stat().st_size or actual.get("Metadata",{}).get("sha256") != digest:
                raise R2UploadError("Uploaded object verification failed")
        except Exception as exc:
            raise self._error(exc, "R2 file upload") from None
        return {"bucket":self.bucket,"key":key,"sha256":digest,"size_bytes":source.stat().st_size}

    def put_json(self, payload, relative):
        import hashlib
        body = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
        digest, key = hashlib.sha256(body).hexdigest(), self.key(relative)
        try:
            self.client.put_object(Bucket=self.bucket,Key=key,Body=body,ContentType="application/json",Metadata={"sha256":digest})
        except Exception as exc:
            raise self._error(exc, "R2 manifest upload") from None
        return {"bucket":self.bucket,"key":key,"sha256":digest,"size_bytes":len(body)}

    def get_json(self, relative):
        try:
            return json.loads(self.client.get_object(Bucket=self.bucket,Key=self.key(relative))["Body"].read())
        except Exception as exc:
            raise self._error(exc, "R2 manifest download") from None

    def download_verified(self, record, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if record.get("bucket") != self.bucket or not record["key"].startswith(self.prefix+"/"):
            raise ValueError("Checkpoint is outside the configured bucket/prefix")
        temporary = destination.with_name(destination.name+".part")
        try:
            self.client.download_file(self.bucket, record["key"], str(temporary))
            if sha256_file(temporary) != record["sha256"]:
                raise R2UploadError("Downloaded checkpoint hash mismatch")
            os.replace(temporary,destination)
        except Exception as exc:
            raise self._error(exc,"R2 checkpoint download") from None
        return destination
