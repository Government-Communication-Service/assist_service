import asyncio
import logging
from typing import BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _check_and_create_s3_bucket(s3_client, bucket_name: str, region: str):
    """
    Check if S3 bucket exists and create it if it doesn't exist.

    This is a synchronous function designed to be executed in a thread pool
    to avoid blocking the async event loop.

    Args:
        s3_client: Boto3 S3 client instance
        bucket_name (str): Name of the S3 bucket to check/create
        region (str): AWS region where the bucket should be created

    Raises:
        ClientError: If bucket creation fails for reasons other than already existing
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # us-east-1 is the default region and doesn't need LocationConstraint
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region})
            logger.info(f"Created S3 bucket: {bucket_name}")


class S3Service:
    """Generic S3 service for uploading files to any bucket"""

    def __init__(self, aws_region: str):
        self.s3_client = boto3.client(
            "s3",
            region_name=aws_region,
            config=Config(signature_version="s3v4"),
        )
        self.aws_region = aws_region

    async def upload_file(self, bucket_name: str, key: str, content: BinaryIO, create_bucket: bool = True) -> bool:
        """
        Upload a file to S3

        Args:
            bucket_name: S3 bucket name
            key: S3 object key (path/filename)
            content: File content as BinaryIO
            create_bucket: Whether to create bucket if it doesn't exist

        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            # Ensure bucket exists if requested
            if create_bucket:
                await asyncio.to_thread(_check_and_create_s3_bucket, self.s3_client, bucket_name, self.aws_region)

            # Upload file
            await asyncio.to_thread(self.s3_client.put_object, Bucket=bucket_name, Key=key, Body=content)

            logger.info(f"Uploaded file to S3: s3://{bucket_name}/{key}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return False
