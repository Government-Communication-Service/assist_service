"""Infrastructure-level unit tests for S3Service.

These tests mock AWS S3 using moto and test the S3Service class directly.
No authentication fixtures since testing the infrastructure layer, not API endpoints.
"""

import asyncio
import io
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from app.aws_services.s3_service import S3Service, _check_and_create_s3_bucket


@pytest.fixture
def sample_content():
    return io.BytesIO(b"test file content")


@mock_aws
def test_upload_file_with_existing_bucket(sample_content):
    """Test upload to existing bucket"""
    s3_service = S3Service(aws_region="us-east-1")

    # Create bucket first
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket="existing-bucket")

    result = asyncio.run(
        s3_service.upload_file(bucket_name="existing-bucket", key="test-file.txt", content=sample_content)
    )

    assert result is True


@mock_aws
def test_upload_file_without_bucket_creation(sample_content):
    """Test upload fails when bucket doesn't exist and create_bucket=False"""
    s3_service = S3Service(aws_region="us-east-1")

    result = asyncio.run(
        s3_service.upload_file(
            bucket_name="nonexistent-bucket", key="test-file.txt", content=sample_content, create_bucket=False
        )
    )

    assert result is False


@mock_aws
def test_create_bucket_when_not_exists():
    """Test bucket creation when it doesn't exist"""
    s3_client = boto3.client("s3", region_name="us-east-1")
    _check_and_create_s3_bucket(s3_client, "new-bucket", "us-east-1")

    # Verify bucket was created
    response = s3_client.list_buckets()
    bucket_names = [bucket["Name"] for bucket in response["Buckets"]]
    assert "new-bucket" in bucket_names


@mock_aws
def test_no_action_when_bucket_exists():
    """Test no action taken when bucket already exists"""
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket="existing-bucket")

    # Should not raise error
    _check_and_create_s3_bucket(s3_client, "existing-bucket", "us-east-1")

    # Bucket should still exist
    response = s3_client.list_buckets()
    bucket_names = [bucket["Name"] for bucket in response["Buckets"]]
    assert "existing-bucket" in bucket_names


@mock_aws
@patch("app.aws_services.s3_service.logger")
def test_logs_bucket_creation(mock_logger):
    """Test that bucket creation is logged"""
    s3_client = boto3.client("s3", region_name="us-east-1")
    _check_and_create_s3_bucket(s3_client, "new-bucket", "us-east-1")

    mock_logger.info.assert_called_with("Created S3 bucket: new-bucket")
