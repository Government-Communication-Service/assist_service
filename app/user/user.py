import asyncio
import io
import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from unstructured.partition.common import UnsupportedFileFormatError
from unstructured_pytesseract.pytesseract import TesseractNotFoundError

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_path_and_header,
    verify_auth_token,
)
from app.aws_services.s3_service import S3Service
from app.config import AWS_DEFAULT_REGION, IS_DEV, S3_ERRORDOCS_BUCKET
from app.database.db_operations import DbOperations
from app.database.db_session import get_db_session
from app.database.models import AuthSession, User
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.document_upload.personal_document_parser import (
    FileFormatError,
    FileInfo,
    NoTextContentError,
    PersonalDocumentParser,
)
from app.opensearch.service import AsyncOpenSearchOperations
from app.user.schemas import (
    DocumentResponse,
    DocumentSchema,
    ListDocumentResponse,
    UploadDocumentResponse,
    UserCreationInput,
    UserCreationResponse,
    UserInput,
)

router = APIRouter()

logger = logging.getLogger(__name__)
document_parser = PersonalDocumentParser()
s3_service = S3Service(AWS_DEFAULT_REGION)


async def upload_failed_document(file_content: bytes, filename: str, user: User, error_type: str):
    """Upload failed document to S3 for debugging (only in production)"""
    if not IS_DEV:
        try:
            content_stream = io.BytesIO(file_content)
            key = f"user_{user.uuid}/{error_type}/{datetime.now().isoformat('T', 'seconds')}_{filename or 'unknown'}"
            await s3_service.upload_file(bucket_name=S3_ERRORDOCS_BUCKET, key=key, content=content_stream)
        except Exception as e:
            logger.error(f"Failed to upload error document to S3: {e}")


@router.put(
    ENDPOINTS.USER,
    response_model=UserCreationResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_auth_token)],
)
async def update_user(
    user_uuid: Annotated[str, Path()],
    userinput: UserInput,
    db_session: AsyncSession = Depends(get_db_session),
) -> UserCreationResponse:
    """
    Update an existing user's profile information.

    Args:
        user_uuid (UUID): The unique identifier of the user to update, provided in the URL path
        userInput (UserInput): The updated user profile information containing:
            - job_title: User's job title
            - region: User's region
            - sector: User's sector
            - organisation: User's organization
            - grade: User's grade
            - communicator_role: User's communicator role

    Returns:
        UserCreationResponse: Response object containing:
            - success (bool): True if update successful, False otherwise
            - message (str): Description of the operation result

    Usage:
        POST /user/{user_uuid}

    Raises:
        HTTPException: 404 if user not found
        HTTPException: 422 if validation fails
    """
    result = await DbOperations.update_user(db_session, user_uuid, userinput)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result.message)
    return result


@router.post(
    ENDPOINTS.USERS,
    response_model=UserCreationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_auth_token)],
)
async def create_user(
    user: UserCreationInput,
    db_session: AsyncSession = Depends(get_db_session),
) -> UserCreationResponse:
    """
    Create a new user with the provided details.

    Args:
        user (UserCreationInput): The user JSON details to create.

    Returns:
        UserCreationResponse: Contains a success flag and msg.

    Usage:
        POST /users

    Raises:
        HTTPException: 409 if user already exists
        HTTPException: 422 if validation fails
    """
    result = await DbOperations.create_user(db_session, user)
    if not result.success:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=result.message)
    return result


@router.get(
    path=ENDPOINTS.USER_DOCUMENTS,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ListDocumentResponse,
)
async def list_documents(
    user=Depends(verify_and_get_user_from_path_and_header),
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Return user documents and central documents available for all users.
    The returned model contains user_documents for user documents and central_documents for central documents
    available for all users.

    Args:
        auth_session(SessionRequest): The session request object for authorization.
        db_session(AsyncSession): The database connection session.

    Returns:
        ListDocumentResponse: A response model user documents and central documents

    Usage:
        GET /user/{user_uuid}/documents
    """

    user_documents = await DbOperations.get_user_documents(db_session, user)
    central_documents = await DbOperations.get_central_documents(db_session)

    # Construct response lists
    user_documents = [
        DocumentSchema(
            uuid=doc.uuid,
            name=doc.name,
            created_at=doc.created_at,
            expired_at=doc.expired_at,
            last_used=doc.last_used,
        )
        for doc in user_documents
    ]
    central_documents = [
        DocumentSchema(uuid=doc.uuid, name=doc.name, created_at=doc.created_at) for doc in central_documents
    ]

    return ListDocumentResponse(user_documents=user_documents, central_documents=central_documents)


@router.delete(
    path=ENDPOINTS.USER_DOCUMENT,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=DocumentResponse,
)
async def delete_document(
    document_uuid: str,
    user=Depends(verify_and_get_user_from_path_and_header),
    db_session: AsyncSession = Depends(get_db_session),
) -> DocumentResponse:
    """
    Deletes a user's document mappings by marking it as deleted.
    If all document-user mappings for a document are marked as deleted,
     then marks document chunks and document as deleted as well and physically deletes the document in OpenSearch.

    Parameters:
    - document_uuid (str): UUID of the document to delete.
    - user (ApiPaths.USER_UUID): The user performing the delete action.
    - auth_session (SessionRequest): Dependency for retrieving the current authentication session.
    - db_session (Session): Dependency for retrieving the current database session.

    Returns:
    - DocumentResponse: A response indicating the document has been marked as deleted.

    Raises:
    - HTTPException (404): If the document or a mapping record not found for the user.

    """
    # Log the inputs
    logger.info(
        "Attempting to delete document mapping",
        extra={"document_uuid": document_uuid, "user_id": user.id},
    )

    # Retrieve document ID based on the UUID
    document_id = await DbOperations.get_document_by_uuid(db_session, document_uuid)

    if document_id is None:
        logger.info(
            "Document not found in the database.",
            extra={"document_uuid": document_id, "user_id": user.id},
        )
        return JSONResponse(
            status_code=404,
            content=DocumentResponse(message="Document not found", document_uuid=document_uuid).model_dump(),
        )

    # mark document mapping as deleted
    result = await DbOperations.mark_user_document_mapping_as_deleted(db_session, document_id, user)

    # Check if any rows were updated and user has document mapping.
    if result.rowcount == 0:
        logger.info(
            "No document mapping found",
            extra={"document_id": document_id, "user_id": user.id},
        )
        return JSONResponse(
            status_code=404,
            content=DocumentResponse(message="No Document mapping found", document_uuid=document_uuid).model_dump(),
        )

    logger.info(
        "User document mapping marked as deleted",
        extra={"document_id": document_id, "user_id": user.id},
    )

    # fetch id_opensearch list from document chunk table.
    id_opensearch = await DbOperations.get_opensearch_ids_from_document_chunks(db_session, document_id)

    await AsyncOpenSearchOperations.delete_document_chunks(PERSONAL_DOCUMENTS_INDEX_NAME, id_opensearch)
    await DbOperations.mark_document_as_deleted(db_session, document_id)

    logger.info(
        "Document marked as deleted.",
        extra={"document_uuid": document_uuid, "user_id": user.id},
    )
    return DocumentResponse(
        message="Document marked as deleted successfully.",
        document_uuid=document_uuid,
    )


@router.post(
    ENDPOINTS.USER_DOCUMENTS,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=UploadDocumentResponse,
)
async def upload_file(
    file: UploadFile = File(...),
    user: User = Depends(verify_and_get_user_from_path_and_header),
    auth_session: AuthSession = Depends(verify_and_get_auth_session_from_header),
) -> UploadDocumentResponse | JSONResponse:
    """
    Parses uploaded file and stores it in the database and opensearch index.

    Args:
        file (UploadFile): The file to be uploaded. It is required to be provided in the POST request.
        user (str): The UUID of the user, retrieved from API paths.

    Returns:
        UploadDocumentResponse: A response model containing a success message and the
        ID of the saved document.

    Raises:
        HTTPException: Raises a 400 error if the file format is unsupported.
        Exception: Re-raises any other exception encountered during processing.

    Usage:
        POST /user/{user_uuid}/documents
        Form Data:
            - description: "Description of the file"
            - file: File to upload
    """
    # Read file content once to avoid stream consumption issues
    file_content = await file.read()
    file_name = file.filename or "unknown"
    try:
        file_info = FileInfo(filename=file_name, content=io.BytesIO(file_content))
        new_document = await document_parser.process_document(file=file_info, auth_session=auth_session, user=user)

        # Return a response model object
        return UploadDocumentResponse(
            message="File parsed and saved successfully",
            document_uuid=str(new_document.uuid),
        )
    except NoTextContentError as ex:
        await upload_failed_document(file_content, file_name, user, "no_text_content")
        logger.info(f"User uploaded file with no text content: {file.filename}")
        content = {
            "error_code": "NO_TEXT_CONTENT_ERROR",
            "status": "failed",
            "status_message": str(ex),
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    except FileFormatError as ex:
        await upload_failed_document(file_content, file_name, user, "unsupported_format")
        logger.info(f"User uploaded file not supported: {file_name}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "failed",
                "error_code": "FILE_FORMAT_NOT_SUPPORTED",
                "supported_formats": ex.supported_formats,
                "status_message": str(ex),
            },
        )
    except asyncio.TimeoutError as ex:
        await upload_failed_document(file_content, file_name, user, "timeout")
        logger.info(f"Processing file timed out: {file.filename}, error: {ex}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "failed",
                "error_code": "FILE_PROCESSING_TIMEOUT_ERROR",
                "status_message": "Uploading document timed out, please try again",
            },
        )
    except TesseractNotFoundError:
        await upload_failed_document(file_content, file_name, user, "ocr_required")
        logger.warning(f"Uploaded document requires OCR tesseract tool, file: {file.filename}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "failed",
                "error_code": "DOCUMENTS_REQUIRING_OCR_NOT_SUPPORTED",
                "status_message": "This document does not contain any text."
                "It may contain scanned text or images of text,"
                " but Assist cannot process these. Please upload a document that contains the information"
                " in text format.",
            },
        )
    except UnsupportedFileFormatError as ex:
        await upload_failed_document(file_content, file_name, user, "unsupported_file_format")
        logger.warning(f"UnsupportedFileFormatError: {ex},", extra={"file_name": f"{file.filename}"})
        if file.filename and file.filename.lower().endswith(".docx"):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "failed",
                    "error_code": "UNSUPPORTED_WORD_DOCUMENT_VERSION",
                    "status_message": "The file uploaded is either not a word document, "
                    "or was generated with an older Word version,"
                    "Please use latest Word version or upload the document in PDF format",
                },
            )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "failed",
                "error_code": "UNSUPPORTED_DOCUMENT",
                "status_message": "Unsupported file uploaded, Please upload the file in Word or PDF format",
            },
        )
    except Exception as e:
        await upload_failed_document(file_content, file_name, user, "general_exception")

        bad_word_error_string = (
            "no relationship of type 'http://schemas.openxmlformats.org/"
            "officeDocument/2006/relationships/officeDocument"
        )

        if file.filename and file.filename.lower().endswith(".docx") and bad_word_error_string in str(e):
            logger.warning("Error uploading file: %s, error: %s", file.filename, e)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "failed",
                    "error_code": "UNSUPPORTED_DOCUMENT",
                    "status_message": "The file uploaded is either not a word document, "
                    "or was generated with an older Word version,"
                    "Please use latest Word version or upload the document in PDF format",
                },
            )

        logger.exception("Error uploading file: %s, error: %s", file.filename, e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "DOCUMENT_UPLOAD_ERROR",
                "status": "failed",
                "status_message": str(e),
            },
        )
