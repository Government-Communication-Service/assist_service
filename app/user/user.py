import asyncio
import io
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile, status
from fastapi.responses import JSONResponse
from unstructured.partition.common import UnsupportedFileFormatError
from unstructured_pytesseract.pytesseract import TesseractNotFoundError

from app.api import ENDPOINTS, endpoint_defaults
from app.api.endpoint_defaults import get_current_session
from app.api.paths import ApiPaths
from app.auth.auth_token import auth_token_validator
from app.auth.session_request import SessionRequest
from app.database.db_operations import DbOperations
from app.database.table import async_db_session
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


@router.put(
    ENDPOINTS.USER,
    response_model=UserCreationResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(auth_token_validator)],
)
async def update_user(user_uuid: Annotated[str, Path()], userinput: UserInput) -> UserCreationResponse:
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

    async with async_db_session() as db_session:
        result = await DbOperations.update_user(db_session, user_uuid, userinput)
        if not result.success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result.message)
        return result


@router.post(
    ENDPOINTS.USERS,
    response_model=UserCreationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(auth_token_validator)],
)
async def create_user(
    user: UserCreationInput,
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

    async with async_db_session() as db_session:
        result = await DbOperations.create_user(db_session, user)
        if not result.success:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=result.message)
        return result


@router.get(ENDPOINTS.USER_DOCUMENTS, response_model=ListDocumentResponse, **endpoint_defaults())
async def list_documents(
    user=ApiPaths.USER_UUID,
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

    async with async_db_session() as db_session:
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


@router.delete(ENDPOINTS.USER_DOCUMENT, response_model=DocumentResponse, **endpoint_defaults())
async def delete_document(
    document_uuid: str,
    user=ApiPaths.USER_UUID,
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

    async with async_db_session() as db_session:
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
    response_model=UploadDocumentResponse,
    **endpoint_defaults(),
)
async def upload_file(
    file: UploadFile = File(...),
    user=ApiPaths.USER_UUID,
    auth_session: SessionRequest = Depends(get_current_session),
) -> UploadDocumentResponse:
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
    try:
        file_info = FileInfo(filename=file.filename, content=io.BytesIO(await file.read()))
        new_document = await document_parser.process_document(file=file_info, auth_session=auth_session, user=user)

        # Return a response model object
        return UploadDocumentResponse(
            message="File parsed and saved successfully",
            document_uuid=str(new_document.uuid),
        )
    except NoTextContentError as ex:
        logger.info(f"User uploaded file with no text content: {file.filename}")
        content = {
            "error_code": "NO_TEXT_CONTENT_ERROR",
            "status": "failed",
            "status_message": str(ex),
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    except FileFormatError as ex:
        logger.info(f"User uploaded file not supported: {file.filename}")
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
        logger.warning(f"UnsupportedFileFormatError: {ex},", extra={"file_name": f"{file.filename}"})
        if file.filename.lower().endswith(".docx"):
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
        bad_word_error_string = (
            "no relationship of type 'http://schemas.openxmlformats.org/"
            "officeDocument/2006/relationships/officeDocument"
        )

        if file.filename.lower().endswith(".docx") and bad_word_error_string in str(e):
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
