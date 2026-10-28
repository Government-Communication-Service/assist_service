# Auth Module: app/auth

The auth module is responsible for ensuring requests sent to the API are from an authorised source.

## Create a new Session-Auth token

Before any meaningful interaction with the Assist API, the user must first retrieve a Session-Auth token. This token is persisted by the client application. This token is required as a header in most endpoints in order for a user to take action.

```
from app.auth.create_auth_session_service import create_auth_session
```

## Verify Session-Auth token

This is required in most endpoints in order for a user to take action.

```
from app.auth.create_auth_session_service import create_auth_session
```

## Verify that a UUID is a valid UUID

This is resued across the app when additional verification is required (e.g. when a Message.uuid provided as a path parameter needs to verified).

```
from app.auth.utils import verify_and_parse_uuid
```
## Verify that the Auth-Token header is present and valid

This is used in most endpoints across the app.

```
from app.auth.verify_service import verify_auth_token
```

## Verify and upsert a user uuid from the path / User-Key-UUID header / both

One of these is used in all endpoints that use a user UUID in the header or path.

```
from app.auth.verify_service import (
    verify_and_get_user_from_header,
    verify_and_get_user_from_path,
    verify_and_get_user_from_path_and_header,
)
```
