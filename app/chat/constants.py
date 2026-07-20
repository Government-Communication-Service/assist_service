DELETION_NOTICE = "[This message content was deleted after 12 months]"

# Machine-readable error code so the frontend can tell "you are not on the private share's
# allow-list" apart from other 403s on the shared-chat endpoint (e.g. sharing turned off,
# which keeps its original plain-string detail for backwards compatibility).
PRIVATE_SHARE_ACCESS_DENIED = "private_share_access_denied"
