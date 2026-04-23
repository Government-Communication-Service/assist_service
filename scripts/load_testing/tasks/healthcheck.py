def healthcheck(user):
    user.client.get("/healthcheck", name="GET /healthcheck")
