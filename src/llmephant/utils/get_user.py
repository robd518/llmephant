from fastapi import Request

# Prep for future front end auth options
CANDIDATE_HEADER_KEYS = [
    "x-openwebui-user-email",  # OpenWebUI
]


def get_user_from_request(req: Request) -> str | None:
    for key in CANDIDATE_HEADER_KEYS:
        value = req.headers.get(key)
        if value:
            return value
    return None
