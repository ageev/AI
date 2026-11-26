"""ASGI middleware: virtual model ids on top of one loaded model.

* POST bodies whose ``model`` is a steering alias are rewritten to the base
  model, and the effective profile is put into the request task's context.
  The engine pool therefore sees an ordinary request for the loaded model: no
  second engine entry, no reload, no eviction.
* ``omlx_steer`` in a body is a sweep override. It is accepted only when sweep
  mode is on and only for sweep profiles; otherwise the request gets 400,
  never a silent ignore.
* Steered responses carry ``X-Omlx-Steer: <alias>@<sha>``.
* An alias whose adapter is not ready answers 503 (oMLX's ``model_fallback``
  would otherwise silently serve the stock model under the alias name).
* ``GET /v1/models`` lists the aliases; ``GET /v1/models/{alias}`` answers
  with the base model's entry under the alias id.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import quote, unquote

from . import rows as R
from .profiles import ProfileError

logger = logging.getLogger("omlx_steer")

_POST_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/responses",
)
_MODEL_PREFIX = "/v1/models/"


class SteerMiddleware:
    def __init__(self, app, *, state):
        self.app = app
        self.state = state  # omlx_steer.State

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        method = scope.get("method", "")
        if method == "GET" and path == "/v1/models":
            return await self._models(scope, receive, send)
        if method == "GET" and path.startswith(_MODEL_PREFIX):
            return await self._model(scope, receive, send, unquote(path[len(_MODEL_PREFIX):]))
        if method == "GET" and path == "/v1/omlx-steer/status":
            return await _json(send, 200, self.state.status())
        if method == "POST" and path in _POST_PATHS:
            return await self._post(scope, receive, send)
        return await self.app(scope, receive, send)

    # -- POST -------------------------------------------------------------
    async def _post(self, scope, receive, send):
        chunks, more = [], True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                return await self.app(scope, _replay([message]), send)
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        replay = _replay([{"type": "http.request", "body": body, "more_body": False}], receive)
        try:
            data = json.loads(body) if body else None
        except ValueError:
            return await self.app(scope, replay, send)
        if not isinstance(data, dict):
            return await self.app(scope, replay, send)
        override = data.pop("omlx_steer", None)
        model = data.get("model")
        store = self.state.store
        aliases = store.aliases() if isinstance(model, str) else {}
        if model not in aliases:
            if override is not None:
                return await _error(send, 400, "omlx_steer is only accepted for steering models")
            return await self.app(scope, replay, send)
        raw = aliases[model]
        if not self.state.ready(raw.get("model_type")):
            return await _error(send, 503, f"steering model '{model}' is not available: {self.state.reason}")
        try:
            profile = store.resolve(model, override)
        except (ProfileError, OSError, ValueError) as exc:
            return await _error(send, 400, f"steering profile error: {exc}")
        data["model"] = raw["base_model"]
        new_body = json.dumps(data).encode()
        headers = [(k, v) for k, v in scope["headers"] if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(new_body)).encode()))
        scope = dict(scope, headers=headers)
        replay = _replay([{"type": "http.request", "body": new_body, "more_body": False}], receive)
        tag = f"{profile.alias}@{profile.sha}".encode()

        async def tagged_send(message):
            if message["type"] == "http.response.start":
                message = dict(message, headers=list(message.get("headers", [])) + [(b"x-omlx-steer", tag)])
            await send(message)

        token = R.REQUEST_PROFILE.set(profile)
        try:
            return await self.app(scope, replay, tagged_send)
        finally:
            R.REQUEST_PROFILE.reset(token)

    # -- GET /v1/models ----------------------------------------------------
    async def _models(self, scope, receive, send):
        start, parts = {}, []

        async def capture(message):
            if message["type"] == "http.response.start":
                start.update(message)
            elif message["type"] == "http.response.body":
                parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    await self._finish_models(send, start, b"".join(parts))

        await self.app(scope, receive, capture)

    # -- GET /v1/models/{id} -----------------------------------------------
    async def _model(self, scope, receive, send, model):
        raw = self.state.store.aliases().get(model)
        if raw is None or not raw.get("listed", True):
            return await self.app(scope, receive, send)
        if not self.state.ready(raw.get("model_type")):
            return await _error(send, 503, f"steering model '{model}' is not available: {self.state.reason}")
        path = _MODEL_PREFIX + raw["base_model"]
        scope = dict(scope, path=path, raw_path=quote(path).encode())
        start, parts = {}, []

        async def capture(message):
            if message["type"] == "http.response.start":
                start.update(message)
            elif message["type"] == "http.response.body":
                parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    await self._finish_model(send, start, b"".join(parts), model)

        await self.app(scope, receive, capture)

    async def _finish_model(self, send, start, body, alias):
        status = start.get("status", 200)
        try:
            data = json.loads(body) if status == 200 else None
        except ValueError:
            data = None
        if isinstance(data, dict):
            body = json.dumps(dict(data, id=alias)).encode()
        headers = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        await send(dict(start, headers=headers))
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _finish_models(self, send, start, body):
        status = start.get("status", 200)
        try:
            data = json.loads(body) if status == 200 else None
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            listed = {m.get("id"): m for m in data["data"] if isinstance(m, dict)}
            for alias, raw in self.state.store.aliases().items():
                base = listed.get(raw.get("base_model"))
                if base is None or alias in listed or not raw.get("listed", True):
                    continue
                if not self.state.ready(raw.get("model_type")):
                    continue
                data["data"].append(dict(base, id=alias))
            body = json.dumps(data).encode()
        headers = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        await send(dict(start, headers=headers))
        await send({"type": "http.response.body", "body": body, "more_body": False})


def _replay(messages, receive=None):
    queue = list(messages)

    async def _receive():
        if queue:
            return queue.pop(0)
        if receive is not None:
            return await receive()
        return {"type": "http.disconnect"}

    return _receive


async def _error(send, status, detail):
    await _json(send, status, {"error": {"message": detail, "type": "omlx_steer"}})


async def _json(send, status, payload):
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})
