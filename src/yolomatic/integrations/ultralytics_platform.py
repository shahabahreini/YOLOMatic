from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import requests

DEFAULT_BASE_URL = "https://platform.ultralytics.com/api"
ENV_API_KEY = "ULTRALYTICS_API_KEY"


class UltralyticsPlatformError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        response_body: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.response_body = response_body


@dataclass(frozen=True)
class DownloadedFile:
    path: Path
    bytes_written: int


def load_api_key(env_path: Path | None = None) -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path or Path(".env"))
    except ImportError:
        pass
    api_key = os.getenv(ENV_API_KEY, "").strip()
    if not api_key:
        raise UltralyticsPlatformError(
            "ULTRALYTICS_API_KEY is missing. Add it to .env or export it before using Ultralytics Platform workflows."
        )
    return api_key


def credential_status(env_path: Path | None = None) -> dict[str, bool]:
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path or Path(".env"))
    except ImportError:
        pass
    return {"api_key": bool(os.getenv(ENV_API_KEY, "").strip())}


class UltralyticsPlatformClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "YOLOmatic/ultralytics-platform",
            }
        )

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        retries = int(kwargs.pop("retries", 1))
        for attempt in range(retries + 1):
            response = self.session.request(
                method,
                self._url(path),
                timeout=kwargs.pop("timeout", self.timeout),
                **kwargs,
            )
            if response.status_code == 429 and attempt < retries:
                retry_after = self._retry_after(response)
                time.sleep(retry_after or min(2.0, 0.5 * (attempt + 1)))
                continue
            if response.status_code >= 400:
                raise self._error_from_response(response)
            return response
        raise UltralyticsPlatformError("Request failed after retry handling.")

    def get_json(self, path: str, **kwargs: Any) -> Any:
        response = self.request("GET", path, **kwargs)
        if not response.content:
            return {}
        return response.json()

    def post_json(self, path: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        response = self.request("POST", path, json=payload or {}, **kwargs)
        if not response.content:
            return {}
        return response.json()

    def paginated_get(self, path: str, *, params: dict[str, Any] | None = None) -> list[Any]:
        results: list[Any] = []
        next_path: str | None = path
        request_params = dict(params or {})
        while next_path:
            payload = self.get_json(next_path, params=request_params)
            items, next_path = self._extract_page(payload)
            results.extend(items)
            request_params = {}
        return results

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.paginated_get("datasets")

    def list_projects(self) -> list[dict[str, Any]]:
        return self.paginated_get("projects")

    def list_models(self, *, completed: bool = False) -> list[dict[str, Any]]:
        return self.paginated_get("models/completed" if completed else "models")

    def dataset_export(self, dataset_id: str, version: int | None = None) -> dict[str, Any]:
        params = {"v": version} if version is not None else None
        payload = self.get_json(f"datasets/{dataset_id}/export", params=params)
        if isinstance(payload, dict):
            return payload
        raise UltralyticsPlatformError("Dataset export response was not an object.")

    def download_dataset_export(self, dataset_id: str, output_path: Path, version: int | None = None) -> DownloadedFile:
        payload = self.dataset_export(dataset_id, version)
        url = _first_present(payload, ("url", "download_url", "signed_url", "signedUrl"))
        if not url:
            raise UltralyticsPlatformError("Dataset export did not include a signed download URL.")
        return self.download_signed_url(str(url), output_path)

    def model_files(self, model_id: str) -> list[dict[str, Any]]:
        payload = self.get_json(f"models/{model_id}/files")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("files", "data", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        raise UltralyticsPlatformError("Model files response did not include a files list.")

    def download_model_files(self, model_id: str, output_dir: Path) -> list[DownloadedFile]:
        output_dir.mkdir(parents=True, exist_ok=True)
        downloads: list[DownloadedFile] = []
        for item in self.model_files(model_id):
            url = _first_present(item, ("url", "download_url", "signed_url", "signedUrl"))
            if not url:
                continue
            filename = str(_first_present(item, ("name", "filename", "path")) or Path(str(url)).name or "model.pt")
            downloads.append(self.download_signed_url(str(url), output_dir / Path(filename).name))
        with contextlib_suppress_platform_error():
            self.post_json(f"models/{model_id}/track-download")
        return downloads

    def create_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.post_json("datasets", payload)
        return result if isinstance(result, dict) else {"result": result}

    def signed_upload_url(self, filename: str, *, content_type: str = "application/octet-stream") -> dict[str, Any]:
        payload = self.post_json("upload/signed-url", {"filename": filename, "content_type": content_type})
        if isinstance(payload, dict):
            return payload
        raise UltralyticsPlatformError("Signed upload response was not an object.")

    def upload_archive(self, archive_path: Path, *, dataset_id: str | None = None) -> dict[str, Any]:
        signed = self.signed_upload_url(archive_path.name)
        upload_url = _first_present(signed, ("url", "upload_url", "signed_url", "signedUrl"))
        if not upload_url:
            raise UltralyticsPlatformError("Signed upload response did not include an upload URL.")
        headers = signed.get("headers") if isinstance(signed.get("headers"), dict) else {}
        with archive_path.open("rb") as file:
            response = requests.put(str(upload_url), data=file, headers=headers, timeout=self.timeout)
        if response.status_code >= 400:
            raise UltralyticsPlatformError(
                f"Signed upload failed with HTTP {response.status_code}.",
                status_code=response.status_code,
                response_body=response.text,
            )
        complete_payload = {
            "upload_id": _first_present(signed, ("upload_id", "uploadId", "id", "key")),
            "dataset_id": dataset_id,
            "filename": archive_path.name,
        }
        complete_payload = {key: value for key, value in complete_payload.items() if value is not None}
        completed = self.post_json("upload/complete", complete_payload)
        if dataset_id:
            with contextlib_suppress_platform_error():
                self.post_json("datasets/ingest", {"dataset_id": dataset_id, "upload": completed})
        return completed if isinstance(completed, dict) else {"result": completed}

    def download_signed_url(self, url: str, output_path: Path) -> DownloadedFile:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=self.timeout) as response:
            if response.status_code >= 400:
                raise UltralyticsPlatformError(
                    f"Signed download failed with HTTP {response.status_code}.",
                    status_code=response.status_code,
                    response_body=response.text,
                )
            bytes_written = 0
            with output_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        bytes_written += len(chunk)
        return DownloadedFile(output_path, bytes_written)

    @staticmethod
    def _extract_page(payload: Any) -> tuple[list[Any], str | None]:
        if isinstance(payload, list):
            return payload, None
        if not isinstance(payload, dict):
            return [], None
        for key in ("data", "results", "items", "datasets", "models", "projects"):
            value = payload.get(key)
            if isinstance(value, list):
                next_value = payload.get("next") or payload.get("next_url") or payload.get("nextUrl")
                return value, str(next_value) if next_value else None
        return [payload], None

    @staticmethod
    def _retry_after(response: requests.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    def _error_from_response(self, response: requests.Response) -> UltralyticsPlatformError:
        retry_after = self._retry_after(response)
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        guidance = {
            401: "Authentication failed. Check ULTRALYTICS_API_KEY.",
            403: "The API key does not have permission for this resource.",
            404: "The requested Platform resource was not found.",
            409: "The request conflicts with an existing Platform resource.",
            429: "Platform rate limit reached. Retry after the indicated delay.",
        }.get(response.status_code)
        if guidance is None and response.status_code >= 500:
            guidance = "Ultralytics Platform returned a server error. Retry later."
        if guidance is None:
            guidance = f"Ultralytics Platform request failed with HTTP {response.status_code}."
        return UltralyticsPlatformError(
            guidance,
            status_code=response.status_code,
            retry_after=retry_after,
            response_body=body,
        )


def _first_present(payload: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        value = payload.get(key)
        if value:
            return value
    return None


class contextlib_suppress_platform_error:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return exc_type is not None and issubclass(exc_type, UltralyticsPlatformError)

