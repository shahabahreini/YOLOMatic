from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from src.cli.ultralytics_platform import build_parser
from src.integrations.ultralytics_platform import (
    UltralyticsPlatformClient,
    UltralyticsPlatformError,
)


class UltralyticsPlatformClientTests(unittest.TestCase):
    def test_headers_use_bearer_auth(self) -> None:
        session = requests.Session()

        UltralyticsPlatformClient("secret", session=session)

        self.assertEqual(session.headers["Authorization"], "Bearer secret")
        self.assertEqual(session.headers["Accept"], "application/json")

    def test_common_api_error_has_actionable_message_and_retry_after(self) -> None:
        session = MagicMock()
        response = MagicMock()
        response.status_code = 429
        response.headers = {"Retry-After": "7"}
        response.json.return_value = {"detail": "slow down"}
        session.request.return_value = response
        client = UltralyticsPlatformClient("secret", session=session)

        with self.assertRaises(UltralyticsPlatformError) as ctx:
            client.get_json("datasets", retries=0)

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.retry_after, 7)
        self.assertIn("rate limit", str(ctx.exception).lower())

    def test_download_dataset_export_uses_signed_url(self) -> None:
        session = MagicMock()
        export_response = MagicMock()
        export_response.status_code = 200
        export_response.content = b"{}"
        export_response.json.return_value = {"signed_url": "https://signed.example/dataset.ndjson"}
        session.request.return_value = export_response
        signed_response = MagicMock()
        signed_response.status_code = 200
        signed_response.iter_content.return_value = [b"row\n"]
        signed_response.__enter__.return_value = signed_response
        signed_response.__exit__.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir, patch("requests.get", return_value=signed_response):
            out = Path(temp_dir) / "dataset.ndjson"
            downloaded = UltralyticsPlatformClient("secret", session=session).download_dataset_export("ds1", out, 3)
            self.assertEqual(out.read_bytes(), b"row\n")

        self.assertEqual(downloaded.bytes_written, 4)
        session.request.assert_called_once()
        self.assertEqual(session.request.call_args.kwargs["params"], {"v": 3})

    def test_signed_upload_completes_flow(self) -> None:
        session = MagicMock()
        signed_response = MagicMock()
        signed_response.status_code = 200
        signed_response.content = b"{}"
        signed_response.json.return_value = {"upload_url": "https://signed.example/upload", "upload_id": "u1"}
        complete_response = MagicMock()
        complete_response.status_code = 200
        complete_response.content = b"{}"
        complete_response.json.return_value = {"ok": True}
        ingest_response = MagicMock()
        ingest_response.status_code = 200
        ingest_response.content = b"{}"
        ingest_response.json.return_value = {"ingest": True}
        session.request.side_effect = [signed_response, complete_response, ingest_response]
        put_response = MagicMock(status_code=200)

        with tempfile.TemporaryDirectory() as temp_dir, patch("requests.put", return_value=put_response) as mock_put:
            archive = Path(temp_dir) / "dataset.zip"
            archive.write_bytes(b"zip")
            result = UltralyticsPlatformClient("secret", session=session).upload_archive(archive, dataset_id="ds1")

        self.assertEqual(result, {"ok": True})
        mock_put.assert_called_once()
        self.assertEqual(session.request.call_args_list[1].args[:2], ("POST", "https://platform.ultralytics.com/api/upload/complete"))


class UltralyticsPlatformCliTests(unittest.TestCase):
    def test_cli_parses_download_dataset_smart_split(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["download-dataset", "ds1", "--version", "5", "--smart-split"])

        self.assertEqual(args.command, "download-dataset")
        self.assertEqual(args.dataset_id, "ds1")
        self.assertEqual(args.version, 5)
        self.assertTrue(args.smart_split)

    def test_cli_parses_uri_helper(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["uri-helper", "alice", "weeds", "--project", "Field", "--name", "run-1"])

        self.assertEqual(args.username, "alice")
        self.assertEqual(args.dataset_slug, "weeds")
        self.assertEqual(args.project, "Field")
        self.assertEqual(args.name, "run-1")


if __name__ == "__main__":
    unittest.main()
