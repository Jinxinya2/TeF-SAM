"""Download a released frozen PRISM artifact."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a frozen PRISM artifact"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output",
        help="Destination path; defaults to artifacts/<manifest filename>",
    )
    parser.add_argument(
        "--url", help="Override the download URL stored in the manifest"
    )
    arguments = parser.parse_args()

    manifest_path = Path(arguments.manifest)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    filename = str(manifest["artifact_filename"])
    url = arguments.url or manifest.get("download_url")
    if not url:
        parser.error(
            "No artifact URL has been published in this manifest yet; pass --url"
        )

    output = (
        Path(arguments.output)
        if arguments.output
        else Path("artifacts") / filename
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(str(url)) as response, temporary.open(
            "wb"
        ) as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Downloaded: {output}")


if __name__ == "__main__":
    main()
