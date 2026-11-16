"""MkDocs hooks: structured data (JSON-LD) for search engines.

Everything is derived from sources that already exist so it cannot drift:
the version comes from pyproject.toml and the FAQ schema is parsed from the
questions and answers visible on docs/faq.md. The template that prints these
is docs/overrides/main.html.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = "https://github.com/shahabahreini/YOLOMatic"
AUTHOR = {"@type": "Person", "name": "Shahab Bahreini Jangjoo", "url": "https://github.com/shahabahreini"}
ARTICLE_SECTIONS = ("getting-started/", "guides/", "reference/", "advanced/")


def on_config(config):
    pyproject = Path(config.config_file_path).parent / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    site = config.site_url
    config.extra["release"] = project["version"]
    config.extra["jsonld_home"] = [
        {
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            "name": "YOLOmatic",
            "alternateName": "YOLOMatic",
            "description": project["description"],
            "applicationCategory": "DeveloperApplication",
            "applicationSubCategory": "Computer vision model training",
            "operatingSystem": "Linux, macOS, Windows",
            "softwareVersion": project["version"],
            "softwareRequirements": "Python 3.12 or 3.13",
            "programmingLanguage": "Python",
            "keywords": ", ".join(project.get("keywords", [])),
            "url": site,
            "downloadUrl": f"{REPO}/releases",
            "installUrl": f"{site}getting-started/install/",
            "codeRepository": REPO,
            "license": "https://www.apache.org/licenses/LICENSE-2.0",
            "image": f"{site}assets/showreel/poster.jpg",
            "screenshot": f"{site}assets/screenshots/wizard-configure-model.png",
            "author": AUTHOR,
            "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        },
        {
            "@context": "https://schema.org",
            "@type": "VideoObject",
            "name": "YOLOmatic showreel",
            "description": "A 40-second tour of YOLOmatic: model selection, the configuration wizard, "
            "detection, segmentation and pose, dataset preparation, benchmarking and export.",
            "thumbnailUrl": f"{site}assets/showreel/poster.jpg",
            "contentUrl": f"{site}assets/showreel/yolomatic-showreel.mp4",
            "uploadDate": "2026-11-15",
            "duration": "PT40S",
        },
        {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": "YOLOmatic",
            "url": site,
        },
    ]
    return config


def _plain(text: str) -> str:
    """Markdown paragraph -> plain text for schema answers."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    text = re.sub(r"[`*_]", "", text)  # code, bold, italics
    return re.sub(r"\s+", " ", text).strip()


def _faq(markdown: str) -> list[dict]:
    items = []
    for block in re.split(r"^### ", markdown, flags=re.M)[1:]:
        question, _, body = block.partition("\n")
        body = body.split("\n## ", 1)[0]  # stop at the next section heading
        parts: list[str] = []
        after_code = False
        for para in re.split(r"\n\s*\n", body.strip()):
            if para.startswith("```"):
                # A lead-in such as "Run:" is completed by its code block and,
                # if present, the sentence that follows it.
                if parts and parts[-1].endswith(":"):
                    parts.append(" ".join(ln for ln in para.splitlines() if not ln.startswith("```")))
                    after_code = True
                continue
            if para.startswith(("|", "#", "!!!", "-", "1.")):
                continue
            if parts and not (parts[-1].endswith(":") or after_code):
                break
            parts.append(_plain(para))
            after_code = False
        answer = " ".join(parts).strip()
        if question.strip() and answer:
            items.append(
                {
                    "@type": "Question",
                    "name": _plain(question),
                    "acceptedAnswer": {"@type": "Answer", "text": answer},
                }
            )
    return items


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_uri
    if src == "faq.md":
        page.meta["jsonld"] = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": _faq(markdown),
        }
    elif src.startswith(ARTICLE_SECTIONS):
        page.meta["jsonld"] = {
            "@context": "https://schema.org",
            "@type": "TechArticle",
            "headline": page.meta.get("title") or page.title,
            "description": page.meta.get("description", ""),
            "url": config.site_url + page.url,
            "author": AUTHOR,
            "publisher": {"@type": "Organization", "name": "YOLOmatic", "url": config.site_url},
            "about": "Computer vision model training with YOLOmatic",
        }
    return markdown
