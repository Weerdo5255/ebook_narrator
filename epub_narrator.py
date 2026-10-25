#!/usr/bin/env python3
"""
epub_narrator.py - illustrate and narrate an EPUB with local AI.

Pipeline
  1. Parse the EPUB and split it into chapters (driven by the book's table of
     contents, including chapters that share one XHTML file).
  2. Ollama (local LLM)
       * reads every chapter in windows, scores how strongly each passage
         describes a visual scene and writes an image prompt for the strong ones;
       * optionally tidies the text for text-to-speech (numbers, abbreviations,
         footnote markers) without rewording it;
       * writes a prompt for a new cover.
  3. ComfyUI (local)
       * renders the scene illustrations and the cover with your image workflow;
       * speaks the narration chunk by chunk with your TTS workflow.
  4. ffmpeg joins the speech into one audio file per chapter.
  5. Everything lands in one folder:
       <Book Title>/
         <Book Title>.epub      the book with the illustrations and new cover
         cover.jpg
         audio/001 - <chapter>.mp3 ...
         audio/<Book Title>.m3u
         chapters.json

Comic edition (--comic)
  Ollama reads the whole book, builds a cast of recurring characters (keeping
  every detail the text gives about their looks and inventing the rest),
  picks the important plot points of each chapter and writes a page-by-page
  comic script with panels, captions and dialogue taken from the book.
  ComfyUI draws a reference portrait of every character and then each panel,
  using those portraits so the characters look the same throughout. The
  program letters the panels (speech balloons, captions, sound effects),
  lays them out as pages and packs everything into <Book Title>.cbz.

Every LLM answer, image and speech chunk is cached, so an interrupted run
picks up where it stopped when you run the same command again.

Requirements: Python 3.10+, `pip install -r requirements.txt`, ffmpeg on PATH,
a running Ollama server and a running ComfyUI server.
"""
from __future__ import annotations

import argparse
import hashlib
import html.entities
import json
import logging
import math
import os
import posixpath
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import wave
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

import requests
from lxml import etree
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

log = logging.getLogger("epub_narrator")

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_VERSION = "1"  # bump to invalidate cached LLM answers after prompt changes

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG: dict = {
    "output_dir": "output",
    "ollama": {
        "url": "http://127.0.0.1:11434",
        "model": "llama3.1:8b",
        "temperature": 0.2,
        "num_ctx": 8192,
        "timeout": 600,
        "unload_when_done": True,  # free VRAM for ComfyUI after the LLM phase
        # "think": False,          # set for "thinking" models (qwen3, deepseek-r1...)
    },
    "comfyui": {
        "url": "http://127.0.0.1:8188",
        "timeout": 1800,  # seconds allowed for a single job
        "poll_interval": 0.5,
        "free_memory_between_phases": True,
    },
    "chapters": {
        "use_toc": True,
        "min_chars": 400,  # shorter sections (title pages, dedications) are skipped
        "include_nonlinear": False,
        "skip_titles": [
            r"^(table of )?contents$",
            r"^copyright",
            r"^index$",
            r"^cover$",
            r"^title page$",
            r"^also by\b",
            r"^(about|other books by) the author",
            r"transcriber.?s notes?",
            r"project gutenberg",
        ],
    },
    "images": {
        "enabled": True,
        "workflow": "workflows/image_sdxl.json",
        "threshold": 7,  # 0-10; only passages scoring at least this get an image
        "min_visual_details": 3,  # and the LLM must list this many concrete details
        "max_per_chapter": 2,
        "max_total": 0,  # 0 = no overall limit
        "min_paragraphs_apart": 10,
        "window_chars": 3500,  # passage size sent to the LLM
        "width": 1216,
        "height": 832,
        "style": "detailed digital painting, book illustration, cinematic lighting, rich color",
        "negative_prompt": "text, letters, words, watermark, signature, logo, caption, frame, "
                           "blurry, deformed, extra limbs, bad anatomy, low quality",
        "max_width": 1600,
        "jpeg_quality": 88,
        "captions": True,
    },
    "cover": {
        "enabled": True,
        "keep_original": False,  # true = keep the book's own cover if it has one
        "workflow": "",  # empty = use images.workflow
        "width": 832,
        "height": 1216,
        "style": "book cover art, dramatic composition, painterly, highly detailed",
        "title_overlay": True,
        "font": "",  # path to a .ttf/.otf; empty = find a serif font automatically
    },
    "narration": {
        "enabled": True,
        "workflow": "workflows/tts_kokoro.json",
        "llm_cleanup": True,  # let Ollama expand numbers/abbreviations for speech
        "cleanup_group_chars": 2500,
        "chunk_chars": 500,  # max characters sent to the TTS model at once
        "speak_chapter_titles": True,
        "seed": 42,
        "lead_in": 0.5,  # seconds of silence at the start of each track
        "sentence_pause": 0.2,
        "paragraph_pause": 0.6,
        "title_pause": 1.2,
        "scene_break_pause": 1.5,
        "tail": 1.5,
        "format": "mp3",  # mp3 | m4a | m4b | opus | flac | wav
        "bitrate": "64k",  # mono speech; 64k is plenty
        "sample_rate": 44100,
        "normalize_loudness": True,
        "embed_cover": True,
        "retries": 2,
    },
    "comic": {
        "enabled": False,  # --comic turns this on (and makes only the comic)
        "panel_workflow": "workflows/comic_panel_sdxl_ipadapter.json",
        "character_workflow": "",  # empty = images.workflow
        "cast_scope": "book",  # book = read the whole book for the cast first; selected = only --chapters
        "read_chars": 6000,  # passage size for reading the plot
        "max_cast": 12,
        "min_presence": 2,  # passages a character must appear in to get a reference portrait
        "pages_per_1000_words": 0.6,
        "min_pages": 1,  # per chapter
        "max_pages": 10,
        "panels_per_page": 5,
        "max_panels_per_page": 6,
        "style": "graphic novel illustration, single panel, bold clean ink lines, flat cel shading, "
                 "rich colors, dramatic lighting",
        "negative_prompt": "text, speech bubble, word balloon, letters, words, caption, signature, watermark, "
                           "logo, multiple panels, panel grid, comic strip layout, split screen, border, "
                           "blurry, deformed, extra limbs, bad anatomy, low quality",
        "portrait_style": "character design portrait, upper body, facing the viewer, neutral expression, "
                          "plain light grey background, even studio lighting",
        "portrait_size": 1024,
        "ref_weight": 0.75,  # how strongly a reference portrait steers a panel with one character
        "ref_weight_multi": 0.6,  # ...and with several characters
        "megapixels": 1.0,  # panel render size (SDXL ~1.0, SD1.5 ~0.4)
        "page_width": 1600,
        "page_height": 2400,
        "margin": 50,
        "gutter": 22,
        "border": 5,
        "font": "",  # lettering font; empty = look for a comic font
        "title_font": "",
        "sfx_font": "",
        "font_size": 30,  # at 1600 px page width
        "uppercase": True,
        "cover": True,
        "cast_page": True,
        "page_numbers": True,
        "split": "book",  # book = one .cbz; chapter = one .cbz per chapter
        "jpeg_quality": 90,
        "cast": {},  # per-character overrides, see config.toml
    },
}


class ConfigError(Exception):
    """Bad configuration or input file."""


class ServiceError(Exception):
    """Ollama, ComfyUI or ffmpeg failed in a way that stops the run."""


class LLMError(Exception):
    """One LLM answer was unusable (the caller decides how to recover)."""


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> tuple[dict, Path]:
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
    else:
        p = SCRIPT_DIR / "config.toml"
        if not p.is_file():
            return json.loads(json.dumps(DEFAULT_CONFIG)), SCRIPT_DIR
    try:
        with open(p, "rb") as fh:
            user = tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{p}: {e}") from e
    return deep_merge(DEFAULT_CONFIG, user), p.resolve().parent


def resolve_file(name: str, *bases: Path) -> Path:
    p = Path(name).expanduser()
    if p.is_absolute():
        return p
    for base in (*bases, SCRIPT_DIR, Path.cwd()):
        if (base / p).is_file():
            return base / p
    return p


# =============================================================================
# Small helpers
# =============================================================================

_WS = re.compile(r"\s+")
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)


def clean_ws(s: str) -> str:
    return _WS.sub(" ", s.translate(_INVISIBLE)).strip()


def sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def safe_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name or "")
    name = clean_ws(name).strip(" .")
    name = name[:max_len].rstrip(" .")
    return name or "untitled"


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_selection(spec: str | None, total: int) -> set[int]:
    """'1-3,5,9-' -> {1,2,3,5,9,...,total}"""
    if not spec:
        return set(range(1, total + 1))
    chosen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d*)\s*-\s*(\d*)", part)
        if m:
            lo = int(m.group(1)) if m.group(1) else 1
            hi = int(m.group(2)) if m.group(2) else total
            chosen.update(range(lo, hi + 1))
        elif part.isdigit():
            chosen.add(int(part))
        else:
            raise ConfigError(f"Bad chapter selection: {part!r} (use e.g. 1-3,5,9-)")
    return {n for n in chosen if 1 <= n <= total}


class Progress:
    """One-line progress on a terminal, periodic log lines otherwise."""

    def __init__(self, label: str, total: int):
        self.label, self.total = label, max(total, 1)
        self.tty = sys.stderr.isatty()
        self.last_logged = -1

    def update(self, done: int) -> None:
        pct = int(100 * done / self.total)
        if self.tty:
            sys.stderr.write(f"\r    {self.label}: {done}/{self.total} ({pct}%)   ")
            sys.stderr.flush()
            if done >= self.total:
                sys.stderr.write("\n")
        elif pct // 25 != self.last_logged // 25 or done >= self.total:
            self.last_logged = pct
            log.info("    %s: %d/%d (%d%%)", self.label, done, self.total, pct)


class Cache:
    """Content-addressed files under the work directory."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def dir(self, ns: str) -> Path:
        d = self.root / ns
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_json(self, ns: str, key: str):
        p = self.dir(ns) / f"{key}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def put_json(self, ns: str, key: str, value) -> None:
        self.put_bytes(ns, f"{key}.json", json.dumps(value, ensure_ascii=False, indent=1).encode("utf-8"))

    def find(self, ns: str, key: str) -> Path | None:
        hits = sorted(p for p in self.dir(ns).glob(f"{key}.*") if not p.name.endswith(".tmp"))
        return hits[0] if hits else None

    def put_bytes(self, ns: str, name: str, data: bytes) -> Path:
        p = self.dir(ns) / name
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        return p


# =============================================================================
# XML / XHTML helpers
# =============================================================================

OPF_NS = "http://www.idpf.org/2007/opf"
OPS_NS = "http://www.idpf.org/2007/ops"
XHTML_NS = "http://www.w3.org/1999/xhtml"

_XML_SAFE_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}


def localname(tag) -> str | None:
    if not isinstance(tag, str):  # comments / processing instructions
        return None
    return etree.QName(tag).localname.lower()


def xchildren(el, name: str) -> list:
    return [c for c in el if localname(c.tag) == name]


def xfind(el, name: str):
    for d in el.iter():
        if localname(d.tag) == name:
            return d
    return None


def xfindall(el, name: str) -> list:
    return [d for d in el.iter() if localname(d.tag) == name]


def _decode_xml(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    m = re.match(rb"^\s*<\?xml[^>]*encoding=[\"']([A-Za-z0-9._-]+)", data)
    enc = m.group(1).decode() if m else "utf-8"
    try:
        return data.decode(enc)
    except (LookupError, UnicodeDecodeError):
        return data.decode("utf-8", errors="replace")


def _entity_to_numeric(m: re.Match) -> str:
    name = m.group(1)
    if name in _XML_SAFE_ENTITIES:
        return m.group(0)
    cp = html.entities.name2codepoint.get(name)
    return f"&#{cp};" if cp else m.group(0)


def parse_xml(data: bytes) -> etree._ElementTree:
    """Parse (X)HTML/XML strictly, then retry with HTML entities fixed, then recover."""
    opts = dict(resolve_entities=False, no_network=True, huge_tree=True)
    try:
        return etree.parse(BytesIO(data), etree.XMLParser(**opts))
    except etree.XMLSyntaxError:
        pass
    text = _decode_xml(data)
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
    text = re.sub(r"&([A-Za-z][A-Za-z0-9]*);", _entity_to_numeric, text)
    fixed = text.encode("utf-8")
    try:
        return etree.parse(BytesIO(fixed), etree.XMLParser(**opts))
    except etree.XMLSyntaxError:
        return etree.parse(BytesIO(fixed), etree.XMLParser(recover=True, **opts))


def serialize_xml(tree: etree._ElementTree) -> bytes:
    return etree.tostring(tree, xml_declaration=True, encoding="utf-8")


# =============================================================================
# EPUB reading
# =============================================================================

IMAGE_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG", "image/gif": "GIF", "image/webp": "WEBP"}


@dataclass
class ManifestItem:
    id: str
    href: str  # full path inside the zip
    media_type: str
    properties: str
    el: object


def split_href(base_dir: str, href: str) -> tuple[str, str]:
    path, _, frag = href.partition("#")
    path = unquote(path).lstrip("/")
    full = posixpath.normpath(posixpath.join(base_dir, path)) if path else ""
    return full, unquote(frag)


class Epub:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.zf = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as e:
            raise ConfigError(f"{path} is not a readable EPUB: {e}") from e
        self.names = self.zf.namelist()
        self._by_lower = {n.lower(): n for n in self.names}
        try:
            container = parse_xml(self.zf.read("META-INF/container.xml")).getroot()
        except KeyError as e:
            raise ConfigError(f"{path}: META-INF/container.xml missing - not an EPUB?") from e
        self._check_drm()
        rootfile = xfind(container, "rootfile")
        if rootfile is None or not rootfile.get("full-path"):
            raise ConfigError(f"{path}: container.xml names no package file")
        self.opf_path = rootfile.get("full-path")
        self.opf_dir = posixpath.dirname(self.opf_path)
        try:
            self.opf_tree = parse_xml(self.read(self.opf_path))
        except KeyError as e:
            raise ConfigError(f"{path}: package file {self.opf_path} is missing from the EPUB") from e
        opf = self.opf_tree.getroot()
        self.version = opf.get("version", "2.0")
        self.metadata_el = (xchildren(opf, "metadata") or [None])[0]
        if not xchildren(opf, "manifest") or not xchildren(opf, "spine"):
            raise ConfigError(f"{path}: package file has no manifest or spine")
        self.manifest_el = xchildren(opf, "manifest")[0]
        self.spine_el = xchildren(opf, "spine")[0]
        self.items: dict[str, ManifestItem] = {}
        for it in xchildren(self.manifest_el, "item"):
            if it.get("id") and it.get("href"):
                full, _ = split_href(self.opf_dir, it.get("href"))
                self.items[it.get("id")] = ManifestItem(
                    it.get("id"), full, (it.get("media-type") or "").lower(), it.get("properties") or "", it)
        self.spine = [(ir.get("idref"), (ir.get("linear") or "yes") != "no")
                      for ir in xchildren(self.spine_el, "itemref")]
        self.title = self._dc("title") or path.stem
        self.author = self._dc("creator") or "Unknown author"
        self.language = self._dc("language") or "en"
        self._docs: dict[str, etree._ElementTree] = {}
        self.toc = self._read_toc()

    def _check_drm(self) -> None:
        """Refuse DRM-protected books (font obfuscation alone is fine)."""
        try:
            enc = parse_xml(self.zf.read("META-INF/encryption.xml")).getroot()
        except KeyError:
            return
        font_algos = {"http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"}
        for data in xfindall(enc, "encrypteddata"):
            method = xfind(data, "encryptionmethod")
            ref = xfind(data, "cipherreference")
            algo = method.get("Algorithm") if method is not None else ""
            uri = (ref.get("URI") or "") if ref is not None else ""
            if algo not in font_algos and not re.search(r"\.(otf|ttf|woff2?)$", uri, re.I):
                raise ConfigError(f"{self.path.name} is DRM-protected ({uri} is encrypted); "
                                  "only DRM-free EPUBs can be processed")

    # -- access ---------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name.lower() in self._by_lower

    def real_name(self, name: str) -> str:
        return name if name in self.names else self._by_lower.get(name.lower(), name)

    def read(self, name: str) -> bytes:
        return self.zf.read(self.real_name(name))

    def doc(self, path: str) -> etree._ElementTree:
        if path not in self._docs:
            self._docs[path] = parse_xml(self.read(path))
        return self._docs[path]

    def _dc(self, name: str) -> str:
        if self.metadata_el is None:
            return ""
        for el in self.metadata_el.iter():
            if localname(el.tag) == name and etree.QName(el).namespace != OPF_NS and (el.text or "").strip():
                return clean_ws(el.text)
        return ""

    # -- table of contents ----------------------------------------------------
    def _read_toc(self) -> list[tuple[str, str, str]]:
        entries: list[tuple[str, str, str]] = []
        nav = next((i for i in self.items.values() if "nav" in i.properties.split()), None)
        if nav is not None:
            try:
                root = self.doc(nav.href).getroot()
                navs = xfindall(root, "nav")
                toc_nav = next((n for n in navs if "toc" in (n.get(f"{{{OPS_NS}}}type") or "").split()),
                               navs[0] if navs else None)
                if toc_nav is not None:
                    base = posixpath.dirname(nav.href)
                    for a in xfindall(toc_nav, "a"):
                        if a.get("href"):
                            path, frag = split_href(base, a.get("href"))
                            entries.append((element_text(a), path, frag))
            except Exception as e:  # noqa: BLE001 - a broken nav is not fatal
                log.debug("Could not read nav document: %s", e)
        if entries:
            return entries
        ncx = self.items.get(self.spine_el.get("toc") or "")
        if ncx is None:
            ncx = next((i for i in self.items.values() if i.media_type == "application/x-dtbncx+xml"), None)
        if ncx is not None:
            try:
                root = parse_xml(self.read(ncx.href)).getroot()
                base = posixpath.dirname(ncx.href)
                for np_ in xfindall(root, "navpoint"):
                    label, content = xfind(np_, "text"), xfind(np_, "content")
                    if content is not None and content.get("src"):
                        path, frag = split_href(base, content.get("src"))
                        entries.append((clean_ws("".join(label.itertext())) if label is not None else "", path, frag))
            except Exception as e:  # noqa: BLE001
                log.debug("Could not read NCX: %s", e)
        return entries

    # -- cover ----------------------------------------------------------------
    def cover_item(self) -> ManifestItem | None:
        for it in self.items.values():
            if "cover-image" in it.properties.split():
                return it
        if self.metadata_el is not None:
            for m in xfindall(self.metadata_el, "meta"):
                if m.get("name") == "cover":
                    it = self.items.get(m.get("content") or "")
                    if it is not None and it.media_type in IMAGE_TYPES:
                        return it
        for it in self.items.values():
            if it.media_type in IMAGE_TYPES and (re.search(r"(^|[^a-z])cover", it.id.lower())
                                                 or re.match(r"cover", posixpath.basename(it.href).lower())):
                return it
        return None


# =============================================================================
# Chapter extraction
# =============================================================================

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CONTAINER_TAGS = {"body", "div", "section", "article", "main", "header", "footer", "aside", "figure",
                  "blockquote", "ul", "ol", "dl", "table", "thead", "tbody", "tfoot", "tr", "hgroup",
                  "center", "details"}
BLOCKISH = HEADING_TAGS | CONTAINER_TAGS | {"p", "li", "pre", "dt", "dd", "figcaption", "caption",
                                           "td", "th", "address", "hr", "summary"}
SKIP_TAGS = {"script", "style", "head", "title", "nav", "svg", "math", "rt", "rp", "noscript",
             "template", "img", "audio", "video", "object", "iframe", "button", "input", "select",
             "textarea", "sup"}
SKIP_TYPES = {"footnote", "footnotes", "endnote", "endnotes", "rearnote", "rearnotes", "noteref",
              "pagebreak", "toc", "landmarks", "page-list", "annotation"}
SKIP_ROLES = {"doc-noteref", "doc-footnote", "doc-endnotes", "doc-pagebreak", "doc-toc"}
SKIP_CLASS_RE = re.compile(r"(^|\s)(pagenum|pageno|page-number|pagebreak|footnotes?|endnotes?|noteref)(\s|$)", re.I)


@dataclass
class Block:
    text: str
    elem: object | None  # element to place an illustration after
    doc: str
    kind: str = "text"  # text | heading | break


@dataclass
class Chapter:
    title: str | None
    blocks: list[Block] = field(default_factory=list)
    number: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text)

    @property
    def chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    @property
    def words(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)


def _skipped(el, tag: str) -> bool:
    if tag in SKIP_TAGS:
        # <sup> is only skipped when it looks like a footnote marker
        if tag == "sup":
            return bool(re.fullmatch(r"\s*[\[(]?[\divx*†‡§]{1,4}[\])]?\s*", "".join(el.itertext())))
        return True
    types = set((el.get(f"{{{OPS_NS}}}type") or "").split())
    if types & SKIP_TYPES or (el.get("role") or "") in SKIP_ROLES:
        return True
    return bool(SKIP_CLASS_RE.search(el.get("class") or ""))


def _gather(e, parts: list[str]) -> None:
    tag = localname(e.tag)
    if tag is not None and not _skipped(e, tag):
        if tag == "br":
            parts.append(" ")
        if e.text:
            parts.append(e.text)
        for c in e:
            _gather(c, parts)
    if e.tail:
        parts.append(e.tail)


def element_text(el) -> str:
    parts = [el.text or ""]
    for c in el:
        _gather(c, parts)
    return clean_ws("".join(parts))


def _has_blockish(el) -> bool:
    for d in el.iterdescendants():
        t = localname(d.tag)
        if t in BLOCKISH:
            return True
    return False


def _iter_blocks(el, frags: set[str], doc: str):
    """Yield ('anchor', id) and ('block', Block) in reading order."""
    tag = localname(el.tag)
    if tag is None or _skipped(el, tag):
        return
    if el.get("id") in frags:
        yield ("anchor", el.get("id"))
    if tag == "hr":
        yield ("block", Block("", el, doc, "break"))
        return
    if tag != "body" and not _has_blockish(el):
        for d in el.iterdescendants():
            if isinstance(d.tag, str) and d.get("id") in frags:
                yield ("anchor", d.get("id"))
        txt = element_text(el)
        if txt:
            yield ("block", Block(txt, el, doc, "heading" if tag in HEADING_TAGS else "text"))
        return
    # Container: walk children, collecting loose inline content into runs.
    # Loose runs get no anchor element: an illustration placed there would land mid-sentence,
    # so _anchor_block() moves it to the nearest real paragraph instead.
    run: list[str] = [el.text or ""]
    for child in el:
        ctag = localname(child.tag)
        is_block = ctag is not None and not _skipped(child, ctag) and (ctag in BLOCKISH or _has_blockish(child))
        if is_block:
            txt = clean_ws("".join(run))
            if txt:
                yield ("block", Block(txt, None, doc, "text"))
            yield from _iter_blocks(child, frags, doc)
            run = [child.tail or ""]
            continue
        if ctag is not None and not _skipped(child, ctag):
            for d in child.iter():
                if isinstance(d.tag, str) and d.get("id") in frags:
                    txt = clean_ws("".join(run))
                    if txt:
                        yield ("block", Block(txt, None, doc, "text"))
                    run = []
                    yield ("anchor", d.get("id"))
            run.append(" " if ctag == "br" else element_text(child))
        run.append(child.tail or "")
    txt = clean_ws("".join(run))
    if txt:
        yield ("block", Block(txt, None, doc, "text"))


def build_chapters(epub: Epub, ccfg: dict) -> list[Chapter]:
    spine_docs = []
    for idref, linear in epub.spine:
        it = epub.items.get(idref)
        if it is None or it.media_type not in ("application/xhtml+xml", "text/html"):
            continue
        if not linear and not ccfg.get("include_nonlinear"):
            continue
        spine_docs.append(it.href)

    toc_map: dict[str, dict[str, str]] = {}
    for title, path, frag in epub.toc:
        if path in spine_docs:
            toc_map.setdefault(path, {})[frag] = title
    use_toc = ccfg.get("use_toc", True) and sum(len(v) for v in toc_map.values()) >= 2

    chapters: list[Chapter] = []
    current: Chapter | None = None

    def start(title: str | None) -> None:
        nonlocal current
        if current is not None and not current.blocks:
            current.title = title or current.title
            return
        current = Chapter(title=title)
        chapters.append(current)

    for doc in spine_docs:
        try:
            root = epub.doc(doc).getroot()
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping unreadable document %s: %s", doc, e)
            continue
        entries = toc_map.get(doc, {}) if use_toc else {}
        if not use_toc:
            start(None)
        elif "" in entries:
            start(entries[""])
        elif current is None:
            start(None)
        body = xfind(root, "body")
        if body is None:
            continue
        frags = {k for k in entries if k}
        for kind, val in _iter_blocks(body, frags, doc):
            if kind == "anchor":
                start(entries[val])
            else:
                current.blocks.append(val)

    skip_res = [re.compile(p, re.I) for p in ccfg.get("skip_titles", [])]
    result: list[Chapter] = []
    for ch in chapters:
        if not ch.title:
            heading = next((b.text for b in ch.blocks[:5] if b.kind == "heading"), "")
            ch.title = heading[:100] if heading else None
        title = clean_ws(ch.title or "")
        if any(r.search(title) for r in skip_res):
            log.debug("Skipping section %r (title filter)", title)
            continue
        if ch.chars < ccfg.get("min_chars", 400):
            log.debug("Skipping short section %r (%d chars)", title, ch.chars)
            continue
        result.append(ch)
    for n, ch in enumerate(result, 1):
        ch.number = n
        ch.title = clean_ws(ch.title or "") or f"Chapter {n}"
    return result


# =============================================================================
# Ollama
# =============================================================================

# Error messages Ollama is known to send, and what to do about each (matched case-insensitively).
OLLAMA_HINTS = [
    (r"not found|try pulling", "run `ollama pull {model}`, or set ollama.model to a model that `ollama list` shows"),
    (r"more system memory|out of memory|cudaMalloc|unable to allocate|insufficient memory",
     "not enough memory for {model}: use a smaller model or a lower ollama.num_ctx, and close other programs "
     "that use the GPU"),
    (r"runner process (has )?(terminated|exited)|unexpectedly stopped|exit status",
     "Ollama's model process crashed, most often because the GPU ran out of memory: try a lower ollama.num_ctx "
     "or a smaller model; Ollama's server log has the cause"),
    (r"does not support thinking", "{model} has no thinking mode: remove `think` from [ollama] in config.toml"),
    (r"does not support (chat|generate|completion)",
     "{model} is not a chat model: set ollama.model to an instruction-tuned model"),
    (r"server busy|maximum pending", "Ollama is busy with other requests: let them finish, then run again"),
    (r"invalid format", "this Ollama is too old for JSON-schema answers: update Ollama (0.5 or newer)"),
]


class Ollama:
    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/")
        self.model = cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.2))
        self.num_ctx = int(cfg.get("num_ctx", 8192))
        self.timeout = float(cfg.get("timeout", 600))
        self.think = cfg.get("think")
        self.session = requests.Session()
        self.used = False

    def read_error(self, failure: requests.Response | requests.RequestException) -> str:
        """Describe a failed Ollama call: Ollama's own error message (it answers {"error": "..."}),
        or what went wrong with the connection, plus a hint for errors with a known fix."""
        if isinstance(failure, requests.ConnectTimeout):
            return f"could not connect to Ollama at {self.url} (is it running?)"
        if isinstance(failure, requests.Timeout):
            return (f"Ollama gave no answer within {self.timeout:g}s: a big model or a long passage may need "
                    "a higher ollama.timeout")
        if isinstance(failure, requests.ConnectionError):
            return f"could not reach Ollama at {self.url} (has it stopped or crashed?)"
        if isinstance(failure, requests.RequestException):
            return clean_ws(str(failure)) or type(failure).__name__
        r = failure
        msg = ""
        try:
            body = r.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("error"):
            err = body["error"]
            msg = err if isinstance(err, str) else json.dumps(err, ensure_ascii=False)
        if not msg:  # not JSON: a plain-text or HTML error page (e.g. from a proxy in front of Ollama)
            msg = re.sub(r"<[^>]+>", " ", r.text or "")
        msg = clean_ws(msg)[:500] or r.reason or "no details given"
        text = msg if r.status_code == 200 else f"HTTP {r.status_code}: {msg}"
        for pattern, hint in OLLAMA_HINTS:
            if re.search(pattern, msg, re.I):
                return f"{text.rstrip('. ')}. Hint: {hint.format(model=self.model)}"
        return text

    def check(self) -> None:
        try:
            r = self.session.get(f"{self.url}/api/tags", timeout=10)
        except requests.RequestException as e:
            log.debug("Ollama check failed: %s", self.read_error(e))
            raise ServiceError(f"Cannot reach Ollama at {self.url}. Is Ollama running (`ollama serve`)?") from e
        if r.status_code != 200:
            raise ServiceError(f"Ollama at {self.url} answered with an error: {self.read_error(r)}")
        names = set()
        for m in r.json().get("models", []):
            names.update(filter(None, (m.get("name"), m.get("model"))))
        wanted = {self.model, f"{self.model}:latest"}
        if not names & wanted:
            raise ServiceError(f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}")

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        self.used = True
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if self.think is not None:
            payload["think"] = bool(self.think)
        problem = ""
        tries = 3
        for attempt in range(1, tries + 1):
            try:
                r = self.session.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                raise ServiceError(f"Ollama request failed: {self.read_error(e)}") from e
            try:
                body = r.json()
            except ValueError:
                body = None
            if r.status_code != 200 or not isinstance(body, dict) or body.get("error"):
                problem = self.read_error(r)
                if r.status_code in (400, 404):
                    raise ServiceError(f"Ollama rejected the request - {problem}")
                log.warning("    Ollama error (try %d of %d): %s", attempt, tries, problem)
                if attempt < tries:
                    time.sleep(2 * attempt)
                continue
            message = body.get("message") or {}
            content = (message.get("content") or "").strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                problem = "empty answer" if not content else f"invalid JSON ({e})"
                if body.get("done_reason") == "length":
                    problem += ". Hint: the answer was cut off at the context limit, raise ollama.num_ctx"
                elif not content and (message.get("thinking") or "").strip():
                    problem += ". Hint: the model only wrote its reasoning, set think = false under [ollama]"
            else:
                if isinstance(data, dict) and all(k in data for k in schema.get("required", [])):
                    return data
                problem = "JSON is missing required fields"
            log.debug("    unusable answer from %s (try %d of %d): %s\n%s", self.model, attempt, tries, problem,
                      content[:2000])
            payload["options"]["temperature"] = min(1.0, self.temperature + 0.3 * attempt)
        raise LLMError(f"unusable answer from {self.model}: {problem}")

    def unload(self) -> None:
        try:
            r = self.session.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
        except requests.RequestException as e:
            log.warning("Could not unload %s from Ollama, so its GPU memory stays in use: %s",
                        self.model, self.read_error(e))
            return
        if r.status_code == 200:
            log.info("Unloaded %s from Ollama to free GPU memory", self.model)
        else:
            log.warning("Could not unload %s from Ollama, so its GPU memory stays in use: %s",
                        self.model, self.read_error(r))


# =============================================================================
# ComfyUI
# =============================================================================

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


def fill_placeholders(obj, values: dict):
    if isinstance(obj, dict):
        return {k: fill_placeholders(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill_placeholders(v, values) for v in obj]
    if isinstance(obj, str):
        whole = PLACEHOLDER_RE.fullmatch(obj.strip())
        if whole and whole.group(1) in values:
            return values[whole.group(1)]  # keeps ints as ints (seed, width...)
        return PLACEHOLDER_RE.sub(lambda m: str(values.get(m.group(1), m.group(0))), obj)
    return obj


class Workflow:
    """A ComfyUI workflow in API format with {{PLACEHOLDER}} strings."""

    def __init__(self, path: Path, required: list[str], allowed: list[str], label: str):
        if not path.is_file():
            raise ConfigError(f"{label} workflow not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}: not valid JSON ({e})") from e
        if isinstance(data, dict) and "nodes" in data and "links" in data:
            raise ConfigError(f"{path} was saved in ComfyUI's normal (UI) format. Open it in ComfyUI and use "
                              f"Workflow > Export (API), then point the config at the exported file.")
        if not isinstance(data, dict) or not data or not all(
                isinstance(v, dict) and "class_type" in v for v in data.values()):
            raise ConfigError(f"{path} does not look like a ComfyUI API-format workflow")
        text = json.dumps(data)
        missing = [p for p in required if "{{%s}}" % p not in text]
        if missing:
            raise ConfigError(f"{label} workflow {path.name} must contain "
                              + ", ".join("{{%s}}" % m for m in missing)
                              + " typed into the matching node input (see README)")
        unknown = set(PLACEHOLDER_RE.findall(text)) - set(allowed)
        if unknown:
            raise ConfigError(f"{label} workflow {path.name} uses unknown placeholder(s) "
                              + ", ".join("{{%s}}" % u for u in sorted(unknown))
                              + "; allowed here: " + ", ".join("{{%s}}" % a for a in allowed))
        self.path, self.data, self.label = path, data, label
        self.hash = sha(data)
        self.placeholders = set(PLACEHOLDER_RE.findall(text))
        # Number boxes in ComfyUI cannot hold "{{SEED}}", so image workflows without those
        # placeholders get seed / width / height filled in by input name instead.
        self.auto_numbers = "WIDTH" in allowed
        # Comic panel workflows can take character reference pictures.
        self.ref_slots = max((int(n) for n in re.findall(r"REF_(?:IMAGE|MASK|WEIGHT)_(\d+)",
                                                         " ".join(self.placeholders))), default=0)
        self.uses_sheet = "REF_SHEET" in self.placeholders
        self.uses_refs = bool(self.ref_slots or self.uses_sheet)
        self.class_types = sorted({str(v.get("class_type")) for v in data.values()})

    def build(self, values: dict, prefix: str) -> dict:
        wf = fill_placeholders(self.data, values)
        for node in wf.values():
            inputs = node.get("inputs") or {}
            if isinstance(inputs.get("filename_prefix"), str):
                inputs["filename_prefix"] = prefix
            if not self.auto_numbers:
                continue
            if "SEED" not in self.placeholders and "SEED" in values:
                for k in ("seed", "noise_seed"):
                    if isinstance(inputs.get(k), int) and not isinstance(inputs.get(k), bool):
                        inputs[k] = values["SEED"]
            if "latent" in str(node.get("class_type", "")).lower():
                for k, ph in (("width", "WIDTH"), ("height", "HEIGHT")):
                    if ph not in self.placeholders and ph in values and isinstance(inputs.get(k), int):
                        inputs[k] = values[ph]
        return wf


class ComfyUI:
    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/")
        self.timeout = float(cfg.get("timeout", 1800))
        self.poll = float(cfg.get("poll_interval", 0.5))
        self.client_id = str(uuid.uuid4())
        self.session = requests.Session()
        self._node_types: set[str] | None = None
        self._uploaded: dict[str, str] = {}

    def check(self) -> None:
        try:
            r = self.session.get(f"{self.url}/system_stats", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            log.debug("ComfyUI check failed: %s", e)
            raise ServiceError(f"Cannot reach ComfyUI at {self.url}. Is ComfyUI running?") from e

    def missing_nodes(self, wf: "Workflow") -> list[str]:
        """Node types the workflow uses that this ComfyUI does not have (empty if unknown)."""
        if self._node_types is None:
            try:
                r = self.session.get(f"{self.url}/object_info", timeout=60)
                r.raise_for_status()
                self._node_types = set(r.json())
            except (requests.RequestException, ValueError) as e:
                log.debug("Could not list ComfyUI nodes: %s", e)
                self._node_types = set()
        if not self._node_types:
            return []
        return [t for t in wf.class_types if t not in self._node_types]

    def upload(self, data: bytes, filename: str, subfolder: str = "epub_narrator") -> str:
        """Put an image into ComfyUI's input folder; returns the name a Load Image node expects."""
        digest = sha(data, filename)
        if digest in self._uploaded:
            return self._uploaded[digest]
        try:
            r = self.session.post(f"{self.url}/upload/image",
                                  files={"image": (filename, data, "application/octet-stream")},
                                  data={"subfolder": subfolder, "type": "input", "overwrite": "true"},
                                  timeout=120)
        except requests.RequestException as e:
            raise ServiceError(f"Could not upload {filename} to ComfyUI: {e}") from e
        if r.status_code != 200:
            raise ServiceError(f"ComfyUI refused the upload of {filename} (HTTP {r.status_code}: {r.text[:200]})")
        info = r.json()
        name = info.get("name", filename)
        sub = info.get("subfolder", subfolder)
        self._uploaded[digest] = f"{sub}/{name}" if sub else name
        return self._uploaded[digest]

    def run(self, workflow: dict) -> list[dict]:
        """Queue a workflow, wait for it, return its output files."""
        try:
            r = self.session.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id},
                                  timeout=60)
        except requests.RequestException as e:
            raise ServiceError(f"ComfyUI request failed: {e}") from e
        if r.status_code != 200:
            raise ServiceError("ComfyUI rejected the workflow: " + self._describe_rejection(r))
        prompt_id = r.json()["prompt_id"]
        deadline = time.time() + self.timeout
        try:
            while time.time() < deadline:
                try:
                    h = self.session.get(f"{self.url}/history/{prompt_id}", timeout=30).json()
                except (requests.RequestException, ValueError):
                    h = {}
                entry = h.get(prompt_id)
                if entry:
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        raise ServiceError("ComfyUI job failed: " + self._describe_failure(status))
                    return self._collect(entry.get("outputs") or {})
                time.sleep(self.poll)
        except KeyboardInterrupt:
            self.interrupt()
            raise
        self.interrupt()
        raise ServiceError(f"ComfyUI job did not finish within {self.timeout:.0f}s (raise comfyui.timeout)")

    @staticmethod
    def _describe_rejection(r: requests.Response) -> str:
        try:
            data = r.json()
        except ValueError:
            return f"HTTP {r.status_code}: {r.text[:300]}"
        err = data.get("error") or {}
        msgs = [err.get("message", "") + (f" ({err.get('details')})" if err.get("details") else "")]
        for node_id, info in (data.get("node_errors") or {}).items():
            for e in info.get("errors", []):
                msgs.append(f"node {node_id} [{info.get('class_type')}]: {e.get('message')} {e.get('details', '')}")
        return "; ".join(m.strip() for m in msgs if m.strip())

    @staticmethod
    def _describe_failure(status: dict) -> str:
        for name, info in status.get("messages") or []:
            if name == "execution_error":
                return f"{info.get('node_type')}: {info.get('exception_message', '').strip()}"
        return "unknown error"

    @staticmethod
    def _collect(outputs: dict) -> list[dict]:
        files = []
        for node_out in outputs.values():
            for key, items in node_out.items():
                if isinstance(items, list):
                    files += [dict(it, kind=key) for it in items if isinstance(it, dict) and "filename" in it]
        files.sort(key=lambda f: f.get("type") != "output")  # prefer saved over preview files
        return files

    def download(self, f: dict) -> bytes:
        params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
        r = self.session.get(f"{self.url}/view", params=params, timeout=120)
        if r.status_code != 200:
            raise ServiceError(f"Could not download {f['filename']} from ComfyUI (HTTP {r.status_code})")
        return r.content

    def interrupt(self) -> None:
        try:
            self.session.post(f"{self.url}/interrupt", json={}, timeout=5)
        except requests.RequestException:
            pass

    def free(self) -> None:
        try:
            self.session.post(f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=10)
        except requests.RequestException:
            pass


# =============================================================================
# Scene analysis (Ollama)
# =============================================================================

SCENE_SYSTEM = """You are the art director for an illustrated edition of a book. You will be given a passage \
whose paragraphs are numbered like [12]. Decide whether it contains a scene described vividly enough to be \
illustrated faithfully.

Fill in the fields in this order:
1. visual_details: the concrete, visible details the text actually states about ONE scene: setting, light, \
weather, objects, clothing, faces, colours, actions, positions. Copy or closely paraphrase the text and invent \
nothing. Leave the list empty if the passage is mostly dialogue, thoughts, exposition or summary.
2. score (0-10): how strongly the text describes that scene visually.
   0-2: no visual description (dialogue, reflection, exposition)
   3-5: a few passing visual hints; an illustrator would have to invent most of the picture
   6-7: a clear scene with several concrete details
   8-10: rich, vivid, specific description of a place, person or dramatic moment
   Be strict: most passages of a typical novel score below 6.
3. paragraph: the number of the paragraph where that scene is described most fully.
4. image_prompt: a prompt for a text-to-image model showing that scene, 40-90 words of comma-separated visual \
phrases: subject and action, setting, time of day and lighting, mood, composition and framing. Describe people by \
appearance (age, build, hair, clothing) instead of by name, because the image model does not know the characters. \
Use only what the text states or clearly implies. Never ask for text, letters, signs or captions. Do not name an \
art style.
5. caption: a caption for the illustration, at most 15 words, like a caption in a printed book.
6. characters: for each named character whose physical appearance is described in this passage, the name and a \
short appearance description taken from the text. Empty list if none."""

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_details": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "integer"},
        "paragraph": {"type": "integer"},
        "image_prompt": {"type": "string"},
        "caption": {"type": "string"},
        "characters": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"name": {"type": "string"}, "appearance": {"type": "string"}},
                      "required": ["name", "appearance"]},
        },
    },
    "required": ["visual_details", "score", "paragraph", "image_prompt", "caption", "characters"],
}


@dataclass
class Scene:
    chapter: int
    block_index: int
    score: int
    effective: int
    prompt: str
    caption: str
    details: list[str]
    image: Path | None = None


def build_windows(ch: Chapter, max_chars: int) -> list[list[int]]:
    windows, cur, size = [], [], 0
    for i, b in enumerate(ch.blocks):
        if b.kind == "break" or not b.text:
            continue
        if cur and size + len(b.text) > max_chars:
            windows.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(b.text)
    if cur:
        windows.append(cur)
    return windows


class CharacterNotes:
    """Appearance notes collected while reading, reused to keep illustrations consistent."""

    def __init__(self):
        self.notes: dict[str, str] = {}

    def relevant(self, passage: str, limit: int = 8) -> dict[str, str]:
        out = {}
        for name, look in self.notes.items():
            tokens = [t for t in re.findall(r"[A-Z][\w'-]+", name) if len(t) > 2] or [name]
            if any(re.search(r"\b" + re.escape(t) + r"\b", passage) for t in tokens):
                out[name] = look
                if len(out) >= limit:
                    break
        return out

    def update(self, items) -> None:
        for c in items or []:
            if not isinstance(c, dict):
                continue
            name, look = clean_ws(str(c.get("name", ""))), clean_ws(str(c.get("appearance", "")))
            if name and len(look) >= 15 and name not in self.notes and len(name) <= 40:
                self.notes[name] = look[:300]


def analyze_chapter(ch: Chapter, book: Epub, llm: Ollama | None, cache: Cache, icfg: dict,
                    chars: CharacterNotes) -> list[Scene] | None:
    """Score every window of a chapter. With llm=None only cached answers are used."""
    threshold = int(icfg["threshold"])
    scenes: list[Scene] = []
    windows = build_windows(ch, int(icfg["window_chars"]))
    model = llm.model if llm else None
    for idxs in windows:
        lines = [f"[{i + 1}] {ch.blocks[i].text[: int(icfg['window_chars'])]}" for i in idxs]
        passage = "\n\n".join(lines)
        if len(passage) < 300:
            continue
        key_model = model or "*"
        key = sha("scene", PROMPT_VERSION, key_model, passage)
        result = cache.get_json("scenes", key) if llm else _find_any_model(cache, passage)
        if result is None:
            if llm is None:
                return None  # not analysed yet
            known = chars.relevant(passage)
            user = f'Book: "{book.title}" by {book.author}\nChapter: {ch.title}\n'
            if known:
                user += ("\nKnown character appearances (use them when these characters appear):\n"
                         + "\n".join(f"- {n}: {a}" for n, a in known.items()) + "\n")
            user += "\nPassage:\n" + passage
            try:
                result = llm.chat_json(SCENE_SYSTEM, user, SCENE_SCHEMA)
            except LLMError as e:
                log.warning("    scene analysis skipped for one passage: %s", e)
                continue
            cache.put_json("scenes", key, result)
            cache.put_json("scenes_by_text", sha(passage), {"key": key})
        chars.update(result.get("characters"))
        details = [d for d in result.get("visual_details") or [] if isinstance(d, str) and d.strip()]
        try:
            score = max(0, min(10, int(result.get("score") or 0)))
        except (TypeError, ValueError):
            score = 0
        effective = score if len(details) >= int(icfg["min_visual_details"]) else min(score, threshold - 1)
        prompt = clean_ws(str(result.get("image_prompt") or ""))
        if len(prompt) < 20:
            effective = 0
        try:
            para = int(result.get("paragraph") or 0) - 1
        except (TypeError, ValueError):
            para = -1
        if para not in idxs:
            para = idxs[len(idxs) // 2]
        scenes.append(Scene(ch.number, para, score, effective, prompt,
                            clean_ws(str(result.get("caption") or ""))[:200], details))
    return scenes


def _find_any_model(cache: Cache, passage: str):
    ref = cache.get_json("scenes_by_text", sha(passage))
    return cache.get_json("scenes", ref["key"]) if ref else None


def pick_scenes(candidates: list[Scene], icfg: dict) -> list[Scene]:
    threshold = int(icfg["threshold"])
    apart = int(icfg["min_paragraphs_apart"])
    strong = sorted((s for s in candidates if s.effective >= threshold),
                    key=lambda s: (-s.effective, s.block_index))
    chosen: list[Scene] = []
    for s in strong:
        if len(chosen) >= int(icfg["max_per_chapter"]):
            break
        if all(abs(s.block_index - o.block_index) >= apart for o in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda s: s.block_index)


# =============================================================================
# Narration text preparation
# =============================================================================

CLEAN_SYSTEM = """You prepare book text for a text-to-speech narrator. You get a JSON list of paragraphs. \
Return the same paragraphs, in the same order and the same number, with ONLY these changes:
- expand abbreviations a speech engine could misread ("Mr." -> "Mister", "Dr." -> "Doctor", "St." -> "Saint" or \
"Street" by context, "etc." -> "et cetera", "e.g." -> "for example");
- write numbers, years, dates, times, money, ordinals and Roman numerals as words, the way a narrator would say them;
- remove footnote markers, reference numbers, page numbers, URLs and decorative symbols.
Do NOT summarise, reword, modernise, translate, censor, add or drop anything else. Keep all punctuation and quotation \
marks. A paragraph that needs no change is returned exactly as given."""

CLEAN_SCHEMA = {
    "type": "object",
    "properties": {"paragraphs": {"type": "array", "items": {"type": "string"}}},
    "required": ["paragraphs"],
}

COVER_SYSTEM = """You design cover art for books. Write a prompt for a text-to-image model for the cover of the \
book described below: 40-90 words of comma-separated visual phrases describing one striking, iconic image that \
captures the book's setting, mood and central subject. Keep the top third of the picture simple (sky, mist, wall, \
darkness) so the title can be printed over it. Describe people by appearance, not by name. Never ask for text, \
letters, a title or typography. Do not name an art style."""

COVER_SCHEMA = {"type": "object", "properties": {"image_prompt": {"type": "string"}}, "required": ["image_prompt"]}

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int | None:
    s = s.upper()
    if not s or any(c not in ROMAN for c in s):
        return None
    total = 0
    for i, c in enumerate(s):
        v = ROMAN[c]
        total += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return total if 0 < total < 400 else None


def speakable_heading(text: str) -> str:
    def fix_roman(m):
        n = roman_to_int(m.group(2))
        return f"{m.group(1)} {n}" if n else m.group(0)

    text = re.sub(r"\b(chapter|part|book|section|volume|act|canto)\s+([IVXLCDM]+)\b", fix_roman, text, flags=re.I)
    text = re.sub(r"^([IVXLCDM]+)([.:]?)$", lambda m: str(roman_to_int(m.group(1)) or m.group(1)) + m.group(2),
                  text.strip())
    if text.isupper():
        text = text.title()
    text = text.strip()
    if text and text[-1] not in ".!?:;":
        text += "."
    return text


FOOTNOTE_MARK_RE = re.compile(r"\[(?:\d{1,3}|[a-z]|[ivx]{1,4})\]|\{\d+\}|[\u00b9\u00b2\u00b3\u2070-\u2079]+")


def basic_tts_clean(text: str) -> str:
    text = FOOTNOTE_MARK_RE.sub("", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[*_#~|^<>=\\]+", " ", text)
    return clean_ws(text)


@dataclass
class Para:
    text: str
    kind: str  # title | text | break


@dataclass
class Chunk:
    text: str
    pause: float


def chapter_paragraphs(ch: Chapter, ncfg: dict) -> list[Para]:
    paras: list[Para] = []
    speak_titles = ncfg.get("speak_chapter_titles", True)
    has_heading = any(b.kind == "heading" for b in ch.blocks[:3])
    if speak_titles and not has_heading and ch.title:
        paras.append(Para(speakable_heading(ch.title), "title"))
    for b in ch.blocks:
        if b.kind == "break" or not re.search(r"\w", b.text):
            paras.append(Para("", "break"))
        elif b.kind == "heading":
            if speak_titles:
                paras.append(Para(speakable_heading(b.text), "title"))
        else:
            paras.append(Para(b.text, "text"))
    return paras


def _cleanup_ok(orig: str, new) -> bool:
    if not isinstance(new, str):
        return False
    if len(orig) < 60:
        return len(new) <= max(3 * len(orig), 60)
    return 0.7 <= len(new) / len(orig) <= 1.7


def llm_cleanup(paras: list[Para], llm: Ollama, cache: Cache, ncfg: dict) -> list[Para]:
    """Ask the LLM to make the text speakable, paragraph by paragraph, with safety checks."""
    limit = int(ncfg.get("cleanup_group_chars", 2500))
    groups: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for i, p in enumerate(paras):
        if p.kind == "break" or not p.text:
            continue
        if cur and size + len(p.text) > limit:
            groups.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(p.text)
    if cur:
        groups.append(cur)

    out = [Para(p.text, p.kind) for p in paras]
    prog = Progress("text cleanup", len(groups))
    for g_no, idxs in enumerate(groups, 1):
        texts = [paras[i].text for i in idxs]
        key = sha("clean", PROMPT_VERSION, llm.model, texts)
        cached = cache.get_json("cleanup", key)
        if cached is None:
            new = None
            try:
                ans = llm.chat_json(CLEAN_SYSTEM, json.dumps(texts, ensure_ascii=False), CLEAN_SCHEMA)
                new = ans.get("paragraphs")
            except LLMError as e:
                log.debug("cleanup failed: %s", e)
            if not isinstance(new, list) or len(new) != len(texts):
                new = texts  # count mismatch: keep the original wording
            fixed = [n if _cleanup_ok(o, n) else o for o, n in zip(texts, new)]
            orig_words, new_words = sum(len(t.split()) for t in texts), sum(len(t.split()) for t in fixed)
            if orig_words and not 0.8 <= new_words / orig_words <= 1.5:
                fixed = texts
            cached = {"paragraphs": fixed}
            cache.put_json("cleanup", key, cached)
        for i, t in zip(idxs, cached["paragraphs"]):
            out[i].text = t
        prog.update(g_no)
    return out


SENT_SPLIT = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"”’»)\]]))\s+")
CLAUSE_SPLIT = re.compile(r"(?<=[,;:—–])\s+")


def _split_long(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    for sent in SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= limit:
            pieces.append(sent)
            continue
        buf = ""
        for part in CLAUSE_SPLIT.split(sent):
            words = [part] if len(part) <= limit else part.split()
            for w in words:
                if buf and len(buf) + 1 + len(w) > limit:
                    pieces.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
        if buf:
            pieces.append(buf)
    return pieces


def make_chunks(paras: list[Para], ncfg: dict) -> list[Chunk]:
    limit = max(80, int(ncfg.get("chunk_chars", 500)))
    p_sent, p_para = float(ncfg["sentence_pause"]), float(ncfg["paragraph_pause"])
    p_title, p_break = float(ncfg["title_pause"]), float(ncfg["scene_break_pause"])
    chunks: list[Chunk] = []
    buf: list[str] = []
    ends_para = True

    def flush():
        nonlocal buf
        if buf:
            chunks.append(Chunk("\n".join(buf), p_para if ends_para else p_sent))
            buf = []

    for p in paras:
        if p.kind == "break":
            flush()
            if chunks:
                chunks[-1].pause = max(chunks[-1].pause, p_break)
            continue
        text = basic_tts_clean(p.text)
        if not re.search(r"\w", text):
            continue
        if p.kind == "title":
            flush()
            chunks.append(Chunk(text, p_title))
            continue
        pieces = _split_long(text, limit)
        for j, piece in enumerate(pieces):
            joined_len = sum(len(x) + 1 for x in buf) + len(piece)
            if buf and joined_len > limit:
                flush()
            if buf and j > 0:
                buf[-1] = f"{buf[-1]} {piece}"  # same paragraph: join with a space
            else:
                buf.append(piece)
            ends_para = j == len(pieces) - 1
    flush()
    return chunks


# =============================================================================
# Audio (ffmpeg)
# =============================================================================

AUDIO_FORMATS = {  # format: (codec, extension, lossy, can_embed_cover)
    "mp3": ("libmp3lame", ".mp3", True, True),
    "m4a": ("aac", ".m4a", True, True),
    "m4b": ("aac", ".m4b", True, True),
    "opus": ("libopus", ".opus", True, False),
    "flac": ("flac", ".flac", False, True),
    "wav": ("pcm_s16le", ".wav", False, False),
}


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ServiceError(f"ffmpeg failed: {proc.stderr.strip()[-800:]}")


def probe_duration(path: Path) -> float:
    if not shutil.which("ffprobe"):
        return 0.0
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                           "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def assemble_track(parts: list[tuple[Path, float]], out_path: Path, meta: dict, ncfg: dict,
                   cover: Path | None) -> float:
    fmt = ncfg.get("format", "mp3").lower()
    codec, _, lossy, can_cover = AUDIO_FORMATS[fmt]
    sr = int(ncfg.get("sample_rate", 44100))
    with tempfile.TemporaryDirectory(prefix="epub_narrator_") as td:
        td_path = Path(td)
        joined = td_path / "joined.wav"
        frames = 0
        with wave.open(str(joined), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)

            def silence(seconds: float):
                nonlocal frames
                n = int(sr * max(0.0, seconds))
                w.writeframes(b"\x00\x00" * n)
                frames += n

            silence(float(ncfg.get("lead_in", 0.5)))
            for i, (src, pause) in enumerate(parts):
                tmp = td_path / f"{i}.wav"
                run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", "-f", "wav", str(tmp)])
                with wave.open(str(tmp), "rb") as r:
                    n = r.getnframes()
                    w.writeframes(r.readframes(n))
                    frames += n
                tmp.unlink()
                silence(pause if i < len(parts) - 1 else float(ncfg.get("tail", 1.5)))

        args = ["-i", str(joined)]
        with_cover = bool(cover and can_cover and ncfg.get("embed_cover", True))
        if with_cover:
            args += ["-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:v", "copy",
                     "-disposition:v:0", "attached_pic"]
        if ncfg.get("normalize_loudness", True):
            args += ["-af", "loudnorm=I=-18:TP=-1.5:LRA=11"]
        args += ["-ar", "48000" if fmt == "opus" else str(sr), "-c:a", codec]
        if lossy:
            args += ["-b:a", str(ncfg.get("bitrate", "64k"))]
        if fmt == "mp3":
            args += ["-id3v2_version", "3"]
        for k, v in meta.items():
            args += ["-metadata", f"{k}={v}"]
        tmp_out = td_path / f"out{out_path.suffix}"
        try:
            run_ffmpeg(args + [str(tmp_out)])
        except ServiceError:
            if not with_cover:
                raise
            log.debug("Embedding the cover failed, encoding without it")
            plain = ["-i", str(joined)] + args[args.index("-disposition:v:0") + 2:]
            run_ffmpeg(plain + [str(tmp_out)])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_out), out_path)
    return frames / sr


# =============================================================================
# Images: cover title overlay
# =============================================================================

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/Library/Fonts/Georgia Bold.ttf",
]


def load_font(path: str, size: int):
    for p in ([path] if path else []) + FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def overlay_title(img: Image.Image, title: str, author: str, font_path: str) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    # Darken the top and bottom so the lettering stays readable on any picture.
    shade = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(shade)
    top, bottom = int(h * 0.38), int(h * 0.80)
    for y in range(top):
        sd.line([(0, y), (w, y)], fill=int(175 * (1 - y / top) ** 1.4))
    for y in range(bottom, h):
        sd.line([(0, y), (w, y)], fill=int(175 * ((y - bottom) / (h - bottom)) ** 1.4))
    img = Image.composite(Image.new("RGB", (w, h), (0, 0, 0)), img, shade)

    draw = ImageDraw.Draw(img)
    size = int(w * 0.115)
    while True:
        font = load_font(font_path, size)
        lines = _wrap(draw, title, font, w * 0.86)
        if len(lines) <= 3 or size <= w * 0.045:
            break
        size = int(size * 0.9)
    y = h * 0.06
    stroke = max(1, size // 22)
    for line in lines:
        lw = draw.textlength(line, font=font)
        draw.text(((w - lw) / 2, y), line, font=font, fill=(255, 255, 255), stroke_width=stroke,
                  stroke_fill=(0, 0, 0))
        y += size * 1.15
    if author:
        asize = int(w * 0.052)
        afont = load_font(font_path, asize)
        lines = _wrap(draw, author, afont, w * 0.86)[:2]
        y = h - h * 0.05 - asize * 1.2 * len(lines)
        for line in lines:
            lw = draw.textlength(line, font=afont)
            draw.text(((w - lw) / 2, y), line, font=afont, fill=(240, 240, 240),
                      stroke_width=max(1, asize // 22), stroke_fill=(0, 0, 0))
            y += asize * 1.2
    return img


# =============================================================================
# EPUB writing
# =============================================================================

COVER_PAGE_EPUB3 = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Cover</title>
<style type="text/css">html,body{{margin:0;padding:0;text-align:center;}} img{{max-width:100%;max-height:100%;}}</style>
</head>
<body epub:type="cover"><div><img src="{src}" alt="Cover"/></div></body>
</html>
"""

COVER_PAGE_EPUB2 = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Cover</title>
<style type="text/css">html,body{{margin:0;padding:0;text-align:center;}} img{{max-width:100%;max-height:100%;}}</style>
</head>
<body><div><img src="{src}" alt="Cover"/></div></body>
</html>
"""

_CLIMB_FROM = {"li", "td", "th", "dt", "dd", "tr", "thead", "tbody", "tfoot", "caption", "figcaption", "summary"}
_CLIMB_PARENTS = {"ul", "ol", "dl", "table", "tr", "thead", "tbody", "tfoot", "p", "span", "a", "em", "i", "b",
                  "strong", "small", "cite", "q", "label", "sub", "sup"} | HEADING_TAGS


class EpubWriter:
    def __init__(self, epub: Epub):
        self.epub = epub
        self.modified_docs: set[str] = set()
        self.replaced: dict[str, bytes] = {}
        self.added: dict[str, bytes] = {}
        self.ids = set(epub.items)
        self.opf_ns = etree.QName(epub.manifest_el).namespace

    def _unique_path(self, path: str) -> str:
        base, ext = posixpath.splitext(path)
        n, cand = 1, path
        while self.epub.has(cand) or cand in self.added:
            n += 1
            cand = f"{base}-{n}{ext}"
        return cand

    def _unique_id(self, base: str) -> str:
        n, cand = 1, base
        while cand in self.ids:
            n += 1
            cand = f"{base}-{n}"
        self.ids.add(cand)
        return cand

    def _opf_el(self, tag: str, parent, **attrs):
        el = etree.SubElement(parent, f"{{{self.opf_ns}}}{tag}" if self.opf_ns else tag)
        for k, v in attrs.items():
            el.set(k.replace("_", "-"), v)
        siblings = list(parent)
        if len(siblings) >= 2:  # keep the file's indentation tidy
            prev = siblings[-2]
            el.tail = prev.tail
            prev.tail = siblings[-3].tail if len(siblings) >= 3 else parent.text
        return el

    def _add_manifest(self, zip_path: str, media_type: str, id_base: str, properties: str = "") -> str:
        item_id = self._unique_id(id_base)
        href = posixpath.relpath(zip_path, self.epub.opf_dir or ".")
        attrs = {"id": item_id, "href": href, "media_type": media_type}
        if properties:
            attrs["properties"] = properties
        self._opf_el("item", self.epub.manifest_el, **attrs)
        return item_id

    def add_illustration(self, block: Block, image_jpeg: bytes, name: str, caption: str, alt: str) -> bool:
        if block.elem is None:
            return False
        img_dir = posixpath.join(self.epub.opf_dir, "images", "illustrations") if self.epub.opf_dir \
            else "images/illustrations"
        zip_path = self._unique_path(posixpath.join(img_dir, f"{name}.jpg"))
        self.added[zip_path] = image_jpeg
        self._add_manifest(zip_path, "image/jpeg", f"ai-{name}")
        src = posixpath.relpath(zip_path, posixpath.dirname(block.doc) or ".")

        target = block.elem
        while target.getparent() is not None:
            parent = target.getparent()
            if localname(target.tag) in _CLIMB_FROM or localname(parent.tag) in _CLIMB_PARENTS:
                target = parent
                continue
            break
        while target.getnext() is not None and "ai-illustration" in (target.getnext().get("class") or ""):
            target = target.getnext()
        ns = etree.QName(target).namespace

        def q(t):
            return f"{{{ns}}}{t}" if ns else t

        div = etree.Element(q("div"))
        div.set("class", "ai-illustration")
        div.set("style", "text-align:center;margin:1.2em 0;page-break-inside:avoid;break-inside:avoid;")
        img = etree.SubElement(div, q("img"))
        img.set("src", src)
        img.set("alt", alt or "Illustration")
        img.set("style", "max-width:100%;height:auto;")
        if caption:
            cap = etree.SubElement(div, q("p"))
            cap.set("class", "ai-illustration-caption")
            cap.set("style", "text-align:center;font-style:italic;font-size:0.9em;margin-top:0.4em;")
            cap.text = caption
        div.tail = "\n"
        target.addnext(div)
        self.modified_docs.add(block.doc)
        return True

    def set_cover(self, cover: Image.Image) -> None:
        epub = self.epub
        existing = epub.cover_item()
        if existing is not None:
            # Overwrite the existing cover image in place, at its original size and format,
            # so every page and reader that already points at it keeps working.
            try:
                old = Image.open(BytesIO(epub.read(existing.href)))
                size = old.size if min(old.size) >= 300 else cover.size
            except Exception:  # noqa: BLE001
                size = cover.size
            fitted = ImageOps.fit(cover.convert("RGB"), size, Image.LANCZOS)
            fmt = IMAGE_TYPES.get(existing.media_type, "JPEG")
            buf = BytesIO()
            fitted.save(buf, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
            self.replaced[epub.real_name(existing.href)] = buf.getvalue()
            self._ensure_cover_meta(existing.id)
            return
        buf = BytesIO()
        cover.convert("RGB").save(buf, "JPEG", quality=92)
        img_path = self._unique_path(posixpath.join(epub.opf_dir, "images", "cover.jpg") if epub.opf_dir
                                     else "images/cover.jpg")
        self.added[img_path] = buf.getvalue()
        epub3 = epub.version.startswith("3")
        img_id = self._add_manifest(img_path, "image/jpeg", "ai-cover-image", "cover-image" if epub3 else "")
        self._ensure_cover_meta(img_id)
        page_path = self._unique_path(posixpath.join(epub.opf_dir, "ai-cover.xhtml") if epub.opf_dir
                                      else "ai-cover.xhtml")
        src = posixpath.relpath(img_path, posixpath.dirname(page_path) or ".")
        self.added[page_path] = (COVER_PAGE_EPUB3 if epub3 else COVER_PAGE_EPUB2).format(src=src).encode("utf-8")
        page_id = self._add_manifest(page_path, "application/xhtml+xml", "ai-cover-page")
        itemref = etree.Element(f"{{{self.opf_ns}}}itemref" if self.opf_ns else "itemref")
        itemref.set("idref", page_id)
        itemref.tail = "\n    "
        epub.spine_el.insert(0, itemref)
        guide = (xchildren(epub.opf_tree.getroot(), "guide") or [None])[0]
        if guide is None and not epub3:
            guide = self._opf_el("guide", epub.opf_tree.getroot())
        if guide is not None:
            for ref in xchildren(guide, "reference"):
                if (ref.get("type") or "").lower() == "cover":
                    guide.remove(ref)
            self._opf_el("reference", guide, type="cover", title="Cover",
                         href=posixpath.relpath(page_path, epub.opf_dir or "."))

    def _ensure_cover_meta(self, item_id: str) -> None:
        md = self.epub.metadata_el
        if md is None:
            return
        for m in xfindall(md, "meta"):
            if m.get("name") == "cover":
                m.set("content", item_id)
                return
        self._opf_el("meta", md, name="cover", content=item_id)

    def _touch_modified_date(self) -> None:
        md = self.epub.metadata_el
        if md is None:
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for m in xfindall(md, "meta"):
            if m.get("property") == "dcterms:modified":
                m.text = now

    def write(self, out_path: Path) -> None:
        self._touch_modified_date()
        replaced = dict(self.replaced)
        for doc in self.modified_docs:
            replaced[self.epub.real_name(doc)] = serialize_xml(self.epub.doc(doc))
        opf_bytes = serialize_xml(self.epub.opf_tree)
        root = self.epub.opf_tree.getroot()
        if root.nsmap.get(None) == OPF_NS and all(el.nsmap.get("opf", OPF_NS) == OPF_NS for el in root.iter()
                                                  if isinstance(el.tag, str)):
            # lxml reuses an "opf:" prefix when one is declared; write plain <meta> etc. instead,
            # which means the same thing here and is what simple readers look for.
            opf_bytes = re.sub(rb"<(/?)opf:(meta|item|itemref|reference)\b", rb"<\1\2", opf_bytes)
        replaced[self.epub.real_name(self.epub.opf_path)] = opf_bytes
        tmp = out_path.with_name(out_path.name + ".tmp")
        with zipfile.ZipFile(tmp, "w") as z:
            z.writestr(zipfile.ZipInfo("mimetype"), b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
            for info in self.epub.zf.infolist():
                if info.filename == "mimetype" or info.is_dir():
                    continue
                data = replaced.get(info.filename)
                if data is None:
                    data = self.epub.zf.read(info.filename)
                z.writestr(info.filename, data, compress_type=zipfile.ZIP_DEFLATED)
            for name, data in self.added.items():
                z.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp, out_path)


# =============================================================================
# Comic edition: prompts
# =============================================================================

COMIC_PROMPT_VERSION = "1"  # bump to invalidate cached comic LLM answers after prompt changes
COMIC_REF_SLOTS = 4  # {{REF_IMAGE_1}} .. {{REF_IMAGE_4}}
COMIC_IMAGE_KEYS = (["PROMPT", "NEGATIVE_PROMPT", "SEED", "WIDTH", "HEIGHT", "REF_SHEET"]
                    + [f"{k}_{i}" for i in range(1, COMIC_REF_SLOTS + 1)
                       for k in ("REF_IMAGE", "REF_MASK", "REF_WEIGHT")])

READ_SYSTEM = """You are adapting a novel into a graphic novel. Read the passage and report three things.

1. characters: every named character who is present, speaks or acts in this passage.
   - name: the fullest form of their name used in the book (for example "Elizabeth Bennet").
   - aliases: other names, nicknames or titles this passage uses for the same person (for example "Lizzy", \
"Miss Elizabeth"). Empty list if none.
   - gender: woman, man, girl, boy, or unknown.
   - appearance: ONLY what this passage says about how they look: age, build, height, hair, eyes, skin, face, \
clothing, distinctive marks. Copy or closely paraphrase the text and invent nothing. Empty string if the passage \
says nothing about their looks.
2. events: what happens in this passage, in story order, one short sentence each that names who does what. \
importance from 1 to 5: 5 = a turning point of the plot, 4 = an important development, 3 = moves the story along, \
2 = a minor detail, 1 = background, memory or reflection. Usually 2 to 6 events.
3. setting: where and when the passage takes place, as an illustrator needs it: the place, time of day, weather \
and light, using only what the text says or clearly implies."""

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "aliases": {"type": "array", "items": {"type": "string"}},
                           "gender": {"type": "string"}, "appearance": {"type": "string"}},
            "required": ["name", "aliases", "gender", "appearance"]}},
        "events": {"type": "array", "items": {
            "type": "object",
            "properties": {"what": {"type": "string"}, "importance": {"type": "integer"}},
            "required": ["what", "importance"]}},
        "setting": {"type": "string"},
    },
    "required": ["characters", "events", "setting"],
}

PROFILE_SYSTEM = """You prepare the art direction for a graphic novel adapted from a book. From the information \
below, give:
- genre: the book's genre in a few words.
- world: one line of visual phrases telling an illustrator when and where the story is set: era, region, \
architecture, technology and typical clothing (for example "Regency-era English countryside, early 19th century, \
country houses, empire-waist gowns and tailcoats"). Only what the book shows or clearly implies.
- mood: a few words about the overall tone (for example "romantic, witty, sunny")."""

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {"genre": {"type": "string"}, "world": {"type": "string"}, "mood": {"type": "string"}},
    "required": ["genre", "world", "mood"],
}

DESIGN_FIELDS = ["gender", "age", "build", "skin", "hair", "eyes", "face", "outfit", "accessories"]

DESIGN_SYSTEM = """You are the character designer for a graphic novel adapted from a book. Design how one \
character looks, so the artist can draw them the same way in every panel.

- Everything the book states about the character's appearance must be kept exactly.
- Invent whatever the book does not state, so the design is complete. Make the invented parts fit the character's \
age, role and personality, the setting and the period, and make the character easy to tell apart from the other \
characters listed.
- Use plain visual words, no names and no art styles.
- Fields: gender (woman, man, girl, boy...); age (for example "about 20 years old"); build (for example "tall, \
slender"); skin (skin tone); hair (colour, length and style); eyes (colour); face (distinctive features, facial \
hair); outfit (their usual clothing in this story, with colours); accessories (hat, glasses, jewellery, weapon...; \
empty if none); short (a 3 to 6 word visual nickname, for example "red-haired girl in green coat").
- invented: the names of the fields you had to invent because the book says nothing about them."""

DESIGN_SCHEMA = {
    "type": "object",
    "properties": {**{k: {"type": "string"} for k in DESIGN_FIELDS},
                   "short": {"type": "string"},
                   "invented": {"type": "array", "items": {"type": "string"}}},
    "required": DESIGN_FIELDS + ["short", "invented"],
}

PLAN_SYSTEM = """You are adapting one chapter of a novel into a graphic novel. You get a numbered list of \
everything that happens in the chapter, each rated from 1 (background) to 5 (turning point). Decide what the comic \
shows and split it into the requested number of pages.

- Keep every event rated 4 or 5 and the events needed to understand them; leave out minor ones when space is short.
- Each page shows one to four consecutive events and reads as a small unit of the story.
- Pages follow the story order, and each event appears on at most one page.
- For each page give the numbers of the events it shows and one sentence describing what the page shows."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"pages": {"type": "array", "items": {
        "type": "object",
        "properties": {"events": {"type": "array", "items": {"type": "integer"}}, "summary": {"type": "string"}},
        "required": ["events", "summary"]}}},
    "required": ["pages"],
}

SCRIPT_SYSTEM = """You write the script for one page of a graphic novel adapted from a novel. You get the events \
this page must show, the original text they come from and the main characters. Write panels that tell those \
events clearly and in order, the way a good comic would: vary the shots, show rather than tell, and let the \
characters speak.

For each panel:
- shot: establishing (a location, people small or absent), wide (the whole scene with full figures), medium \
(people from the waist up) or close-up (a face, or an important object).
- setting: where it is, the time of day and the light, as visual phrases. Keep it the same from panel to panel \
while the place does not change.
- action: what the picture shows: who is there, what they do, their poses and expressions. Use the characters' \
names. Describe only what can be drawn; never ask for written words, signs or letters in the picture.
- characters: everyone visible, with their position in the frame (left, center, right or background) and their \
expression. For anyone who is not in the main character list, describe their looks in "look" (from the text, or \
invented to suit them); leave "look" empty for main characters.
- caption: a short narration box (at most 20 words), only where it is needed to explain time, place or what \
happens; otherwise empty.
- dialogue: at most 3 speech balloons, in reading order. speaker is the character's name exactly as in the list. \
text is what they say: take it from the book's own dialogue wherever possible, shortened if it is long (at most \
25 words). Do not put quotation marks around it. style is speech, thought, shout or whisper.
- sfx: a sound effect word (for example CRASH or KNOCK) only when the scene has a striking sound; otherwise empty."""

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {"panels": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "shot": {"type": "string", "enum": ["establishing", "wide", "medium", "close-up"]},
            "setting": {"type": "string"},
            "action": {"type": "string"},
            "characters": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                               "position": {"type": "string", "enum": ["left", "center", "right", "background"]},
                               "expression": {"type": "string"},
                               "look": {"type": "string"}},
                "required": ["name", "position", "expression", "look"]}},
            "caption": {"type": "string"},
            "dialogue": {"type": "array", "items": {
                "type": "object",
                "properties": {"speaker": {"type": "string"}, "text": {"type": "string"},
                               "style": {"type": "string", "enum": ["speech", "thought", "shout", "whisper"]}},
                "required": ["speaker", "text", "style"]}},
            "sfx": {"type": "string"},
        },
        "required": ["shot", "setting", "action", "characters", "caption", "dialogue", "sfx"]}}},
    "required": ["panels"],
}

SHOT_WORDS = {
    "establishing": "establishing shot, wide view of the location, any figures small in the scene",
    "wide": "wide shot, full figures in their surroundings",
    "medium": "medium shot, characters seen from the waist up",
    "close-up": "close-up shot, face and shoulders filling the frame",
}
TAIL_TARGET_Y = {"establishing": 0.62, "wide": 0.5, "medium": 0.38, "close-up": 0.45}
POSITION_RANK = {"left": 0, "center": 1, "right": 2}


# =============================================================================
# Comic edition: who is who
# =============================================================================

TITLES_MALE = {"mr", "mister", "sir", "lord", "master", "king", "prince", "duke", "earl", "count", "baron",
               "father", "uncle", "brother", "monsieur", "herr", "senor", "señor", "don"}
TITLES_FEMALE = {"mrs", "miss", "ms", "lady", "madam", "madame", "mistress", "queen", "princess", "duchess",
                 "countess", "baroness", "mother", "aunt", "sister", "mademoiselle", "mlle", "frau", "fraulein",
                 "senora", "señora", "senorita", "señorita", "dona", "doña"}
TITLES_OTHER = {"dr", "doctor", "professor", "prof", "captain", "capt", "colonel", "col", "general", "major",
                "lieutenant", "lt", "sergeant", "sgt", "reverend", "rev", "saint", "st", "the", "old", "young",
                "little", "big", "agent", "detective", "inspector", "officer", "judge"}
NAME_TITLES = TITLES_MALE | TITLES_FEMALE | TITLES_OTHER
NAME_FILLERS = {"of", "de", "van", "von", "der", "la", "le", "du", "da", "di", "and"}
NOT_NAMES = {"i", "me", "you", "he", "she", "it", "we", "they", "him", "her", "them", "narrator", "the narrator",
             "unknown", "someone", "somebody", "everyone", "everybody", "nobody", "crowd", "the crowd", "all",
             "voice", "a voice", "stranger", "a stranger", "none", "narration", "caption"}


def name_key(s) -> str:
    s = str(s or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"[^\w\s'-]", " ", s)
    return clean_ws(s)


def split_name(key: str) -> tuple[frozenset, frozenset]:
    """'mrs elizabeth bennet' -> ({'mrs'}, {'elizabeth', 'bennet'})"""
    toks = key.split()
    titles = []
    while len(toks) > 1 and toks[0] in NAME_TITLES:
        titles.append(toks.pop(0))
    core = [t for t in toks if t not in NAME_FILLERS] or toks
    return frozenset(titles), frozenset(core)


def names_compatible(a: str, b: str) -> bool:
    """Could these two names (keys) belong to the same person?"""
    ta, ca = split_name(a)
    tb, cb = split_name(b)
    if not (ca <= cb or cb <= ca):
        return False
    return not (ta and tb and ta != tb)


@dataclass
class ComicWindow:
    chapter: int
    number: int
    text: str
    events: list
    setting: str
    characters: list


@dataclass
class CastMember:
    name: str
    aliases: list[str] = field(default_factory=list)
    gender: str = ""
    stated: list[str] = field(default_factory=list)  # appearance details the book gives
    presence: int = 0  # number of passages the character appears in
    first_chapter: int = 0
    design: dict = field(default_factory=dict)
    portrait: Path | None = None
    portrait_hash: str = ""
    seed: int | None = None
    image_override: Path | None = None

    @property
    def tag(self) -> str:
        return self.design.get("tag") or self.name

    @property
    def short(self) -> str:
        return self.design.get("short") or self.name

    @property
    def names(self) -> list[str]:
        return [self.name] + self.aliases


def build_cast(windows: list[ComicWindow], max_cast: int, min_presence: int) -> list[CastMember]:
    """Merge the per-passage character lists into one list of people, most present first."""
    mentions = []
    for w in windows:
        for c in w.characters:
            if not isinstance(c, dict):
                continue
            name = clean_ws(str(c.get("name") or ""))[:60]
            key = name_key(name)
            if not key or key in NOT_NAMES or len(key) < 2:
                continue
            aliases = [clean_ws(a)[:60] for a in (c.get("aliases") or []) if isinstance(a, str) and a.strip()]
            alias_keys = {name_key(a) for a in aliases} - NOT_NAMES - {""}
            keys = {key} | alias_keys | {t for t in split_name(key)[1] if len(t) >= 3}
            mentions.append({"name": name, "key": key, "aliases": aliases, "keys": keys,
                             "gender": name_key(c.get("gender")), "look": clean_ws(str(c.get("appearance") or "")),
                             "where": (w.chapter, w.number)})
    if not mentions:
        return []

    # A key is ambiguous when it is used for people whose main names cannot be the same person
    # ("Miss Bennet" for both Jane and Elizabeth, "Bennet" for the whole family).
    owners: dict[str, set[str]] = defaultdict(set)
    for m in mentions:
        for k in m["keys"]:
            owners[k].add(m["key"])
    ambiguous = set()
    for k, prims in owners.items():
        prims = sorted(prims)
        if any(not names_compatible(a, b) for i, a in enumerate(prims) for b in prims[i + 1:]):
            ambiguous.add(k)

    parent = list(range(len(mentions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_owner: dict[str, int] = {}
    for i, m in enumerate(mentions):
        for k in m["keys"]:
            if k in ambiguous and k != m["key"]:
                continue
            if k in first_owner:
                parent[find(i)] = find(first_owner[k])
            else:
                first_owner[k] = i

    groups: dict[int, list[dict]] = defaultdict(list)
    for i, m in enumerate(mentions):
        groups[find(i)].append(m)

    cast: list[CastMember] = []
    for ms in groups.values():
        counts = Counter(m["name"] for m in ms)
        top = counts.most_common(1)[0][1]
        name = max((n for n, c in counts.items() if c >= 0.3 * top),
                   key=lambda n: (len(split_name(name_key(n))[1]), counts[n], len(n)))
        alias_counts = Counter(a for m in ms for a in [m["name"]] + m["aliases"])
        aliases = [a for a, _ in alias_counts.most_common() if name_key(a) != name_key(name)]
        seen_alias, uniq = set(), []
        for a in aliases:
            if name_key(a) not in seen_alias:
                seen_alias.add(name_key(a))
                uniq.append(a)
        genders = Counter(m["gender"] for m in ms if m["gender"] and m["gender"] != "unknown")
        looks, seen_look = [], set()
        for m in ms:
            lk = m["look"]
            if len(lk) >= 3 and lk.lower() not in seen_look and lk.lower() not in ("none", "n/a", "unknown"):
                seen_look.add(lk.lower())
                looks.append(lk[:300])
        cast.append(CastMember(name=name, aliases=uniq[:8], gender=genders.most_common(1)[0][0] if genders else "",
                               stated=looks[:20], presence=len({m["where"] for m in ms}),
                               first_chapter=min(m["where"][0] for m in ms)))
    cast.sort(key=lambda c: (-c.presence, c.first_chapter, c.name))
    keep = [c for c in cast if c.presence >= min_presence]
    if len(keep) < 2:  # short book or a small model: take the most present anyway
        keep = cast[:max(2, len(keep))]
    return keep[:max_cast]


class CastIndex:
    """Finds the cast member a name in the script refers to."""

    def __init__(self, cast: list[CastMember]):
        self.cast = cast
        self.exact: dict[str, CastMember] = {}
        clash = set()
        for m in cast:
            for k in {name_key(n) for n in m.names}:
                if k in self.exact and self.exact[k] is not m:
                    clash.add(k)
                self.exact[k] = m
        for k in clash:
            self.exact.pop(k, None)

    def find(self, name: str) -> CastMember | None:
        key = name_key(name)
        if not key or key in NOT_NAMES:
            return None
        if key in self.exact:
            return self.exact[key]
        _, core = split_name(key)
        hits = []
        for m in self.cast:
            for n in m.names:
                nk = name_key(n)
                if core and core <= split_name(nk)[1] and names_compatible(key, nk):
                    hits.append(m)
                    break
        return hits[0] if len(hits) == 1 else None

    def same(self, a: str, b: str) -> bool:
        ma, mb = self.find(a), self.find(b)
        if ma is not None or mb is not None:
            return ma is mb
        ka, kb = name_key(a), name_key(b)
        return bool(ka) and (ka == kb or names_compatible(ka, kb))


def compose_tag(d: dict) -> str:
    """Design fields -> the fixed description used in every prompt for this character."""

    def f(k: str) -> str:
        v = clean_ws(str(d.get(k) or "")).strip(" ,.;")[:120]
        return "" if v.lower() in ("", "none", "unknown", "n/a", "not stated", "-") else v

    parts = [f("gender"), f("age")]
    build = f("build")
    if build:
        parts.append(build if re.search(r"\bbuild\b", build, re.I) else f"{build} build")
    skin = f("skin")
    if skin:
        parts.append(skin if re.search(r"skin|complexion", skin, re.I) else f"{skin} skin")
    hair = f("hair")
    if hair:
        parts.append(hair if re.search(r"hair|bald|shaved", hair, re.I) else f"{hair} hair")
    eyes = f("eyes")
    if eyes:
        parts.append(eyes if re.search(r"\beyes?\b", eyes, re.I) else f"{eyes} eyes")
    parts.append(f("face"))
    outfit = f("outfit")
    if outfit:
        parts.append(outfit if re.match(r"(wearing|in|dressed)\b", outfit, re.I) else f"wearing {outfit}")
    parts.append(f("accessories"))
    return ", ".join(p for p in parts if p)


def _trim_words(s: str, n: int) -> str:
    words = s.split()
    return s if len(words) <= n else " ".join(words[:n]).rstrip(",;:") + "\u2026"


def _strip_quotes(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] in "\"\u201c\u201d'\u2018" and s[-1] in "\"\u201c\u201d'\u2019":
        s = s[1:-1].strip()
    return s.strip("\"\u201c\u201d").strip()


def clean_panel(p: dict) -> dict | None:
    """Normalise one panel from the LLM; None if it has nothing to show."""
    shot = p.get("shot") if p.get("shot") in SHOT_WORDS else "medium"
    chars = []
    for c in p.get("characters") or []:
        if not isinstance(c, dict):
            continue
        name = clean_ws(str(c.get("name") or ""))[:60]
        if not name or name_key(name) in NOT_NAMES:
            continue
        pos = c.get("position") if c.get("position") in ("left", "center", "right", "background") else "center"
        chars.append({"name": name, "position": pos, "expression": clean_ws(str(c.get("expression") or ""))[:80],
                      "look": clean_ws(str(c.get("look") or ""))[:200]})
    caption = _strip_quotes(clean_ws(str(p.get("caption") or "")))
    dialogue = []
    for line in p.get("dialogue") or []:
        if not isinstance(line, dict):
            continue
        text = _strip_quotes(clean_ws(str(line.get("text") or "")))
        speaker = clean_ws(str(line.get("speaker") or ""))[:60]
        if not text:
            continue
        if name_key(speaker) in ("narrator", "the narrator", "narration", "caption"):
            caption = f"{caption} {text}".strip()
            continue
        style = line.get("style") if line.get("style") in ("speech", "thought", "shout", "whisper") else "speech"
        dialogue.append({"speaker": speaker, "text": _trim_words(text, 40), "style": style})
    sfx = clean_ws(str(p.get("sfx") or "")).strip(" !.\"'*")
    if len(sfx) > 16 or len(sfx.split()) > 2:
        sfx = ""
    panel = {"shot": shot, "setting": clean_ws(str(p.get("setting") or ""))[:300],
             "action": clean_ws(str(p.get("action") or ""))[:400], "characters": chars[:6],
             "caption": _trim_words(caption, 35), "dialogue": dialogue[:3], "sfx": sfx}
    if not panel["action"] and not panel["dialogue"] and not panel["caption"]:
        return None
    return panel


def visible_layout(panel: dict) -> list[tuple[dict, float]]:
    """Characters in the foreground, left to right, with the x position (0-1) of each."""
    vis = [c for c in panel.get("characters") or [] if c.get("position") != "background"][:4]
    vis = sorted(vis, key=lambda c: POSITION_RANK.get(c.get("position"), 1))  # stable: keeps script order
    n = len(vis)
    if n == 1:
        return [(vis[0], {"left": 0.3, "right": 0.7}.get(vis[0].get("position"), 0.5))]
    return [(c, (i + 0.5) / n) for i, c in enumerate(vis)]


def position_phrase(i: int, n: int, pos: str) -> str:
    if n == 1:
        return {"left": "on the left", "right": "on the right"}.get(pos, "")
    if n == 2:
        return ["on the left", "on the right"][i]
    if n == 3:
        return ["on the left", "in the center", "on the right"][i]
    return ["at the far left", "left of center", "right of center", "at the far right"][min(i, 3)]


def gen_size(aspect: float, megapixels: float) -> tuple[int, int]:
    """A render size near `megapixels` with this aspect ratio, in multiples of 64."""
    aspect = min(max(aspect, 0.42), 2.4)
    px = max(0.2, megapixels) * 1024 * 1024
    h = math.sqrt(px / aspect)
    w = h * aspect
    return max(256, int(round(w / 64)) * 64), max(256, int(round(h / 64)) * 64)


def layout_page(panels: list[dict], box: tuple[int, int, int, int], gutter: int,
                variant: int = 0) -> list[tuple[int, int, int, int]]:
    """Panel rectangles for one page: wide shots get a row of their own, others share rows."""
    x0, y0, x1, y1 = box
    shots = [p.get("shot", "medium") for p in panels]
    n = len(shots)
    if n == 0:
        return []
    rows: list[list[int]] = []
    i = 0
    while i < n:
        if shots[i] in ("establishing", "wide") or i == n - 1 or shots[i + 1] == "establishing":
            rows.append([i])
            i += 1
        elif i + 2 < n and n >= 5 and shots[i] == shots[i + 1] == shots[i + 2] == "close-up":
            rows.append([i, i + 1, i + 2])
            i += 3
        else:
            rows.append([i, i + 1])
            i += 2
    while len(rows) > 4:
        cands = [(len(rows[r]) + len(rows[r + 1]), r) for r in range(len(rows) - 1)
                 if len(rows[r]) + len(rows[r + 1]) <= 3]
        if not cands:
            break
        _, r = min(cands)
        rows[r:r + 2] = [rows[r] + rows[r + 1]]
    weights = []
    for row in rows:
        if len(row) == 1:
            weights.append(1.25 if shots[row[0]] in ("establishing", "wide") else 1.1)
        else:
            weights.append(1.0 if len(row) == 2 else 0.85)
    avail_h = (y1 - y0) - gutter * (len(rows) - 1)
    total = sum(weights)
    rects: list = [None] * n
    y = y0
    for r, row in enumerate(rows):
        h = round(avail_h * weights[r] / total) if r < len(rows) - 1 else y1 - y
        k = len(row)
        avail_w = (x1 - x0) - gutter * (k - 1)
        if k == 2:
            fr = [(0.5, 0.5), (0.58, 0.42), (0.42, 0.58)][(variant + r) % 3]
            a, b = shots[row[0]], shots[row[1]]
            if a == "close-up" and b != "close-up":
                fr = (0.42, 0.58)
            elif b == "close-up" and a != "close-up":
                fr = (0.58, 0.42)
        else:
            fr = [1 / k] * k
        x = x0
        for j, idx in enumerate(row):
            w = round(avail_w * fr[j]) if j < k - 1 else x1 - x
            rects[idx] = (x, y, x + w, y + h)
            x += w + gutter
        y += h + gutter
    return rects


# =============================================================================
# Comic edition: lettering
# =============================================================================

FONT_DIRS = [str(SCRIPT_DIR / "fonts"), "/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts",
             "~/.local/share/fonts", "/Library/Fonts", "/System/Library/Fonts", "~/Library/Fonts",
             "C:/Windows/Fonts", "%LOCALAPPDATA%/Microsoft/Windows/Fonts"]
LETTERING_FONTS = ["animeace2_bld.ttf", "animeace2_reg.ttf", "ComicNeue-Bold.ttf", "ComicNeue-Bold.otf",
                   "ComicNeue_Bold.otf", "Comic Neue Bold.ttf", "comicbd.ttf", "Comic Sans MS Bold.ttf"]
TITLE_FONTS = ["Bangers-Regular.ttf", "Bangers.ttf", "impact.ttf", "Impact.ttf"]
SANS_FONTS = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
              "FreeSansBold.ttf", "NotoSans-Bold.ttf"]
_FONT_INDEX: dict[str, str] | None = None


def find_font(names: list[str]) -> str:
    """Path of the first installed font file in `names` (searched in the usual font folders)."""
    global _FONT_INDEX
    if _FONT_INDEX is None:
        _FONT_INDEX = {}
        for d in FONT_DIRS:
            root = Path(os.path.expandvars(os.path.expanduser(d)))
            if not root.is_dir():
                continue
            try:
                for p in root.rglob("*"):
                    if p.suffix.lower() in (".ttf", ".otf", ".ttc"):
                        _FONT_INDEX.setdefault(p.name.lower(), str(p))
            except OSError:
                continue
    for n in names:
        hit = _FONT_INDEX.get(n.lower())
        if hit:
            return hit
    return ""


def open_font(path: str, size: int):
    for p in (path, find_font(LETTERING_FONTS + SANS_FONTS)):
        if p:
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                log.debug("Could not open font %s", p)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inside(pt, poly) -> bool:
    x, y = pt
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def _bezier(p0, p1, p2, steps: int = 10) -> list[tuple[float, float]]:
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / steps for i in range(steps + 1))]


class Letterer:
    """Draws captions, speech balloons and sound effects onto finished panels."""

    SS = 3  # supersampling, for smooth balloon outlines
    BLACK = (0, 0, 0, 255)
    WHITE = (255, 255, 255, 255)

    def __init__(self, ccfg: dict, page_w: int):
        scale = page_w / 1600
        self.size = max(12, round(float(ccfg.get("font_size", 30)) * scale))
        self.upper = bool(ccfg.get("uppercase", True))
        self.bw = max(2, round(3 * scale))
        self.font_path = ccfg.get("font") or find_font(LETTERING_FONTS) or find_font(SANS_FONTS)
        self.title_path = ccfg.get("title_font") or find_font(TITLE_FONTS) or self.font_path
        self.sfx_path = ccfg.get("sfx_font") or find_font(TITLE_FONTS) or self.font_path
        self._fonts: dict = {}

    def font(self, kind: str, size: int):
        path = {"text": self.font_path, "title": self.title_path, "sfx": self.sfx_path}[kind]
        key = (path, int(size))
        if key not in self._fonts:
            self._fonts[key] = open_font(path, int(size))
        return self._fonts[key]

    def text(self, s: str) -> str:
        return s.upper() if self.upper else s

    @staticmethod
    def wrap(text: str, font, max_w: float) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            while font.getlength(word) > max_w and len(word) > 2:  # a word wider than the balloon
                cut = len(word) - 1
                while cut > 1 and font.getlength(word[:cut] + "-") > max_w:
                    cut -= 1
                if line:
                    lines.append(line)
                    line = ""
                lines.append(word[:cut] + "-")
                word = word[cut:]
            trial = f"{line} {word}".strip()
            if line and font.getlength(trial) > max_w:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        return lines or [""]

    # -- one panel -------------------------------------------------------------
    def letter_panel(self, page: Image.Image, rect, panel: dict, speaker_x, tail_y: float,
                     heads: list[float] = ()) -> None:
        """speaker_x(name) -> horizontal position 0-1 of that speaker in the panel, or None if not shown.
        heads: x positions (0-1) of the characters drawn in the panel; balloons try not to cover them."""
        x0, y0, x1, y1 = rect
        w0, h0 = x1 - x0, y1 - y0
        if w0 < 60 or h0 < 60 or not (panel.get("caption") or panel.get("dialogue") or panel.get("sfx")):
            return
        S = self.SS
        W, H = w0 * S, h0 * S
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        inset = round(self.size * 0.45 * S)
        gap = round(self.size * 0.35 * S)
        obstacles: list[tuple] = []
        if panel.get("caption"):
            box = self._caption(d, self.text(panel["caption"]), W, H, inset)
            obstacles.append((box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap))
        # Where the faces probably are (from the positions the script asked for), to keep balloons off them.
        close = panel.get("shot") == "close-up"
        half_w = (0.22 if close else min(0.12, 0.45 / max(1, len(heads)))) * W
        top, bottom = ((0.2, 0.7) if close else (tail_y - 0.14, tail_y + 0.12))
        faces = [(x * W - half_w, top * H, x * W + half_w, bottom * H) for x in heads]
        tail_y = max(0.05, tail_y - 0.08)  # tails point at the top of the head
        prev = None
        for n, line in enumerate(panel.get("dialogue") or []):
            style = line.get("style", "speech")
            sx = speaker_x(line.get("speaker", ""))
            want = (sx if sx is not None else (0.3 if n % 2 == 0 else 0.7)) * W
            if prev is None:
                min_y = inset
            elif want >= prev[0] - W * 0.05:
                min_y = prev[1]
            else:  # a later speaker further left must sit lower, so the balloons read in order
                min_y = prev[1] + prev[3] * 0.6
            best = None
            for scale in (1.0, 0.9, 0.8, 0.7, 0.62):  # shrink the lettering until the balloon has room
                g = self._fit(self.text(line["text"]), style, scale, W - 2 * inset)
                if 2 * g["ea"] > W - 2 * inset or 2 * g["eb"] > H * 0.85:  # too big for the panel at all
                    best = (g, *self._place(2 * g["ea"], 2 * g["eb"], want, min_y, obstacles, faces,
                                            W, H, inset)[:2])
                    continue
                bx, by, free = self._place(2 * g["ea"], 2 * g["eb"], want, min_y, obstacles, faces, W, H, inset)
                if free or scale == 0.62:
                    best = (g, bx, by)
                if free:
                    break
            g, bx, by = best
            cx, cy = bx + g["ea"], by + g["eb"]
            if sx is not None:
                target = (sx * W, tail_y * H)
            else:  # speaker outside the frame: point at the nearer edge
                target = (-0.3 * W if cx < W / 2 else 1.3 * W, cy + g["b"] * 0.4)
            self._balloon(d, g, style, cx, cy, target, H)
            obstacles.append((bx - gap, by - gap, bx + 2 * g["ea"] + gap, by + 2 * g["eb"] + gap))
            prev = (cx, by, 2 * g["ea"], 2 * g["eb"])
        if panel.get("sfx"):
            self._sfx(layer, panel["sfx"].upper(), W, H, inset, obstacles)
        small = layer.resize((w0, h0), Image.LANCZOS)
        page.paste(small, (x0, y0), small)

    # -- pieces ----------------------------------------------------------------
    def _caption(self, d: ImageDraw.ImageDraw, text: str, W: int, H: int, inset: int):
        S = self.SS
        for scale in (0.92, 0.8, 0.7):
            fs = round(self.size * S * scale)
            font = self.font("text", fs)
            lh = fs * 1.16
            pad = round(fs * 0.45)
            max_w = (W - 2 * inset) * (0.66 if W > fs * 22 else 1.0) - 2 * pad
            lines = self.wrap(text, font, max_w)
            if lh * len(lines) + 2 * pad <= H * 0.45:
                break
        tw = max(font.getlength(ln) for ln in lines)
        box = (inset, inset, inset + tw + 2 * pad, inset + lh * len(lines) + 2 * pad)
        d.rectangle(box, fill=(255, 243, 186, 255), outline=self.BLACK, width=self.bw * S)
        y = inset + pad + lh / 2
        for ln in lines:
            d.text((inset + pad, y), ln, font=font, fill=self.BLACK, anchor="lm")
            y += lh
        return box

    def _fit(self, text: str, style: str, scale: float, max_w: float) -> dict:
        S = self.SS
        fs = round(self.size * S * scale * {"shout": 1.08, "whisper": 0.92}.get(style, 1.0))
        font = self.font("text", fs)
        lh = fs * 1.16
        total = font.getlength(text)
        longest = max((font.getlength(wd) for wd in text.split()), default=0)
        target = math.sqrt(max(total, 1) * lh * 2.0)  # text block about twice as wide as high
        target = min(max(target, fs * 4, longest), max(max_w * 0.62, min(longest, max_w)))
        lines = self.wrap(text, font, target)
        tw = max(font.getlength(ln) for ln in lines)
        th = lh * len(lines)
        pad = fs * 0.55
        a, b = tw / 2 * 1.30 + pad, th / 2 * 1.30 + pad * 0.8
        bw = self.bw * S
        extra = {"shout": (a * 0.25, b * 0.25), "thought": (fs * 0.65, fs * 0.65)}.get(style, (0, 0))
        return {"font": font, "fs": fs, "lh": lh, "lines": lines, "th": th, "a": a, "b": b,
                "ea": a + extra[0] + bw, "eb": b + extra[1] + bw}

    def _place(self, w: float, h: float, want_cx: float, min_y: float, obstacles, faces, W: int, H: int,
               inset: int) -> tuple[float, float, bool]:
        """Top-left corner for a balloon, as high as possible near want_cx, and whether the spot is free.
        Tries, in order: the top of the panel clear of faces, the top at all, anywhere clear of faces,
        anywhere clear of other lettering."""
        step = max(4, round(self.size * self.SS * 0.35))
        offsets = [0.0]
        k = 1
        while k * step * 2 < W:
            offsets += [k * step * 2.0, -k * step * 2.0]
            k += 1

        def x_for(dx):
            return min(max(want_cx - w / 2 + dx, inset), max(inset, W - inset - w))

        for limit, blocked in ((0.62, obstacles + faces), (0.62, obstacles), (1.0, obstacles + faces),
                               (1.0, obstacles)):
            y = max(min_y, inset)
            while y + h <= H * limit - (inset if limit == 1.0 else 0):
                for dx in offsets:
                    x = x_for(dx)
                    if not any(_overlaps((x, y, x + w, y + h), o) for o in blocked):
                        return x, y, True
                y += step
        return x_for(0), min(max(min_y, inset), max(inset, H - inset - h)), False

    def _tail(self, cx, cy, a, b, tx, ty, fs, H, curve=True):
        ang = math.atan2((ty - cy) / b, (tx - cx) / a)
        ex, ey = cx + a * math.cos(ang), cy + b * math.sin(ang)
        dx, dy = tx - ex, ty - ey
        dist = math.hypot(dx, dy)
        if dist < fs * 0.6:  # the speaker is right under the balloon: short tail straight down
            ang, ex, ey, dx, dy, dist = math.pi / 2, cx, cy + b, 0.0, 1.0, 1.0
        ux, uy = dx / dist, dy / dist
        length = min(max(dist * 0.8, fs * 1.0), max(fs * 1.2, min(H * 0.2, fs * 6)))
        tip = (ex + ux * length, ey + uy * length)
        px, py = -uy, ux
        hw = min(0.3 * a, 0.3 * b, fs * 0.7)
        bx, by = cx + a * 0.72 * math.cos(ang), cy + b * 0.72 * math.sin(ang)
        b1, b2 = (bx + px * hw, by + py * hw), (bx - px * hw, by - py * hw)
        if not curve:
            return [b1, tip, b2]
        bend = 0.18 * length * (1 if tx >= cx else -1)
        c1 = ((b1[0] + tip[0]) / 2 + px * bend, (b1[1] + tip[1]) / 2 + py * bend)
        c2 = ((b2[0] + tip[0]) / 2 + px * bend, (b2[1] + tip[1]) / 2 + py * bend)
        return _bezier(b1, c1, tip) + _bezier(tip, c2, b2)[1:]

    def _balloon(self, d: ImageDraw.ImageDraw, g: dict, style: str, cx, cy, target, H) -> None:
        S = self.SS
        bw = self.bw * S
        a, b, fs = g["a"], g["b"], g["fs"]
        K, Wt = self.BLACK, self.WHITE
        tx, ty = target
        ell = (cx - a, cy - b, cx + a, cy + b)
        grown = (cx - a - bw, cy - b - bw, cx + a + bw, cy + b + bw)
        if style == "thought":
            r = fs * 0.62
            per = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
            k = max(9, int(per / (r * 1.5)))
            bumps = [(cx + a * 0.97 * math.cos(2 * math.pi * i / k), cy + b * 0.97 * math.sin(2 * math.pi * i / k), r)
                     for i in range(k)]
            ang = math.atan2(ty - cy, tx - cx)
            under = ((tx - cx) / (a + r)) ** 2 + ((ty - cy) / (b + r)) ** 2 < 1.4  # thinker right under the cloud
            if under:
                ang = math.pi / 2
            ex, ey = cx + (a + r) * math.cos(ang), cy + (b + r) * math.sin(ang)
            length = fs * 1.8 if under else min(max(math.hypot(tx - ex, ty - ey) * 0.8, fs * 1.6), H * 0.2)
            dots = [(ex + math.cos(ang) * length * t, ey + math.sin(ang) * length * t, fs * rr)
                    for t, rr in ((0.3, 0.42), (0.62, 0.3), (0.92, 0.2))]
            for (x, y, rr) in bumps + dots:
                d.ellipse((x - rr - bw, y - rr - bw, x + rr + bw, y + rr + bw), fill=K)
            d.ellipse(grown, fill=K)
            for (x, y, rr) in bumps + dots:
                d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=Wt)
            d.ellipse(ell, fill=Wt)
        elif style == "shout":
            rnd = random.Random(int(cx) * 7 + int(cy))
            pts = []
            for i in range(44):
                ang = math.pi * i / 22
                rr = 1.0 if i % 2 else 1.2 + rnd.uniform(-0.04, 0.08)
                pts.append((cx + a * rr * math.cos(ang), cy + b * rr * math.sin(ang)))
            tail = self._tail(cx, cy, a, b, tx, ty, fs, H, curve=False)
            d.polygon(pts, fill=K)
            d.line(pts + [pts[0]], fill=K, width=2 * bw, joint="curve")
            d.polygon(tail, fill=K)
            d.line(tail, fill=K, width=2 * bw, joint="curve")
            d.polygon(pts, fill=Wt)
            d.polygon(tail, fill=Wt)
        else:
            tail = self._tail(cx, cy, a, b, tx, ty, fs, H)
            if style == "whisper":  # dashed outline, open where the tail joins
                d.polygon(tail, fill=Wt)
                for i in range(0, len(tail) - 1, 2):
                    d.line([tail[i], tail[i + 1]], fill=K, width=bw)
                d.ellipse(ell, fill=Wt)
                for start in range(0, 360, 15):
                    mid = math.radians(start + 4.5)
                    if not _inside((cx + a * math.cos(mid), cy + b * math.sin(mid)), tail):
                        d.arc(ell, start, start + 9, fill=K, width=bw)
            else:
                d.ellipse(grown, fill=K)
                d.polygon(tail, fill=K)
                d.line(tail, fill=K, width=2 * bw, joint="curve")
                d.ellipse(ell, fill=Wt)
                d.polygon(tail, fill=Wt)
        y = cy - g["th"] / 2 + g["lh"] / 2
        for ln in g["lines"]:
            d.text((cx, y), ln, font=g["font"], fill=K, anchor="mm")
            y += g["lh"]

    def _sfx(self, layer: Image.Image, text: str, W: int, H: int, inset: int, obstacles) -> None:
        S = self.SS
        fs = round(self.size * S * 2.4)
        font = self.font("sfx", fs)
        while font.getlength(text) > W * 0.6 and fs > self.size * S:
            fs = int(fs * 0.85)
            font = self.font("sfx", fs)
        stroke = max(2, round(fs * 0.09))
        tw, th = int(font.getlength(text) + 2 * stroke + 8), int(fs * 1.4 + 2 * stroke)
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((tw / 2, th / 2), text, font=font, fill=(255, 214, 40, 255),
                                 stroke_width=stroke, stroke_fill=self.BLACK, anchor="mm")
        img = img.rotate(8, expand=True, resample=Image.BICUBIC)
        if img.width > W or img.height > H:
            f = min(W / img.width, H / img.height) * 0.9
            img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)
        w, h = img.size
        spots = [(W - inset - w, H - inset - h), (inset, H - inset - h), (W - inset - w, (H - h) // 2),
                 ((W - w) // 2, H - inset - h), (inset, (H - h) // 2)]
        spots = [(int(max(0, x)), int(max(0, y))) for x, y in spots]
        spot = next((s for s in spots if not any(_overlaps((s[0], s[1], s[0] + w, s[1] + h), o) for o in obstacles)),
                    spots[0])
        layer.alpha_composite(img, spot)


def comic_info_xml(title: str, series: str, number: int | None, summary: str, writer: str, genre: str,
                   language: str, characters: list[str], pages: list[dict]) -> bytes:
    """ComicInfo.xml (the ComicRack metadata most comic readers understand)."""
    root = etree.Element("ComicInfo", nsmap={"xsd": "http://www.w3.org/2001/XMLSchema",
                                             "xsi": "http://www.w3.org/2001/XMLSchema-instance"})

    def add(tag, value):
        if value not in (None, ""):
            etree.SubElement(root, tag).text = str(value)

    add("Title", title)
    add("Series", series)
    add("Number", number)
    add("Summary", summary)
    add("Notes", "Comic adaptation made with epub_narrator using local AI (Ollama and ComfyUI).")
    add("Writer", writer)
    add("Genre", genre)
    add("PageCount", len(pages))
    add("LanguageISO", (language or "").split("-")[0])
    add("Manga", "No")
    add("Characters", ", ".join(characters))
    pages_el = etree.SubElement(root, "Pages")
    for i, pg in enumerate(pages):
        el = etree.SubElement(pages_el, "Page")
        el.set("Image", str(i))
        if pg.get("type"):
            el.set("Type", pg["type"])
        el.set("ImageSize", str(len(pg["data"])))
        el.set("ImageWidth", str(pg["width"]))
        el.set("ImageHeight", str(pg["height"]))
        if pg.get("bookmark"):
            el.set("Bookmark", pg["bookmark"])
    return etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True)


def write_cbz(path: Path, pages: list[dict], info_xml: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().timetuple()[:6]

    def entry(name: str) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(name, date_time=stamp)
        info.external_attr = 0o644 << 16
        return info

    with zipfile.ZipFile(tmp, "w") as z:
        for i, pg in enumerate(pages):
            z.writestr(entry(f"{i:04d}.jpg"), pg["data"], compress_type=zipfile.ZIP_STORED)
        z.writestr(entry("ComicInfo.xml"), info_xml, compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, path)


# =============================================================================
# Comic edition: the maker
# =============================================================================

class ComicMaker:
    def __init__(self, pipe: "Pipeline"):
        self.p = pipe
        self.cfg = pipe.cfg["comic"]
        self.cache = pipe.cache
        self.llm = pipe.llm
        self.epub = pipe.epub
        self.windows: dict[int, list[ComicWindow]] = {}
        self.cast: list[CastMember] = []
        self.index = CastIndex([])
        self.profile = {"genre": "", "world": "", "mood": ""}
        self.scripts: dict[int, dict] = {}
        self.cover_prompt: str | None = None
        self.cover_image: Image.Image | None = None
        self.failed: set[int] = set()
        self.panel_wf: Workflow | None = None
        self.char_wf: Workflow | None = None
        self.pw, self.ph = int(self.cfg["page_width"]), int(self.cfg["page_height"])
        s = self.pw / 1600
        self.margin = round(int(self.cfg["margin"]) * s)
        self.gutter = round(int(self.cfg["gutter"]) * s)
        self.border = max(1, round(int(self.cfg["border"]) * s))
        self.banner_h = round(self.ph * 0.07)

    # -- setup ----------------------------------------------------------------
    def load_workflows(self, image_keys: list[str]) -> None:
        self.panel_wf = Workflow(resolve_file(self.cfg["panel_workflow"], self.p.cfg_dir), ["PROMPT"],
                                 COMIC_IMAGE_KEYS, "Comic panel")
        cw = self.cfg.get("character_workflow") or self.p.cfg["images"]["workflow"]
        self.char_wf = Workflow(resolve_file(cw, self.p.cfg_dir), ["PROMPT"], image_keys, "Character portrait")
        if not self.panel_wf.uses_refs:
            log.warning("The comic panel workflow %s takes no character pictures ({{REF_IMAGE_1}}...); characters "
                        "are kept consistent through their descriptions only.", self.panel_wf.path.name)

    def workflows(self) -> list[Workflow]:
        return [w for w in (self.panel_wf, self.char_wf) if w is not None]

    def _ask(self, kind: str, system: str, user: str, schema: dict, key_parts) -> dict | None:
        key = sha("comic", kind, COMIC_PROMPT_VERSION, self.llm.model, key_parts)
        hit = self.cache.get_json("comic_llm", key)
        if hit is not None:
            return hit
        try:
            ans = self.llm.chat_json(system, user, schema)
        except LLMError as e:
            log.warning("    %s step failed: %s", kind, e)
            return None
        self.cache.put_json("comic_llm", key, ans)
        return ans

    def _book_line(self) -> str:
        line = f'Book: "{self.epub.title}" by {self.epub.author}.'
        if self.profile.get("genre"):
            line += f" Genre: {self.profile['genre']}."
        if self.profile.get("world"):
            line += f" Setting: {self.profile['world']}."
        return line

    # =========================================================================
    # Phase 1: everything that needs the LLM
    # =========================================================================
    @property
    def wants_cover(self) -> bool:  # per-chapter files open on their first page instead
        return bool(self.cfg.get("cover", True)) and self.cfg.get("split", "book") == "book"

    def plan(self) -> None:
        self.read_book()
        self.make_profile()
        self.make_cast()
        self.write_scripts()
        if self.wants_cover:
            self.make_cover_prompt()

    # -- reading --------------------------------------------------------------
    def read_book(self) -> None:
        whole = self.cfg.get("cast_scope", "book") != "selected"
        size = int(self.cfg.get("read_chars", 6000))
        live = [c for c in self.p.chapters if whole or c.number in self.p.selected]
        total = sum(len(build_windows(c, size)) for c in live)
        log.info("Reading %d chapter(s) for the plot and the characters with %s%s", len(live), self.llm.model,
                 " (the whole book, so the cast is complete)" if whole and len(live) > len(self.p.selected) else "")
        prog = Progress("reading", total)
        done = 0
        for ch in self.p.chapters:
            is_live = whole or ch.number in self.p.selected
            out: list[ComicWindow] = []
            complete = True
            for n, idxs in enumerate(build_windows(ch, size), 1):
                text = "\n\n".join(ch.blocks[i].text for i in idxs)
                by_text = sha("comic-read", text)
                if is_live:
                    user = f'Book: "{self.epub.title}" by {self.epub.author}\nChapter: {ch.title}\n\nPassage:\n{text}'
                    ans = self._ask("read", READ_SYSTEM, user, READ_SCHEMA, [text])
                    if ans is not None and self.cache.get_json("comic_read_by_text", by_text) is None:
                        self.cache.put_json("comic_read_by_text", by_text, ans)
                    done += 1
                    prog.update(done)
                else:
                    ans = self.cache.get_json("comic_read_by_text", by_text)
                if ans is None:
                    complete = False
                    continue
                events = []
                for e in ans.get("events") or []:
                    if isinstance(e, dict) and clean_ws(str(e.get("what") or "")):
                        try:
                            imp = max(1, min(5, int(e.get("importance") or 3)))
                        except (TypeError, ValueError):
                            imp = 3
                        events.append({"what": clean_ws(str(e["what"]))[:300], "importance": imp})
                out.append(ComicWindow(ch.number, n, text, events, clean_ws(str(ans.get("setting") or ""))[:300],
                                       [c for c in ans.get("characters") or [] if isinstance(c, dict)]))
            if out and (is_live or complete):
                self.windows[ch.number] = out

    def make_profile(self) -> None:
        frozen = self.cache.get_json("comic", "profile")
        if frozen:
            self.profile = frozen
            return
        settings = []
        for ws in self.windows.values():
            for w in ws:
                if w.setting and w.setting not in settings:
                    settings.append(w.setting)
        titles = ", ".join(c.title for c in self.p.chapters[:40])
        user = (f'Title: "{self.epub.title}"\nAuthor: {self.epub.author}\nChapters: {titles}\n\n'
                "Settings of some passages:\n" + "\n".join(f"- {s}" for s in settings[:20])
                + f"\n\nOpening of the book:\n{self.p.chapters[0].text[:2000]}")
        ans = self._ask("profile", PROFILE_SYSTEM, user, PROFILE_SCHEMA, [user])
        if ans:
            self.profile = {k: clean_ws(str(ans.get(k) or ""))[:300] for k in ("genre", "world", "mood")}
            self.cache.put_json("comic", "profile", self.profile)

    # -- cast -----------------------------------------------------------------
    def make_cast(self) -> None:
        everything = [w for n in sorted(self.windows) for w in self.windows[n]]
        self.cast = build_cast(everything, int(self.cfg.get("max_cast", 12)), int(self.cfg.get("min_presence", 2)))
        self.index = CastIndex(self.cast)
        if not self.cast:
            log.warning("No recurring characters found; panels will describe people from the script alone")
            return
        designs = self.cache.get_json("comic", "designs") or {}
        new = [m for m in self.cast if not any(name_key(n) in designs for n in m.names)]
        if new:
            log.info("Designing the look of %d character(s)", len(new))
        for m in self.cast:
            known = next((designs[name_key(n)] for n in m.names if name_key(n) in designs), None)
            if known is None:
                known = self._design(m, everything)
                designs[name_key(m.name)] = known
                self.cache.put_json("comic", "designs", designs)
            m.design = dict(known)
        self._apply_overrides()
        for m in self.cast:
            m.design["tag"] = m.design.get("look") or compose_tag(m.design) or m.name
            if not m.design.get("short"):
                m.design["short"] = clean_ws(f"{m.design.get('hair', '')} {m.design.get('gender', '')}") or m.name
        log.info("Cast: %s", ", ".join(m.name for m in self.cast))

    def _design(self, m: CastMember, windows: list[ComicWindow]) -> dict:
        tokens = {t for n in m.names for t in split_name(name_key(n))[1] if len(t) >= 3}
        pats = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in sorted(tokens)]
        role = []
        for w in windows:
            for e in w.events:
                if any(p.search(e["what"]) for p in pats) and e["what"] not in role:
                    role.append(e["what"])
        others = [f"- {o.name}: {o.design.get('short')}" for o in self.cast if o is not m and o.design]
        stated = "\n".join(f"- {s}" for s in m.stated) or \
            "(The book does not describe how this character looks: invent everything.)"
        user = (self._book_line() + "\n\n"
                + f"Character: {m.name}" + (f" (also called {', '.join(m.aliases[:5])})" if m.aliases else "")
                + f"\nGender according to the text: {m.gender or 'unknown'}\n\n"
                + f"What the book says about their appearance:\n{stated}\n\n"
                + "What they do in the story:\n" + ("\n".join(f"- {r}" for r in role[:8]) or "- (unknown)") + "\n")
        if others:
            user += f"\nCharacters already designed (make {m.name} look clearly different):\n" + "\n".join(others)
        ans = self._ask("design", DESIGN_SYSTEM, user, DESIGN_SCHEMA, [m.name, m.stated, self.profile])
        if not ans:
            return {"gender": m.gender, "look": ", ".join([m.gender] + m.stated[:4]).strip(", ") or m.name,
                    "short": m.name, "invented": []}
        design = {k: clean_ws(str(ans.get(k) or ""))[:160] for k in DESIGN_FIELDS + ["short"]}
        design["invented"] = [str(x) for x in ans.get("invented") or [] if str(x) in DESIGN_FIELDS]
        return design

    def _apply_overrides(self) -> None:
        for name, ov in (self.cfg.get("cast") or {}).items():
            if not isinstance(ov, dict):
                continue
            m = self.index.find(name)
            if m is None:
                log.warning("[comic.cast] %r does not match anyone in the cast (%s)", name,
                            ", ".join(c.name for c in self.cast))
                continue
            for k in DESIGN_FIELDS + ["short", "look"]:
                if ov.get(k):
                    m.design[k] = clean_ws(str(ov[k]))
                    if k in m.design.get("invented", []):
                        m.design["invented"] = [x for x in m.design["invented"] if x != k]
            if ov.get("seed") is not None:
                m.seed = int(ov["seed"])
            if ov.get("image"):
                p = resolve_file(str(ov["image"]), self.p.cfg_dir)
                if p.is_file():
                    m.image_override = p
                else:
                    log.warning("[comic.cast] reference image for %s not found: %s", m.name, p)

    # -- scripts --------------------------------------------------------------
    def write_scripts(self) -> None:
        todo = [c for c in self.p.chapters if c.number in self.p.selected]
        log.info("Writing the comic script for %d chapter(s)", len(todo))
        for ch in self.p.chapters:
            if ch.number in self.p.selected:
                log.info("  [%d/%d] %s", ch.number, len(self.p.chapters), ch.title)
                script = self._script_chapter(ch)
                if script:
                    self.cache.put_json("comic_scripts", f"ch{ch.number:04d}", script)
                    self.scripts[ch.number] = script
                    n_pan = sum(len(pg["panels"]) for pg in script["pages"])
                    log.info("    %d page(s), %d panel(s)", len(script["pages"]), n_pan)
            else:
                s = self.cache.get_json("comic_scripts", f"ch{ch.number:04d}")
                if s and s.get("text_hash") == sha(ch.text) and s.get("pages"):
                    self.scripts[ch.number] = s

    def _pages_for(self, ch: Chapter, n_events: int) -> int:
        est = round(ch.words / 1000 * float(self.cfg.get("pages_per_1000_words", 0.6)))
        n = min(max(est, int(self.cfg.get("min_pages", 1))), int(self.cfg.get("max_pages", 10)))
        return max(1, min(n, n_events))

    def _script_chapter(self, ch: Chapter) -> dict | None:
        windows = self.windows.get(ch.number) or []
        events = [(w.number, e) for w in windows for e in w.events]
        if not events:
            log.warning("    no events found in this chapter; skipped")
            return None
        n_pages = self._pages_for(ch, len(events))
        plan = self._plan_chapter(ch, events, n_pages)
        pages, prev = [], []
        for k, (ids, summary) in enumerate(plan, 1):
            panels = self._script_page(ch, k, len(plan), ids, events, windows, prev)
            if not panels:
                log.warning("    page %d: no usable panels, skipped", k)
                continue
            pages.append({"summary": summary, "events": [events[i - 1][1]["what"] for i in ids], "panels": panels})
            prev.append(summary)
        if not pages:
            return None
        return {"chapter": ch.number, "title": ch.title, "text_hash": sha(ch.text), "model": self.llm.model,
                "pages": pages}

    def _plan_chapter(self, ch: Chapter, events: list, n_pages: int) -> list[tuple[list[int], str]]:
        listing = "\n".join(f"{i}. [{e['importance']}] {e['what']}" for i, (_, e) in enumerate(events, 1))
        user = (self._book_line() + f"\nChapter {ch.number}: {ch.title}\n\n"
                f"Number of pages: {n_pages}\n\nEvents:\n{listing}")
        ans = self._ask("plan", PLAN_SYSTEM, user, PLAN_SCHEMA, [user])
        plan, last = [], 0
        for pg in (ans or {}).get("pages") or []:
            if not isinstance(pg, dict):
                continue
            ids = []
            for x in pg.get("events") or []:
                try:
                    ids.append(int(x))
                except (TypeError, ValueError):
                    continue
            ids = sorted({i for i in ids if 1 <= i <= len(events) and i > last})
            if ids:
                plan.append((ids, clean_ws(str(pg.get("summary") or ""))[:300]))
                last = ids[-1]
        while len(plan) > n_pages:  # merge the two smallest neighbouring pages
            r = min(range(len(plan) - 1), key=lambda j: len(plan[j][0]) + len(plan[j + 1][0]))
            plan[r:r + 2] = [(plan[r][0] + plan[r + 1][0], f"{plan[r][1]} {plan[r + 1][1]}".strip())]
        if plan:
            return [(ids, summary or "; ".join(events[i - 1][1]["what"] for i in ids)) for ids, summary in plan]
        # Fallback: keep the most important events in story order and share them out.
        budget = n_pages * 3
        ranked = sorted(range(1, len(events) + 1), key=lambda i: (-events[i - 1][1]["importance"], i))[:budget]
        keep = sorted(ranked)
        per = math.ceil(len(keep) / n_pages)
        chunks = [keep[j:j + per] for j in range(0, len(keep), per)]
        return [(c, "; ".join(events[i - 1][1]["what"] for i in c)[:300]) for c in chunks]

    def _script_page(self, ch: Chapter, k: int, total: int, ids: list[int], events: list,
                     windows: list[ComicWindow], prev: list[str]) -> list[dict]:
        wanted = sorted({events[i - 1][0] for i in ids})
        parts = [w.text for w in windows if w.number in wanted]
        each = max(1500, 9000 // max(1, len(parts)))
        src = "\n\n[...]\n\n".join(t[:each] for t in parts)
        # Names only: changing a character's look later must not rewrite the script.
        cast_lines = "\n".join(f"- {m.name}" + (f" (also called {', '.join(m.aliases[:3])})" if m.aliases else "")
                               for m in self.cast) or "- (none)"
        lo = max(1, int(self.cfg.get("panels_per_page", 5)) - 1)
        hi = max(lo, int(self.cfg.get("max_panels_per_page", 6)))
        user = (self._book_line() + f"\nChapter {ch.number}: {ch.title} - page {k} of {total}.\n"
                + (f"Earlier pages of this chapter: {' '.join(prev[-2:])}\n" if prev else "")
                + f"\nMain characters (use exactly these names):\n{cast_lines}\n"
                + "\nThis page must show:\n" + "\n".join(f"- {events[i - 1][1]['what']}" for i in ids)
                + f"\n\nWrite {lo} to {hi} panels.\n\nOriginal text of this part of the chapter:\n<<<\n{src}\n>>>")
        ans = self._ask("script", SCRIPT_SYSTEM, user, SCRIPT_SCHEMA, [user])
        panels = [clean_panel(p) for p in (ans or {}).get("panels") or [] if isinstance(p, dict)]
        return [p for p in panels if p][:hi]

    def make_cover_prompt(self) -> None:
        frozen = self.cache.get_json("comic", "cover_prompt")
        if frozen:
            self.cover_prompt = frozen["image_prompt"]
            return
        scenes = [pg["summary"] for n in sorted(self.scripts) for pg in self.scripts[n]["pages"]][:10]
        user = (self._book_line() + "\n\nMain characters (describe them with exactly these looks):\n"
                + "\n".join(f"- {m.name}: {m.tag}" for m in self.cast[:3])
                + ("\n\nKey scenes:\n" + "\n".join(f"- {s}" for s in scenes) if scenes else "")
                + f"\n\nOpening of the book:\n{self.p.chapters[0].text[:2000]}")
        ans = self._ask("cover", COVER_SYSTEM, user, COVER_SCHEMA, [user])
        if ans and clean_ws(str(ans.get("image_prompt") or "")):
            self.cover_prompt = clean_ws(str(ans["image_prompt"]))
            self.cache.put_json("comic", "cover_prompt", {"image_prompt": self.cover_prompt})
        else:
            self.cover_prompt = f"dramatic scene from {self.epub.title}, {self.profile.get('world', '')}"

    # =========================================================================
    # Phase 2: pictures
    # =========================================================================
    def render(self) -> None:
        self.render_portraits()
        self.render_panels()
        if self.wants_cover and self.cover_prompt:
            self.render_cover()

    def render_portraits(self) -> None:
        if not self.cast:
            return
        size = int(self.cfg.get("portrait_size", 1024))
        neg = self.cfg.get("negative_prompt", "")
        log.info("Drawing reference portraits for %d character(s)", len(self.cast))
        prog = Progress("portraits", len(self.cast))
        for i, m in enumerate(self.cast, 1):
            try:
                if m.image_override is not None:
                    m.portrait = self._import_portrait(m.image_override, size)
                else:
                    prompt = ", ".join(x for x in (m.tag, self.cfg.get("portrait_style", ""),
                                                   self.profile.get("world", ""), self.cfg.get("style", "")) if x)
                    prompt = re.sub(r"[()\[\]{}<>]", " ", prompt)
                    seed = m.seed if m.seed is not None else int(sha("portrait", name_key(m.name))[:8], 16)
                    m.portrait = self.p.render_image(self.char_wf, prompt, size, size,
                                                     f"cast_{safe_filename(m.name, 40).replace(' ', '_')}",
                                                     negative=neg, seed=seed)
                if m.portrait:
                    m.portrait_hash = sha(Path(m.portrait).read_bytes())
            except ServiceError as e:
                log.warning("    portrait of %s failed: %s", m.name, e)
            prog.update(i)

    def _import_portrait(self, src: Path, size: int) -> Path:
        data = src.read_bytes()
        key = sha("portrait-file", data, size)
        hit = self.cache.find("images", key)
        if hit:
            return hit
        im = ImageOps.fit(Image.open(BytesIO(data)).convert("RGB"), (size, size), Image.LANCZOS, centering=(0.5, 0.3))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=92)
        return self.cache.put_bytes("images", f"{key}.jpg", buf.getvalue())

    def page_rects(self, first: bool, panels: list[dict], variant: int) -> list[tuple[int, int, int, int]]:
        top = self.margin + (self.banner_h + self.gutter if first else 0)
        bottom = self.ph - self.margin
        return layout_page(panels, (self.margin, top, self.pw - self.margin, bottom), self.gutter, variant)

    def _refs_for(self, panel: dict) -> list[tuple[CastMember, int, int]]:
        """(cast member, slot index, number of slots) for every drawn character that has a portrait."""
        lay = visible_layout(panel)
        refs = []
        for i, (c, _) in enumerate(lay):
            m = self.index.find(c["name"])
            if m is not None and m.portrait and m.portrait_hash and all(r[0] is not m for r in refs):
                refs.append((m, i, len(lay)))
        if not self.panel_wf.uses_refs:
            return []
        return refs[:self.panel_wf.ref_slots or COMIC_REF_SLOTS]

    def _ref_weight(self, refs) -> float:
        return float(self.cfg.get("ref_weight", 0.75) if len(refs) <= 1 else self.cfg.get("ref_weight_multi", 0.6))

    @staticmethod
    def _png(im: Image.Image) -> bytes:
        buf = BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def _mask(self, w: int, h: int, i: int | None, n: int) -> bytes:
        im = Image.new("L", (w, h), 0)
        if i is not None:
            if n <= 1:
                im.paste(255, (0, 0, w, h))
            else:
                x0, x1 = max(0, int((i / n - 0.06) * w)), min(w, int(((i + 1) / n + 0.06) * w))
                im.paste(255, (x0, 0, x1, h))
                im = im.filter(ImageFilter.GaussianBlur(radius=max(2, w * 0.03)))
        return self._png(im)

    def _ref_values(self, refs, w: int, h: int) -> dict:
        """Upload the reference pictures and masks for one render, and return the workflow values."""
        comfy, wf = self.p.comfy, self.panel_wf
        weight = self._ref_weight(refs)
        blank = comfy.upload(self._png(Image.new("RGB", (64, 64), (128, 128, 128))), "blank.png")
        values: dict = {}
        for s in range(1, wf.ref_slots + 1):
            if s <= len(refs):
                m, i, n = refs[s - 1]
                values[f"REF_IMAGE_{s}"] = comfy.upload(Path(m.portrait).read_bytes(), f"ref_{m.portrait_hash}.jpg")
                values[f"REF_MASK_{s}"] = comfy.upload(self._mask(w, h, i, n), f"mask_{w}x{h}_{i}of{n}.png")
                values[f"REF_WEIGHT_{s}"] = weight
            else:
                values[f"REF_IMAGE_{s}"] = blank
                values[f"REF_MASK_{s}"] = comfy.upload(self._mask(w, h, None, 0), f"mask_{w}x{h}_off.png")
                values[f"REF_WEIGHT_{s}"] = 0.0
        if wf.uses_sheet:
            if refs:
                tiles = [Image.open(m.portrait).convert("RGB").resize((512, 512), Image.LANCZOS) for m, _, _ in refs]
                sheet = Image.new("RGB", (512 * len(tiles), 512), "white")
                for j, t in enumerate(tiles):
                    sheet.paste(t, (512 * j, 0))
                values["REF_SHEET"] = comfy.upload(self._png(sheet), f"sheet_{sha([m.portrait_hash for m, _, _ in refs])}.png")
            else:
                values["REF_SHEET"] = blank
        return values

    def _ref_key(self, refs) -> list | None:
        if not self.panel_wf.uses_refs:
            return None
        return [[m.portrait_hash, i, n] for m, i, n in refs] + [self._ref_weight(refs) if refs else 0.0]

    def panel_prompt(self, panel: dict) -> str:
        lay = visible_layout(panel)
        swap: dict[str, str] = {}
        for m in self.cast:
            for n in m.names:
                if len(n) >= 2:
                    swap.setdefault(n, f"the {m.short}")
        for c in panel.get("characters") or []:
            if c["name"] not in swap and self.index.find(c["name"]) is None and c.get("look"):
                swap[c["name"]] = "the " + " ".join(c["look"].split()[:6])
        action = panel.get("action", "")
        if swap:
            pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(swap, key=len, reverse=True)) + r")\b")
            action = pat.sub(lambda mt: swap[mt.group(1)], action)
        parts = [SHOT_WORDS.get(panel.get("shot"), ""), action]
        for i, (c, _) in enumerate(lay):
            m = self.index.find(c["name"])
            look = m.tag if m else (c.get("look") or "a person")
            where = position_phrase(i, len(lay), c.get("position", "center"))
            text = f"{where}: {look}" if where else look
            parts.append(f"{text}, {c['expression']}" if c.get("expression") else text)
        for c in panel.get("characters") or []:
            if c.get("position") == "background":
                m = self.index.find(c["name"])
                parts.append("in the background " + (m.short if m else (c.get("look") or "a figure")))
        parts += [panel.get("setting", ""), self.profile.get("world", ""), self.cfg.get("style", "")]
        prompt = ", ".join(p.strip(" ,") for p in parts if p and p.strip(" ,"))
        prompt = re.sub(r"[()\[\]{}<>]", " ", prompt)
        return clean_ws(re.sub(r"\s*,\s*(,\s*)+", ", ", prompt))

    def render_panels(self) -> None:
        jobs = []
        for ch_no in sorted(self.scripts):
            for pi, page in enumerate(self.scripts[ch_no]["pages"]):
                rects = self.page_rects(pi == 0, page["panels"], ch_no + pi)
                for qi, (panel, rect) in enumerate(zip(page["panels"], rects)):
                    jobs.append((ch_no, pi, qi, panel, rect))
        live = sum(1 for j in jobs if j[0] in self.p.selected)
        if live:
            log.info("Drawing %d comic panel(s) with ComfyUI", live)
        prog = Progress("panels", live)
        done = 0
        neg = self.cfg.get("negative_prompt", "")
        mp = float(self.cfg.get("megapixels", 1.0))
        for ch_no, pi, qi, panel, rect in jobs:
            is_live = ch_no in self.p.selected
            w, h = gen_size((rect[2] - rect[0]) / max(1, rect[3] - rect[1]), mp)
            prompt = self.panel_prompt(panel)
            refs = self._refs_for(panel)
            panel["_prompt"] = prompt
            try:
                panel["_image"] = self.p.render_image(
                    self.panel_wf, prompt, w, h, f"comic_c{ch_no:03d}_p{pi + 1:02d}_{qi + 1}",
                    cached_only=not is_live, negative=neg, seed=int(sha(prompt, ch_no, pi, qi)[:8], 16),
                    key_extra=self._ref_key(refs),
                    extra_values=lambda refs=refs, w=w, h=h: self._ref_values(refs, w, h))
            except ServiceError as e:
                log.warning("    chapter %d page %d panel %d failed: %s", ch_no, pi + 1, qi + 1, e)
                panel["_image"] = None
                self.failed.add(ch_no)
            if is_live:
                done += 1
                prog.update(done)

    def render_cover(self) -> None:
        log.info("Drawing the comic cover")
        w, h = gen_size(self.pw / self.ph, float(self.cfg.get("megapixels", 1.0)))
        hero = next((m for m in self.cast if m.portrait and m.portrait_hash), None)
        refs = [(hero, 0, 1)] if hero is not None and self.panel_wf.uses_refs else []
        prompt = re.sub(r"[()\[\]{}<>]", " ", f"{self.cover_prompt}, {self.cfg.get('style', '')}")
        try:
            path = self.p.render_image(self.panel_wf, prompt, w, h, "comic_cover",
                                       negative=self.cfg.get("negative_prompt", ""), key_extra=self._ref_key(refs),
                                       extra_values=lambda: self._ref_values(refs, w, h))
        except ServiceError as e:
            log.warning("Comic cover failed: %s", e)
            return
        img = ImageOps.fit(Image.open(path).convert("RGB"), (self.pw, self.ph), Image.LANCZOS)
        let = Letterer(self.cfg, self.pw)
        self.cover_image = overlay_title(img, self.epub.title, self.epub.author, let.title_path)

    # =========================================================================
    # Phase 3: pages and the .cbz
    # =========================================================================
    def _speaker_x(self, panel: dict):
        lay = visible_layout(panel)

        def find(speaker: str):
            for c, x in lay:
                if self.index.same(c["name"], speaker):
                    return x
            return None
        return find

    def compose_chapter(self, ch_no: int, script: dict, let: Letterer) -> list[Image.Image]:
        pages = []
        for pi, page in enumerate(script["pages"]):
            img = Image.new("RGB", (self.pw, self.ph), "white")
            d = ImageDraw.Draw(img)
            if pi == 0:
                self._banner(d, let, let.text(script.get("title") or f"Chapter {ch_no}"))
            rects = self.page_rects(pi == 0, page["panels"], ch_no + pi)
            for panel, rect in zip(page["panels"], rects):
                x0, y0, x1, y1 = rect
                art_path = panel.get("_image")
                if art_path and Path(art_path).is_file():
                    art = Image.open(art_path).convert("RGB")
                else:
                    art = Image.new("RGB", (x1 - x0, y1 - y0), (205, 205, 205))
                    ImageDraw.Draw(art).text(((x1 - x0) / 2, (y1 - y0) / 2), "(panel could not be drawn)",
                                             font=let.font("text", let.size), fill=(90, 90, 90), anchor="mm")
                img.paste(ImageOps.fit(art, (x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0))
                let.letter_panel(img, rect, panel, self._speaker_x(panel),
                                 TAIL_TARGET_Y.get(panel.get("shot"), 0.45),
                                 [x for _, x in visible_layout(panel)])
                d.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 0, 0), width=self.border)
            pages.append(img)
        return pages

    def _banner(self, d: ImageDraw.ImageDraw, let: Letterer, text: str) -> None:
        size = round(self.banner_h * 0.62)
        font = let.font("title", size)
        while font.getlength(text) > (self.pw - 2 * self.margin) * 0.92 and size > 12:
            size = int(size * 0.9)
            font = let.font("title", size)
        cy = self.margin + self.banner_h / 2
        d.text((self.pw / 2, cy), text, font=font, fill=(0, 0, 0), anchor="mm")
        lw = font.getlength(text)
        for x0, x1 in ((self.margin, self.pw / 2 - lw / 2 - self.gutter),
                       (self.pw / 2 + lw / 2 + self.gutter, self.pw - self.margin)):
            if x1 - x0 > 20:
                d.line((x0, cy, x1, cy), fill=(0, 0, 0), width=max(2, self.border))

    def _page_number(self, img: Image.Image, let: Letterer, n: int) -> None:
        d = ImageDraw.Draw(img)
        d.text((self.pw / 2, self.ph - self.margin / 2), str(n), font=let.font("text", round(let.size * 0.8)),
               fill=(60, 60, 60), anchor="mm")

    def cast_page(self, let: Letterer) -> Image.Image | None:
        members = [m for m in self.cast if m.portrait and Path(m.portrait).is_file()][:12]
        if not members:
            return None
        img = Image.new("RGB", (self.pw, self.ph), "white")
        d = ImageDraw.Draw(img)
        self._banner(d, let, let.text("The Characters"))
        n = len(members)
        cols = 2 if n <= 4 else 3
        rows = math.ceil(n / cols)
        top = self.margin + self.banner_h + self.gutter * 2
        cw = (self.pw - 2 * self.margin - self.gutter * (cols - 1)) / cols
        chh = (self.ph - self.margin - top - self.gutter * (rows - 1)) / rows
        name_fs = round(let.size * 1.15)
        desc_fs = round(let.size * 0.8)
        text_h = name_fs * 1.5 + desc_fs * 1.2 * 2 + self.gutter
        side = int(max(80, min(cw, chh - text_h)))
        chh = min(chh, side + text_h)  # few characters: centre the grid on the page
        top += max(0, (self.ph - self.margin - top - rows * chh - (rows - 1) * self.gutter) / 2)
        for i, m in enumerate(members):
            r, c = divmod(i, cols)
            x = self.margin + c * (cw + self.gutter) + (cw - side) / 2
            y = top + r * (chh + self.gutter)
            pic = ImageOps.fit(Image.open(m.portrait).convert("RGB"), (side, side), Image.LANCZOS)
            img.paste(pic, (int(x), int(y)))
            d.rectangle((int(x), int(y), int(x) + side - 1, int(y) + side - 1), outline=(0, 0, 0), width=self.border)
            cx = self.margin + c * (cw + self.gutter) + cw / 2
            ty = y + side + name_fs * 0.85
            name_font = let.font("title", name_fs)
            label = let.text(m.name)
            while name_font.getlength(label) > cw and name_fs > 10:
                name_fs = int(name_fs * 0.9)
                name_font = let.font("title", name_fs)
            d.text((cx, ty), label, font=name_font, fill=(0, 0, 0), anchor="mm")
            dfont = let.font("text", desc_fs)
            for j, ln in enumerate(Letterer.wrap(m.short, dfont, cw * 0.95)[:2]):
                d.text((cx, ty + name_fs * 0.75 + desc_fs * 1.2 * (j + 0.5)), ln, font=dfont, fill=(50, 50, 50),
                       anchor="mm")
        return img

    def _jpeg(self, img: Image.Image) -> dict:
        buf = BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=int(self.cfg.get("jpeg_quality", 90)), optimize=True)
        return {"data": buf.getvalue(), "width": img.width, "height": img.height}

    def package(self) -> None:
        let = Letterer(self.cfg, self.pw)
        if not let.font_path or "dejavu" in let.font_path.lower() or "liberation" in let.font_path.lower():
            log.info("Tip: for real comic lettering put a comic font (e.g. Comic Neue Bold, Anime Ace) in the "
                     "fonts/ folder or set comic.font")
        comic_dir = self.p.out_dir / "comic"
        chapters: dict[int, list[Image.Image]] = {}
        for ch in self.p.chapters:
            script = self.scripts.get(ch.number)
            if not script:
                continue
            ready = all(p.get("_image") for pg in script["pages"] for p in pg["panels"])
            if not ready and ch.number not in self.p.selected:
                log.debug("Chapter %d: not all panels are drawn yet; left out of the comic", ch.number)
                continue
            chapters[ch.number] = self.compose_chapter(ch.number, script, let)
        self.write_index(comic_dir)
        if not chapters:
            log.warning("No comic pages to write yet")
            return
        titles = {c.number: c.title for c in self.p.chapters}
        numbers = bool(self.cfg.get("page_numbers", True))
        info = dict(writer=self.epub.author, genre=self.profile.get("genre", ""), language=self.epub.language,
                    characters=[m.name for m in self.cast])
        if self.cfg.get("split", "book") == "chapter":
            for n, imgs in chapters.items():
                pages = []
                for k, im in enumerate(imgs, 1):
                    if numbers:
                        self._page_number(im, let, k)
                    pages.append(dict(self._jpeg(im), type="FrontCover" if k == 1 else "Story",
                                      bookmark=titles[n] if k == 1 else ""))
                out = comic_dir / f"{n:02d} - {safe_filename(titles[n], 70)}.cbz"
                write_cbz(out, pages, comic_info_xml(titles[n], self.epub.title, n,
                                                     " ".join(pg["summary"] for pg in self.scripts[n]["pages"]),
                                                     pages=pages, **info))
            log.info("Wrote %d comic file(s) to %s", len(chapters), comic_dir)
        else:
            pages = []
            if self.cover_image is not None:
                pages.append(dict(self._jpeg(self.cover_image), type="FrontCover"))
            cast_img = self.cast_page(let) if self.cfg.get("cast_page", True) else None
            if cast_img is not None:
                pages.append(dict(self._jpeg(cast_img), type="Other", bookmark="The Characters"))
            k = 0
            for n, imgs in chapters.items():
                for j, im in enumerate(imgs):
                    k += 1
                    if numbers:
                        self._page_number(im, let, k)
                    pages.append(dict(self._jpeg(im), type="Story", bookmark=titles[n] if j == 0 else ""))
            summary = " ".join(self.scripts[n]["pages"][0]["summary"] for n in list(chapters)[:3])
            out = self.p.out_dir / f"{self.p.book_name}.cbz"
            write_cbz(out, pages, comic_info_xml(self.epub.title, self.epub.title, None, summary, pages=pages, **info))
            log.info("Wrote %s (%d pages, %d chapter(s))", out.name, len(pages), len(chapters))

    def write_index(self, comic_dir: Path) -> None:
        cast_dir = comic_dir / "cast"
        cast_out = []
        for m in self.cast:
            entry = {"name": m.name, "aliases": m.aliases, "passages": m.presence, "stated_in_book": m.stated,
                     "design": {k: m.design.get(k, "") for k in DESIGN_FIELDS + ["short"]},
                     "invented": m.design.get("invented", []), "prompt": m.tag, "portrait": None}
            if m.design.get("look"):
                entry["design"]["look"] = m.design["look"]
            if m.portrait and Path(m.portrait).is_file():
                cast_dir.mkdir(parents=True, exist_ok=True)
                name = f"{safe_filename(m.name, 60)}.jpg"
                shutil.copyfile(m.portrait, cast_dir / name)
                entry["portrait"] = f"cast/{name}"
            cast_out.append(entry)
        chapters = []
        for n in sorted(self.scripts):
            pages = []
            for pg in self.scripts[n]["pages"]:
                panels = []
                for p in pg["panels"]:
                    q = {k: v for k, v in p.items() if not k.startswith("_")}
                    if p.get("_prompt"):
                        q["image_prompt"] = p["_prompt"]
                    panels.append(q)
                pages.append({"summary": pg["summary"], "events": pg.get("events", []), "panels": panels})
            chapters.append({"number": n, "title": self.scripts[n].get("title"), "pages": pages})
        data = {"title": self.epub.title, "author": self.epub.author, "art_direction": self.profile,
                "created": datetime.now().isoformat(timespec="seconds"), "cast": cast_out, "chapters": chapters}
        comic_dir.mkdir(parents=True, exist_ok=True)
        (comic_dir / "comic.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def print_script(self) -> None:
        print(f"\nArt direction: {self.profile.get('genre', '')} | {self.profile.get('world', '')} | "
              f"{self.profile.get('mood', '')}")
        print(f"\nCast ({len(self.cast)}):")
        for m in self.cast:
            also = f" (also: {', '.join(m.aliases[:4])})" if m.aliases else ""
            print(f"  {m.name}{also} - in {m.presence} passage(s)")
            print(f"      look: {m.tag}")
            if m.design.get("invented"):
                print(f"      invented (not in the book): {', '.join(m.design['invented'])}")
        for ch in self.p.chapters:
            if ch.number not in self.p.selected:
                continue
            print(f"\n{ch.number}. {ch.title}")
            script = self.scripts.get(ch.number)
            if not script:
                print("   (no script)")
                continue
            for pi, page in enumerate(script["pages"], 1):
                print(f"   Page {pi}: {page['summary']}")
                for qi, p in enumerate(page["panels"], 1):
                    who = ", ".join(f"{c['name']} ({c['position']})" for c in p["characters"])
                    print(f"      {qi}. [{p['shot']}] {p['action']}" + (f"  <{who}>" if who else ""))
                    if p["caption"]:
                        print(f"           caption: {p['caption']}")
                    for line in p["dialogue"]:
                        print(f"           {line['speaker']} ({line['style']}): {line['text']}")
                    if p["sfx"]:
                        print(f"           sfx: {p['sfx']}")
        print()


# =============================================================================
# Pipeline
# =============================================================================

NODE_PACKS = {
    "IPAdapter": "ComfyUI_IPAdapter_plus (in ComfyUI Manager search for \"IPAdapter plus\") plus its models",
    "Kokoro": "comfyui-kokoro (in ComfyUI Manager search for \"Kokoro\")",
}


class Pipeline:
    def __init__(self, args: argparse.Namespace, cfg: dict, cfg_dir: Path):
        self.args, self.cfg, self.cfg_dir = args, cfg, cfg_dir
        src = Path(args.epub).expanduser()
        if not src.is_file():
            raise ConfigError(f"EPUB not found: {src}")
        self.epub = Epub(src)
        self.chapters = build_chapters(self.epub, cfg["chapters"])
        if not self.chapters:
            raise ConfigError("No chapters with text found in this EPUB (try --min-chars 0)")
        self.selected = parse_selection(args.chapters, len(self.chapters))
        self.book_name = safe_filename(self.epub.title)
        out_root = Path(args.output or cfg.get("output_dir", "output")).expanduser()
        self.out_dir = out_root / self.book_name
        file_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:10]
        work = Path(args.work_dir).expanduser() if args.work_dir else out_root / ".cache" / f"{self.book_name}-{file_hash}"
        self.cache = Cache(work)
        self.prefix = f"epub_narrator/{safe_filename(self.book_name, 40).replace(' ', '_')}"

        self.do_images = cfg["images"]["enabled"]
        self.do_audio = cfg["narration"]["enabled"]
        has_cover = self.epub.cover_item() is not None
        self.do_cover = cfg["cover"]["enabled"] and not (cfg["cover"]["keep_original"] and has_cover)
        self.do_comic = bool(cfg["comic"]["enabled"])
        self.llm = Ollama(cfg["ollama"])
        self.comfy = ComfyUI(cfg["comfyui"])
        self.image_wf = self.cover_wf = self.tts_wf = None
        self.new_cover: Image.Image | None = None
        self.comic = ComicMaker(self) if self.do_comic or getattr(args, "script_only", False) else None
        if self.do_comic and cfg["comic"].get("split", "book") not in ("book", "chapter"):
            raise ConfigError("comic.split must be \"book\" or \"chapter\"")

    # -- setup ----------------------------------------------------------------
    def _load_workflows(self) -> None:
        image_keys = ["PROMPT", "NEGATIVE_PROMPT", "SEED", "WIDTH", "HEIGHT"]
        if self.do_images or self.do_cover:
            self.image_wf = Workflow(resolve_file(self.cfg["images"]["workflow"], self.cfg_dir),
                                     ["PROMPT"], image_keys, "Image")
        if self.do_cover:
            cw = self.cfg["cover"].get("workflow")
            self.cover_wf = (Workflow(resolve_file(cw, self.cfg_dir), ["PROMPT"], image_keys, "Cover")
                             if cw else self.image_wf)
        if self.do_audio:
            self.tts_wf = Workflow(resolve_file(self.cfg["narration"]["workflow"], self.cfg_dir),
                                   ["TEXT"], ["TEXT", "SEED"], "TTS")
            if self.cfg["narration"].get("format", "mp3").lower() not in AUDIO_FORMATS:
                raise ConfigError(f"narration.format must be one of {', '.join(AUDIO_FORMATS)}")
        if self.do_comic:
            self.comic.load_workflows(image_keys)

    def _preflight(self) -> None:
        needs_llm = (self.do_images or self.do_cover or self.do_comic
                     or (self.do_audio and self.cfg["narration"]["llm_cleanup"]))
        if needs_llm:
            self.llm.check()
        if self.do_images or self.do_cover or self.do_audio or self.do_comic:
            self.comfy.check()
            workflows = [self.image_wf, self.cover_wf, self.tts_wf] + (self.comic.workflows() if self.do_comic else [])
            for wf in {id(w): w for w in workflows if w is not None}.values():
                missing = self.comfy.missing_nodes(wf)
                if missing:
                    packs = sorted({pack for node in missing for prefix, pack in NODE_PACKS.items()
                                    if node.startswith(prefix)})
                    hint = f" Install {' and '.join(packs)}, then restart ComfyUI." if packs else \
                        " Install the custom nodes that provide them (ComfyUI Manager > Install Missing Custom Nodes)."
                    if self.comic is not None and wf is self.comic.panel_wf:
                        hint += (" Or draw panels from text alone with "
                                 "comic.panel_workflow = \"workflows/image_sdxl.json\".")
                    raise ConfigError(f"{wf.label} workflow {wf.path.name} uses node(s) this ComfyUI does not "
                                      f"have: {', '.join(missing)}.{hint}")
        if self.do_audio and not shutil.which("ffmpeg"):
            raise ServiceError("ffmpeg was not found on PATH (needed to build the audio tracks)")

    # -- main -----------------------------------------------------------------
    def run(self) -> int:
        args = self.args
        log.info('"%s" by %s - %d chapters', self.epub.title, self.epub.author, len(self.chapters))
        if args.list_chapters:
            self.print_chapters()
            return 0
        if args.scenes_only:
            self.llm.check()
            self.print_scenes(self.phase_scenes())
            return 0
        if getattr(args, "script_only", False):
            self.llm.check()
            self.comic.plan()
            self.comic.print_script()
            self.comic.write_index(self.out_dir / "comic")
            log.info("Script saved to %s", (self.out_dir / "comic" / "comic.json").resolve())
            return 0
        self._load_workflows()
        self._preflight()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        make_book = self.do_images or self.do_cover or self.do_audio

        # Phase 1 - everything that needs the LLM, so it can be unloaded before ComfyUI runs.
        scenes = self.phase_scenes() if self.do_images else {}
        narration = self.phase_narration_text() if self.do_audio else {}
        cover_prompt = self.phase_cover_prompt(scenes) if self.do_cover else None
        if self.do_comic:
            self.comic.plan()
        if self.llm.used and self.cfg["ollama"].get("unload_when_done", True):
            self.llm.unload()

        # Phase 2 - images.
        if self.do_images:
            self.phase_images(scenes)
        cover_path = self.phase_cover(cover_prompt) if make_book else None
        if self.do_comic:
            self.comic.render()
        if self.cfg["comfyui"].get("free_memory_between_phases", True) and self.do_audio:
            self.comfy.free()

        # Phase 3 - narration.
        failed: list[int] = []
        if self.do_audio:
            failed = self.phase_audio(narration, cover_path)

        # Phase 4 - package.
        if make_book:
            self.package(scenes, cover_path)
        if self.do_comic:
            self.comic.package()
        log.info("Done: %s", self.out_dir.resolve())
        status = 0
        if failed:
            log.warning("Narration failed for chapter(s) %s - run the same command again to retry them.",
                        ", ".join(map(str, failed)))
            status = 1
        if self.do_comic and self.comic.failed:
            log.warning("Some comic panels failed in chapter(s) %s - run the same command again to retry them.",
                        ", ".join(map(str, sorted(self.comic.failed))))
            status = 1
        return status

    def print_chapters(self) -> None:
        print(f"\n{'#':>4}  {'words':>7}  {'~min':>5}  title")
        for ch in self.chapters:
            mark = "*" if ch.number in self.selected else " "
            print(f"{ch.number:>4}{mark} {ch.words:>7}  {ch.words / 155:>5.0f}  {ch.title}")
        total = sum(c.words for c in self.chapters)
        print(f"\n{total} words, roughly {fmt_duration(total / 155 * 60)} of narration\n")

    # -- phase 1 --------------------------------------------------------------
    def phase_scenes(self) -> dict[int, list[Scene]]:
        icfg = self.cfg["images"]
        notes = CharacterNotes()
        result: dict[int, list[Scene]] = {}
        todo = [c for c in self.chapters if c.number in self.selected]
        log.info("Looking for strongly described scenes in %d chapter(s) with %s", len(todo), self.llm.model)
        for ch in self.chapters:
            live = ch.number in self.selected
            if live:
                log.info("  [%d/%d] %s", ch.number, len(self.chapters), ch.title)
            candidates = analyze_chapter(ch, self.epub, self.llm if live else None, self.cache, icfg, notes)
            if candidates is None:
                continue
            chosen = pick_scenes(candidates, icfg)
            result[ch.number] = chosen
            if live:
                best = max((s.effective for s in candidates), default=0)
                log.info("    %d passages scored (best %d/10) -> %d illustration(s)", len(candidates), best, len(chosen))
        cap = int(icfg.get("max_total") or 0)
        if cap > 0:
            everything = sorted((s for v in result.values() for s in v), key=lambda s: (-s.effective, s.chapter))
            keep = {id(s) for s in everything[:cap]}
            result = {k: [s for s in v if id(s) in keep] for k, v in result.items()}
        return result

    def print_scenes(self, scenes: dict[int, list[Scene]]) -> None:
        for ch in self.chapters:
            if ch.number not in self.selected:
                continue
            print(f"\n{ch.number}. {ch.title}")
            for s in scenes.get(ch.number, []):
                print(f"   score {s.score}/10, after paragraph {s.block_index + 1}: {s.caption}")
                print(f"      prompt: {s.prompt}")
            if not scenes.get(ch.number):
                print("   (no passage passed the threshold)")
        print()

    def phase_narration_text(self) -> dict[int, list[Chunk]]:
        ncfg = self.cfg["narration"]
        out: dict[int, list[Chunk]] = {}
        todo = [c for c in self.chapters if c.number in self.selected]
        if ncfg.get("llm_cleanup"):
            log.info("Preparing narration text for %d chapter(s) with %s", len(todo), self.llm.model)
        for ch in todo:
            paras = chapter_paragraphs(ch, ncfg)
            if ncfg.get("llm_cleanup"):
                log.info("  [%d/%d] %s", ch.number, len(self.chapters), ch.title)
                paras = llm_cleanup(paras, self.llm, self.cache, ncfg)
            out[ch.number] = make_chunks(paras, ncfg)
        return out

    def phase_cover_prompt(self, scenes: dict[int, list[Scene]]) -> str | None:
        key = sha("cover", PROMPT_VERSION, self.llm.model, self.epub.title, self.epub.author)
        cached = self.cache.get_json("cover", key)
        if cached:
            return cached["image_prompt"]
        notes = [s.caption for v in scenes.values() for s in sorted(v, key=lambda s: -s.score) if s.caption][:12]
        opening = self.chapters[0].text[:2500]
        user = f'Title: "{self.epub.title}"\nAuthor: {self.epub.author}\n'
        if notes:
            user += "\nKey scenes:\n" + "\n".join(f"- {n}" for n in notes) + "\n"
        user += f"\nOpening of the book:\n{opening}"
        log.info("Writing a cover prompt")
        try:
            ans = self.llm.chat_json(COVER_SYSTEM, user, COVER_SCHEMA)
        except LLMError as e:
            log.warning("Could not get a cover prompt (%s); using a generic one", e)
            return f"atmospheric scene evoking the book {self.epub.title}"
        prompt = clean_ws(str(ans.get("image_prompt") or "")) or f"atmospheric scene evoking {self.epub.title}"
        self.cache.put_json("cover", key, {"image_prompt": prompt})
        return prompt

    # -- phase 2 --------------------------------------------------------------
    def render_image(self, wf: Workflow, prompt: str, width: int, height: int, name: str,
                     cached_only: bool = False, negative: str | None = None, seed: int | None = None,
                     key_extra=None, extra_values=None) -> Path | None:
        """Render (or fetch from the cache) one image. extra_values() supplies more workflow values
        (comic reference pictures); it is only called when the image really has to be rendered."""
        icfg = self.cfg["images"]
        if negative is None:
            negative = icfg.get("negative_prompt", "")
        if seed is None:
            seed = int(sha(prompt)[:8], 16)
        key_parts = ["img", wf.hash, prompt, negative, width, height, seed]
        if key_extra is not None:
            key_parts.append(key_extra)
        key = sha(*key_parts)
        hit = self.cache.find("images", key)
        if hit or cached_only:
            return hit
        values = {"PROMPT": prompt, "NEGATIVE_PROMPT": negative, "SEED": seed, "WIDTH": int(width),
                  "HEIGHT": int(height)}
        if extra_values is not None:
            values.update(extra_values())
        files = self.comfy.run(wf.build(values, f"{self.prefix}/{name}"))
        images = [f for f in files if Path(f["filename"]).suffix.lower() in IMAGE_EXTS]
        if not images:
            raise ServiceError(f"{wf.path.name} produced no image - it needs a Save Image node")
        im = Image.open(BytesIO(self.comfy.download(images[0]))).convert("RGB")
        max_w = int(icfg.get("max_width", 1600))
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, "JPEG", quality=int(icfg.get("jpeg_quality", 88)), optimize=True, progressive=True)
        return self.cache.put_bytes("images", f"{key}.jpg", buf.getvalue())

    def phase_images(self, scenes: dict[int, list[Scene]]) -> None:
        icfg = self.cfg["images"]
        style = icfg.get("style", "")
        todo = [s for n, v in scenes.items() for s in v if n in self.selected]
        if todo:
            log.info("Rendering %d illustration(s) with ComfyUI", len(todo))
        prog = Progress("illustrations", len(todo))
        done = 0
        for n, items in scenes.items():
            for k, s in enumerate(items, 1):
                prompt = f"{s.prompt}, {style}" if style else s.prompt
                live = n in self.selected
                try:
                    s.image = self.render_image(self.image_wf, prompt, icfg["width"], icfg["height"],
                                                f"ch{n:03d}_{k:02d}", cached_only=not live)
                except ServiceError as e:
                    log.warning("    illustration for chapter %d failed: %s", n, e)
                if live:
                    done += 1
                    prog.update(done)

    def phase_cover(self, prompt: str | None) -> Path | None:
        ccfg = self.cfg["cover"]
        out = self.out_dir / "cover.jpg"
        if prompt is not None:
            log.info("Rendering the cover")
            style = ccfg.get("style", "")
            try:
                raw = self.render_image(self.cover_wf, f"{prompt}, {style}" if style else prompt,
                                        ccfg["width"], ccfg["height"], "cover")
                img = Image.open(raw)
                if ccfg.get("title_overlay", True):
                    img = overlay_title(img, self.epub.title, self.epub.author, ccfg.get("font", ""))
                img = img.convert("RGB")
                img.save(out, "JPEG", quality=92)
                self.new_cover = img
                return out
            except ServiceError as e:
                log.warning("Cover generation failed (%s); keeping the original cover", e)
        existing = self.epub.cover_item()
        if existing is None:
            return None
        try:  # reuse the book's own cover for cover.jpg and the audio tags
            Image.open(BytesIO(self.epub.read(existing.href))).convert("RGB").save(out, "JPEG", quality=92)
            return out
        except Exception:  # noqa: BLE001
            return None

    # -- phase 3 --------------------------------------------------------------
    def tts(self, text: str) -> Path:
        ncfg = self.cfg["narration"]
        key = sha("tts", self.tts_wf.hash, text, ncfg.get("seed", 42))
        hit = self.cache.find("tts", key)
        if hit:
            return hit
        values = {"TEXT": text, "SEED": int(ncfg.get("seed", 42))}
        last: Exception | None = None
        for attempt in range(int(ncfg.get("retries", 2)) + 1):
            try:
                files = self.comfy.run(self.tts_wf.build(values, f"{self.prefix}/tts/{key}"))
                audio = [f for f in files if Path(f["filename"]).suffix.lower() in AUDIO_EXTS or f["kind"] == "audio"]
                if not audio:
                    raise ServiceError(f"{self.tts_wf.path.name} produced no audio - it needs a Save Audio node")
                data = self.comfy.download(audio[0])
                ext = Path(audio[0]["filename"]).suffix.lower() or ".flac"
                return self.cache.put_bytes("tts", f"{key}{ext}", data)
            except ServiceError as e:
                last = e
                log.debug("TTS attempt %d failed: %s", attempt + 1, e)
                time.sleep(2)
        raise ServiceError(f"TTS failed for: {text[:80]!r}... ({last})")

    def track_path(self, ch: Chapter) -> Path:
        ext = AUDIO_FORMATS[self.cfg["narration"].get("format", "mp3").lower()][1]
        width = max(2, len(str(len(self.chapters))))
        return self.out_dir / "audio" / f"{ch.number:0{width}d} - {safe_filename(ch.title, 70)}{ext}"

    def phase_audio(self, narration: dict[int, list[Chunk]], cover: Path | None) -> list[int]:
        ncfg = self.cfg["narration"]
        failed = []
        total_chunks = sum(len(v) for v in narration.values())
        log.info("Narrating %d chapter(s) (%d speech chunks) with ComfyUI", len(narration), total_chunks)
        for ch in self.chapters:
            chunks = narration.get(ch.number)
            if not chunks:
                continue
            log.info("  [%d/%d] %s", ch.number, len(self.chapters), ch.title)
            try:
                prog = Progress("speech", len(chunks))
                parts = []
                for i, c in enumerate(chunks, 1):
                    parts.append((self.tts(c.text), c.pause))
                    prog.update(i)
                meta = {
                    "title": ch.title, "album": self.epub.title, "artist": self.epub.author,
                    "album_artist": self.epub.author, "track": f"{ch.number}/{len(self.chapters)}",
                    "genre": "Audiobook",
                }
                out = self.track_path(ch)
                cover_hash = sha(cover.read_bytes()) if cover and cover.is_file() else None
                track_key = sha([p.name for p, _ in parts], [round(s, 3) for _, s in parts], ncfg, meta, cover_hash)
                record = {"key": track_key, "file": out.name}
                if out.is_file() and self.cache.get_json("tracks", f"ch{ch.number:04d}") == record:
                    log.info("    -> %s (unchanged)", out.name)
                    continue
                seconds = assemble_track(parts, out, meta, ncfg, cover)
                self.cache.put_json("tracks", f"ch{ch.number:04d}", record)
                log.info("    -> %s (%s)", out.name, fmt_duration(seconds))
            except ServiceError as e:
                log.error("    chapter %d failed: %s", ch.number, e)
                failed.append(ch.number)
        return failed

    # -- phase 4 --------------------------------------------------------------
    def package(self, scenes: dict[int, list[Scene]], cover_path: Path | None) -> None:
        writer = EpubWriter(self.epub)
        placed: dict[int, list[dict]] = {}
        captions = self.cfg["images"].get("captions", True)
        for ch in self.chapters:
            for k, s in enumerate(scenes.get(ch.number, []), 1):
                if not s.image or not Path(s.image).is_file():
                    continue
                block = self._anchor_block(ch, s.block_index)
                if block is None:
                    continue
                name = f"ch{ch.number:03d}-{k:02d}"
                ok = writer.add_illustration(block, Path(s.image).read_bytes(), name,
                                             s.caption if captions else "", s.caption or "Illustration")
                if ok:
                    placed.setdefault(ch.number, []).append(
                        {"caption": s.caption, "score": s.score, "after_paragraph": s.block_index + 1,
                         "prompt": s.prompt})
        if self.new_cover is not None:
            writer.set_cover(self.new_cover)
        epub_out = self.out_dir / f"{self.book_name}.epub"
        writer.write(epub_out)
        n_img = sum(len(v) for v in placed.values())
        log.info("Wrote %s (%d illustration(s)%s)", epub_out.name, n_img,
                 ", new cover" if self.new_cover is not None else "")

        # Playlist and index of everything in the folder.
        entries = []
        playlist = ["#EXTM3U", f"#PLAYLIST:{self.epub.title}"]
        for ch in self.chapters:
            track = self.track_path(ch)
            duration = probe_duration(track) if track.is_file() else 0.0
            if track.is_file():
                playlist += [f"#EXTINF:{int(duration) or -1},{self.epub.author} - {ch.title}",
                             f"{track.name}"]
            entries.append({
                "number": ch.number, "title": ch.title, "words": ch.words,
                "audio": f"audio/{track.name}" if track.is_file() else None,
                "duration_seconds": round(duration, 1) if track.is_file() else None,
                "illustrations": placed.get(ch.number, []),
            })
        if len(playlist) > 2:
            (self.out_dir / "audio" / f"{self.book_name}.m3u").write_text("\n".join(playlist) + "\n",
                                                                       encoding="utf-8")
        index = {
            "title": self.epub.title, "author": self.epub.author, "language": self.epub.language,
            "epub": epub_out.name, "cover": "cover.jpg" if (self.out_dir / "cover.jpg").is_file() else None,
            "created": datetime.now().isoformat(timespec="seconds"),
            "chapters": entries,
        }
        (self.out_dir / "chapters.json").write_text(json.dumps(index, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")

    @staticmethod
    def _anchor_block(ch: Chapter, idx: int) -> Block | None:
        order = [idx] + [i for d in range(1, len(ch.blocks)) for i in (idx - d, idx + d)]
        for i in order:
            if 0 <= i < len(ch.blocks) and ch.blocks[i].elem is not None and ch.blocks[i].kind == "text":
                return ch.blocks[i]
        return None


# =============================================================================
# Command line
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Illustrate and narrate an EPUB with a local Ollama LLM and ComfyUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  %(prog)s book.epub --list-chapters\n"
               "  %(prog)s book.epub --scenes-only --chapters 1-3\n"
               "  %(prog)s book.epub\n"
               "  %(prog)s book.epub --no-audio --threshold 8\n"
               "  %(prog)s book.epub --script-only --chapters 1\n"
               "  %(prog)s book.epub --comic --chapters 1-2\n")
    p.add_argument("epub", help="the EPUB file to process")
    p.add_argument("-c", "--config", help="TOML config file (default: config.toml next to this script)")
    p.add_argument("-o", "--output", help="output folder (default from config: ./output)")
    p.add_argument("--work-dir", help="cache folder (default: <output>/.cache/<book>)")
    p.add_argument("--chapters", help="only process these chapters, e.g. 1-3,7 (numbers from --list-chapters)")
    p.add_argument("--list-chapters", action="store_true", help="show how the book was split and exit")
    p.add_argument("--scenes-only", action="store_true",
                   help="run the scene analysis and print which images would be made, then exit")
    p.add_argument("--no-images", action="store_true", help="skip illustrations")
    p.add_argument("--no-cover", action="store_true", help="keep the original cover")
    p.add_argument("--no-audio", action="store_true", help="skip narration")
    p.add_argument("--no-cleanup", action="store_true", help="send the text to TTS without the LLM cleanup pass")
    p.add_argument("--model", help="Ollama model (overrides config)")
    p.add_argument("--threshold", type=int, help="minimum scene score 0-10 for an illustration")
    p.add_argument("--max-per-chapter", type=int, help="maximum illustrations per chapter")
    p.add_argument("--min-chars", type=int, help="skip sections shorter than this many characters")
    p.add_argument("--format", choices=sorted(AUDIO_FORMATS), help="audio format for the chapter tracks")
    p.add_argument("--comic", action="store_true",
                   help="make the comic edition (.cbz) instead of the illustrated EPUB and audiobook")
    p.add_argument("--script-only", action="store_true",
                   help="plan the comic with the LLM (cast and page scripts), print it and exit")
    p.add_argument("--pages", type=int, help="comic pages per chapter (default: estimated from its length)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def apply_overrides(cfg: dict, a: argparse.Namespace) -> None:
    if a.no_images:
        cfg["images"]["enabled"] = False
    if a.no_cover:
        cfg["cover"]["enabled"] = False
    if a.no_audio:
        cfg["narration"]["enabled"] = False
    if a.no_cleanup:
        cfg["narration"]["llm_cleanup"] = False
    if a.model:
        cfg["ollama"]["model"] = a.model
    if a.threshold is not None:
        cfg["images"]["threshold"] = a.threshold
    if a.max_per_chapter is not None:
        cfg["images"]["max_per_chapter"] = a.max_per_chapter
    if a.min_chars is not None:
        cfg["chapters"]["min_chars"] = a.min_chars
    if a.format:
        cfg["narration"]["format"] = a.format
    if a.scenes_only:
        cfg["images"]["enabled"] = True
    if a.comic or a.script_only:
        cfg["comic"]["enabled"] = True
    if a.comic:  # comic only; set comic.enabled in config.toml to make it alongside everything else
        cfg["images"]["enabled"] = cfg["cover"]["enabled"] = cfg["narration"]["enabled"] = False
    if a.pages is not None:
        cfg["comic"]["min_pages"] = cfg["comic"]["max_pages"] = max(1, a.pages)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except AttributeError:
            pass
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    try:
        cfg, cfg_dir = load_config(args.config)
        apply_overrides(cfg, args)
        return Pipeline(args, cfg, cfg_dir).run()
    except KeyboardInterrupt:
        log.warning("Interrupted. Run the same command again to resume - finished work is cached.")
        return 130
    except (ConfigError, ServiceError) as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
