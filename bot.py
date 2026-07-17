#!/usr/bin/env python3
"""
AI Code Bot — Telegram bot that repairs and upgrades source files and whole projects.

AI layer
--------
Multi-provider, OpenAI-compatible (`POST /chat/completions`) with automatic failover:

    DeepSeek  (primary,  DEEPSEEK_API_KEY)
        └── on timeout / rate-limit / API error / outage ──▶ Groq (fallback, GROQ_API_KEY)

`AIProviderManager` owns the routing: it health-checks every provider, keeps a circuit
breaker + moving-average latency per provider, switches instantly when one degrades and
brings it back automatically once it recovers. The bot never stops because one API is down.
At least one API key must be present; both are supported, either can be missing.

Flows
-----
1) "تصحيح أخطاء"   : file/archive -> problem description (+ error message) -> corrected file.
2) "تحديث ملف"     : file/archive -> update prompt                          -> updated file.
3) "سجل العمليات"  : the user's latest operations (type, duration, provider, result).
4) "/ai_status"    : admin-only dashboard of the AI providers.

Capabilities
------------
* Single source files (30+ languages) and whole projects uploaded as `.zip`
  (Python / PHP / Node.js / React / Vue / Flutter ... ), where the relationships
  between files (imports/requires/uses) are fed to the model as read-only context.
* Large files are split at top-level boundaries, repaired part by part
  ("⏳ جاري تحليل الجزء 2/5") and merged back into one file.
* Every answer is validated before delivery: no markdown, no placeholders, no lost
  functions/classes, syntax check when the language allows it, plus one automatic
  repair round when validation fails.
* FIFO queue with workers: queue position, estimated wait and automatic start.
* Live progress inside a single edited message + "إلغاء العملية الحالية" button that
  aborts the AI request and frees the job memory immediately.

Engineering notes
-----------------
* Fully async (aiogram 3 + aiohttp); one pooled HTTP session shared by every provider.
* CPU-bound work (zip, ast, merge) runs in worker threads so the event loop never blocks.
* Hardened networking: per-provider timeouts, exponential backoff + jitter, `Retry-After`,
  retry only on transient statuses, automatic payload downgrade when an endpoint rejects an
  optional field, then failover.
* Security: extension allow-list, size/entry/zip-bomb caps, path-traversal-safe names,
  binary sniffing, per-user flood control, optional user allow-list, admin allow-list,
  secrets from the environment only (never in code, never logged).
* Persistence: only metadata logs are written to `logs/` (user, date, provider, latency,
  status). User code is never written to disk — it lives in memory and is released on
  completion, failure or cancellation.

.env template
-------------
    TELEGRAM_BOT_TOKEN=123456:ABC...        # required (BOT_TOKEN still accepted)

    DEEPSEEK_API_KEY=                       # primary provider   (optional)
    DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions
    DEEPSEEK_MODEL=deepseek-chat
    DEEPSEEK_MAX_TOKENS=8192
    DEEPSEEK_TEMPERATURE=0.2
    DEEPSEEK_THINKING=false                 # true only for reasoning models
    DEEPSEEK_REASONING_EFFORT=high          # high | max
    DEEPSEEK_ENABLED=true

    GROQ_API_KEY=                           # fallback provider  (optional)
    GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
    GROQ_MODEL=llama-3.3-70b-versatile
    GROQ_MAX_TOKENS=32768
    GROQ_TEMPERATURE=0.2
    GROQ_ENABLED=true

    AI_ROUTING=priority                     # priority (DeepSeek first) | fastest
    AI_HEALTH_INTERVAL=300                  # seconds between health checks (0 = off)
    AI_PROBE_TIMEOUT=15
    AI_TEMPERATURE=0.2                      # default for every provider

    REQUEST_TIMEOUT=600
    MAX_RETRIES=3
    RETRY_DELAY=2.0
    VALIDATION_RETRIES=1
    MAX_FILE_SIZE=200000                    # bytes, per source file
    MAX_ARCHIVE_SIZE=15000000               # bytes, uploaded .zip
    MAX_ARCHIVE_ENTRIES=2000
    MAX_EXTRACTED_SIZE=25000000             # bytes, uncompressed
    MAX_PROJECT_FILES=20                    # source files sent to the AI per project
    PROJECT_PARALLELISM=3
    MAX_CONTEXT_FILES=6                     # related files attached as read-only context
    MAX_CONTEXT_CHARS=40000
    MAX_CHUNKS=24
    MAX_CONCURRENT_JOBS=4                   # queue workers
    MAX_HISTORY=10                          # remembered operations per user
    USER_COOLDOWN=1.0                       # seconds between updates per user
    ALLOWED_USER_IDS=                       # empty = open to everyone
    ADMIN_IDS=                              # required for /ai_status
    LOG_DIR=logs
    LOG_TO_FILE=true
    LOG_LEVEL=INFO

Run: python bot.py
"""

from __future__ import annotations

import ast
import asyncio
import itertools
import json
import logging
import math
import os
import random
import re
import sys
import time
import zipfile
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from html import escape
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Final, Iterable, Sequence
from xml.etree import ElementTree

import aiohttp
from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    TelegramObject,
    User,
)
from aiogram.utils.chat_action import ChatActionSender
from dotenv import load_dotenv

LOGGER: Final = logging.getLogger("ai_code_bot")


# --------------------------------------------------------------------------- #
# Errors                                                                       #
# --------------------------------------------------------------------------- #


class ConfigError(RuntimeError):
    """Raised when the environment configuration is missing or invalid."""


class JobError(RuntimeError):
    """User-facing failure: the message is shown to the user as-is."""


class AIError(JobError):
    """No AI provider could deliver a usable answer."""


class AIContentError(JobError):
    """
    The answer itself is unusable (truncated, filtered).

    Deterministic: switching provider would not help, so it never triggers a failover.
    """


class ProviderUnavailable(RuntimeError):
    """
    One provider failed and the manager must move to the next one.

    `fatal`        -> configuration problem (bad key/model): the provider is parked until a
                      health check proves it works again.
    `rate_limited` -> the service is alive but throttling us.
    `cooldown`     -> seconds to keep the provider out of the rotation.
    """

    def __init__(
        self,
        message: str,
        *,
        fatal: bool = False,
        rate_limited: bool = False,
        cooldown: float = 20.0,
    ) -> None:
        super().__init__(message)
        self.fatal = fatal
        self.rate_limited = rate_limited
        self.cooldown = cooldown


# Backwards-compatible alias: older code raised/caught DeepSeekError.
DeepSeekError = AIError


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #


def _env_str(key: str, default: str = "", *, required: bool = False) -> str:
    """Read a stripped environment string, optionally enforcing its presence."""
    value = (os.getenv(key) or "").strip()
    if not value:
        if required:
            raise ConfigError(f"متغير البيئة المطلوب غير موجود: {key}")
        return default
    return value


def _env_any(*keys: str, default: str = "", required: bool = False) -> str:
    """Read the first key that carries a value (used for renamed variables)."""
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    if required:
        raise ConfigError(f"متغير البيئة المطلوب غير موجود: {' أو '.join(keys)}")
    return default


def _env_int(key: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read an integer setting, clamped into a safe range."""
    raw = _env_str(key)
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError as exc:
        raise ConfigError(f"قيمة غير صالحة لـ {key}: {raw!r} (يجب أن تكون عدداً صحيحاً)") from exc


def _env_float(key: str, default: float, *, minimum: float, maximum: float) -> float:
    """Read a float setting, clamped into a safe range."""
    raw = _env_str(key)
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, float(raw)))
    except ValueError as exc:
        raise ConfigError(f"قيمة غير صالحة لـ {key}: {raw!r} (يجب أن تكون رقماً)") from exc


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean setting written as 1/true/yes/on."""
    raw = _env_str(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "y"}


def _env_ids(key: str) -> frozenset[int]:
    """Read a separated list of Telegram user ids."""
    raw = _env_str(key)
    if not raw:
        return frozenset()
    ids: set[int] = set()
    for chunk in re.split(r"[,\s;]+", raw):
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError as exc:
            raise ConfigError(f"معرّف مستخدم غير صالح في {key}: {chunk!r}") from exc
    return frozenset(ids)


ROUTING_PRIORITY: Final = "priority"
ROUTING_FASTEST: Final = "fastest"

# Every supported provider: env prefix -> display name + defaults. Both speak the same
# OpenAI-compatible dialect, so adding a third one is a single line here.
PROVIDER_BLUEPRINTS: Final[tuple[dict[str, Any], ...]] = (
    {
        "prefix": "DEEPSEEK",
        "name": "DeepSeek",
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "max_tokens": 8_192,
        "thinking": False,
    },
    {
        "prefix": "GROQ",
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 32_768,
        "thinking": False,
    },
)


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Immutable settings of one OpenAI-compatible endpoint."""

    name: str
    prefix: str
    api_key: str
    api_url: str
    model: str
    max_tokens: int
    temperature: float
    thinking: bool
    reasoning_effort: str
    priority: int

    @property
    def masked_key(self) -> str:
        """The key as it may appear in a log: never the secret itself."""
        if len(self.api_key) <= 8:
            return "***"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"


def _load_provider(blueprint: dict[str, Any], priority: int) -> ProviderConfig | None:
    """Build one provider from the environment, or None when it is absent/disabled."""
    prefix: str = blueprint["prefix"]
    api_key = _env_str(f"{prefix}_API_KEY")
    if not api_key or not _env_bool(f"{prefix}_ENABLED", True):
        return None

    api_url = _env_str(f"{prefix}_API_URL", blueprint["url"])
    if not api_url.startswith(("http://", "https://")):
        raise ConfigError(f"رابط غير صالح لـ {prefix}_API_URL: {api_url!r}")

    effort = _env_str(f"{prefix}_REASONING_EFFORT", "high").lower()
    if effort not in {"high", "max"}:
        effort = "high"

    default_temperature = _env_float("AI_TEMPERATURE", 0.2, minimum=0.0, maximum=2.0)
    return ProviderConfig(
        name=blueprint["name"],
        prefix=prefix,
        api_key=api_key,
        api_url=api_url,
        model=_env_str(f"{prefix}_MODEL", blueprint["model"]),
        max_tokens=_env_int(
            f"{prefix}_MAX_TOKENS", blueprint["max_tokens"], minimum=1_024, maximum=384_000
        ),
        temperature=_env_float(
            f"{prefix}_TEMPERATURE", default_temperature, minimum=0.0, maximum=2.0
        ),
        thinking=_env_bool(f"{prefix}_THINKING", blueprint["thinking"]),
        reasoning_effort=effort,
        priority=priority,
    )


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration, loaded once at startup."""

    bot_token: str
    providers: tuple[ProviderConfig, ...]
    routing: str
    health_interval: int
    probe_timeout: int
    request_timeout: int
    max_retries: int
    retry_delay: float
    validation_retries: int
    max_file_size: int
    max_archive_size: int
    max_archive_entries: int
    max_extracted_size: int
    max_project_files: int
    project_parallelism: int
    max_context_files: int
    max_context_chars: int
    max_chunks: int
    max_concurrent_jobs: int
    max_history: int
    user_cooldown: float
    allowed_users: frozenset[int]
    admin_ids: frozenset[int]
    log_dir: Path
    log_to_file: bool
    log_level: str

    @classmethod
    def load(cls) -> "Settings":
        """Build the settings object from the process environment."""
        providers = tuple(
            provider
            for provider in (
                _load_provider(blueprint, priority)
                for priority, blueprint in enumerate(PROVIDER_BLUEPRINTS, start=1)
            )
            if provider is not None
        )
        if not providers:
            raise ConfigError(
                "لا يوجد أي مزود ذكاء اصطناعي مفعّل. "
                "أضف DEEPSEEK_API_KEY أو GROQ_API_KEY (أو كليهما) إلى ملف .env"
            )

        routing = _env_str("AI_ROUTING", ROUTING_PRIORITY).lower()
        if routing not in {ROUTING_PRIORITY, ROUTING_FASTEST}:
            routing = ROUTING_PRIORITY

        return cls(
            bot_token=_env_any("TELEGRAM_BOT_TOKEN", "BOT_TOKEN", required=True),
            providers=providers,
            routing=routing,
            health_interval=_env_int("AI_HEALTH_INTERVAL", 300, minimum=0, maximum=86_400),
            probe_timeout=_env_int("AI_PROBE_TIMEOUT", 15, minimum=5, maximum=120),
            request_timeout=_env_int("REQUEST_TIMEOUT", 600, minimum=30, maximum=3_600),
            max_retries=_env_int("MAX_RETRIES", 3, minimum=1, maximum=10),
            retry_delay=_env_float("RETRY_DELAY", 2.0, minimum=0.5, maximum=30.0),
            validation_retries=_env_int("VALIDATION_RETRIES", 1, minimum=0, maximum=3),
            max_file_size=_env_int("MAX_FILE_SIZE", 200_000, minimum=1_024, maximum=2_000_000),
            max_archive_size=_env_int(
                "MAX_ARCHIVE_SIZE", 15_000_000, minimum=10_000, maximum=20_000_000
            ),
            max_archive_entries=_env_int(
                "MAX_ARCHIVE_ENTRIES", 2_000, minimum=1, maximum=20_000
            ),
            max_extracted_size=_env_int(
                "MAX_EXTRACTED_SIZE", 25_000_000, minimum=10_000, maximum=100_000_000
            ),
            max_project_files=_env_int("MAX_PROJECT_FILES", 20, minimum=1, maximum=200),
            project_parallelism=_env_int("PROJECT_PARALLELISM", 3, minimum=1, maximum=16),
            max_context_files=_env_int("MAX_CONTEXT_FILES", 6, minimum=0, maximum=30),
            max_context_chars=_env_int(
                "MAX_CONTEXT_CHARS", 40_000, minimum=0, maximum=400_000
            ),
            max_chunks=_env_int("MAX_CHUNKS", 24, minimum=1, maximum=64),
            max_concurrent_jobs=_env_int("MAX_CONCURRENT_JOBS", 4, minimum=1, maximum=64),
            max_history=_env_int("MAX_HISTORY", 10, minimum=1, maximum=50),
            user_cooldown=_env_float("USER_COOLDOWN", 1.0, minimum=0.0, maximum=30.0),
            allowed_users=_env_ids("ALLOWED_USER_IDS"),
            admin_ids=_env_ids("ADMIN_IDS"),
            log_dir=Path(_env_str("LOG_DIR", "logs")),
            log_to_file=_env_bool("LOG_TO_FILE", True),
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        )


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

MODE_FIX: Final = "fix"
MODE_UPDATE: Final = "update"
MODE_TITLES: Final[dict[str, str]] = {MODE_FIX: "تصحيح أخطاء", MODE_UPDATE: "تحديث ملف"}

STATUS_SUCCESS: Final = "نجاح"
STATUS_FAILED: Final = "فشل"
STATUS_CANCELLED: Final = "ملغاة"

# ASCII statuses written to logs/ (grep-friendly, matches the requested log format).
LOG_STATUS: Final[dict[str, str]] = {
    STATUS_SUCCESS: "Success",
    STATUS_FAILED: "Failed",
    STATUS_CANCELLED: "Cancelled",
}
LOG_EVENT_AI: Final = "AI"
LOG_EVENT_JOB: Final = "JOB"

ARCHIVE_EXTENSION: Final = ".zip"
ZIP_MAGIC: Final = b"PK\x03\x04"

# Every language the bot accepts, mapped to the label injected into the prompt.
LANGUAGE_BY_EXTENSION: Final[dict[str, str]] = {
    ".py": "Python",
    ".pyi": "Python",
    ".php": "PHP",
    ".js": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".jsx": "React (JavaScript)",
    ".tsx": "React (TypeScript)",
    ".ts": "TypeScript",
    ".vue": "Vue",
    ".dart": "Dart (Flutter)",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".gradle": "Gradle",
    ".swift": "Swift",
    ".c": "C",
    ".h": "C/C++ header",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".hh": "C++",
    ".cs": "C#",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".xml": "XML",
    ".toml": "TOML",
    ".ini": "INI configuration",
    ".cfg": "INI configuration",
    ".conf": "configuration",
    ".env": "environment configuration",
    ".sql": "SQL",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".ps1": "PowerShell",
    ".psm1": "PowerShell",
    ".md": "Markdown",
    ".txt": "plain-text",
}

# Files that carry no extension (or a misleading one) but are still source files.
LANGUAGE_BY_FILENAME: Final[dict[str, str]] = {
    "dockerfile": "Dockerfile",
    "docker-compose.yml": "Docker Compose",
    "docker-compose.yaml": "Docker Compose",
    "compose.yml": "Docker Compose",
    "compose.yaml": "Docker Compose",
    "nginx.conf": "NGINX configuration",
    "default.conf": "NGINX configuration",
    ".htaccess": "Apache configuration",
    "apache2.conf": "Apache configuration",
    "httpd.conf": "Apache configuration",
    "makefile": "Makefile",
    "procfile": "Procfile",
    ".env": "environment configuration",
    ".env.example": "environment configuration",
}

# Markers used to name the project type shown in the prompt and in the result caption.
PROJECT_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ("pubspec.yaml", "Flutter / Dart"),
    ("composer.json", "PHP"),
    ("package.json", "Node.js"),
    ("requirements.txt", "Python"),
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("manage.py", "Python (Django)"),
    ("go.mod", "Go"),
    ("cargo.toml", "Rust"),
    ("pom.xml", "Java (Maven)"),
    ("build.gradle", "Java / Kotlin (Gradle)"),
    ("gemfile", "Ruby"),
)

# Directories never extracted: heavy, generated, or irrelevant to a code review.
IGNORED_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        ".git", ".svn", ".hg", ".idea", ".vscode", ".gradle", ".dart_tool",
        "node_modules", "vendor", "venv", ".venv", "env", "__pycache__",
        "build", "dist", "out", "target", "bin", "obj", "coverage", "pods",
        ".next", ".nuxt", ".expo", ".mypy_cache", ".pytest_cache", ".tox",
    }
)

# utf-8-sig first so a BOM is stripped instead of leaking into the source.
DECODE_ENCODINGS: Final[tuple[str, ...]] = ("utf-8-sig", "utf-8", "cp1256", "latin-1")

# --- HTTP classification --------------------------------------------------- #

RATE_LIMIT_STATUS: Final = 429
# Transient: worth retrying on the same provider before failing over.
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 500, 502, 503, 504, 529})
# Configuration problems: the provider is parked until a health check clears it.
FATAL_STATUS: Final[frozenset[int]] = frozenset({400, 401, 402, 403, 404, 413, 422})
# Statuses that mean "your key/URL/model is wrong", not "the service hiccuped".
DISABLING_STATUS: Final[frozenset[int]] = frozenset({401, 403, 404})

FATAL_STATUS_MESSAGES: Final[dict[int, str]] = {
    400: "طلب غير صالح تجاه {provider}.",
    401: "مفتاح {provider} غير صحيح أو منتهي ({prefix}_API_KEY).",
    402: "رصيد حساب {provider} غير كافٍ.",
    403: "الوصول مرفوض من {provider}.",
    404: "رابط أو موديل {provider} غير صحيح ({prefix}_API_URL / {prefix}_MODEL).",
    413: "حجم الطلب أكبر من الحد المسموح لدى {provider}.",
    422: "معطيات الطلب غير صالحة لدى {provider}.",
}

# Optional body fields: dropped automatically when an endpoint rejects them with a 400.
OPTIONAL_PAYLOAD_KEYS: Final[tuple[str, ...]] = ("max_tokens", "thinking", "reasoning_effort")

COOLDOWN_TRANSIENT: Final = 20.0
COOLDOWN_RATE_LIMIT: Final = 45.0
COOLDOWN_FATAL: Final = 600.0
MAX_COOLDOWN: Final = 1_800.0
MAX_TIMEOUTS_PER_PROVIDER: Final = 1     # one timeout is enough: fail over instead of waiting
CONNECT_TIMEOUT: Final = 20
PROBE_MAX_TOKENS: Final = 8
DEFAULT_LATENCY: Final = 999.0           # unknown provider sorts last under "fastest" routing

# --- Regexes and limits ---------------------------------------------------- #

FENCED_BLOCK_RE: Final = re.compile(r"```(?P<tag>[^\n`]*)\n(?P<body>.*?)\n?```", re.DOTALL)
CONTROL_CHARS_RE: Final = re.compile(r"[\x00-\x1f\x7f]")

# Markers proving the model shortened the file instead of returning it in full.
PLACEHOLDER_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?:\.\.\.|…)\s*(?:rest|remaining|existing|previous|unchanged|same)\b", re.I),
    re.compile(r"\b(?:rest|remainder)\s+of\s+(?:the\s+)?(?:code|file|class|function)\b", re.I),
    re.compile(r"\bcode\s+(?:omitted|unchanged|remains\s+the\s+same)\b", re.I),
    re.compile(r"\b(?:keep|leave)\s+(?:the\s+)?(?:rest|existing)\s+(?:as\s+is|unchanged)\b", re.I),
    re.compile(r"^[ \t]*(?:#|//|--|;)[ \t]*(?:\.\.\.|…)[ \t]*$", re.M),
    re.compile(r"بقية\s+الكود|باقي\s+الكود|بدون\s+تغيير"),
)

# Best-effort symbol extraction, used to prove nothing was deleted.
SYMBOL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+(\w+)", re.M),
    re.compile(r"^[ \t]*class[ \t]+(\w+)", re.M),
    re.compile(r"^[ \t]*(?:[\w\s]*?)function[ \t]+(\w+)[ \t]*\(", re.M),
    re.compile(r"^[ \t]*func[ \t]+(?:\([^)]*\)[ \t]*)?(\w+)", re.M),
    re.compile(r"^[ \t]*fn[ \t]+(\w+)", re.M),
)

# Import/require/use statements, used to link the files of a project together.
IMPORT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^[ \t]*from[ \t]+([.\w]+)[ \t]+import\b", re.M),
    re.compile(r"^[ \t]*import[ \t]+([.\w]+)", re.M),
    re.compile(r"""import[^'"\n]*['"]([^'"\n]+)['"]""", re.M),
    re.compile(r"""require(?:_once)?[ \t]*\(?[ \t]*['"]([^'"\n]+)['"]""", re.M),
    re.compile(r"""include(?:_once)?[ \t]*\(?[ \t]*['"]([^'"\n]+)['"]""", re.M),
    re.compile(r"^[ \t]*use[ \t]+([\w\\]+)[ \t]*;", re.M),
)

# Signals that the user pasted a real error/stack trace, so it gets its own prompt section.
ERROR_HINT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bTraceback \(most recent call last\)", re.I),
    re.compile(r"\b\w*(?:Error|Exception|Warning)\b\s*:", re.I),
    re.compile(r"\b(?:Fatal|Parse|Uncaught|Unhandled)\s+error\b", re.I),
    re.compile(r"\bat\s+line\s+\d+|\bline\s+\d+\b.*\bcolumn\b", re.I),
    re.compile(r"\bstack\s*trace\b|\bsegmentation fault\b", re.I),
    re.compile(r"^\s*File\s+\".+\",\s+line\s+\d+", re.M),
)

MIN_INSTRUCTION_LEN: Final = 2
MAX_INSTRUCTION_LEN: Final = 1_500
MAX_FILENAME_LEN: Final = 96
MAX_ERROR_PREVIEW: Final = 300
MAX_LOG_DETAIL: Final = 160
MAX_BACKOFF: Final = 30.0
MAX_PROMPT_CHARS: Final = 600_000
MAX_TREE_ENTRIES: Final = 200
CHARS_PER_TOKEN: Final = 3.0
MIN_CHUNK_BUDGET: Final = 4_000
CHUNK_CONTEXT_LINES: Final = 25
MIN_LENGTH_RATIO: Final = 0.4
ZIP_BOMB_RATIO: Final = 300
ZIP_BOMB_FLOOR: Final = 1_000_000
DEFAULT_JOB_SECONDS: Final = 90.0
EMA_ALPHA: Final = 0.3
MIN_EDIT_INTERVAL: Final = 1.0
MAX_HISTORY_USERS: Final = 5_000
LOG_MAX_BYTES: Final = 5_000_000
LOG_BACKUPS: Final = 5

STAGE_RECEIVE: Final = 0
STAGE_READ: Final = 1
STAGE_ANALYZE: Final = 2
STAGE_REQUEST: Final = 3
STAGE_PROCESS: Final = 4
STAGE_VALIDATE: Final = 5
STAGE_SEND: Final = 6

STAGE_LABELS: Final[tuple[str, ...]] = (
    "📂 استلام الملف...",
    "📖 قراءة الملف...",
    "🔍 تحليل الكود...",
    "🤖 إرسال الطلب إلى الذكاء الاصطناعي...",
    "⚙️ معالجة النتائج...",
    "✅ التحقق النهائي...",
    "📤 إرسال الملف...",
)

# --------------------------------------------------------------------------- #
# User-facing texts (Arabic)                                                   #
# --------------------------------------------------------------------------- #

TEXT_WELCOME: Final = (
    "مرحباً {name}\n\n"
    "أهلاً بك في بوت تصحيح وتطوير الأكواد.\n\n"
    "يمكنك استخدام الأزرار التالية:\n\n"
    "• تصحيح أخطاء الملفات البرمجية بواسطة الذكاء الاصطناعي.\n"
    "• تحديث أو تطوير أي ملف برمجي حسب البرومبت الذي تكتبه.\n"
    "• عرض سجل العمليات السابقة.\n\n"
    "يمكنك رفع ملف واحد أو مشروع كامل بصيغة zip.\n\n"
    "المزودات: {providers} — مع تبديل تلقائي عند تعطل أحدها.\n\n"
    "اختر العملية المطلوبة من الأسفل."
)
TEXT_ASK_FILE_FIX: Final = "📂 أرسل الملف البرمجي المراد تصحيحه، أو مشروعاً بصيغة zip."
TEXT_ASK_FILE_UPDATE: Final = "📂 أرسل الملف المراد تطويره، أو مشروعاً بصيغة zip."
TEXT_ASK_PROBLEM: Final = (
    "اكتب نوع المشكلة التي تريد إصلاحها، وألصق رسالة الخطأ إن وجدت "
    "(سيتم إرسالها للذكاء الاصطناعي كما هي).\n\n"
    "مثلاً:\n"
    "- أخطاء Syntax\n"
    "- أخطاء Runtime\n"
    "- تحسين الأداء\n"
    "- تنظيف الكود\n"
    "- إصلاح شامل\n"
    "- إصلاح جميع المشاكل"
)
TEXT_ASK_PROMPT: Final = "اكتب البرومبت الذي يصف التحديث المطلوب."
TEXT_FILE_ONLY: Final = "يجب إرسال ملف برمجي أو مشروع zip، لا رسائل نصية."
TEXT_TEXT_ONLY: Final = "أرسل وصفاً نصياً فقط في هذه المرحلة."
TEXT_UNSUPPORTED: Final = (
    "صيغة الملف غير مدعومة.\n\n"
    "مدعوم: py, php, js, ts, jsx, tsx, vue, dart, go, rs, java, kt, swift, c, cpp, cs, "
    "html, css, scss, json, yml, xml, toml, ini, conf, env, sql, sh, ps1, md, txt, "
    "Dockerfile, docker-compose, nginx, apache — أو مشروع كامل بصيغة zip."
)
TEXT_TOO_BIG: Final = "الملف كبير جداً. الحد الأقصى {limit}."
TEXT_EMPTY_FILE: Final = "الملف فارغ."
TEXT_NOT_TEXT: Final = "الملف ليس نصياً (يبدو ملفاً ثنائياً أو تالفاً)."
TEXT_BAD_ARCHIVE: Final = "ملف zip تالف أو غير صالح."
TEXT_DOWNLOAD_FAILED: Final = "تعذّر تنزيل الملف من تيليجرام. أعد المحاولة."
TEXT_SHORT_INSTRUCTION: Final = "الوصف قصير جداً. اكتب وصفاً أوضح."
TEXT_LONG_INSTRUCTION: Final = f"الوصف طويل جداً. الحد الأقصى {MAX_INSTRUCTION_LEN} حرف."
TEXT_BUSY: Final = "لديك عملية قيد التنفيذ أو في الانتظار. انتظر انتهاءها أو ألغِها."
TEXT_THROTTLED: Final = "الرجاء التمهّل قليلاً."
TEXT_SESSION_LOST: Final = "انتهت الجلسة. ابدأ من جديد."
TEXT_CANCELLED: Final = "تم إلغاء العملية وتنظيف الذاكرة."
TEXT_NOTHING_TO_CANCEL: Final = "لا توجد عملية جارية لإلغائها."
TEXT_MENU: Final = "اختر العملية المطلوبة من الأسفل."
TEXT_FAILED: Final = "❌ فشلت العملية.\n\nالسبب: {reason}"
TEXT_UNEXPECTED: Final = "حدث خطأ غير متوقع. أعد المحاولة."
TEXT_STARTING: Final = "⏳ جاري التحضير..."
TEXT_FIX_DONE: Final = "تم تصحيح الملف بنجاح."
TEXT_UPDATE_DONE: Final = "تم تحديث الملف بنجاح."
TEXT_EMPTY_HISTORY: Final = "لا توجد عمليات سابقة."
TEXT_HISTORY_TITLE: Final = "<b>سجل العمليات</b> — آخر {count} عملية"
TEXT_QUEUED: Final = (
    "<b>⏳ في قائمة الانتظار</b>\n\n"
    "الملف: {file}\n"
    "الترتيب: {position}\n"
    "الوقت المتوقع: ~{eta}\n\n"
    "سيبدأ التنفيذ تلقائياً عند توفر منفذ."
)
TEXT_NO_SUPPORTED_FILES: Final = "لا توجد ملفات مدعومة داخل المشروع."
TEXT_TOO_MANY_FILES: Final = (
    "المشروع يحتوي على {count} ملف مدعوم، والحد الأقصى {limit}. "
    "قلّل عدد الملفات أو ارفع MAX_PROJECT_FILES."
)
TEXT_TOO_MANY_CHUNKS: Final = "الملف كبير جداً للمعالجة ({parts} جزء، الحد {limit})."
TEXT_ARCHIVE_TOO_LARGE: Final = "محتوى المشروع بعد فك الضغط تجاوز الحد المسموح."
TEXT_ARCHIVE_BOMB: Final = "المشروع مرفوض: نسبة ضغط مشبوهة (zip bomb)."
TEXT_CHUNK_PROGRESS: Final = "⏳ جاري تحليل الجزء {index}/{total}"
TEXT_CHUNK_TRUNCATED: Final = "الجزء {index}/{total} رجع ناقصاً من مزود الذكاء الاصطناعي."
TEXT_PROVIDER_SWITCH: Final = "⚠️ {failed} غير متاح — التحويل تلقائياً إلى {next}..."
TEXT_ALL_PROVIDERS_DOWN: Final = (
    "تعذّر الاتصال بجميع مزودي الذكاء الاصطناعي.\n{reasons}"
)
TEXT_TRUNCATED_ANSWER: Final = (
    "الرد وصل للحد الأقصى وتم قطعه لدى {provider}. "
    "ارفع {prefix}_MAX_TOKENS أو قلّل حجم الملف."
)
TEXT_FILTERED_ANSWER: Final = "تم حجب المحتوى من مرشحات {provider}."
TEXT_ADMIN_ONLY: Final = "هذا الأمر مخصص للمشرفين فقط (ADMIN_IDS)."
TEXT_PROBING: Final = "جاري فحص المزودات..."
TEXT_BTN_FIX: Final = "تصحيح أخطاء"
TEXT_BTN_UPDATE: Final = "تحديث ملف"
TEXT_BTN_HISTORY: Final = "سجل العمليات"
TEXT_BTN_CANCEL: Final = "إلغاء العملية الحالية"
TEXT_BTN_REFRESH: Final = "🔄 إعادة الفحص"

TEXT_AVAILABLE: Final = "✅ Available"
TEXT_UNAVAILABLE: Final = "❌ Unavailable"
TEXT_DISABLED: Final = "⛔ Disabled"
TEXT_COOLING: Final = "⏳ Cooling down"
TEXT_UNTESTED: Final = "❔ Not tested"

# --------------------------------------------------------------------------- #
# Prompt engineering                                                           #
# --------------------------------------------------------------------------- #

OUTPUT_CONTRACT: Final = """OUTPUT CONTRACT (breaking any rule makes the answer worthless):
1. Output the COMPLETE final content of the target file, from its first character to its last.
2. Output raw file content ONLY: no markdown, no ``` fences, no prose, no notes, no diff.
3. NEVER shorten, summarise, elide or omit anything. Placeholders such as "...",
   "// rest of the code", "# unchanged", "code omitted" are strictly forbidden.
4. NEVER delete a function, class, method, route, constant, config key or feature.
   Every symbol present in the input MUST still exist in the output.
5. Preserve the original language, framework, style, structure, comments, encoding and
   indentation. Keep the public API stable.
6. If a rule conflicts with the user request, rules 1-3 always win."""

FIX_SYSTEM_PROMPT: Final = """You are a Principal {language} Engineer performing a surgical repair of one file.

{contract}

ANALYSIS RULES (think before you write):
- Identify the real root cause of the reported problem, not only its symptom.
- Map the error message (when provided) to the exact line and construct that produces it.
- Then apply the minimal, correct fix.

REPAIR RULES:
- Fix every syntax, runtime and logical error.
- Fix broken imports, exception handling and formatting.
- Remove ONLY provably dead or incorrect code, never a working feature.
- Improve security, readability and performance without changing behaviour.
- Respect how this file is used by the rest of the project (see the read-only context)."""

UPDATE_SYSTEM_PROMPT: Final = """You are a Principal Software Engineer updating one {language} file.

{contract}

UPDATE RULES:
- Implement exactly what the user asked for, nothing less and nothing extra.
- Keep every existing feature working; never break the current behaviour.
- Delete code only when the user explicitly asked for its removal.
- Keep the file consistent with the rest of the project (see the read-only context)."""

CHUNK_SYSTEM_PROMPT: Final = """You are a Principal {language} Engineer editing PART {index}/{total} of the file "{name}".

The parts are concatenated in order to rebuild the whole file, so your answer replaces this part verbatim.

PART CONTRACT (breaking any rule corrupts the file):
1. Return the COMPLETE corrected content of THIS PART ONLY.
2. Never return other parts, never repeat the read-only context, never add a header or footer.
3. Keep the exact leading indentation of the first line and the trailing newline of the last line.
4. Never shorten, summarise or use placeholders such as "..." or "rest of the code".
5. Never delete a function, class or feature that exists in this part.
6. Output raw code only: no markdown, no ``` fences, no explanations.

TASK: {task}"""

REPAIR_SYSTEM_SUFFIX: Final = """

CRITICAL — your previous answer was REJECTED by the automatic validator:
{issues}
Return the corrected content again, fully, obeying every rule above."""


def build_system_prompt(mode: str, file_name: str) -> str:
    """Pick the instruction set for the mode and specialise it for the file language."""
    template = FIX_SYSTEM_PROMPT if mode == MODE_FIX else UPDATE_SYSTEM_PROMPT
    return template.format(language=resolve_language(file_name), contract=OUTPUT_CONTRACT)


def build_chunk_system_prompt(mode: str, file_name: str, index: int, total: int) -> str:
    """Instruction set used when a large file is repaired part by part."""
    task = (
        "Fix every error in this part while keeping all of its functionality."
        if mode == MODE_FIX
        else "Apply the user's requested update to this part while keeping all of its functionality."
    )
    return CHUNK_SYSTEM_PROMPT.format(
        language=resolve_language(file_name),
        index=index,
        total=total,
        name=file_name,
        task=task,
    )


def looks_like_error(text: str) -> bool:
    """True when the user pasted a compiler/runtime error or a stack trace."""
    return any(pattern.search(text) for pattern in ERROR_HINT_PATTERNS)


def build_user_prompt(
    mode: str,
    instruction: str,
    file_name: str,
    code: str,
    *,
    project_type: str = "",
    project_map: str = "",
    related: str = "",
    chunk_before: str = "",
    chunk_after: str = "",
) -> str:
    """
    Assemble the user turn: request, error message, project knowledge, then the content.

    The model always receives the file name, its language, the code itself and — when the
    user pasted one — the error message in its own clearly labelled section.
    """
    instruction = instruction.strip()
    sections: list[str] = [
        "=== TASK ===",
        f"Operation: {'FIX' if mode == MODE_FIX else 'UPDATE'}",
        f"Target file: {file_name}",
        f"Language: {resolve_language(file_name)}",
    ]
    if project_type:
        sections.append(f"Project type: {project_type}")

    sections += ["", "=== USER REQUEST ===", instruction]

    if mode == MODE_FIX and looks_like_error(instruction):
        sections += [
            "",
            "=== ERROR MESSAGE / TRACEBACK (reported by the user) ===",
            instruction,
            "",
            "Explain nothing: diagnose this error internally, find its root cause in the "
            "code below and return the corrected file.",
        ]

    if project_map:
        sections += ["", "=== PROJECT MAP (read-only) ===", project_map]
    if related:
        sections += [
            "",
            "=== RELATED FILES (read-only context — never output them) ===",
            related,
        ]
    if chunk_before:
        sections += [
            "",
            "=== PREVIOUS PART (read-only — never output it) ===",
            chunk_before,
        ]
    if chunk_after:
        sections += ["", "=== NEXT PART (read-only — never output it) ===", chunk_after]

    sections += [
        "",
        "=== TARGET CONTENT (return its complete corrected version, raw, nothing else) ===",
        code,
    ]
    prompt = "\n".join(sections)
    if len(prompt) > MAX_PROMPT_CHARS:  # last-resort guard against a context overflow
        prompt = prompt[:MAX_PROMPT_CHARS]
    return prompt


# --------------------------------------------------------------------------- #
# Generic helpers                                                              #
# --------------------------------------------------------------------------- #


def resolve_language(file_name: str) -> str:
    """Map a file name to the human language label used inside the prompts."""
    lower = PurePosixPath(file_name.lower()).name
    if lower in LANGUAGE_BY_FILENAME:
        return LANGUAGE_BY_FILENAME[lower]
    if lower.startswith(".env"):
        return "environment configuration"
    if lower.startswith("dockerfile"):
        return "Dockerfile"
    return LANGUAGE_BY_EXTENSION.get(PurePosixPath(lower).suffix, "Software")


def is_supported(file_name: str) -> bool:
    """Allow-list check for a single source file (archives are handled separately)."""
    lower = PurePosixPath(file_name.lower()).name
    if not lower:
        return False
    if lower in LANGUAGE_BY_FILENAME or lower.startswith(".env") or lower.startswith("dockerfile"):
        return True
    return PurePosixPath(lower).suffix in LANGUAGE_BY_EXTENSION


def is_archive(file_name: str) -> bool:
    """True when the upload is a project archive."""
    return PurePosixPath(file_name.lower()).suffix == ARCHIVE_EXTENSION


def safe_file_name(raw: str | None) -> str:
    """Strip directories and control characters so a name can never escape anywhere."""
    candidate = PurePosixPath((raw or "").replace("\\", "/")).name
    candidate = CONTROL_CHARS_RE.sub("", candidate).strip()
    return candidate[:MAX_FILENAME_LEN] or "output.txt"


def safe_archive_path(raw: str) -> str:
    """Normalise a zip entry path, returning '' when it is unsafe or absolute."""
    cleaned = CONTROL_CHARS_RE.sub("", raw.replace("\\", "/")).strip()
    if not cleaned or cleaned.endswith("/"):
        return ""
    if cleaned.startswith("/") or re.match(r"^[A-Za-z]:", cleaned):
        return ""
    parts = [part for part in cleaned.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return ""
    return "/".join(parts)


def is_ignored_path(path: str) -> bool:
    """True for generated/vendor directories that must never be processed or shipped."""
    return any(part.lower() in IGNORED_DIRECTORIES for part in PurePosixPath(path).parts[:-1])


def decode_source(raw: bytes) -> str:
    """Decode uploaded bytes as text, rejecting binaries early."""
    if b"\x00" in raw:
        raise ValueError("binary content")
    for encoding in DECODE_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("undecodable content")


def unwrap_code(text: str, file_name: str = "", *, keep_indent: bool = False) -> str:
    """
    Drop a markdown fence when the model wrapped the whole answer in one.

    `keep_indent` is used for chunks, whose first line may legitimately be indented:
    only newlines are trimmed, never the indentation itself. A `.md` file may open and
    close with a fence, so its wrapper is removed only when unambiguous.
    """
    body = text.lstrip("\ufeff")
    body = body.strip("\n") if keep_indent else body.strip()
    match = FENCED_BLOCK_RE.fullmatch(body.strip() if keep_indent else body)
    if match:
        inner = match.group("body")
        tag = match.group("tag").strip().lower()
        is_markdown = PurePosixPath(file_name.lower()).suffix == ".md"
        if not is_markdown or "```" not in inner or tag in {"markdown", "md"}:
            body = inner
    return body if body.endswith("\n") else body + "\n"


def restore_chunk_edges(part: str, chunk: str) -> str:
    """
    Give a produced part the exact leading/trailing newlines of the original chunk.

    Parts are concatenated to rebuild the file, and the model (like `unwrap_code`) freely
    trims the blank lines around its answer. Without this, every cut boundary would quietly
    swallow the blank lines that separate two definitions.
    """
    if not chunk.strip("\n"):
        return chunk  # the chunk is only newlines: nothing to rebuild
    core = part.strip("\n")
    if not core:
        return chunk  # empty answer: keep the original part rather than corrupt the file
    lead = chunk[: len(chunk) - len(chunk.lstrip("\n"))]
    trail = chunk[len(chunk.rstrip("\n")) :]
    return f"{lead}{core}{trail}"


def human_size(size: int) -> str:
    """Render a byte count for humans."""
    if size >= 1_048_576:
        return f"{size / 1_048_576:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def human_duration(seconds: float) -> str:
    """Render a duration for humans."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total} ث"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} د {secs} ث"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} س {minutes} د"


def one_line(text: str, limit: int = MAX_LOG_DETAIL) -> str:
    """Collapse a value into a single, bounded log line."""
    collapsed = " ".join(str(text).split())
    return collapsed[:limit]


def is_cut_boundary(line: str) -> bool:
    """A blank line or a column-0 line is a safe place to cut a source file."""
    if not line.strip():
        return True
    return line[0] not in " \t"


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last-resort splitter for a block without any usable boundary."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and current:
            parts.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        parts.append("".join(current))
    return parts


def split_source(code: str, max_chars: int) -> list[str]:
    """
    Split a large file into ordered parts that rebuild it exactly when concatenated.

    Cuts are placed before a top-level line (column 0) or a blank line, so definitions
    are not sliced in half.
    """
    if max_chars <= 0 or len(code) <= max_chars:
        return [code]

    lines = code.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    boundary = 0  # index inside `current` where the last safe cut was seen

    for line in lines:
        if size + len(line) > max_chars and current:
            cut = boundary if 0 < boundary < len(current) else len(current)
            chunks.append("".join(current[:cut]))
            current = current[cut:]
            size = sum(len(item) for item in current)
            boundary = 0
        if is_cut_boundary(line):
            boundary = len(current)
        current.append(line)
        size += len(line)

    if current:
        chunks.append("".join(current))

    result: list[str] = []
    for chunk in chunks:
        result.extend(_hard_split(chunk, max_chars) if len(chunk) > max_chars else [chunk])
    return [chunk for chunk in result if chunk]


def extract_symbols(code: str, file_name: str) -> set[str]:
    """
    Collect the names of the functions/classes declared in a file.

    Python is parsed with `ast` (exact); every other language falls back to anchored
    regexes. Used to prove that the model deleted nothing.
    """
    if PurePosixPath(file_name.lower()).suffix in {".py", ".pyi"}:
        with suppress(SyntaxError, ValueError, RecursionError):
            tree = ast.parse(code)
            names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
            return names

    symbols: set[str] = set()
    for pattern in SYMBOL_PATTERNS:
        symbols.update(match.group(1) for match in pattern.finditer(code))
    return symbols


def check_syntax(file_name: str, code: str) -> str | None:
    """Validate the produced file when the language allows a cheap parse."""
    suffix = PurePosixPath(file_name.lower()).suffix
    try:
        if suffix in {".py", ".pyi"}:
            ast.parse(code)
        elif suffix == ".json":
            json.loads(code)
        elif suffix == ".xml":
            ElementTree.fromstring(code)
    except SyntaxError as exc:
        return f"خطأ صياغة Python (سطر {exc.lineno}): {exc.msg}"
    except json.JSONDecodeError as exc:
        return f"خطأ صياغة JSON (سطر {exc.lineno}): {exc.msg}"
    except ElementTree.ParseError as exc:
        return f"خطأ صياغة XML: {exc}"
    except (ValueError, RecursionError) as exc:
        return f"تعذّر تحليل الملف: {exc}"
    return None


def validate_result(
    original: str, produced: str, file_name: str, mode: str
) -> tuple[list[str], list[str]]:
    """
    Audit the model answer before it reaches the user.

    Returns (errors, warnings). Errors trigger an automatic repair round and, if they
    survive it, abort the file so the user never receives a corrupted result.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not produced.strip():
        return ["الرد فارغ."], warnings

    # A placeholder is only suspicious when the model introduced it.
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.search(produced) and not pattern.search(original):
            errors.append("الرد يحتوي على اختصار/Placeholder بدل الكود الكامل.")
            break

    if produced.lstrip().startswith("```"):
        errors.append("الرد أُعيد داخل Markdown.")

    ratio = len(produced) / max(len(original), 1)
    if ratio < MIN_LENGTH_RATIO:
        message = f"الرد أقصر من الأصل بنسبة كبيرة ({ratio:.0%})."
        (warnings if mode == MODE_UPDATE else errors).append(message)

    missing = sorted(extract_symbols(original, file_name) - extract_symbols(produced, file_name))
    if missing:
        preview = "، ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        message = f"اختفت دوال/كلاسات من الملف: {preview}"
        (warnings if mode == MODE_UPDATE else errors).append(message)

    syntax_error = check_syntax(file_name, produced)
    if syntax_error:
        errors.append(syntax_error)

    return errors, warnings


# --------------------------------------------------------------------------- #
# Project model                                                                #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ProjectBundle:
    """Everything extracted from an upload: editable sources plus untouched blobs."""

    sources: dict[str, str] = field(default_factory=dict)
    blobs: dict[str, bytes] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    skipped: int = 0
    project_type: str = ""
    is_project: bool = False

    def release(self) -> None:
        """Drop every byte held by the job (called on success, failure and cancel)."""
        self.sources.clear()
        self.blobs.clear()
        self.order.clear()


def detect_project_type(paths: Iterable[str]) -> str:
    """Name the stack from its marker files, falling back to the dominant language."""
    names = {PurePosixPath(path).name.lower() for path in paths}
    for marker, label in PROJECT_MARKERS:
        if marker in names:
            return label

    counts: dict[str, int] = {}
    for path in paths:
        language = LANGUAGE_BY_EXTENSION.get(PurePosixPath(path.lower()).suffix)
        if language:
            counts[language] = counts.get(language, 0) + 1
    if not counts:
        return "Unknown"
    return max(counts.items(), key=lambda item: item[1])[0]


def build_project_map(paths: Sequence[str]) -> str:
    """Render the file tree handed to the model as read-only context."""
    listed = sorted(paths)[:MAX_TREE_ENTRIES]
    lines = [f"- {path}" for path in listed]
    if len(paths) > len(listed):
        lines.append(f"- ... (+{len(paths) - len(listed)} more)")
    return "\n".join(lines)


def _normalise_reference(reference: str) -> str:
    """Reduce an import target to a comparable path fragment."""
    ref = reference.strip().strip("'\"")
    ref = re.sub(r"^package:[^/]+/", "", ref)      # dart: package:app/widgets/home.dart
    ref = ref.replace("\\", "/")                    # php: App\Http\Controller
    ref = ref.lstrip("./")                          # js:  ./utils/format
    if "/" not in ref and "." in ref:
        ref = ref.replace(".", "/")                 # python: app.services.mailer
    ref = re.sub(r"\.(js|jsx|ts|tsx|mjs|cjs|dart|php|py|vue)$", "", ref, flags=re.I)
    return ref.strip("/")


def build_relations(sources: dict[str, str]) -> dict[str, set[str]]:
    """
    Best-effort dependency graph: file -> files it imports plus files importing it.

    Only used to enrich the prompt, so an unresolved reference costs nothing.
    """
    index: dict[str, str] = {}
    for path in sources:
        without_ext = re.sub(r"\.[^./]+$", "", path)
        index[without_ext.lower()] = path
        index[PurePosixPath(without_ext).name.lower()] = path

    relations: dict[str, set[str]] = {path: set() for path in sources}
    for path, code in sources.items():
        for pattern in IMPORT_PATTERNS:
            for match in pattern.finditer(code):
                reference = _normalise_reference(match.group(1))
                if not reference:
                    continue
                target = index.get(reference.lower()) or index.get(
                    PurePosixPath(reference).name.lower()
                )
                if target and target != path:
                    relations[path].add(target)
                    relations[target].add(path)
    return relations


def build_related_context(
    bundle: ProjectBundle,
    target: str,
    relations: dict[str, set[str]],
    settings: Settings,
) -> str:
    """Attach the neighbours of `target` (truncated) so the model sees the relationships."""
    if not settings.max_context_files or not settings.max_context_chars:
        return ""

    neighbours = sorted(relations.get(target, set()))[: settings.max_context_files]
    if not neighbours:
        return ""

    budget = settings.max_context_chars
    per_file = max(1_000, budget // max(len(neighbours), 1))
    blocks: list[str] = []
    for path in neighbours:
        content = bundle.sources.get(path, "")
        if not content:
            continue
        snippet = content[:per_file]
        if len(content) > per_file:
            snippet += "\n... (truncated context)"
        blocks.append(f"--- {path} ---\n{snippet}")
        budget -= len(snippet)
        if budget <= 0:
            break
    return "\n\n".join(blocks)


def read_archive(raw: bytes, settings: Settings) -> ProjectBundle:
    """
    Extract a project archive in memory, with zip-bomb, traversal and size guards.

    Runs inside a worker thread: it is pure CPU work and must not block the event loop.
    """
    bundle = ProjectBundle(is_project=True)
    extracted = 0

    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            entries = [info for info in archive.infolist() if not info.is_dir()]
            if len(entries) > settings.max_archive_entries:
                raise JobError(
                    f"عدد ملفات المشروع {len(entries)} يتجاوز الحد {settings.max_archive_entries}."
                )

            for info in entries:
                # Symlinks are never followed nor re-shipped.
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    continue

                path = safe_archive_path(info.filename)
                if not path or is_ignored_path(path):
                    bundle.skipped += 1
                    continue

                compressed = max(info.compress_size, 1)
                if (
                    info.file_size > ZIP_BOMB_FLOOR
                    and info.file_size / compressed > ZIP_BOMB_RATIO
                ):
                    raise JobError(TEXT_ARCHIVE_BOMB)

                extracted += info.file_size
                if extracted > settings.max_extracted_size:
                    raise JobError(TEXT_ARCHIVE_TOO_LARGE)

                data = archive.read(info)
                if len(data) != info.file_size:  # declared size lied: treat as hostile
                    raise JobError(TEXT_ARCHIVE_BOMB)

                bundle.order.append(path)
                if is_supported(path) and len(data) <= settings.max_file_size:
                    try:
                        bundle.sources[path] = decode_source(data)
                        continue
                    except ValueError:
                        pass  # a binary named like a source file stays an untouched blob
                bundle.blobs[path] = data
    except JobError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError, ValueError) as exc:
        raise JobError(TEXT_BAD_ARCHIVE) from exc

    if not bundle.order:
        raise JobError(TEXT_BAD_ARCHIVE)

    bundle.project_type = detect_project_type(bundle.order)
    return bundle


def build_archive(bundle: ProjectBundle) -> bytes:
    """Rebuild the project zip, keeping the original entry order. Thread-friendly."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle.order:
            if path in bundle.sources:
                archive.writestr(path, bundle.sources[path].encode("utf-8"))
            elif path in bundle.blobs:
                archive.writestr(path, bundle.blobs[path])
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Request log (logs/)                                                          #
# --------------------------------------------------------------------------- #


class RequestLogger:
    """
    Append-only, daily-rotated record of every AI call and every job.

    Metadata only — user code is never written to disk:

        2026-07-14 22:31:07
        Event:AI
        User:123456
        Provider:Groq
        Status:Success
        Time:3.2s
        Detail:bot.py
        ------------------------------
    """

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self._directory = directory
        self._enabled = enabled
        self._lock = asyncio.Lock()
        if enabled:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                LOGGER.warning("could not create %s: %s — request log disabled", directory, exc)
                self._enabled = False

    @property
    def enabled(self) -> bool:
        """False when the directory is not writable; the bot keeps running either way."""
        return self._enabled

    async def record(
        self,
        *,
        event: str,
        user_id: int,
        provider: str,
        status: str,
        seconds: float,
        detail: str = "",
    ) -> None:
        """Append one record; a broken disk must never break a job."""
        if not self._enabled:
            return

        now = time.localtime()
        lines = [
            time.strftime("%Y-%m-%d %H:%M:%S", now),
            f"Event:{event}",
            f"User:{user_id}",
            f"Provider:{provider or '-'}",
            f"Status:{status}",
            f"Time:{max(seconds, 0.0):.1f}s",
        ]
        if detail:
            lines.append(f"Detail:{one_line(detail)}")
        lines += ["-" * 30, ""]

        path = self._directory / f"requests-{time.strftime('%Y-%m-%d', now)}.log"
        block = "\n".join(lines)
        async with self._lock:
            try:
                await asyncio.to_thread(self._append, path, block)
            except OSError as exc:
                LOGGER.warning("request log write failed: %s", exc)

    @staticmethod
    def _append(path: Path, block: str) -> None:
        """Blocking write, executed in a worker thread."""
        with path.open("a", encoding="utf-8") as handle:
            handle.write(block)


# --------------------------------------------------------------------------- #
# AI layer: providers, health, failover                                        #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Completion:
    """One successful model answer plus its accounting."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    elapsed: float
    retries: int
    provider: str = ""
    model: str = ""


@dataclass(slots=True)
class ProviderHealth:
    """Live state of one provider: circuit breaker + statistics."""

    available: bool = True
    disabled: bool = False          # bad key / URL / model: parked until a probe clears it
    checked: bool = False           # a probe or a real call has already been attempted
    consecutive_failures: int = 0
    cooldown_until: float = 0.0     # monotonic
    requests: int = 0
    successes: int = 0
    failures: int = 0
    ema_latency: float = 0.0
    last_latency: float = 0.0
    last_error: str = ""
    last_success: float = 0.0       # wall clock

    def usable(self, now: float) -> bool:
        """True when the provider may take traffic right now."""
        return not self.disabled and now >= self.cooldown_until

    def cooldown_left(self, now: float) -> float:
        """Seconds remaining before the provider re-enters the rotation."""
        return max(0.0, self.cooldown_until - now)

    def label(self, now: float) -> str:
        """Human status used by /ai_status."""
        if self.disabled:
            return TEXT_DISABLED
        if self.cooldown_until > now:
            return TEXT_COOLING
        if not self.checked:
            return TEXT_UNTESTED
        return TEXT_AVAILABLE if self.available else TEXT_UNAVAILABLE

    @property
    def success_rate(self) -> float:
        """Share of successful calls, 0..1."""
        return self.successes / self.requests if self.requests else 0.0


class HttpClient:
    """One pooled aiohttp session shared by every provider (keeps connections warm)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def session(self) -> aiohttp.ClientSession:
        """Lazily create the session inside the running loop."""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    limit = (
                        self._settings.max_concurrent_jobs
                        * self._settings.project_parallelism
                        + 8
                    )
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(
                            total=self._settings.request_timeout, connect=CONNECT_TIMEOUT
                        ),
                        connector=aiohttp.TCPConnector(
                            limit=limit, ttl_dns_cache=300, keepalive_timeout=60
                        ),
                        headers={"Accept": "application/json"},
                    )
        return self._session

    async def close(self) -> None:
        """Release every socket on shutdown."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0)  # let the connector tear its transports down
        self._session = None


class AIProvider:
    """
    Resilient async wrapper around one OpenAI-compatible `/chat/completions` endpoint.

    Retries transient failures itself; anything that means "this provider cannot serve the
    request right now" is raised as `ProviderUnavailable` so the manager can fail over.
    """

    def __init__(self, config: ProviderConfig, settings: Settings, http: HttpClient) -> None:
        self._config = config
        self._settings = settings
        self._http = http
        # Sticky: once an endpoint proves it rejects the extended body, stop sending it.
        self._minimal = False

    @property
    def config(self) -> ProviderConfig:
        """The immutable configuration of this provider."""
        return self._config

    @property
    def name(self) -> str:
        """Display name (DeepSeek / Groq)."""
        return self._config.name

    @property
    def priority(self) -> int:
        """1 = primary, 2 = fallback ..."""
        return self._config.priority

    # -- request building --------------------------------------------------- #

    def _headers(self) -> dict[str, str]:
        """OpenAI-compatible auth headers; the key never leaves this method."""
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Build the request body.

        Baseline is the exact OpenAI-compatible shape both providers accept; the optional
        fields (max_tokens / thinking) are added only while the endpoint tolerates them.
        """
        config = self._config
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if self._minimal:
            payload["temperature"] = config.temperature
            return payload

        payload["max_tokens"] = config.max_tokens
        if config.thinking:
            # Thinking mode ignores temperature/top_p by design.
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = config.reasoning_effort
        else:
            payload["temperature"] = config.temperature
        return payload

    async def _post(
        self, payload: dict[str, Any], timeout: float
    ) -> tuple[int, str, dict[str, Any] | None, str | None]:
        """One HTTP round-trip -> (status, error preview, decoded body, Retry-After)."""
        session = await self._http.session()
        async with session.post(
            self._config.api_url,
            json=payload,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT),
        ) as response:
            retry_after = response.headers.get("Retry-After")
            if response.status == 200:
                try:
                    return 200, "", await response.json(content_type=None), retry_after
                except (aiohttp.ContentTypeError, ValueError, json.JSONDecodeError):
                    return 200, "invalid json", None, retry_after
            detail = (await response.text())[:MAX_ERROR_PREVIEW].replace("\n", " ")
            return response.status, detail, None, retry_after

    # -- public API --------------------------------------------------------- #

    async def complete(self, system_prompt: str, user_prompt: str) -> Completion:
        """Send one completion request; raise ProviderUnavailable to trigger a failover."""
        settings = self._settings
        started = time.monotonic()
        attempts = settings.max_retries
        last_error = f"{self.name}: خطأ غير معروف"
        retries = 0
        timeouts = 0
        downgraded = False
        attempt = 0

        while attempt < attempts:
            attempt += 1
            retry_after: str | None = None
            try:
                status, detail, data, retry_after = await self._post(
                    self._payload(system_prompt, user_prompt), settings.request_timeout
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                timeouts += 1
                last_error = f"انتهت مهلة الاتصال بـ {self.name}."
                if timeouts >= MAX_TIMEOUTS_PER_PROVIDER:
                    raise ProviderUnavailable(last_error, cooldown=COOLDOWN_TRANSIENT) from None
            except aiohttp.ClientError as exc:
                last_error = f"خطأ شبكة مع {self.name}: {type(exc).__name__}"
            else:
                if status == 200 and data is not None:
                    return self._parse(data, time.monotonic() - started, retries)

                if status == 200:
                    last_error = f"رد غير صالح من {self.name}."
                elif status == 400 and not downgraded and not self._minimal:
                    # The endpoint refused an optional field (max_tokens / thinking).
                    # Drop them and retry once: this round is free.
                    LOGGER.warning(
                        "%s rejected the extended payload (400: %s) — retrying without "
                        "the optional fields",
                        self.name, one_line(detail, 120),
                    )
                    self._minimal = True
                    downgraded = True
                    attempt -= 1
                    continue
                elif status == RATE_LIMIT_STATUS:
                    raise ProviderUnavailable(
                        f"{self.name}: تجاوز حد الطلبات (429).",
                        rate_limited=True,
                        cooldown=self._cooldown_from(retry_after, COOLDOWN_RATE_LIMIT),
                    )
                elif status in FATAL_STATUS:
                    template = FATAL_STATUS_MESSAGES.get(status)
                    message = (
                        template.format(provider=self.name, prefix=self._config.prefix)
                        if template
                        else f"{self.name} HTTP {status} — {detail}"
                    )
                    raise ProviderUnavailable(
                        message,
                        fatal=status in DISABLING_STATUS,
                        cooldown=COOLDOWN_FATAL,
                    )
                elif status in RETRYABLE_STATUS:
                    last_error = f"{self.name} HTTP {status}"
                else:
                    raise ProviderUnavailable(
                        f"{self.name} HTTP {status} — {detail}", cooldown=COOLDOWN_TRANSIENT
                    )

            if attempt >= attempts:
                break

            retries += 1
            delay = self._backoff(attempt, retry_after)
            LOGGER.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                self.name, attempt, attempts, last_error, delay,
            )
            await asyncio.sleep(delay)

        raise ProviderUnavailable(last_error, cooldown=COOLDOWN_TRANSIENT)

    async def probe(self) -> float:
        """
        Cheap availability check; returns the round-trip time in seconds.

        A 429 means the service is alive, so it is reported as such (rate_limited).
        """
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": PROBE_MAX_TOKENS,
            "stream": False,
        }
        started = time.monotonic()
        try:
            status, detail, _, retry_after = await self._post(
                payload, float(self._settings.probe_timeout)
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise ProviderUnavailable(
                f"انتهت مهلة فحص {self.name}.", cooldown=COOLDOWN_TRANSIENT
            ) from None
        except aiohttp.ClientError as exc:
            raise ProviderUnavailable(
                f"خطأ شبكة مع {self.name}: {type(exc).__name__}", cooldown=COOLDOWN_TRANSIENT
            ) from exc

        if status == 200:
            return time.monotonic() - started
        if status == RATE_LIMIT_STATUS:
            raise ProviderUnavailable(
                f"{self.name}: تجاوز حد الطلبات (429).",
                rate_limited=True,
                cooldown=self._cooldown_from(retry_after, COOLDOWN_RATE_LIMIT),
            )
        template = FATAL_STATUS_MESSAGES.get(status)
        message = (
            template.format(provider=self.name, prefix=self._config.prefix)
            if template
            else f"{self.name} HTTP {status} — {detail}"
        )
        raise ProviderUnavailable(
            message,
            fatal=status in DISABLING_STATUS,
            cooldown=COOLDOWN_FATAL if status in FATAL_STATUS else COOLDOWN_TRANSIENT,
        )

    # -- internals ---------------------------------------------------------- #

    def _parse(self, data: Any, elapsed: float, retries: int) -> Completion:
        """Validate the envelope and refuse anything that would corrupt a file."""
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailable(
                f"رد غير متوقع من {self.name}.", cooldown=COOLDOWN_TRANSIENT
            ) from exc

        # Deterministic failures: another provider would truncate/filter it too.
        if finish_reason == "length":
            raise AIContentError(
                TEXT_TRUNCATED_ANSWER.format(provider=self.name, prefix=self._config.prefix)
            )
        if finish_reason == "content_filter":
            raise AIContentError(TEXT_FILTERED_ANSWER.format(provider=self.name))
        if not content.strip():
            raise ProviderUnavailable(
                f"رجع رد فارغ من {self.name}.", cooldown=COOLDOWN_TRANSIENT
            )

        usage = data.get("usage") or {}
        return Completion(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            elapsed=elapsed,
            retries=retries,
            provider=self.name,
            model=self._config.model,
        )

    def _cooldown_from(self, retry_after: str | None, default: float) -> float:
        """Honour `Retry-After` when the server sent one."""
        if retry_after:
            with suppress(ValueError):
                return min(max(float(retry_after), 1.0), MAX_COOLDOWN)
        return default

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        """Honour `Retry-After`, otherwise exponential backoff with jitter."""
        if retry_after:
            with suppress(ValueError):
                return min(float(retry_after), MAX_BACKOFF)
        base = self._settings.retry_delay * (2 ** (attempt - 1))
        return min(base, MAX_BACKOFF) + random.uniform(0.0, 0.5)


class AIProviderManager:
    """
    Routes every AI request across the configured providers.

    * Availability  : each provider is health-checked at startup and every
                      `AI_HEALTH_INTERVAL` seconds, so an outage is noticed before a user is.
    * Selection     : `priority` routing keeps DeepSeek first and Groq as the fallback;
                      `fastest` routing picks the healthy provider with the lowest moving
                      average latency.
    * Failover      : timeout, rate limit, API error or outage -> the next provider takes
                      over inside the same request, transparently for the user.
    * Circuit breaker: a failing provider is parked for a growing cooldown and comes back
                      automatically once a health check succeeds.
    """

    def __init__(self, settings: Settings, request_log: RequestLogger) -> None:
        self._settings = settings
        self._log = request_log
        self._http = HttpClient(settings)
        self._providers: tuple[AIProvider, ...] = tuple(
            AIProvider(config, settings, self._http) for config in settings.providers
        )
        self._health: dict[str, ProviderHealth] = {
            provider.name: ProviderHealth() for provider in self._providers
        }
        self._probe_locks: dict[str, asyncio.Lock] = {
            provider.name: asyncio.Lock() for provider in self._providers
        }
        self._current: str = self._providers[0].name
        self._monitor: asyncio.Task[None] | None = None

    # -- introspection ------------------------------------------------------ #

    @property
    def providers(self) -> tuple[AIProvider, ...]:
        """Every enabled provider, in configuration order."""
        return self._providers

    @property
    def current_provider(self) -> str:
        """Name of the provider that served the last successful request."""
        return self._current

    def model_of(self, name: str) -> str:
        """Model configured for a provider name."""
        for provider in self._providers:
            if provider.name == name:
                return provider.config.model
        return ""

    def average_latency(self) -> float:
        """Mean response time across the providers that already answered."""
        samples = [
            health.ema_latency
            for health in self._health.values()
            if health.successes and health.ema_latency
        ]
        return sum(samples) / len(samples) if samples else 0.0

    def chunk_budget(self) -> int:
        """
        Characters one answer may safely contain.

        Derived from the *weakest* provider so a chunk built for DeepSeek can still be
        completed by Groq after a mid-job failover.
        """
        budgets = [
            int(
                provider.config.max_tokens
                * CHARS_PER_TOKEN
                * (0.5 if provider.config.thinking else 0.7)
            )
            for provider in self._providers
        ]
        return max(MIN_CHUNK_BUDGET, min(budgets) if budgets else MIN_CHUNK_BUDGET)

    # -- routing ------------------------------------------------------------ #

    def _order(self) -> list[AIProvider]:
        """Healthy providers, best first."""
        now = time.monotonic()
        usable = [
            provider
            for provider in self._providers
            if self._health[provider.name].usable(now)
        ]
        if self._settings.routing == ROUTING_FASTEST:
            usable.sort(
                key=lambda provider: (
                    self._health[provider.name].ema_latency or DEFAULT_LATENCY,
                    provider.priority,
                )
            )
        else:
            usable.sort(key=lambda provider: provider.priority)
        return usable

    def _forced_order(self, tried: set[str]) -> list[AIProvider]:
        """Last resort: ignore the cooldowns rather than fail the job outright."""
        return [
            provider
            for provider in sorted(self._providers, key=lambda item: item.priority)
            if provider.name not in tried and not self._health[provider.name].disabled
        ]

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        user_id: int = 0,
        detail: str = "",
        notify: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> Completion:
        """
        Answer with the first provider that can, switching automatically on failure.

        `notify(failed, next)` is awaited before each switch so the UI can show it.
        """
        reasons: list[str] = []
        tried: set[str] = set()

        for stage in (0, 1):
            candidates = self._order() if stage == 0 else self._forced_order(tried)
            if stage == 1 and candidates:
                LOGGER.warning("every provider is cooling down — forcing a last attempt")

            for index, provider in enumerate(candidates):
                tried.add(provider.name)
                started = time.monotonic()
                try:
                    completion = await provider.complete(system_prompt, user_prompt)
                except asyncio.CancelledError:
                    raise
                except AIContentError as exc:
                    # The answer is unusable but the provider is fine: no failover.
                    self._record_success(provider, time.monotonic() - started, count=False)
                    await self._log.record(
                        event=LOG_EVENT_AI,
                        user_id=user_id,
                        provider=provider.name,
                        status=LOG_STATUS[STATUS_FAILED],
                        seconds=time.monotonic() - started,
                        detail=f"{detail} | {exc}" if detail else str(exc),
                    )
                    raise
                except ProviderUnavailable as exc:
                    elapsed = time.monotonic() - started
                    self._record_failure(provider, exc)
                    reasons.append(f"{provider.name}: {exc}")
                    LOGGER.warning("provider %s unavailable: %s", provider.name, exc)
                    await self._log.record(
                        event=LOG_EVENT_AI,
                        user_id=user_id,
                        provider=provider.name,
                        status=LOG_STATUS[STATUS_FAILED],
                        seconds=elapsed,
                        detail=f"{detail} | {exc}" if detail else str(exc),
                    )
                    following = self._next_name(candidates, index, tried)
                    if notify and following:
                        with suppress(Exception):
                            await notify(provider.name, following)
                    continue

                self._record_success(provider, completion.elapsed)
                await self._log.record(
                    event=LOG_EVENT_AI,
                    user_id=user_id,
                    provider=provider.name,
                    status=LOG_STATUS[STATUS_SUCCESS],
                    seconds=completion.elapsed,
                    detail=detail,
                )
                return completion

        raise AIError(
            TEXT_ALL_PROVIDERS_DOWN.format(
                reasons="\n".join(f"- {reason}" for reason in reasons) or "- سبب غير معروف"
            )
        )

    @staticmethod
    def _next_name(candidates: Sequence[AIProvider], index: int, tried: set[str]) -> str:
        """Name of the provider that will be attempted after `index`, if any."""
        for provider in candidates[index + 1 :]:
            if provider.name not in tried:
                return provider.name
        return ""

    # -- health ------------------------------------------------------------- #

    def _record_success(self, provider: AIProvider, elapsed: float, *, count: bool = True) -> None:
        """A working provider clears its breaker and becomes the current one."""
        health = self._health[provider.name]
        health.available = True
        health.disabled = False
        health.checked = True
        health.consecutive_failures = 0
        health.cooldown_until = 0.0
        health.last_error = ""
        if count:
            health.requests += 1
            health.successes += 1
            health.last_latency = elapsed
            health.last_success = time.time()
            health.ema_latency = (
                elapsed
                if not health.ema_latency
                else EMA_ALPHA * elapsed + (1 - EMA_ALPHA) * health.ema_latency
            )
        self._current = provider.name

    def _record_failure(self, provider: AIProvider, error: ProviderUnavailable) -> None:
        """Open the circuit breaker with an exponentially growing cooldown."""
        health = self._health[provider.name]
        health.requests += 1
        health.failures += 1
        health.consecutive_failures += 1
        health.available = False
        health.checked = True
        health.disabled = error.fatal
        health.last_error = one_line(str(error), 120)
        cooldown = min(
            error.cooldown * (2 ** (health.consecutive_failures - 1)), MAX_COOLDOWN
        )
        health.cooldown_until = time.monotonic() + cooldown

    def _mark_alive(self, provider: AIProvider, latency: float) -> None:
        """A successful probe re-opens the circuit without touching the call statistics."""
        health = self._health[provider.name]
        health.available = True
        health.disabled = False
        health.checked = True
        health.consecutive_failures = 0
        health.cooldown_until = 0.0
        health.last_error = ""
        health.last_latency = latency

    async def probe(self, provider: AIProvider) -> bool:
        """Health-check one provider; a rate-limited provider still counts as alive."""
        async with self._probe_locks[provider.name]:
            try:
                latency = await provider.probe()
            except asyncio.CancelledError:
                raise
            except ProviderUnavailable as exc:
                if exc.rate_limited:
                    self._mark_alive(provider, 0.0)
                    self._health[provider.name].last_error = one_line(str(exc), 120)
                    return True
                self._record_failure(provider, exc)
                LOGGER.warning("health check failed for %s: %s", provider.name, exc)
                return False
            except Exception as exc:  # noqa: BLE001 - a probe must never crash the bot
                self._record_failure(provider, ProviderUnavailable(str(exc)))
                LOGGER.exception("health check crashed for %s", provider.name)
                return False

            self._mark_alive(provider, latency)
            LOGGER.info("health check ok: %s (%.2fs)", provider.name, latency)
            return True

    async def probe_all(self) -> None:
        """Health-check every provider concurrently."""
        await asyncio.gather(
            *(self.probe(provider) for provider in self._providers), return_exceptions=True
        )
        healthy = self._order()
        if healthy:
            # Keep "current" meaningful even before the first user request.
            if self._current not in {provider.name for provider in healthy}:
                self._current = healthy[0].name
        else:
            LOGGER.error("no AI provider is currently available")

    def start_monitor(self) -> None:
        """Spawn the background health monitor (auto-recovery)."""
        if self._settings.health_interval <= 0 or self._monitor is not None:
            return
        self._monitor = asyncio.create_task(self._monitor_loop(), name="ai-health-monitor")

    async def _monitor_loop(self) -> None:
        """Re-check every provider periodically so a recovered API is used again."""
        interval = self._settings.health_interval
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.probe_all()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - the monitor must never die
                    LOGGER.exception("health monitor iteration failed")
        except asyncio.CancelledError:
            pass

    # -- reporting ---------------------------------------------------------- #

    def render_status(self) -> str:
        """The /ai_status dashboard."""
        now = time.monotonic()
        lines = ["<b>🤖 AI Status</b>", ""]

        for provider in self._providers:
            health = self._health[provider.name]
            config = provider.config
            lines.append(f"<b>{escape(provider.name)}</b>")
            lines.append(health.label(now))
            lines.append(f"الموديل: <code>{escape(config.model)}</code>")
            lines.append(
                f"الطلبات: {health.requests} | نجاح: {health.successes} | فشل: {health.failures}"
            )
            if health.successes:
                lines.append(
                    f"متوسط الاستجابة: {health.ema_latency:.1f}s "
                    f"(آخر طلب {health.last_latency:.1f}s)"
                )
            if health.cooldown_until > now:
                lines.append(f"يعود بعد: {human_duration(health.cooldown_left(now))}")
            if health.last_error:
                lines.append(f"آخر خطأ: <i>{escape(health.last_error)}</i>")
            lines.append("")

        average = self.average_latency()
        healthy = self._order()
        current = self._current if healthy else "-"
        lines.append("<b>Current Provider:</b>")
        lines.append(escape(current))
        lines.append("")
        lines.append("<b>Average Response:</b>")
        lines.append(f"{average:.1f} seconds" if average else "لا توجد بيانات بعد")
        lines.append("")
        lines.append(
            f"<b>Routing:</b> {escape(self._settings.routing)} | "
            f"<b>Health check:</b> {self._settings.health_interval}s"
        )
        return "\n".join(lines)

    # -- lifecycle ---------------------------------------------------------- #

    async def close(self) -> None:
        """Stop the monitor and release every socket."""
        if self._monitor is not None:
            self._monitor.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor
            self._monitor = None
        await self._http.close()


# --------------------------------------------------------------------------- #
# History                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HistoryEntry:
    """One finished operation, shown by the "سجل العمليات" button."""

    file_name: str
    mode: str
    started_at: float
    duration: float
    status: str
    detail: str = ""
    provider: str = ""


class HistoryStore:
    """Bounded in-memory history: N operations per user, LRU-capped users."""

    def __init__(self, per_user: int) -> None:
        self._per_user = per_user
        self._entries: dict[int, deque[HistoryEntry]] = {}

    def add(self, user_id: int, entry: HistoryEntry) -> None:
        """Record one finished operation, evicting the oldest user when full."""
        bucket = self._entries.get(user_id)
        if bucket is None:
            if len(self._entries) >= MAX_HISTORY_USERS:
                self._entries.pop(next(iter(self._entries)), None)
            bucket = deque(maxlen=self._per_user)
            self._entries[user_id] = bucket
        bucket.appendleft(entry)

    def get(self, user_id: int) -> list[HistoryEntry]:
        """Return the user's operations, newest first."""
        return list(self._entries.get(user_id, ()))

    def render(self, user_id: int) -> str:
        """Format the user's history as an HTML message."""
        entries = self.get(user_id)
        if not entries:
            return TEXT_EMPTY_HISTORY

        icons = {STATUS_SUCCESS: "✅", STATUS_FAILED: "❌", STATUS_CANCELLED: "⛔"}
        lines = [TEXT_HISTORY_TITLE.format(count=len(entries)), ""]
        for index, entry in enumerate(entries, start=1):
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.started_at))
            lines.append(
                f"{index}. {icons.get(entry.status, '•')} <b>{escape(entry.file_name)}</b>"
                f" — {MODE_TITLES.get(entry.mode, entry.mode)}"
            )
            summary = f"    {stamp} • {human_duration(entry.duration)} • {entry.status}"
            if entry.provider:
                summary += f" • {escape(entry.provider)}"
            lines.append(summary)
            if entry.detail:
                lines.append(f"    <i>{escape(entry.detail[:120])}</i>")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Keyboards                                                                    #
# --------------------------------------------------------------------------- #


class ActionCB(CallbackData, prefix="act"):
    """Main menu callback."""

    mode: str


class CancelCB(CallbackData, prefix="cancel"):
    """Cancel button attached to a running/queued job."""

    job_id: int


class StatusCB(CallbackData, prefix="aist"):
    """Refresh button of the /ai_status dashboard."""

    action: str


def main_menu() -> InlineKeyboardMarkup:
    """The only menu of the bot."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=TEXT_BTN_FIX, callback_data=ActionCB(mode=MODE_FIX).pack()
                ),
                InlineKeyboardButton(
                    text=TEXT_BTN_UPDATE, callback_data=ActionCB(mode=MODE_UPDATE).pack()
                ),
            ],
            [
                InlineKeyboardButton(
                    text=TEXT_BTN_HISTORY, callback_data=ActionCB(mode="history").pack()
                )
            ],
        ]
    )


def cancel_menu(job_id: int) -> InlineKeyboardMarkup:
    """Keyboard shown while a job is queued or running."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=TEXT_BTN_CANCEL, callback_data=CancelCB(job_id=job_id).pack()
                )
            ]
        ]
    )


def status_menu() -> InlineKeyboardMarkup:
    """Keyboard of the admin AI dashboard."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=TEXT_BTN_REFRESH, callback_data=StatusCB(action="refresh").pack()
                )
            ]
        ]
    )


# --------------------------------------------------------------------------- #
# Progress reporting                                                           #
# --------------------------------------------------------------------------- #


class ProgressReporter:
    """
    Owns the single status message of a job and edits it in place.

    Edits are serialised and rate-limited so Telegram is never flooded; terminal
    renders are forced so the last state is always visible.
    """

    def __init__(self, message: Message, title: str, job_id: int) -> None:
        self._message = message
        self._title = title
        self._job_id = job_id
        self._stage = STAGE_RECEIVE
        self._detail = ""
        self._started = time.monotonic()
        self._last_edit = 0.0
        self._last_text = ""
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def message(self) -> Message:
        """The status message this reporter owns."""
        return self._message

    def close(self) -> None:
        """Freeze the message: only terminal renders may still write to it."""
        self._closed = True

    def _body(self) -> str:
        """Render the checklist with the current stage highlighted."""
        lines = [f"<b>{escape(self._title)}</b>", ""]
        for index, label in enumerate(STAGE_LABELS):
            if index < self._stage:
                marker = "✔"
            elif index == self._stage:
                marker = "⏳"
            else:
                marker = "▫️"
            lines.append(f"{marker} {label}")
        if self._detail:
            lines += ["", escape(self._detail)]
        lines += ["", f"⏱ {human_duration(time.monotonic() - self._started)}"]
        return "\n".join(lines)

    async def _render(
        self,
        text: str,
        keyboard: InlineKeyboardMarkup | None,
        *,
        force: bool = False,
        terminal: bool = False,
    ) -> None:
        """Edit the status message, respecting the rate limit and the closed flag."""
        if self._closed and not terminal:
            return

        async with self._lock:
            # Frequent updates (detail lines) are dropped when they come too fast;
            # forced ones (stage changes, terminal states) always land — a 429 is
            # rare and handled below, and sleeping here would delay the job itself.
            if not force and time.monotonic() - self._last_edit < MIN_EDIT_INTERVAL:
                return
            if text == self._last_text:
                return

            try:
                await self._message.edit_text(text, reply_markup=keyboard)
            except TelegramRetryAfter as exc:
                await asyncio.sleep(min(float(exc.retry_after), MAX_BACKOFF))
                with suppress(TelegramAPIError):
                    await self._message.edit_text(text, reply_markup=keyboard)
            except TelegramBadRequest:
                pass  # identical text, or the message was removed by the user
            except TelegramAPIError as exc:
                LOGGER.debug("progress edit failed for job %s: %s", self._job_id, exc)

            self._last_text = text
            self._last_edit = time.monotonic()

    async def queued(self, position: int, eta: float, file_name: str) -> None:
        """Show the queue position and the estimated wait."""
        text = TEXT_QUEUED.format(
            file=escape(file_name), position=position, eta=human_duration(eta)
        )
        await self._render(text, cancel_menu(self._job_id), force=True)

    async def stage(self, stage: int, detail: str = "", *, force: bool = False) -> None:
        """Advance the checklist to `stage`."""
        self._stage = stage
        self._detail = detail
        await self._render(self._body(), cancel_menu(self._job_id), force=force)

    async def detail(self, detail: str, *, force: bool = False) -> None:
        """Update only the detail line (files x/y, part i/n, provider switch)."""
        self._detail = detail
        await self._render(self._body(), cancel_menu(self._job_id), force=force)

    async def complete(self) -> None:
        """Terminal render: every stage done, no keyboard."""
        self._stage = len(STAGE_LABELS)
        self._detail = ""
        await self._render(self._body(), None, force=True, terminal=True)
        self.close()

    async def fail(self, reason: str) -> None:
        """Terminal render: failure + the main menu."""
        await self._render(
            TEXT_FAILED.format(reason=escape(reason)), main_menu(), force=True, terminal=True
        )
        self.close()

    async def cancelled(self) -> None:
        """Terminal render: cancellation + the main menu."""
        await self._render(f"⛔ {TEXT_CANCELLED}", main_menu(), force=True, terminal=True)
        self.close()


# --------------------------------------------------------------------------- #
# Jobs and queue                                                               #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Job:
    """One user request travelling through the queue."""

    job_id: int
    user_id: int
    user_name: str
    chat_id: int
    mode: str
    instruction: str
    file_name: str
    raw: bytes
    archive: bool
    reporter: ProgressReporter
    created_at: float = field(default_factory=time.monotonic)
    started_at: float = 0.0
    cancelled: bool = False
    task: "asyncio.Task[None] | None" = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    providers: list[str] = field(default_factory=list)

    @property
    def providers_label(self) -> str:
        """Every provider that contributed to this job, in order of first use."""
        return " + ".join(self.providers)

    def release(self) -> None:
        """Free the uploaded bytes as soon as they are no longer needed."""
        self.raw = b""


class JobQueue:
    """
    FIFO queue with a fixed worker pool.

    Enforces one active job per user, reports the queue position and an ETA derived from
    an exponential moving average of the previous durations, and supports cancellation of
    both queued and running jobs.
    """

    def __init__(self, workers: int, processor: "Processor", history: HistoryStore) -> None:
        self._workers = workers
        self._processor = processor
        self._history = history
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._pending: list[Job] = []
        self._running: dict[int, Job] = {}
        self._by_user: dict[int, Job] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._average = DEFAULT_JOB_SECONDS
        self._ids = itertools.count(1)

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Spawn the workers (must run inside the event loop)."""
        self._tasks = [
            asyncio.create_task(self._worker(index), name=f"worker-{index}")
            for index in range(self._workers)
        ]

    async def stop(self) -> None:
        """Cancel running jobs and workers, then wait for them to unwind."""
        for job in list(self._running.values()):
            job.cancelled = True
            if job.task and not job.task.done():
                job.task.cancel()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # -- public API --------------------------------------------------------- #

    def next_id(self) -> int:
        """Allocate the identifier carried by the cancel button."""
        return next(self._ids)

    def has_active(self, user_id: int) -> bool:
        """True while the user already has a queued or running job."""
        return user_id in self._by_user

    def active_job(self, user_id: int) -> Job | None:
        """The user's queued or running job, if any."""
        return self._by_user.get(user_id)

    def position(self, job: Job) -> int:
        """1-based position in the waiting list (0 when it is already running)."""
        try:
            return self._pending.index(job) + 1
        except ValueError:
            return 0

    def eta(self, position: int) -> float:
        """Estimated wait before the job starts, from the moving average."""
        ahead = max(position - 1, 0) + len(self._running)
        waves = math.ceil(ahead / self._workers) if ahead else 0
        return waves * self._average

    def is_immediate(self, position: int) -> bool:
        """True when a worker is free right now."""
        return max(position - 1, 0) + len(self._running) < self._workers

    async def submit(self, job: Job) -> int:
        """Register the job and hand it to the workers. Returns its queue position."""
        self._by_user[job.user_id] = job
        self._pending.append(job)
        await self._queue.put(job)
        return self.position(job)

    async def cancel(self, job_id: int, user_id: int) -> bool:
        """
        Abort a job owned by `user_id`, whether it is queued or running.

        The AI request is aborted through task cancellation, the job memory is released,
        and the status message becomes the cancellation notice.
        """
        job = self._by_user.get(user_id)
        if job is None or job.job_id != job_id or job.cancelled:
            return False

        job.cancelled = True
        self._by_user.pop(user_id, None)
        if job in self._pending:
            self._pending.remove(job)
            self._history.add(
                user_id,
                HistoryEntry(
                    file_name=job.file_name,
                    mode=job.mode,
                    started_at=time.time(),
                    duration=0.0,
                    status=STATUS_CANCELLED,
                ),
            )
        if job.task and not job.task.done():
            job.task.cancel()
        job.release()

        await job.reporter.cancelled()
        await self._refresh_pending()
        LOGGER.info("job=%s cancelled by user=%s", job.job_id, user_id)
        return True

    # -- internals ---------------------------------------------------------- #

    def _record_duration(self, seconds: float) -> None:
        """Blend the finished duration into the moving average used for the ETA."""
        self._average = EMA_ALPHA * seconds + (1 - EMA_ALPHA) * self._average

    async def _refresh_pending(self) -> None:
        """Repaint the queue position of everybody still waiting."""
        for index, job in enumerate(list(self._pending), start=1):
            if job.cancelled:
                continue
            with suppress(TelegramAPIError):
                await job.reporter.queued(index, self.eta(index), job.file_name)

    async def _worker(self, index: int) -> None:
        """Pull jobs forever; each job runs as its own cancellable task."""
        while True:
            job = await self._queue.get()
            try:
                if job.cancelled:
                    continue
                if job in self._pending:
                    self._pending.remove(job)

                self._running[job.job_id] = job
                job.started_at = time.monotonic()
                LOGGER.info(
                    "job=%s started on worker %d after %s in queue",
                    job.job_id, index, human_duration(job.started_at - job.created_at),
                )
                job.task = asyncio.create_task(
                    self._processor.run(job), name=f"job-{job.job_id}"
                )
                try:
                    await job.task
                except asyncio.CancelledError:
                    if not job.cancelled:
                        raise  # the bot itself is shutting down
                except Exception:  # noqa: BLE001 - a job must never kill its worker
                    LOGGER.exception("worker %d crashed on job %s", index, job.job_id)
            finally:
                if job.task and not job.task.done():
                    job.task.cancel()
                self._running.pop(job.job_id, None)
                if self._by_user.get(job.user_id) is job:
                    self._by_user.pop(job.user_id, None)
                if job.started_at and not job.cancelled:
                    self._record_duration(time.monotonic() - job.started_at)
                job.release()
                self._queue.task_done()
                with suppress(Exception):
                    await self._refresh_pending()


# --------------------------------------------------------------------------- #
# Processor: the pipeline that turns a job into a delivered file               #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FileOutcome:
    """Result of processing one source file."""

    path: str
    content: str
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    parts: int = 1

    @property
    def ok(self) -> bool:
        """True when the file was rewritten and passed validation."""
        return not self.error


class Processor:
    """Runs the seven stages of a job and delivers the final artefact."""

    def __init__(
        self,
        bot: Bot,
        settings: Settings,
        ai: AIProviderManager,
        history: HistoryStore,
        request_log: RequestLogger,
    ) -> None:
        self._bot = bot
        self._settings = settings
        self._ai = ai
        self._history = history
        self._request_log = request_log

    # -- entry point -------------------------------------------------------- #

    async def run(self, job: Job) -> None:
        """Execute one job end to end; never raises except on cancellation."""
        started = time.monotonic()
        status = STATUS_SUCCESS
        detail = ""
        bundle: ProjectBundle | None = None

        try:
            await job.reporter.stage(STAGE_RECEIVE)
            await job.reporter.stage(STAGE_READ)
            bundle = await self._read(job)

            await job.reporter.stage(STAGE_ANALYZE)
            targets, relations, project_map = self._analyse(bundle)

            await job.reporter.stage(STAGE_REQUEST, force=True)
            outcomes = await self._process(job, bundle, targets, relations, project_map)

            await job.reporter.stage(STAGE_SEND, force=True)
            await self._deliver(job, bundle, outcomes, time.monotonic() - started)
            await job.reporter.complete()

            failed = [item for item in outcomes if not item.ok]
            if failed:
                detail = f"فشل {len(failed)} من {len(outcomes)} ملف"
        except asyncio.CancelledError:
            status = STATUS_CANCELLED
            raise
        except JobError as exc:
            status, detail = STATUS_FAILED, str(exc)
            LOGGER.warning("job=%s user=%s failed: %s", job.job_id, job.user_id, exc)
            with suppress(TelegramAPIError):
                await job.reporter.fail(str(exc))
        except TelegramAPIError as exc:
            status, detail = STATUS_FAILED, f"Telegram: {type(exc).__name__}"
            LOGGER.error("job=%s telegram failure: %s", job.job_id, exc)
        except Exception:  # noqa: BLE001 - the queue must survive any bug
            status, detail = STATUS_FAILED, TEXT_UNEXPECTED
            LOGGER.exception("job=%s unexpected failure", job.job_id)
            with suppress(TelegramAPIError):
                await job.reporter.fail(TEXT_UNEXPECTED)
        finally:
            duration = time.monotonic() - started
            if bundle is not None:
                bundle.release()
            job.release()
            self._history.add(
                job.user_id,
                HistoryEntry(
                    file_name=job.file_name,
                    mode=job.mode,
                    started_at=time.time() - duration,
                    duration=duration,
                    status=status,
                    detail=detail,
                    provider=job.providers_label,
                ),
            )
            LOGGER.info(
                "job=%s user=%s(%s) mode=%s file=%s provider=%s status=%s duration=%.1fs "
                "tokens_in=%d tokens_out=%d retries=%d",
                job.job_id, job.user_name, job.user_id, job.mode, job.file_name,
                job.providers_label or "-", status, duration,
                job.prompt_tokens, job.completion_tokens, job.retries,
            )
            # Shielded: a cancelled job must still leave its trace in logs/.
            with suppress(Exception):
                await asyncio.shield(
                    self._request_log.record(
                        event=LOG_EVENT_JOB,
                        user_id=job.user_id,
                        provider=job.providers_label,
                        status=LOG_STATUS.get(status, status),
                        seconds=duration,
                        detail=f"{job.file_name} — {MODE_TITLES.get(job.mode, job.mode)}",
                    )
                )

    # -- stages ------------------------------------------------------------- #

    async def _read(self, job: Job) -> ProjectBundle:
        """📖 Decode a single file, or extract a project archive in a worker thread."""
        raw = job.raw
        if not raw:
            raise JobError(TEXT_EMPTY_FILE)

        if job.archive:
            bundle = await asyncio.to_thread(read_archive, raw, self._settings)
        else:
            try:
                code = decode_source(raw)
            except ValueError as exc:
                raise JobError(TEXT_NOT_TEXT) from exc
            if not code.strip():
                raise JobError(TEXT_EMPTY_FILE)
            bundle = ProjectBundle(
                sources={job.file_name: code},
                order=[job.file_name],
                project_type=resolve_language(job.file_name),
            )
        job.release()  # the raw upload is no longer needed
        return bundle

    def _analyse(
        self, bundle: ProjectBundle
    ) -> tuple[list[str], dict[str, set[str]], str]:
        """🔍 Pick the target files and map the relationships between them."""
        targets = [path for path, code in bundle.sources.items() if code.strip()]
        if not targets:
            raise JobError(TEXT_NO_SUPPORTED_FILES if bundle.is_project else TEXT_EMPTY_FILE)

        if bundle.is_project and len(targets) > self._settings.max_project_files:
            raise JobError(
                TEXT_TOO_MANY_FILES.format(
                    count=len(targets), limit=self._settings.max_project_files
                )
            )

        targets.sort()
        if not bundle.is_project:
            return targets, {}, ""

        relations = build_relations(bundle.sources)
        project_map = build_project_map(bundle.order)
        return targets, relations, project_map

    async def _process(
        self,
        job: Job,
        bundle: ProjectBundle,
        targets: Sequence[str],
        relations: dict[str, set[str]],
        project_map: str,
    ) -> list[FileOutcome]:
        """🤖/⚙️/✅ Send every target to the AI, then validate what comes back."""
        total = len(targets)
        done = 0
        lock = asyncio.Lock()

        async def worker(path: str) -> FileOutcome:
            """Process one target file and advance the shared counter."""
            nonlocal done
            outcome = await self._process_file(
                job, bundle, path, relations, project_map, total
            )
            async with lock:
                done += 1
                if total > 1:
                    await job.reporter.detail(f"الملفات: {done}/{total}")
            return outcome

        if total == 1:
            outcomes = [await worker(targets[0])]
        else:
            await job.reporter.detail(f"الملفات: 0/{total}")
            semaphore = asyncio.Semaphore(self._settings.project_parallelism)

            async def guarded(path: str) -> FileOutcome:
                """Bound how many files of one project hit the AI at once."""
                async with semaphore:
                    return await worker(path)

            outcomes = list(await asyncio.gather(*(guarded(path) for path in targets)))

        await job.reporter.stage(STAGE_PROCESS)
        for outcome in outcomes:
            if outcome.ok:
                bundle.sources[outcome.path] = outcome.content

        await job.reporter.stage(STAGE_VALIDATE)
        if not bundle.is_project and outcomes and not outcomes[0].ok:
            raise JobError(outcomes[0].error)
        if all(not outcome.ok for outcome in outcomes):
            raise JobError(outcomes[0].error if outcomes else TEXT_UNEXPECTED)
        return outcomes

    async def _process_file(
        self,
        job: Job,
        bundle: ProjectBundle,
        path: str,
        relations: dict[str, set[str]],
        project_map: str,
        total_files: int,
    ) -> FileOutcome:
        """Repair or update a single file, chunking it when it is too large."""
        original = bundle.sources[path]
        related = (
            build_related_context(bundle, path, relations, self._settings)
            if bundle.is_project
            else ""
        )
        project_type = bundle.project_type if bundle.is_project else ""

        chunks = split_source(original, self._ai.chunk_budget())
        parts = len(chunks)

        try:
            if parts > self._settings.max_chunks:
                raise JobError(
                    TEXT_TOO_MANY_CHUNKS.format(parts=parts, limit=self._settings.max_chunks)
                )
            if parts == 1:
                produced = await self._single_pass(
                    job, path, original, project_type, project_map, related
                )
            else:
                produced = await self._chunked_pass(
                    job, path, original, chunks, project_type, project_map, related, total_files
                )
        except asyncio.CancelledError:
            raise
        except JobError as exc:
            LOGGER.warning("job=%s file=%s failed: %s", job.job_id, path, exc)
            return FileOutcome(path=path, content=original, error=str(exc), parts=parts)

        errors, warnings = await asyncio.to_thread(
            validate_result, original, produced, path, job.mode
        )
        if errors:
            LOGGER.warning("job=%s file=%s rejected: %s", job.job_id, path, "; ".join(errors))
            return FileOutcome(
                path=path, content=original, error="؛ ".join(errors), parts=parts
            )
        return FileOutcome(path=path, content=produced, warnings=warnings, parts=parts)

    async def _ask(self, job: Job, system_prompt: str, user_prompt: str, detail: str) -> str:
        """
        One AI round-trip, with its accounting folded into the job.

        The provider is chosen — and silently replaced on failure — by the manager; the
        user only sees a short notice when a switch happens.
        """

        async def notify(failed: str, following: str) -> None:
            """Surface the failover inside the live progress message."""
            await job.reporter.detail(
                TEXT_PROVIDER_SWITCH.format(failed=failed, next=following), force=True
            )

        completion = await self._ai.complete(
            system_prompt,
            user_prompt,
            user_id=job.user_id,
            detail=detail,
            notify=notify,
        )
        job.prompt_tokens += completion.prompt_tokens
        job.completion_tokens += completion.completion_tokens
        job.retries += completion.retries
        if completion.provider and completion.provider not in job.providers:
            job.providers.append(completion.provider)
        return completion.content

    async def _single_pass(
        self,
        job: Job,
        path: str,
        original: str,
        project_type: str,
        project_map: str,
        related: str,
    ) -> str:
        """Whole-file repair, with automatic repair rounds when validation fails."""
        system_prompt = build_system_prompt(job.mode, path)
        user_prompt = build_user_prompt(
            job.mode,
            job.instruction,
            path,
            original,
            project_type=project_type,
            project_map=project_map,
            related=related,
        )

        issues: list[str] = []
        produced = ""
        for attempt in range(self._settings.validation_retries + 1):
            prompt = system_prompt
            if issues:
                prompt += REPAIR_SYSTEM_SUFFIX.format(issues="\n".join(f"- {i}" for i in issues))
            raw = await self._ask(job, prompt, user_prompt, path)
            produced = unwrap_code(raw, path)

            issues, _ = await asyncio.to_thread(
                validate_result, original, produced, path, job.mode
            )
            if not issues:
                return produced
            LOGGER.warning(
                "job=%s file=%s validation failed (attempt %d): %s",
                job.job_id, path, attempt + 1, "; ".join(issues),
            )
        return produced

    async def _chunked_pass(
        self,
        job: Job,
        path: str,
        original: str,
        chunks: list[str],
        project_type: str,
        project_map: str,
        related: str,
        total_files: int,
    ) -> str:
        """Repair a large file part by part, then merge the parts back into one file."""
        total = len(chunks)
        produced: list[str] = []

        for index, chunk in enumerate(chunks, start=1):
            if total_files == 1:
                await job.reporter.detail(
                    TEXT_CHUNK_PROGRESS.format(index=index, total=total), force=True
                )

            before = ""
            if produced:  # the tail the model just wrote, so both parts stay coherent
                before = "\n".join(produced[-1].splitlines()[-CHUNK_CONTEXT_LINES:])
            after = ""
            if index < total:
                after = "\n".join(chunks[index].splitlines()[:CHUNK_CONTEXT_LINES])

            system_prompt = build_chunk_system_prompt(job.mode, path, index, total)
            user_prompt = build_user_prompt(
                job.mode,
                job.instruction,
                path,
                chunk,
                project_type=project_type,
                project_map=project_map,
                related=related if index == 1 else "",
                chunk_before=before,
                chunk_after=after,
            )
            raw = await self._ask(job, system_prompt, user_prompt, f"{path} [{index}/{total}]")
            part = unwrap_code(raw, path, keep_indent=True)

            # A part must never come back shorter than a fraction of what was sent.
            if len(part.strip()) < len(chunk.strip()) * MIN_LENGTH_RATIO:
                raise JobError(TEXT_CHUNK_TRUNCATED.format(index=index, total=total))

            # Re-apply the original blank lines around the part so the merge is lossless.
            produced.append(restore_chunk_edges(part, chunk))

        return "".join(produced)

    # -- delivery ----------------------------------------------------------- #

    async def _deliver(
        self,
        job: Job,
        bundle: ProjectBundle,
        outcomes: Sequence[FileOutcome],
        elapsed: float,
    ) -> None:
        """📤 Build the artefact and send it back with the summary caption."""
        if bundle.is_project:
            payload = await asyncio.to_thread(build_archive, bundle)
        else:
            payload = bundle.sources[outcomes[0].path].encode("utf-8")

        caption = self._caption(job, bundle, outcomes, payload, elapsed)
        document = BufferedInputFile(payload, filename=job.file_name)

        async with ChatActionSender.upload_document(bot=self._bot, chat_id=job.chat_id):
            await self._bot.send_document(
                chat_id=job.chat_id,
                document=document,
                caption=caption,
                reply_markup=main_menu(),
            )

    def _caption(
        self,
        job: Job,
        bundle: ProjectBundle,
        outcomes: Sequence[FileOutcome],
        payload: bytes,
        elapsed: float,
    ) -> str:
        """Summarise the operation: provider, model, size, duration, tokens, warnings."""
        done = TEXT_FIX_DONE if job.mode == MODE_FIX else TEXT_UPDATE_DONE
        lines = [done, ""]

        if job.providers:
            labels = [
                f"{name} ({self._ai.model_of(name)})".strip()
                for name in job.providers
            ]
            lines.append(f"المزود: {escape(' + '.join(labels))}")

        if bundle.is_project:
            failed = [item for item in outcomes if not item.ok]
            lines.append(f"المشروع: {escape(bundle.project_type)}")
            lines.append(
                f"الملفات: {len(outcomes) - len(failed)} نجحت / {len(failed)} فشلت"
            )
        chunked = [item for item in outcomes if item.parts > 1]
        if chunked:
            lines.append(f"أجزاء: {sum(item.parts for item in chunked)}")

        lines.append(f"الحجم: {human_size(len(payload))}")
        lines.append(f"المدة: {human_duration(elapsed)}")
        lines.append(
            f"Tokens: {job.prompt_tokens:,} ⇦ / {job.completion_tokens:,} ⇨"
        )

        warnings = [
            f"{item.path}: {warning}" for item in outcomes for warning in item.warnings
        ][:3]
        if warnings:
            lines += ["", "⚠️ تنبيهات:"] + [f"- {escape(item)}" for item in warnings]

        failures = [f"{item.path}: {item.error}" for item in outcomes if not item.ok][:3]
        if failures:
            lines += ["", "❌ ملفات لم تُعدّل (بقيت كما هي):"] + [
                f"- {escape(item)}" for item in failures
            ]

        if bundle.skipped:
            lines += ["", f"تم استثناء {bundle.skipped} مسار (vendor / build / git)."]

        caption = "\n".join(lines)
        return caption[:1_020]  # Telegram caption hard limit


# --------------------------------------------------------------------------- #
# Middlewares                                                                  #
# --------------------------------------------------------------------------- #


class ThrottlingMiddleware(BaseMiddleware):
    """Drop updates coming faster than the configured per-user cooldown (anti-flood)."""

    def __init__(self, cooldown: float, capacity: int = 10_000) -> None:
        self._cooldown = cooldown
        self._capacity = capacity
        self._last_seen: dict[int, float] = {}

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get("event_from_user")
        if user is None or self._cooldown <= 0:
            return await handler(event, data)

        now = time.monotonic()
        last = self._last_seen.get(user.id)
        if last is not None and now - last < self._cooldown:
            if isinstance(event, CallbackQuery):
                with suppress(TelegramAPIError):
                    await event.answer(TEXT_THROTTLED)
            return None

        self._last_seen[user.id] = now
        self._prune(now)
        return await handler(event, data)

    def _prune(self, now: float) -> None:
        """Keep the tracking dict from growing without bound."""
        if len(self._last_seen) <= self._capacity:
            return
        cutoff = now - 3_600
        for user_id in [uid for uid, seen in self._last_seen.items() if seen < cutoff]:
            self._last_seen.pop(user_id, None)


class AccessMiddleware(BaseMiddleware):
    """Optional allow-list; an empty list keeps the bot public. Admins always pass."""

    def __init__(self, allowed: frozenset[int], admins: frozenset[int]) -> None:
        self._allowed = allowed
        self._admins = admins

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not self._allowed:
            return await handler(event, data)

        user: User | None = data.get("event_from_user")
        if user is None or (user.id not in self._allowed and user.id not in self._admins):
            LOGGER.warning("blocked update from user %s", getattr(user, "id", "unknown"))
            return None
        return await handler(event, data)


# --------------------------------------------------------------------------- #
# Handlers                                                                     #
# --------------------------------------------------------------------------- #


class Flow(StatesGroup):
    """Both buttons share one state machine: pick mode -> file -> instruction."""

    waiting_file = State()
    waiting_instruction = State()


router: Final = Router(name="ai_code_bot")


def is_admin(user_id: int, settings: Settings) -> bool:
    """Admin commands are opt-in: without ADMIN_IDS nobody is an admin."""
    return user_id in settings.admin_ids


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, ai: AIProviderManager) -> None:
    """Greet the user and show the menu."""
    await state.clear()
    name = escape(message.from_user.full_name) if message.from_user else ""
    providers = " + ".join(provider.name for provider in ai.providers)
    await message.answer(
        TEXT_WELCOME.format(name=name, providers=escape(providers)),
        reply_markup=main_menu(),
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext, queue: JobQueue) -> None:
    """Cancel the active job, or simply reset the state machine."""
    user_id = message.from_user.id if message.from_user else 0
    job = queue.active_job(user_id)
    if job is not None and await queue.cancel(job.job_id, user_id):
        return
    await state.clear()
    await message.answer(TEXT_NOTHING_TO_CANCEL, reply_markup=main_menu())


@router.message(Command("ai_status"))
async def cmd_ai_status(message: Message, settings: Settings, ai: AIProviderManager) -> None:
    """Admin dashboard: availability, current provider and average response time."""
    user_id = message.from_user.id if message.from_user else 0
    if not is_admin(user_id, settings):
        await message.answer(TEXT_ADMIN_ONLY)
        return

    placeholder = await message.answer(TEXT_PROBING)
    await ai.probe_all()
    with suppress(TelegramAPIError):
        await placeholder.edit_text(ai.render_status(), reply_markup=status_menu())


@router.callback_query(StatusCB.filter())
async def on_status_refresh(
    callback: CallbackQuery, settings: Settings, ai: AIProviderManager
) -> None:
    """Re-run the health checks and repaint the dashboard."""
    if not is_admin(callback.from_user.id, settings):
        await callback.answer(TEXT_ADMIN_ONLY, show_alert=True)
        return

    await callback.answer(TEXT_PROBING)
    await ai.probe_all()
    if isinstance(callback.message, Message):
        with suppress(TelegramAPIError):
            await callback.message.edit_text(ai.render_status(), reply_markup=status_menu())


@router.callback_query(CancelCB.filter())
async def on_cancel(callback: CallbackQuery, callback_data: CancelCB, queue: JobQueue) -> None:
    """Cancel button attached to the status message."""
    cancelled = await queue.cancel(callback_data.job_id, callback.from_user.id)
    await callback.answer(TEXT_CANCELLED if cancelled else TEXT_NOTHING_TO_CANCEL)


@router.callback_query(ActionCB.filter())
async def on_action(
    callback: CallbackQuery,
    callback_data: ActionCB,
    state: FSMContext,
    queue: JobQueue,
    history: HistoryStore,
) -> None:
    """Main menu: start a flow or show the history."""
    await callback.answer()
    target = callback.message
    if not isinstance(target, Message):
        return

    if callback_data.mode == "history":
        await target.answer(history.render(callback.from_user.id), reply_markup=main_menu())
        return

    if queue.has_active(callback.from_user.id):
        await target.answer(TEXT_BUSY)
        return

    mode = callback_data.mode if callback_data.mode in MODE_TITLES else MODE_FIX
    await state.set_state(Flow.waiting_file)
    await state.set_data({"mode": mode})
    await target.answer(TEXT_ASK_FILE_FIX if mode == MODE_FIX else TEXT_ASK_FILE_UPDATE)


@router.message(Flow.waiting_file, F.document)
async def on_file_received(
    message: Message, state: FSMContext, bot: Bot, settings: Settings
) -> None:
    """Validate the upload, download it, and ask for the instruction."""
    document = message.document
    if document is None or not document.file_name:
        await message.answer(TEXT_UNSUPPORTED)
        return

    file_name = safe_file_name(document.file_name)
    archive = is_archive(file_name)
    if not archive and not is_supported(file_name):
        await message.answer(TEXT_UNSUPPORTED)
        return

    limit = settings.max_archive_size if archive else settings.max_file_size
    if (document.file_size or 0) > limit:
        await message.answer(TEXT_TOO_BIG.format(limit=human_size(limit)))
        return

    buffer = BytesIO()
    try:
        await bot.download(document, destination=buffer)
        raw = buffer.getvalue()
    except TelegramAPIError as exc:
        LOGGER.error("download failed for %s: %s", file_name, exc)
        await message.answer(TEXT_DOWNLOAD_FAILED)
        return
    finally:
        buffer.close()

    if len(raw) > limit:  # file_size is optional in the Telegram payload
        await message.answer(TEXT_TOO_BIG.format(limit=human_size(limit)))
        return
    if not raw.strip():
        await message.answer(TEXT_EMPTY_FILE)
        return
    if archive and not raw.startswith(ZIP_MAGIC):
        await message.answer(TEXT_BAD_ARCHIVE)
        return
    if not archive:
        try:
            decode_source(raw)
        except ValueError:
            await message.answer(TEXT_NOT_TEXT)
            return

    data = await state.get_data()
    mode = data.get("mode", MODE_FIX)
    await state.update_data(file_name=file_name, raw=raw, archive=archive)
    await state.set_state(Flow.waiting_instruction)
    await message.answer(TEXT_ASK_PROBLEM if mode == MODE_FIX else TEXT_ASK_PROMPT)


@router.message(Flow.waiting_file)
async def on_file_expected(message: Message) -> None:
    """Anything that is not a document while a file is expected."""
    await message.answer(TEXT_FILE_ONLY)


@router.message(Flow.waiting_instruction, F.text)
async def on_instruction_received(
    message: Message, state: FSMContext, queue: JobQueue
) -> None:
    """Turn the (file + instruction) pair into a queued job."""
    instruction = (message.text or "").strip()
    if len(instruction) < MIN_INSTRUCTION_LEN:
        await message.answer(TEXT_SHORT_INSTRUCTION)
        return
    if len(instruction) > MAX_INSTRUCTION_LEN:
        await message.answer(TEXT_LONG_INSTRUCTION)
        return

    user = message.from_user
    user_id = user.id if user else 0
    if queue.has_active(user_id):
        await message.answer(TEXT_BUSY)
        return

    data = await state.get_data()
    mode: str = data.get("mode", "")
    file_name: str = data.get("file_name", "")
    raw: bytes = data.get("raw", b"")
    archive: bool = bool(data.get("archive", False))
    if not mode or not file_name or not raw:
        await state.clear()
        await message.answer(TEXT_SESSION_LOST, reply_markup=main_menu())
        return

    # Clear early: frees the cached upload from storage and blocks double submits.
    await state.clear()

    job_id = queue.next_id()
    status = await message.answer(TEXT_STARTING, reply_markup=cancel_menu(job_id))
    job = Job(
        job_id=job_id,
        user_id=user_id,
        user_name=(user.full_name if user else "unknown"),
        chat_id=message.chat.id,
        mode=mode,
        instruction=instruction,
        file_name=file_name,
        raw=raw,
        archive=archive,
        reporter=ProgressReporter(status, f"{MODE_TITLES[mode]} — {file_name}", job_id),
    )

    position = await queue.submit(job)
    LOGGER.info(
        "job=%s queued user=%s(%s) file=%s size=%s position=%d",
        job_id, job.user_name, user_id, file_name, human_size(len(raw)), position,
    )
    if not queue.is_immediate(position):
        await job.reporter.queued(position, queue.eta(position), file_name)


@router.message(Flow.waiting_instruction)
async def on_instruction_expected(message: Message) -> None:
    """Anything that is not text while the instruction is expected."""
    await message.answer(TEXT_TEXT_ONLY)


@router.message()
async def on_fallback(message: Message, queue: JobQueue) -> None:
    """Any message outside a flow brings the menu back."""
    user_id = message.from_user.id if message.from_user else 0
    if queue.has_active(user_id):
        await message.answer(TEXT_BUSY)
        return
    await message.answer(TEXT_MENU, reply_markup=main_menu())


@router.errors()
async def on_error(event: ErrorEvent) -> bool:
    """Last-resort logger so no exception escapes the dispatcher."""
    LOGGER.exception("unhandled error: %r", event.exception, exc_info=event.exception)
    return True


# --------------------------------------------------------------------------- #
# Bootstrap                                                                    #
# --------------------------------------------------------------------------- #


def setup_logging(settings: Settings) -> None:
    """Structured, greppable logs on stdout plus a rotating file for a 24/7 VPS."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.log_to_file:
        try:
            settings.log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(
                    settings.log_dir / "bot.log",
                    maxBytes=LOG_MAX_BYTES,
                    backupCount=LOG_BACKUPS,
                    encoding="utf-8",
                )
            )
        except OSError as exc:  # read-only volume: keep running on stdout only
            print(f"[log] could not open {settings.log_dir}: {exc}", file=sys.stderr)

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


async def announce(bot: Bot, settings: Settings, ai: AIProviderManager) -> None:
    """Log the effective configuration once, with the secrets masked."""
    me = await bot.get_me()
    for provider in ai.providers:
        config = provider.config
        LOGGER.info(
            "provider %s | model=%s | url=%s | key=%s | max_tokens=%d | thinking=%s | priority=%d",
            config.name, config.model, config.api_url, config.masked_key,
            config.max_tokens, config.thinking, config.priority,
        )
    LOGGER.info(
        "started @%s | providers=%s | routing=%s | workers=%d | chunk_budget=%d chars | "
        "max_file=%s | max_zip=%s | admins=%d",
        me.username,
        ", ".join(provider.name for provider in ai.providers),
        settings.routing,
        settings.max_concurrent_jobs,
        ai.chunk_budget(),
        human_size(settings.max_file_size),
        human_size(settings.max_archive_size),
        len(settings.admin_ids),
    )
    if not settings.admin_ids:
        LOGGER.warning("ADMIN_IDS is empty: /ai_status is disabled for everyone")


async def main() -> None:
    """Wire every component together and poll until stopped."""
    load_dotenv()
    settings = Settings.load()
    setup_logging(settings)

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher(storage=MemoryStorage())

    access = AccessMiddleware(settings.allowed_users, settings.admin_ids)
    throttling = ThrottlingMiddleware(settings.user_cooldown)
    for observer in (dispatcher.message, dispatcher.callback_query):
        observer.outer_middleware(access)
        observer.outer_middleware(throttling)
    dispatcher.include_router(router)

    request_log = RequestLogger(settings.log_dir, enabled=settings.log_to_file)
    ai = AIProviderManager(settings, request_log)
    history = HistoryStore(settings.max_history)
    processor = Processor(bot, settings, ai, history, request_log)
    queue = JobQueue(settings.max_concurrent_jobs, processor, history)

    try:
        await announce(bot, settings, ai)

        # Know which APIs are alive before the first user does.
        await ai.probe_all()
        ai.start_monitor()
        queue.start()

        with suppress(TelegramAPIError):
            await bot.set_my_commands(
                [
                    BotCommand(command="start", description="القائمة الرئيسية"),
                    BotCommand(command="cancel", description="إلغاء العملية الحالية"),
                    BotCommand(command="ai_status", description="حالة مزودات الذكاء الاصطناعي"),
                ]
            )
        await bot.delete_webhook(drop_pending_updates=True)
        await dispatcher.start_polling(
            bot,
            settings=settings,
            ai=ai,
            queue=queue,
            history=history,
            allowed_updates=dispatcher.resolve_used_update_types(),
        )
    finally:
        await queue.stop()
        await ai.close()
        await bot.session.close()
        LOGGER.info("shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ConfigError as error:
        print(f"[config] {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except (KeyboardInterrupt, SystemExit):
        pass
