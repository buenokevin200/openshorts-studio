"""Universal LLM backend (BYOK): Gemini, Huawei Cloud MaaS, OpenAI, Anthropic,
OpenRouter, Groq, DeepSeek, GLM/Zhipu, and any local/custom OpenAI-compatible
server (Ollama, LM Studio, vLLM, llama.cpp, LocalAI).

When ``LLM_BASE_URL`` is set or ``LLM_PROVIDER`` is a non-Gemini provider,
transcript scoring, detail passes, script generation, and analysis route
through the configured Universal LLM provider.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes

HUAWEI_MAAS_ENDPOINTS = [
    "https://api-ap-southeast-1.modelarts-maas.com/openai/v1",
    "https://api.modelarts-maas.com/openai/v1",
    "https://api-ap-southeast-1.modelarts-maas.com/v1",
    "https://api.modelarts-maas.com/v1",
    "https://api-ap-southeast-1.modelarts-maas.com/v2",
    "https://api.modelarts-maas.com/v2",
]

PROVIDER_PRESETS = {
    "gemini": {
        "label": "Google Gemini",
        "base_url": "",
        "default_model": "gemini-3.1-flash-lite",
        "models": ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        "protocol": "gemini",
    },
    "huawei_maas": {
        "label": "Huawei Cloud MaaS",
        "base_url": "https://api-ap-southeast-1.modelarts-maas.com/openai/v1",
        "default_model": "glm-5.2",
        "models": ["glm-5.2", "glm-5", "DeepSeek-V3", "DeepSeek-R1", "Qwen2.5-72B-Instruct"],
        "protocol": "openai",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "o3-mini"],
        "protocol": "openai",
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-sonnet-4-20250514",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-latest", "claude-3-7-sonnet-latest"],
        "protocol": "anthropic",
    },
    "openrouter": {
        "label": "OpenRouter (Universal)",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemini-2.5-flash",
        "models": [
            "google/gemini-2.5-flash",
            "anthropic/claude-3.7-sonnet",
            "openai/gpt-4o-mini",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "protocol": "openai",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen-qwq-32b"],
        "protocol": "openai",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "protocol": "openai",
    },
    "glm": {
        "label": "Zhipu / GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-plus",
        "models": ["glm-4-plus", "glm-4-flash", "glm-4-air"],
        "protocol": "openai",
    },
    "custom": {
        "label": "Local / Custom (Ollama, LM Studio, vLLM)",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1:8b",
        "models": ["llama3.1:8b", "qwen2.5:14b", "deepseek-r1:14b", "mistral:7b"],
        "protocol": "openai",
    },
}


def _clean_api_key(raw_key: Optional[str]) -> str:
    """Strip accidental 'Bearer ' prefix, surrounding quotes, or whitespace."""
    k = (raw_key or "").strip().strip('"').strip("'").strip()
    if k.lower().startswith("bearer "):
        k = k[7:].strip()
    return k


def _clean_base_url(raw_url: Optional[str]) -> str:
    """Normalize base URL and strip accidental /chat/completions suffix."""
    u = (raw_url or "").strip().rstrip("/")
    if u.lower().endswith("/chat/completions"):
        u = u[:-17].rstrip("/")
    return u


def provider_id() -> str:
    """Return the normalized provider ID."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in PROVIDER_PRESETS:
        return explicit
    if explicit in ("huawei", "maas", "modelarts"):
        return "huawei_maas"
    if explicit in ("ollama", "local", "openai-compatible"):
        return "custom"
    if (os.environ.get("LLM_BASE_URL") or "").strip():
        return "openai"
    return "gemini"


def provider() -> str:
    """``"openai"`` when a non-Gemini endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in (
        "openai",
        "ollama",
        "local",
        "openai-compatible",
        "openrouter",
        "groq",
        "deepseek",
        "glm",
        "huawei_maas",
        "huawei",
        "maas",
        "modelarts",
        "custom",
        "anthropic",
    ):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    explicit_url = _clean_base_url(os.environ.get("LLM_BASE_URL"))
    if explicit_url:
        return explicit_url
    pid = provider_id()
    if pid != "gemini" and pid in PROVIDER_PRESETS:
        return _clean_base_url(PROVIDER_PRESETS[pid]["base_url"])
    return ""


def model_name() -> str:
    explicit_model = (os.environ.get("LLM_MODEL") or "").strip()
    if explicit_model:
        return explicit_model
    pid = provider_id()
    if pid in PROVIDER_PRESETS and pid != "custom":
        return PROVIDER_PRESETS[pid]["default_model"]
    return DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the non-Gemini LLM backend."""
    return provider() == "openai" and bool(base_url())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {
        "provider": provider_id(),
        "model": model_name(),
        "baseUrl": base_url(),
    }


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers(api_key_override: Optional[str] = None, base_url_override: Optional[str] = None) -> dict:
    raw_key = api_key_override if api_key_override is not None else os.environ.get("LLM_API_KEY")
    key = _clean_api_key(raw_key) or "ollama"
    pid = provider_id()
    b_url = (base_url_override or base_url()).lower()
    if pid == "anthropic" or "api.anthropic.com" in b_url:
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if pid == "openrouter" or "openrouter.ai" in b_url:
        headers["HTTP-Referer"] = "https://openshorts.app"
        headers["X-Title"] = "OpenShorts"
    return headers


def _candidate_base_urls(primary_url: str, prov: Optional[str] = None) -> List[str]:
    """Return ordered list of base URLs to try (auto-resolves Huawei Cloud MaaS regions)."""
    clean_primary = _clean_base_url(primary_url)
    pid = (prov or provider_id()).lower()
    is_huawei = pid in ("huawei_maas", "huawei", "maas") or "modelarts-maas.com" in clean_primary.lower()
    if not is_huawei:
        return [clean_primary] if clean_primary else []

    candidates = []
    if clean_primary:
        candidates.append(clean_primary)
    for ep in HUAWEI_MAAS_ENDPOINTS:
        if ep not in candidates:
            candidates.append(ep)
    return candidates


def _is_huawei_region_auth_error(resp: httpx.Response) -> bool:
    """True when a Huawei MaaS regional endpoint rejects a key meant for another region."""
    if resp.status_code in (401, 404):
        txt = resp.text
        if "ModelArts.81003" in txt or "ModelArts.81002" in txt or resp.status_code == 404:
            return True
    return False


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def _is_anthropic_endpoint(url_base: str) -> bool:
    return provider_id() == "anthropic" and "api.anthropic.com" in url_base.lower()


def _call_anthropic_messages(url_base: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 8192) -> Tuple[str, dict]:
    url = f"{url_base.rstrip('/')}/messages"
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    with _client() as client:
        resp = client.post(url, json=body, headers=_headers(base_url_override=url_base))
        if resp.status_code >= 400:
            raise RuntimeError(f"Anthropic API {resp.status_code} from {url}: {resp.text[:300]}")
        data = resp.json()
    content_blocks = data.get("content") or []
    text = "".join(b.get("text", "") for b in content_blocks if isinstance(b, dict) and b.get("type") == "text")
    usage = data.get("usage") or {}
    return text, {
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
    }


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``."""
    import gemini_worker  # local import: keeps this module free of the google SDK

    b_url = base_url()
    model = model or model_name()

    if _is_anthropic_endpoint(b_url):
        schema_str = json.dumps(schema.model_json_schema())
        sys_msg = (
            "You answer with a single valid JSON object matching this JSON schema and nothing else: "
            + schema_str
        )
        text, usage = _call_anthropic_messages(b_url, model, sys_msg, prompt)
        parsed = gemini_worker._parse_json_response_text(text)
        validated = schema.model_validate(parsed).model_dump()
        cost = {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "thinking_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "model": model,
            "price_estimated": False,
            "local": True,
        }
        return validated, cost

    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    candidates = _candidate_base_urls(b_url)
    last_error: Optional[str] = None
    data = None

    with _client() as client:
        for cand_base in candidates:
            url = f"{cand_base}/chat/completions"
            region_failed = False
            last_rejection: Optional[str] = None
            for fmt in _response_formats(schema):
                body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
                if fmt is not None:
                    body["response_format"] = fmt
                resp = client.post(url, json=body, headers=_headers(base_url_override=cand_base))
                if len(candidates) > 1 and _is_huawei_region_auth_error(resp):
                    last_error = f"LLM server {resp.status_code} from {url}: {resp.text[:300]}"
                    region_failed = True
                    break
                if fmt is not None and _is_format_rejection(resp):
                    last_rejection = resp.text[:200]
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
                data = resp.json()
                if cand_base != b_url:
                    os.environ["LLM_BASE_URL"] = cand_base
                break
            if data is not None:
                break
            if not region_failed and last_rejection is not None:
                raise RuntimeError(
                    f"LLM server rejected every response_format variant: {last_rejection}")

    if data is None:
        raise RuntimeError(last_error or "LLM request failed")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost


def generate_raw_text(prompt: str, system_prompt: str = "You are a helpful AI assistant.", json_mode: bool = False) -> str:
    """Universal text/JSON completion helper for modules like SaaSShorts, Hooks, and Thumbnail Studio."""
    b_url = base_url()
    model = model_name()

    if _is_anthropic_endpoint(b_url):
        sys_msg = system_prompt + (" Respond with valid JSON only." if json_mode else "")
        text, _ = _call_anthropic_messages(b_url, model, sys_msg, prompt)
        return text

    sys_msg = system_prompt + (" Respond with valid JSON only." if json_mode else "")
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    candidates = _candidate_base_urls(b_url)
    last_err = None

    with _client() as client:
        for cand_base in candidates:
            url = f"{cand_base}/chat/completions"
            for use_fmt in ([True, False] if json_mode else [False]):
                body = {"model": model, "messages": messages, "temperature": 0.3, "stream": False}
                if use_fmt:
                    body["response_format"] = {"type": "json_object"}
                resp = client.post(url, json=body, headers=_headers(base_url_override=cand_base))
                if len(candidates) > 1 and _is_huawei_region_auth_error(resp):
                    last_err = f"LLM server {resp.status_code} from {url}: {resp.text[:300]}"
                    break
                if use_fmt and _is_format_rejection(resp):
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
                if cand_base != b_url:
                    os.environ["LLM_BASE_URL"] = cand_base
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content") or ""
                    if isinstance(content, list):
                        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                    return content
                return ""
    if last_err:
        raise RuntimeError(last_err)
    return ""


def test_connection(
    prov: str,
    api_key: str = "",
    custom_base_url: str = "",
    custom_model: str = "",
) -> dict:
    """Test an LLM provider connection and return latency + sample reply."""
    prov = (prov or "gemini").strip().lower()
    preset = PROVIDER_PRESETS.get(prov, PROVIDER_PRESETS["custom"])
    eff_model = (custom_model or "").strip() or preset["default_model"]
    eff_base_url = _clean_base_url(custom_base_url) or _clean_base_url(preset["base_url"])
    clean_key = _clean_api_key(api_key)
    t0 = time.monotonic()

    if prov == "gemini" and not eff_base_url:
        if not clean_key:
            raise ValueError("Missing Google Gemini API Key")
        from google import genai
        client = genai.Client(api_key=clean_key)
        resp = client.models.generate_content(
            model=eff_model,
            contents=["Reply with the single word: READY"],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "ok": True,
            "provider": prov,
            "model": eff_model,
            "latency_ms": latency_ms,
            "reply": (resp.text or "READY").strip()[:80],
        }

    if not eff_base_url:
        raise ValueError("Base URL is required for this provider")

    if prov == "anthropic" and "api.anthropic.com" in eff_base_url.lower():
        if not clean_key:
            raise ValueError("Missing Anthropic API Key")
        url = f"{eff_base_url}/messages"
        headers = {
            "x-api-key": clean_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": eff_model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Reply with the single word: READY"}],
        }
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, json=body, headers=headers)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:250]}")
            data = resp.json()
        blocks = data.get("content") or []
        reply = "".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "ok": True,
            "provider": prov,
            "model": eff_model,
            "base_url": eff_base_url,
            "latency_ms": latency_ms,
            "reply": (reply or "READY")[:80],
        }

    if prov == "huawei_maas" and not clean_key:
        raise ValueError("Missing Huawei Cloud MaaS API Key")

    candidates = _candidate_base_urls(eff_base_url, prov=prov)
    key = clean_key or "ollama"
    body = {
        "model": eff_model,
        "messages": [{"role": "user", "content": "Reply with the single word: READY"}],
        "max_tokens": 32,
        "temperature": 0.1,
    }

    last_err_msg = ""
    with httpx.Client(timeout=25.0) as client:
        for cand_base in candidates:
            url = f"{cand_base}/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            if prov == "openrouter" or "openrouter.ai" in cand_base.lower():
                headers["HTTP-Referer"] = "https://openshorts.app"
                headers["X-Title"] = "OpenShorts"
            resp = client.post(url, json=body, headers=headers)
            if len(candidates) > 1 and _is_huawei_region_auth_error(resp):
                last_err_msg = f"HTTP {resp.status_code} ({cand_base}): {resp.text[:220]}"
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:250]}")
            data = resp.json()
            os.environ["LLM_BASE_URL"] = cand_base
            choices = data.get("choices") or []
            reply = ""
            if choices:
                reply = (choices[0].get("message") or {}).get("content") or ""
            latency_ms = int((time.monotonic() - t0) * 1000)
            return {
                "ok": True,
                "provider": prov,
                "model": eff_model,
                "base_url": cand_base,
                "latency_ms": latency_ms,
                "reply": (str(reply).strip() or "READY")[:80],
            }

    raise RuntimeError(
        last_err_msg
        or "Huawei Cloud MaaS rejected the API key across all regional endpoints. Verify the key in ModelArts Studio -> API Keys."
    )
