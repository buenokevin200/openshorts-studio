// Centralized API client for cloud and self-hosted Universal BYOK mode.
// Adds Authorization: Bearer and Universal BYOK headers (LLM + ElevenLabs),
// and turns a 402 (quota exceeded) into a typed QuotaError.
import { getApiUrl } from '../config';

export const AUTH_TOKEN_KEY = 'openshorts_auth';
export const UNIVERSAL_BYOK_KEY = 'openshorts_universal_byok_v1';

const SECRET_KEY = import.meta.env.VITE_ENCRYPTION_KEY || 'OpenShorts-Static-Salt-Change-Me';
const ENCRYPTION_PREFIX = 'ENC:';

export const encryptKey = (text) => {
  if (!text) return '';
  try {
    const xor = text
      .split('')
      .map((c, i) => String.fromCharCode(c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length)))
      .join('');
    return ENCRYPTION_PREFIX + btoa(xor);
  } catch {
    return text;
  }
};

export const decryptKey = (text) => {
  if (!text) return '';
  if (text.startsWith(ENCRYPTION_PREFIX)) {
    try {
      const raw = text.slice(ENCRYPTION_PREFIX.length);
      const xor = atob(raw);
      return xor
        .split('')
        .map((c, i) => String.fromCharCode(c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length)))
        .join('');
    } catch {
      return '';
    }
  }
  return text;
};

export function getUniversalByok() {
  try {
    const raw = localStorage.getItem(UNIVERSAL_BYOK_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    const legacyGemini = localStorage.getItem('gemini_key') || '';
    const legacyEleven = decryptKey(localStorage.getItem('elevenLabsKey_v1') || '');
    return {
      llmProvider: parsed.llmProvider || 'gemini',
      llmApiKey: parsed.llmApiKey ? decryptKey(parsed.llmApiKey) : legacyGemini,
      llmBaseUrl: parsed.llmBaseUrl || '',
      llmModel: parsed.llmModel || 'gemini-3.1-flash-lite',
      geminiFallbackKey: parsed.geminiFallbackKey ? decryptKey(parsed.geminiFallbackKey) : legacyGemini,
      elevenLabsApiKey: parsed.elevenLabsApiKey ? decryptKey(parsed.elevenLabsApiKey) : legacyEleven,
      elevenLabsVoiceId: parsed.elevenLabsVoiceId || '21m00Tcm4TlvDq8ikWAM',
      elevenLabsModelId: parsed.elevenLabsModelId || 'eleven_multilingual_v2',
      elevenLabsStability: parsed.elevenLabsStability ?? 0.5,
      elevenLabsSimilarity: parsed.elevenLabsSimilarity ?? 0.75,
      elevenLabsStyle: parsed.elevenLabsStyle ?? 0.4,
      elevenLabsSpeakerBoost: parsed.elevenLabsSpeakerBoost ?? true,
    };
  } catch {
    return {
      llmProvider: 'gemini',
      llmApiKey: localStorage.getItem('gemini_key') || '',
      llmBaseUrl: '',
      llmModel: 'gemini-3.1-flash-lite',
      geminiFallbackKey: localStorage.getItem('gemini_key') || '',
      elevenLabsApiKey: decryptKey(localStorage.getItem('elevenLabsKey_v1') || ''),
      elevenLabsVoiceId: '21m00Tcm4TlvDq8ikWAM',
      elevenLabsModelId: 'eleven_multilingual_v2',
      elevenLabsStability: 0.5,
      elevenLabsSimilarity: 0.75,
      elevenLabsStyle: 0.4,
      elevenLabsSpeakerBoost: true,
    };
  }
}

export function saveUniversalByok(next) {
  const cur = getUniversalByok();
  const merged = { ...cur, ...next };
  try {
    localStorage.setItem(
      UNIVERSAL_BYOK_KEY,
      JSON.stringify({
        llmProvider: merged.llmProvider,
        llmApiKey: encryptKey(merged.llmApiKey || ''),
        llmBaseUrl: merged.llmBaseUrl || '',
        llmModel: merged.llmModel || '',
        geminiFallbackKey: encryptKey(merged.geminiFallbackKey || ''),
        elevenLabsApiKey: encryptKey(merged.elevenLabsApiKey || ''),
        elevenLabsVoiceId: merged.elevenLabsVoiceId || '21m00Tcm4TlvDq8ikWAM',
        elevenLabsModelId: merged.elevenLabsModelId || 'eleven_multilingual_v2',
        elevenLabsStability: merged.elevenLabsStability ?? 0.5,
        elevenLabsSimilarity: merged.elevenLabsSimilarity ?? 0.75,
        elevenLabsStyle: merged.elevenLabsStyle ?? 0.4,
        elevenLabsSpeakerBoost: merged.elevenLabsSpeakerBoost ?? true,
      }),
    );
    if (merged.llmProvider === 'gemini' && merged.llmApiKey) {
      localStorage.setItem('gemini_key', merged.llmApiKey);
    } else if (merged.geminiFallbackKey) {
      localStorage.setItem('gemini_key', merged.geminiFallbackKey);
    }
    if (merged.elevenLabsApiKey) {
      localStorage.setItem('elevenLabsKey_v1', encryptKey(merged.elevenLabsApiKey));
    }
  } catch {
    /* ignore storage error */
  }
  return merged;
}

export const getToken = () => localStorage.getItem(AUTH_TOKEN_KEY) || '';
export const setToken = (t) => localStorage.setItem(AUTH_TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(AUTH_TOKEN_KEY);

export class QuotaError extends Error {
  constructor(detail) {
    super('quota_exceeded');
    this.name = 'QuotaError';
    this.minutesRequired = detail?.minutes_required;
    this.minutesRemaining = detail?.minutes_remaining;
    // Minutes of the source the server offers to clip instead (0 = no offer).
    this.partialMinutes = Number(detail?.partial_minutes) || 0;
  }
}

// A 402 caught outside the main clip flow (thumbnails, clip editor, autopilot):
// ask the app shell to open the quota wall instead of a dead-end error. `source`
// tags CheckoutStarted so each surface can be measured.
export const QUOTA_WALL_EVENT = 'os:quota-wall';
export function openQuotaWall(err, source) {
  try {
    window.dispatchEvent(new CustomEvent(QUOTA_WALL_EVENT, {
      detail: { required: err?.minutesRequired, remaining: err?.minutesRemaining, source },
    }));
  } catch (_) { /* ignore */ }
}

// Drop-in fetch wrapper. Always attaches the bearer token and Universal BYOK headers when present.
export async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);

  // Attach Universal BYOK headers (LLM + ElevenLabs) if not already set on the request
  const byok = getUniversalByok();
  if (byok.llmProvider && !headers.has('X-LLM-Provider')) {
    headers.set('X-LLM-Provider', byok.llmProvider);
  }
  if (byok.llmApiKey && !headers.has('X-LLM-Key')) {
    headers.set('X-LLM-Key', byok.llmApiKey);
  }
  if (byok.llmBaseUrl && !headers.has('X-LLM-Base-Url')) {
    headers.set('X-LLM-Base-Url', byok.llmBaseUrl);
  }
  if (byok.llmModel && !headers.has('X-LLM-Model')) {
    headers.set('X-LLM-Model', byok.llmModel);
  }
  const gemKey = byok.llmProvider === 'gemini' ? byok.llmApiKey : byok.geminiFallbackKey;
  if (gemKey && !headers.has('X-Gemini-Key')) {
    headers.set('X-Gemini-Key', gemKey);
  }
  if (byok.elevenLabsApiKey && !headers.has('X-ElevenLabs-Key')) {
    headers.set('X-ElevenLabs-Key', byok.elevenLabsApiKey);
  }
  if (byok.elevenLabsVoiceId && !headers.has('X-ElevenLabs-Voice-Id')) {
    headers.set('X-ElevenLabs-Voice-Id', byok.elevenLabsVoiceId);
  }
  if (byok.elevenLabsModelId && !headers.has('X-ElevenLabs-Model-Id')) {
    headers.set('X-ElevenLabs-Model-Id', byok.elevenLabsModelId);
  }

  const res = await fetch(getApiUrl(path), { ...options, headers });

  if (res.status === 402) {
    let detail = {};
    try {
      const body = await res.clone().json();
      detail = body.detail || body;
    } catch (_) { /* ignore */ }
    throw new QuotaError(detail);
  }
  return res;
}

// A failed request, carrying the server's own explanation.
export class ApiError extends Error {
  constructor(status, detail, raw) {
    super(`${status}: ${raw}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

// Convenience JSON helper.
export async function apiJson(path, options = {}) {
  const res = await apiFetch(path, options);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    let detail = '';
    try {
      const d = JSON.parse(text).detail;
      if (typeof d === 'string') detail = d;
    } catch (_) { /* not JSON */ }
    throw new ApiError(res.status, detail, text);
  }
  return res.json();
}
