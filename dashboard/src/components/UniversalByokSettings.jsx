import React, { useState, useEffect, useRef } from 'react';
import {
  Key,
  Eye,
  EyeOff,
  Check,
  Cpu,
  Volume2,
  Loader2,
  Play,
  Square,
  RefreshCw,
  Download,
  AlertCircle,
  Globe,
  Sliders,
  Sparkles,
} from 'lucide-react';
import { apiFetch, apiJson, getUniversalByok, saveUniversalByok } from '../lib/api';
import { getApiUrl } from '../config';

const LLM_PROVIDERS = [
  {
    id: 'gemini',
    name: 'google gemini',
    tag: 'NATIVE · MULTIMODAL',
    baseUrl: '',
    defaultModel: 'gemini-3.1-flash-lite',
    models: ['gemini-3.1-flash-lite', 'gemini-2.5-flash', 'gemini-2.5-pro'],
    placeholderKey: 'AIzaSy...',
    docsUrl: 'https://aistudio.google.com/app/apikey',
    hideBaseUrl: true,
  },
  {
    id: 'huawei_maas',
    name: 'huawei cloud maas',
    tag: 'MODELARTS · GLM-5.2',
    baseUrl: 'https://api-ap-southeast-1.modelarts-maas.com/openai/v1',
    defaultModel: 'glm-5.2',
    models: ['glm-5.2', 'glm-5', 'DeepSeek-V3', 'DeepSeek-R1', 'Qwen2.5-72B-Instruct'],
    placeholderKey: 'Paste your Huawei Cloud MaaS API Key...',
    docsUrl: 'https://console-intl.huaweicloud.com/modelarts/',
    hideBaseUrl: true,
  },
  {
    id: 'openai',
    name: 'openai',
    tag: 'OPENAI API',
    baseUrl: 'https://api.openai.com/v1',
    defaultModel: 'gpt-4o-mini',
    models: ['gpt-4o-mini', 'gpt-4o', 'o3-mini'],
    placeholderKey: 'sk-proj-...',
    docsUrl: 'https://platform.openai.com/api-keys',
  },
  {
    id: 'anthropic',
    name: 'anthropic claude',
    tag: 'MESSAGES API',
    baseUrl: 'https://api.anthropic.com/v1',
    defaultModel: 'claude-sonnet-4-20250514',
    models: ['claude-sonnet-4-20250514', 'claude-3-7-sonnet-latest', 'claude-3-5-haiku-latest'],
    placeholderKey: 'sk-ant-...',
    docsUrl: 'https://console.anthropic.com/settings/keys',
  },
  {
    id: 'openrouter',
    name: 'openrouter',
    tag: 'UNIVERSAL ROUTER',
    baseUrl: 'https://openrouter.ai/api/v1',
    defaultModel: 'google/gemini-2.5-flash',
    models: [
      'google/gemini-2.5-flash',
      'anthropic/claude-3.7-sonnet',
      'openai/gpt-4o-mini',
      'deepseek/deepseek-chat',
      'meta-llama/llama-3.3-70b-instruct',
    ],
    placeholderKey: 'sk-or-v1-...',
    docsUrl: 'https://openrouter.ai/keys',
  },
  {
    id: 'groq',
    name: 'groq lpu',
    tag: 'ULTRA FAST',
    baseUrl: 'https://api.groq.com/openai/v1',
    defaultModel: 'llama-3.3-70b-versatile',
    models: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'qwen-qwq-32b'],
    placeholderKey: 'gsk_...',
    docsUrl: 'https://console.groq.com/keys',
  },
  {
    id: 'deepseek',
    name: 'deepseek',
    tag: 'OPENAI COMPAT',
    baseUrl: 'https://api.deepseek.com/v1',
    defaultModel: 'deepseek-chat',
    models: ['deepseek-chat', 'deepseek-reasoner'],
    placeholderKey: 'sk-...',
    docsUrl: 'https://platform.deepseek.com/api_keys',
  },
  {
    id: 'glm',
    name: 'zhipu / glm',
    tag: 'OPENAI COMPAT',
    baseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    defaultModel: 'glm-4-plus',
    models: ['glm-4-plus', 'glm-4-flash', 'glm-4-air'],
    placeholderKey: 'id.secret...',
    docsUrl: 'https://open.bigmodel.cn/',
  },
  {
    id: 'custom',
    name: 'local / custom',
    tag: 'OLLAMA · VLLM · LM STUDIO',
    baseUrl: 'http://localhost:11434/v1',
    defaultModel: 'llama3.1:8b',
    models: ['llama3.1:8b', 'qwen2.5:14b', 'deepseek-r1:14b', 'mistral:7b'],
    placeholderKey: 'ollama (optional for local)',
    docsUrl: 'https://ollama.com/',
  },
];

const ELEVENLABS_MODELS = [
  { id: 'eleven_multilingual_v2', label: 'eleven multilingual v2', tag: '29 LANGS · HIGH QUALITY' },
  { id: 'eleven_turbo_v2_5', label: 'eleven turbo v2.5', tag: 'LOW LATENCY · MULTILINGUAL' },
  { id: 'eleven_flash_v2_5', label: 'eleven flash v2.5', tag: 'ULTRA FAST · 75MS' },
];

export default function UniversalByokSettings({
  onGeminiKeyChange,
  onElevenLabsKeyChange,
  onByokChange,
}) {
  const initial = getUniversalByok();

  // Universal LLM State
  const [llmProvider, setLlmProvider] = useState(initial.llmProvider || 'gemini');
  const [llmApiKey, setLlmApiKey] = useState(initial.llmApiKey || '');
  const [llmBaseUrl, setLlmBaseUrl] = useState(initial.llmBaseUrl || '');
  const [llmModel, setLlmModel] = useState(initial.llmModel || 'gemini-3.1-flash-lite');
  const [geminiFallbackKey, setGeminiFallbackKey] = useState(initial.geminiFallbackKey || '');
  const [showLlmKey, setShowLlmKey] = useState(false);
  const [showFallbackKey, setShowFallbackKey] = useState(false);
  const [llmSaved, setLlmSaved] = useState(false);
  const [testingLlm, setTestingLlm] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState(null);

  // ElevenLabs Voice Generation & Dubbing State
  const [elevenLabsKey, setElevenLabsKey] = useState(initial.elevenLabsApiKey || '');
  const [showElevenKey, setShowElevenKey] = useState(false);
  const [elevenSaved, setElevenSaved] = useState(false);
  const [voices, setVoices] = useState([]);
  const [voicesSource, setVoicesSource] = useState('defaults');
  const [loadingVoices, setLoadingVoices] = useState(false);
  const [selectedVoiceId, setSelectedVoiceId] = useState(initial.elevenLabsVoiceId || '21m00Tcm4TlvDq8ikWAM');
  const [selectedModelId, setSelectedModelId] = useState(initial.elevenLabsModelId || 'eleven_multilingual_v2');
  const [stability, setStability] = useState(initial.elevenLabsStability ?? 0.5);
  const [similarity, setSimilarity] = useState(initial.elevenLabsSimilarity ?? 0.75);
  const [styleExaggeration, setStyleExaggeration] = useState(initial.elevenLabsStyle ?? 0.4);
  const [speakerBoost, setSpeakerBoost] = useState(initial.elevenLabsSpeakerBoost ?? true);

  // Server .env status
  const [serverEnvStatus, setServerEnvStatus] = useState({
    envGemini: false,
    envLlmKey: false,
    envElevenLabs: false,
  });

  // Live TTS Voice Generator / Tester State
  const [ttsText, setTtsText] = useState(
    '¿Sabías que puedes generar voces hiperrealistas y cortar clips virales automáticamente con tu propio modelo de inteligencia artificial?'
  );
  const [generatingTts, setGeneratingTts] = useState(false);
  const [ttsError, setTtsError] = useState('');
  const [ttsResult, setTtsResult] = useState(null);
  const [playingPreviewUrl, setPlayingPreviewUrl] = useState(null);
  const previewAudioRef = useRef(null);

  const currentProviderPreset =
    LLM_PROVIDERS.find((p) => p.id === llmProvider) || LLM_PROVIDERS[0];

  // Fetch server BYOK status and voices on mount
  useEffect(() => {
    apiJson('/api/byok/config')
      .then((data) => {
        setServerEnvStatus({
          envGemini: !!data.env_gemini_configured,
          envLlmKey: !!data.env_llm_key_configured,
          envElevenLabs: !!data.env_elevenlabs_configured,
        });
      })
      .catch(() => {});
    fetchElevenLabsVoices(initial.elevenLabsApiKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchElevenLabsVoices = async (keyOverride) => {
    setLoadingVoices(true);
    try {
      const k = keyOverride !== undefined ? keyOverride : elevenLabsKey;
      const headers = k ? { 'X-ElevenLabs-Key': k } : {};
      const res = await apiFetch('/api/byok/elevenlabs/voices', { headers });
      if (res.ok) {
        const data = await res.json();
        const list = data.voices || [];
        setVoices(list);
        setVoicesSource(data.source || 'defaults');
        if (list.length > 0 && !list.some((v) => v.voice_id === selectedVoiceId)) {
          setSelectedVoiceId(list[0].voice_id);
        }
      }
    } catch {
      /* keep current voices */
    } finally {
      setLoadingVoices(false);
    }
  };

  const handleSelectProvider = (provId) => {
    const preset = LLM_PROVIDERS.find((p) => p.id === provId) || LLM_PROVIDERS[0];
    setLlmProvider(provId);
    setLlmBaseUrl(preset.baseUrl);
    setLlmModel(preset.defaultModel);
    setLlmTestResult(null);
  };

  const handleSaveLlm = async () => {
    const updated = saveUniversalByok({
      llmProvider,
      llmApiKey: llmApiKey.trim(),
      llmBaseUrl: llmBaseUrl.trim(),
      llmModel: llmModel.trim() || currentProviderPreset.defaultModel,
      geminiFallbackKey: geminiFallbackKey.trim(),
    });
    const effectiveGemini =
      llmProvider === 'gemini' ? llmApiKey.trim() : geminiFallbackKey.trim() || llmApiKey.trim();
    if (onGeminiKeyChange) onGeminiKeyChange(effectiveGemini);
    if (onByokChange) onByokChange(updated);

    try {
      await apiFetch('/api/byok/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          llm_provider: llmProvider,
          llm_api_key: llmApiKey.trim(),
          llm_base_url: llmBaseUrl.trim(),
          llm_model: llmModel.trim() || currentProviderPreset.defaultModel,
          gemini_api_key: effectiveGemini,
        }),
      });
    } catch {
      /* saved in localStorage even if backend is offline */
    }

    setLlmSaved(true);
    setTimeout(() => setLlmSaved(false), 2200);
  };

  const handleTestLlm = async () => {
    setTestingLlm(true);
    setLlmTestResult(null);
    try {
      await handleSaveLlm();
      const res = await apiFetch('/api/byok/test-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: llmProvider,
          api_key: llmApiKey.trim(),
          base_url: llmBaseUrl.trim(),
          model: llmModel.trim() || currentProviderPreset.defaultModel,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Connection failed');
      }
      if (data.base_url && data.base_url !== llmBaseUrl) {
        setLlmBaseUrl(data.base_url);
        saveUniversalByok({ llmBaseUrl: data.base_url });
      }
      setLlmTestResult({
        ok: true,
        latency: data.latency_ms,
        model: data.model,
        reply: data.reply,
      });
    } catch (err) {
      setLlmTestResult({
        ok: false,
        error: err.message || 'Could not reach LLM endpoint',
      });
    } finally {
      setTestingLlm(false);
    }
  };

  const handleSaveElevenLabs = async () => {
    const updated = saveUniversalByok({
      elevenLabsApiKey: elevenLabsKey.trim(),
      elevenLabsVoiceId: selectedVoiceId,
      elevenLabsModelId: selectedModelId,
      elevenLabsStability: stability,
      elevenLabsSimilarity: similarity,
      elevenLabsStyle: styleExaggeration,
      elevenLabsSpeakerBoost: speakerBoost,
    });
    if (onElevenLabsKeyChange) onElevenLabsKeyChange(elevenLabsKey.trim());
    if (onByokChange) onByokChange(updated);

    try {
      await apiFetch('/api/byok/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          elevenlabs_api_key: elevenLabsKey.trim(),
          elevenlabs_voice_id: selectedVoiceId,
          elevenlabs_model_id: selectedModelId,
          elevenlabs_stability: stability,
          elevenlabs_similarity: similarity,
          elevenlabs_style: styleExaggeration,
        }),
      });
    } catch {
      /* saved locally */
    }

    await fetchElevenLabsVoices(elevenLabsKey.trim());
    setElevenSaved(true);
    setTimeout(() => setElevenSaved(false), 2200);
  };

  const handlePlayVoiceSample = (voice) => {
    if (!voice?.preview_url) return;
    if (playingPreviewUrl === voice.preview_url && previewAudioRef.current) {
      previewAudioRef.current.pause();
      setPlayingPreviewUrl(null);
      return;
    }
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
    }
    const audio = new Audio(voice.preview_url);
    previewAudioRef.current = audio;
    setPlayingPreviewUrl(voice.preview_url);
    audio.onended = () => setPlayingPreviewUrl(null);
    audio.onerror = () => setPlayingPreviewUrl(null);
    audio.play().catch(() => setPlayingPreviewUrl(null));
  };

  const handleGenerateTts = async () => {
    if (!ttsText.trim()) return;
    setGeneratingTts(true);
    setTtsError('');
    setTtsResult(null);
    try {
      await handleSaveElevenLabs();
      const headers = { 'Content-Type': 'application/json' };
      if (elevenLabsKey.trim()) {
        headers['X-ElevenLabs-Key'] = elevenLabsKey.trim();
      }
      const res = await apiFetch('/api/byok/elevenlabs/tts', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          text: ttsText.trim(),
          voice_id: selectedVoiceId,
          model_id: selectedModelId,
          stability,
          similarity_boost: similarity,
          style: styleExaggeration,
          use_speaker_boost: speakerBoost,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'TTS generation failed');
      }
      setTtsResult({
        ...data,
        fullUrl: getApiUrl(data.audio_url),
      });
    } catch (err) {
      setTtsError(err.message || 'Error generating voice audio');
    } finally {
      setGeneratingTts(false);
    }
  };

  const selectedVoiceObj = voices.find((v) => v.voice_id === selectedVoiceId);
  const llmReady =
    !!llmApiKey.trim() ||
    llmProvider === 'custom' ||
    (llmProvider === 'gemini' && serverEnvStatus.envGemini) ||
    serverEnvStatus.envLlmKey;
  const elevenReady = !!elevenLabsKey.trim() || serverEnvStatus.envElevenLabs;

  return (
    <div className="space-y-8">
      {/* ═══════════════════════════════════════════════════════════════════
          CARD 1: UNIVERSAL LLM BYOK
         ═══════════════════════════════════════════════════════════════════ */}
      <div className="card p-4 sm:p-6 animate-fade">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0 text-brass">
              <Cpu size={18} />
            </div>
            <div>
              <p className="eyebrow">01 · UNIVERSAL LLM BYOK</p>
              <h2 className="font-display lowercase text-xl text-ink">
                universal language model provider
              </h2>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="readout">{llmProvider.toUpperCase()}</span>
            <span className={llmReady ? 'badge-ok' : 'badge-warn'}>
              {llmReady ? 'READY' : 'KEY NEEDED'}
            </span>
          </div>
        </div>

        <p className="text-xs text-muted mb-5 leading-relaxed">
          Connect any LLM provider for viral moment detection, scriptwriting, hooks, and SaaS analysis.
          Choose native <strong>Google Gemini</strong>, any cloud provider (<strong>OpenAI</strong>,{' '}
          <strong>Anthropic Claude</strong>, <strong>OpenRouter</strong>, <strong>Groq</strong>,{' '}
          <strong>DeepSeek</strong>, <strong>GLM</strong>), or run 100% locally via{' '}
          <strong>Ollama / LM Studio / vLLM</strong>.
        </p>

        {/* Provider Grid */}
        <div className="mb-5">
          <label className="eyebrow block mb-2">PROVIDER</label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {LLM_PROVIDERS.map((prov) => {
              const active = llmProvider === prov.id;
              return (
                <button
                  key={prov.id}
                  type="button"
                  onClick={() => handleSelectProvider(prov.id)}
                  className={`p-3 rounded-input border text-left transition-colors flex flex-col justify-between gap-1 ${
                    active
                      ? 'bg-paper3 border-brass text-ink'
                      : 'bg-paper border-rule text-muted hover:text-ink2 hover:bg-paper3/50'
                  }`}
                >
                  <span className="text-sm font-medium lowercase text-ink truncate">
                    {prov.name}
                  </span>
                  <span className="readout text-[10px] truncate">{prov.tag}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* API Key + Base URL + Model */}
        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs text-muted lowercase">
                {currentProviderPreset.name} api key
              </label>
              {currentProviderPreset.docsUrl && (
                <a
                  href={currentProviderPreset.docsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-brass hover:underline lowercase"
                >
                  get key →
                </a>
              )}
            </div>
            <div className="relative">
              <input
                type={showLlmKey ? 'text' : 'password'}
                value={llmApiKey}
                onChange={(e) => {
                  setLlmApiKey(e.target.value);
                  setLlmSaved(false);
                }}
                placeholder={currentProviderPreset.placeholderKey}
                className="input-field pr-10 font-mono text-sm"
              />
              <button
                type="button"
                onClick={() => setShowLlmKey(!showLlmKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
                aria-label="toggle key visibility"
              >
                {showLlmKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {/* Base URL (hidden for Gemini and Huawei Cloud MaaS since it is automatic) */}
          {!currentProviderPreset.hideBaseUrl && (
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs text-muted lowercase">
                  endpoint base url (openai / anthropic compatible)
                </label>
                <span className="readout">ENDPOINT</span>
              </div>
              <input
                type="text"
                value={llmBaseUrl}
                onChange={(e) => {
                  setLlmBaseUrl(e.target.value);
                  setLlmSaved(false);
                }}
                placeholder={currentProviderPreset.baseUrl || 'https://api.openai.com/v1'}
                className="input-field font-mono text-sm"
              />
            </div>
          )}

          {/* Model Selection + Custom Model Input */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs text-muted lowercase">model identifier</label>
              <span className="readout">MODEL</span>
            </div>
            <input
              type="text"
              value={llmModel}
              onChange={(e) => {
                setLlmModel(e.target.value);
                setLlmSaved(false);
              }}
              placeholder={currentProviderPreset.defaultModel}
              className="input-field font-mono text-sm mb-2"
            />
            <div className="flex flex-wrap gap-1.5">
              {currentProviderPreset.models.map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => {
                    setLlmModel(m);
                    setLlmSaved(false);
                  }}
                  className={`px-2.5 py-1 rounded-full font-mono text-[11px] border transition-colors ${
                    llmModel === m
                      ? 'bg-paper3 border-brass text-ink'
                      : 'bg-paper border-rule text-muted hover:text-ink2'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          {/* Optional Multimodal Gemini Fallback Key when using a text-only provider */}
          {llmProvider !== 'gemini' && (
            <div className="pt-2 border-t border-rule">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs text-muted lowercase">
                  optional gemini key for vision (auto-layout &amp; silent videos)
                </label>
                <span className="readout">OPTIONAL VISION</span>
              </div>
              <div className="relative">
                <input
                  type={showFallbackKey ? 'text' : 'password'}
                  value={geminiFallbackKey}
                  onChange={(e) => {
                    setGeminiFallbackKey(e.target.value);
                    setLlmSaved(false);
                  }}
                  placeholder="AIzaSy... (optional, only for frame inspection)"
                  className="input-field pr-10 font-mono text-sm"
                />
                <button
                  type="button"
                  onClick={() => setShowFallbackKey(!showFallbackKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
                >
                  {showFallbackKey ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              type="button"
              onClick={handleSaveLlm}
              className={llmSaved ? 'badge-ok px-4 py-2' : 'btn-primary py-2 px-5 text-sm'}
            >
              {llmSaved ? (
                <>
                  <Check size={14} /> saved
                </>
              ) : (
                <>
                  <Key size={14} /> save llm config
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleTestLlm}
              disabled={testingLlm}
              className="btn-ghost py-2 px-4 text-sm"
            >
              {testingLlm ? (
                <>
                  <Loader2 size={14} className="animate-spin" /> testing…
                </>
              ) : (
                <>
                  <Sparkles size={14} /> test connection
                </>
              )}
            </button>
          </div>

          {/* Connection Test Feedback */}
          {llmTestResult && (
            <div
              className={`p-3 rounded-input border text-xs flex items-center justify-between gap-3 ${
                llmTestResult.ok
                  ? 'bg-paper3 border-rule2 text-ink2'
                  : 'bg-paper3 border-danger/40 text-danger'
              }`}
            >
              {llmTestResult.ok ? (
                <>
                  <div className="flex items-center gap-2 min-w-0">
                    <Check size={14} className="text-ok shrink-0" />
                    <span className="truncate">
                      connected to <strong className="text-ink font-mono">{llmTestResult.model}</strong>{' '}
                      — response: <span className="font-mono text-ink">"{llmTestResult.reply}"</span>
                    </span>
                  </div>
                  <span className="readout shrink-0">{llmTestResult.latency}MS · OK</span>
                </>
              ) : (
                <div className="flex items-center gap-2">
                  <AlertCircle size={14} className="shrink-0" />
                  <span>{llmTestResult.error}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          CARD 2: ELEVENLABS — VOICE GENERATION (TTS) & DUBBING BYOK
         ═══════════════════════════════════════════════════════════════════ */}
      <div className="card p-4 sm:p-6 animate-fade">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0 text-brass">
              <Volume2 size={18} />
            </div>
            <div>
              <p className="eyebrow">02 · ELEVENLABS BYOK · VOICE GENERATION &amp; DUBBING</p>
              <h2 className="font-display lowercase text-xl text-ink">
                elevenlabs voice synthesis &amp; translation
              </h2>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="readout">
              {voicesSource === 'elevenlabs' ? `${voices.length} VOICES` : 'BYOK'}
            </span>
            <span className={elevenReady ? 'badge-ok' : 'badge-warn'}>
              {elevenLabsKey.trim()
                ? 'KEY SET'
                : serverEnvStatus.envElevenLabs
                  ? 'ENV KEY ACTIVE'
                  : 'KEY NEEDED'}
            </span>
          </div>
        </div>

        <p className="text-xs text-muted mb-5 leading-relaxed">
          Configure <strong>ElevenLabs</strong> for AI voiceover generation (TTS in AI Shorts &amp; curiosity
          explainers) and multi-language video dubbing (30+ languages). Select your default voice, TTS model,
          acoustic parameters, and test voice generation right below.
        </p>

        <div className="space-y-5">
          {/* API Key Input */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs text-muted lowercase">elevenlabs api key</label>
              <a
                href="https://elevenlabs.io/app/settings/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-brass hover:underline lowercase"
              >
                get elevenlabs key →
              </a>
            </div>
            <div className="flex flex-col sm:flex-row gap-2">
              <div className="relative flex-1">
                <input
                  type={showElevenKey ? 'text' : 'password'}
                  value={elevenLabsKey}
                  onChange={(e) => {
                    setElevenLabsKey(e.target.value);
                    setElevenSaved(false);
                  }}
                  placeholder={
                    serverEnvStatus.envElevenLabs
                      ? 'Using server .env key (or paste key to override)...'
                      : 'sk_...'
                  }
                  className="input-field pr-10 font-mono text-sm"
                />
                <button
                  type="button"
                  onClick={() => setShowElevenKey(!showElevenKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
                  aria-label="toggle elevenlabs key visibility"
                >
                  {showElevenKey ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              <button
                type="button"
                onClick={handleSaveElevenLabs}
                className={elevenSaved ? 'badge-ok px-4 py-2' : 'btn-primary py-2 px-4 text-sm shrink-0'}
              >
                {elevenSaved ? (
                  <>
                    <Check size={14} /> saved
                  </>
                ) : (
                  'save & load voices'
                )}
              </button>
            </div>
          </div>

          {/* TTS Model Selector */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="eyebrow">TTS ENGINE MODEL</label>
              <span className="readout">{selectedModelId.toUpperCase()}</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {ELEVENLABS_MODELS.map((m) => {
                const active = selectedModelId === m.id;
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setSelectedModelId(m.id)}
                    className={`p-3 rounded-input border text-left transition-colors flex flex-col justify-between gap-1 ${
                      active
                        ? 'bg-paper3 border-brass text-ink'
                        : 'bg-paper border-rule text-muted hover:text-ink2'
                    }`}
                  >
                    <span className="text-sm font-medium lowercase text-ink">{m.label}</span>
                    <span className="readout text-[10px]">{m.tag}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Voice Selector + Sample Preview */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="eyebrow">DEFAULT VOICE FOR TTS GENERATION</label>
              <button
                type="button"
                onClick={() => fetchElevenLabsVoices()}
                disabled={loadingVoices}
                className="text-xs text-brass hover:underline flex items-center gap-1 lowercase"
              >
                <RefreshCw size={12} className={loadingVoices ? 'animate-spin' : ''} />
                refresh voices
              </button>
            </div>

            <div className="flex flex-col sm:flex-row gap-2">
              <select
                value={selectedVoiceId}
                onChange={(e) => setSelectedVoiceId(e.target.value)}
                className="input-field flex-1 cursor-pointer"
              >
                {voices.map((v) => {
                  const gender = v.labels?.gender ? ` · ${v.labels.gender}` : '';
                  const accent = v.labels?.accent ? ` (${v.labels.accent})` : '';
                  const cat = v.category ? ` [${v.category}]` : '';
                  return (
                    <option key={v.voice_id} value={v.voice_id}>
                      {v.name}
                      {gender}
                      {accent}
                      {cat}
                    </option>
                  );
                })}
              </select>

              {selectedVoiceObj?.preview_url && (
                <button
                  type="button"
                  onClick={() => handlePlayVoiceSample(selectedVoiceObj)}
                  className="btn-ghost py-2 px-4 text-xs shrink-0"
                >
                  {playingPreviewUrl === selectedVoiceObj.preview_url ? (
                    <>
                      <Square size={13} /> stop sample
                    </>
                  ) : (
                    <>
                      <Play size={13} /> preview sample
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Acoustic Voice Settings Sliders */}
          <div className="p-3.5 rounded-input bg-paper border border-rule space-y-3">
            <div className="flex items-center justify-between">
              <span className="eyebrow flex items-center gap-1.5">
                <Sliders size={12} /> VOICE SYNTHESIS PARAMETERS
              </span>
              <label className="flex items-center gap-2 text-xs text-muted lowercase cursor-pointer">
                <input
                  type="checkbox"
                  checked={speakerBoost}
                  onChange={(e) => setSpeakerBoost(e.target.checked)}
                  className="accent-brass"
                />
                speaker boost
              </label>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted lowercase">stability</span>
                  <span className="readout tabular-nums">{Math.round(stability * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={stability}
                  onChange={(e) => setStability(parseFloat(e.target.value))}
                  className="w-full accent-brass cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted lowercase">similarity boost</span>
                  <span className="readout tabular-nums">{Math.round(similarity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={similarity}
                  onChange={(e) => setSimilarity(parseFloat(e.target.value))}
                  className="w-full accent-brass cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-muted lowercase">style exaggeration</span>
                  <span className="readout tabular-nums">{Math.round(styleExaggeration * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={styleExaggeration}
                  onChange={(e) => setStyleExaggeration(parseFloat(e.target.value))}
                  className="w-full accent-brass cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Live Voice Generation Studio */}
          <div className="p-4 rounded-input bg-paper border border-rule space-y-3">
            <div className="flex items-center justify-between gap-2">
              <span className="eyebrow flex items-center gap-1.5">
                <Volume2 size={12} /> LIVE VOICE GENERATOR (TTS STUDIO)
              </span>
              <span className="readout tabular-nums">{ttsText.length} CHARS</span>
            </div>

            <textarea
              rows={3}
              value={ttsText}
              onChange={(e) => setTtsText(e.target.value)}
              placeholder="Write text in Spanish or any language to generate speech with ElevenLabs..."
              className="input-field text-sm resize-y"
            />

            <div className="flex flex-wrap items-center justify-between gap-3">
              <button
                type="button"
                onClick={handleGenerateTts}
                disabled={generatingTts || !ttsText.trim() || !elevenReady}
                className="btn-primary py-2 px-4 text-xs"
              >
                {generatingTts ? (
                  <>
                    <Loader2 size={14} className="animate-spin" /> generating voice…
                  </>
                ) : (
                  <>
                    <Volume2 size={14} /> generate voice audio
                  </>
                )}
              </button>

              {ttsResult && (
                <a
                  href={ttsResult.fullUrl}
                  download={ttsResult.filename || 'elevenlabs-voice.mp3'}
                  className="btn-ghost py-1.5 px-3 text-xs"
                >
                  <Download size={13} /> download mp3
                </a>
              )}
            </div>

            {ttsError && (
              <div className="p-2.5 rounded-input bg-paper3 border border-danger/40 text-xs text-danger flex items-center gap-2">
                <AlertCircle size={14} className="shrink-0" />
                <span>{ttsError}</span>
              </div>
            )}

            {ttsResult && (
              <div className="pt-2 space-y-2">
                <audio
                  controls
                  autoPlay
                  src={ttsResult.fullUrl}
                  className="w-full h-10 rounded-input"
                />
                <div className="flex items-center justify-between text-[11px] text-muted font-mono uppercase">
                  <span>VOICE: {selectedVoiceObj?.name || ttsResult.voice_id}</span>
                  <span>MODEL: {ttsResult.model_id}</span>
                  <span>{ttsResult.characters} CHARS</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
