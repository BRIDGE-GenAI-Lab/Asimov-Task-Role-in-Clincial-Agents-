from __future__ import annotations
import os
from typing import Callable, Dict
from .base import ModelAdapter


def _retry(fn):
    # tenacity imported lazily so the core/tests never depend on it
    from tenacity import retry, stop_after_attempt, wait_exponential
    return retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=30), reraise=True)(fn)


def _anthropic_builder(model_id: str) -> Callable[[], ModelAdapter]:
    def build() -> ModelAdapter:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        class A(ModelAdapter):
            name = model_id

            @_retry
            def generate(self, prompt: str) -> str:
                r = client.messages.create(model=model_id, max_tokens=1024,
                                           messages=[{"role": "user", "content": prompt}])
                return r.content[0].text
        return A()
    return build


def _openai_builder(model_id: str) -> Callable[[], ModelAdapter]:
    def build() -> ModelAdapter:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        class O(ModelAdapter):
            name = model_id

            @_retry
            def generate(self, prompt: str) -> str:
                r = client.chat.completions.create(model=model_id,
                                                   messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
        return O()
    return build


def _openai_compatible_builder(model_id: str, key_env: str, base_url: str) -> Callable[[], ModelAdapter]:
    # DeepSeek, xAI, local vLLM, etc. all expose an OpenAI-compatible /v1 endpoint.
    def build() -> ModelAdapter:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ[key_env], base_url=base_url)

        class C(ModelAdapter):
            name = model_id

            @_retry
            def generate(self, prompt: str) -> str:
                r = client.chat.completions.create(model=model_id,
                                                   messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
        return C()
    return build


def _google_builder(model_id: str) -> Callable[[], ModelAdapter]:
    def build() -> ModelAdapter:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        gm = genai.GenerativeModel(model_id)

        class G(ModelAdapter):
            name = model_id

            @_retry
            def generate(self, prompt: str) -> str:
                return gm.generate_content(prompt).text
        return G()
    return build


def _ollama_builder(tag: str, temperature: float = 0.7,
                    num_predict: Optional[int] = None,
                    think: Optional[bool] = None) -> Callable[[], ModelAdapter]:
    def build() -> ModelAdapter:
        from .local import OllamaAdapter
        return OllamaAdapter(tag, temperature=temperature, num_predict=num_predict, think=think)
    return build


def _openrouter_builder(model_id: str, temperature: float = 0.7) -> Callable[[], ModelAdapter]:
    def build() -> ModelAdapter:
        from .openrouter import OpenRouterAdapter
        return OpenRouterAdapter(model_id, temperature=temperature)
    return build


def _rev_builder(model_id: str, provider: dict = None, extra_body: dict = None,
                 supports_temperature: bool = True) -> Callable[[], ModelAdapter]:
    def build() -> ModelAdapter:
        from .openrouter import OpenRouterAdapter
        t = os.environ.get("WARD_TEMPERATURE", "0.7")
        temp = None if (t.strip().lower() == "none" or not supports_temperature) else float(t)
        return OpenRouterAdapter(model_id, temperature=temp, extra_body=extra_body, provider=provider,
                                 max_tokens=int(os.environ.get("WARD_MAX_TOKENS", "4096")),
                                 record_usage=True)
    return build


def _rev_entries() -> Dict[str, Callable[[], ModelAdapter]]:
    import json
    path = os.environ.get("WARD_REV_ROSTER",
                          "experiment2/model_roster.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        roster = json.load(fh)
    return {m["key"]: _rev_builder(m["slug"], m.get("provider"), m.get("extra_body"),
                                   m.get("supports_temperature") is not False)
            for m in roster.get("models", []) if m.get("status") == "locked"}


_QWEN_NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}  # Qwen3.5 reasoning runs away on ward prompts (3k+ tok/step)


def _o2_llama_builder(model_id: str, port: int, temperature: float = 0.7,
                      key_env: str = "WARD_O2_KEY", extra_body: dict = None) -> Callable[[], ModelAdapter]:
    # Self-hosted llama-server on Harvard O2 (HMS RC cluster), reached via an SSH
    # port-forward to localhost:<port>. Same OpenAI-compatible /v1 surface as
    # OpenRouter, so we reuse OpenRouterAdapter with a custom base. The key is a
    # placeholder: llama-server ignores auth unless started with --api-key.
    # Self-hosting pins exact weights+quant+seed -> reproducible (no hosted drift).
    def build() -> ModelAdapter:
        os.environ.setdefault(key_env, "local")
        from .openrouter import OpenRouterAdapter
        return OpenRouterAdapter(model_id, key_env=key_env,
                                 base="http://127.0.0.1:%d/v1" % port, temperature=temperature,
                                 extra_body=extra_body)
    return build


def _gemini_dual_builder(cli_model: str, api_factory: Callable[[], ModelAdapter]) -> Callable[[], ModelAdapter]:
    # WARD_GEMINI_CLI=1 -> route through the authenticated local `gemini` binary (free quota);
    # resume across backends (the CLI does not expose temperature; backend recorded via run
    # provenance). Free-CLI quota currently serves gemini-2.5-{flash,pro}; 3.x ids 404 (2026-06).
    def build() -> ModelAdapter:
        if os.environ.get("WARD_GEMINI_CLI") == "1":
            from .gemini_cli import GeminiCLIAdapter
            return GeminiCLIAdapter(cli_model)
        return api_factory()
    return build


def _gemini_flash_builder() -> Callable[[], ModelAdapter]:
    from .openrouter import OpenRouterAdapter
    return _gemini_dual_builder("gemini-2.5-flash",
                                lambda: OpenRouterAdapter("google/gemini-2.5-flash", temperature=0.7))


def _agy_builder(model_id: str) -> Callable[[], ModelAdapter]:
    # gemini CLI (which stops free serving 2026-06-18). Exposes Gemini 3.5 Flash, Gemini 3.1 Pro,
    # Claude Opus/Sonnet 4.6, GPT-OSS 120B via `agy --model <id> -p`.
    def build() -> ModelAdapter:
        from .antigravity_cli import AntigravityCLIAdapter
        return AntigravityCLIAdapter(model_id)
    return build


def _claude_cli_builder(model_id: str) -> Callable[[], ModelAdapter]:
    # `--model opus` resolves to the current Opus (4.8). Runs from a neutral cwd (see claude_cli.py)
    # so the CLI does not detect the study repo and refuse to role-play a study subject.
    def build() -> ModelAdapter:
        from .claude_cli import ClaudeCLIAdapter
        return ClaudeCLIAdapter(model_id)
    return build


def _mock_builder() -> Callable[[], ModelAdapter]:
    # Offline self-test: returns a valid no-op ward action for any prompt (no model, no network,
    # no cost). Lets a collaborator validate the harness plumbing end-to-end before spending quota.
    def build() -> ModelAdapter:
        from .mock import MockAdapter
        return MockAdapter("mock", rules=[],
                           default='{"rationale": "offline self-test", "action": {"verb": "wait", "target": null}}')
    return build


def _codex_cli_builder(model_id: str, effort: str = "medium") -> Callable[[], ModelAdapter]:
    # model_id + effort are overridable at run time via CODEX_WARD_MODEL / CODEX_WARD_EFFORT so a
    # collaborator can point it at whatever exact GPT-5.6 id their Codex build exposes.
    def build() -> ModelAdapter:
        from .codex_cli import CodexCLIAdapter
        return CodexCLIAdapter(model_id, effort)
    return build


# Public registry. Model ids are configurable; cloud entries are exercised only when their key is set.
BUILDERS: Dict[str, Callable[[], ModelAdapter]] = {
    # Anthropic
    "claude-opus-4-8": _anthropic_builder("claude-opus-4-8"),
    "claude-haiku-4-5": _anthropic_builder("claude-haiku-4-5-20251001"),
    "claude-opus-4.8-cc": _claude_cli_builder("opus"),

    # NOTE: confirm the exact model slugs your Codex build exposes (model picker / `codex exec -m ...`);
    # override without editing here via CODEX_WARD_MODEL / CODEX_WARD_EFFORT. Luna = the S10o config.
    "gpt-5.6-luna-cc": _codex_cli_builder("gpt-5.6-luna", "medium"),
    "gpt-5.6-sol-cc": _codex_cli_builder("gpt-5.6-sol", "medium"),
    "gpt-5.6-terra-cc": _codex_cli_builder("gpt-5.6-terra", "medium"),
    "mock": _mock_builder(),  # offline harness self-test (no model / no network / no cost)
    # OpenAI
    "gpt-5.1-direct": _openai_builder("gpt-5.1"),  # needs OPENAI_API_KEY (unset) — kept for later
    "gpt-5.1": _openrouter_builder("openai/gpt-5.1"),
    # Google (Gemini API)
    "gemini-2.5-pro": _gemini_dual_builder("gemini-2.5-pro", _google_builder("gemini-2.5-pro")),
    "gemini-2.5-flash-lite": _gemini_dual_builder("gemini-2.5-flash-lite",
                                                  _openrouter_builder("google/gemini-2.5-flash-lite")),
    "gemini-3.1-flash-lite": _gemini_dual_builder("gemini-3.1-flash-lite",
                                                  _openrouter_builder("google/gemini-3.1-flash-lite")),
    "gemini-3.5-flash": _agy_builder("gemini-3.5-flash"),  # newest Gemini Flash
    "gemini-3.1-pro": _agy_builder("gemini-3.1-pro-high"),      # Gemini 3.1 Pro (High reasoning) via agy subscription
    "claude-opus-4.6-agy": _agy_builder("claude-opus-4.6"),      # free frontier Anthropic via agy subscription
    "claude-sonnet-4.6-agy": _agy_builder("claude-sonnet-4.6"),  # free frontier Anthropic via agy subscription
    # OpenAI-compatible vendors
    "deepseek-chat": _openai_compatible_builder("deepseek-chat", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
    "grok-4": _openai_compatible_builder("grok-4", "XAI_API_KEY", "https://api.x.ai/v1"),
    # Local (Ollama) — no API key needed
    "gemma3-12b-qat": _ollama_builder("hf.co/unsloth/gemma-3-12b-it-qat-GGUF:Q4_K_M"),
    "gemma4-26b-qat": _ollama_builder("hf.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL",
                                      num_predict=800),  # verbose MoE; 400 truncated JSON -> 100% parse fail (2026-06-10)
    # qwen3.5 is a *thinking* model family: without think=False the reasoning channel eats the whole
    "qwen3.5-4b": _ollama_builder("qwen3.5:4b", num_predict=800, think=False),   # distinct-family local judge
    # 16GB second-Mac packet panel (capability low end)
    "qwen3.5-9b": _ollama_builder("qwen3.5:9b", num_predict=800, think=False),
    "qwen3.5-2b": _ollama_builder("qwen3.5:2b", num_predict=800, think=False),
    "qwen3.5-0.8b": _ollama_builder("qwen3.5:0.8b", num_predict=800, think=False),
    # gemma4 e-series is also a *thinking* family (separate `thinking` channel); on the long ward
    # Gemma-4 is a thinking family. Probed 2026-06-17: with num_predict=8000 and think=True,
    # e4b parses 3/3 (reasons AND emits a valid action), matching the reasoning-on frontier models
    # (GPT-5.1/DeepSeek). So thinking is ENABLED here. (Qwen3.5 still fails think=True -> empty content
    "gemma4-e4b-qat": _ollama_builder("gemma4:e4b-it-qat", num_predict=8000, think=True),  # ~4B edge anchor (reasoning on)
    "gemma4-e2b-qat": _ollama_builder("gemma4:e2b-it-qat", num_predict=8000, think=True),
    # OpenRouter free panel (needs OPENROUTER_API_KEY; ":free" models are rate-capped)
    "owl-alpha": _openrouter_builder("openrouter/owl-alpha"),
    "nemotron-ultra-550b": _openrouter_builder("nvidia/nemotron-3-ultra-550b-a55b:free"),
    "nemotron-super-120b": _openrouter_builder("nvidia/nemotron-3-super-120b-a12b:free"),
    "qwen3-next-80b": _openrouter_builder("qwen/qwen3-next-80b-a3b-instruct:free"),
    "gemma4-31b": _openrouter_builder("google/gemma-4-31b-it:free"),
    # Self-hosted twin of gemma4-31b: same model at pinned Q8 weights on an O2 L40S,
    # reached via SSH port-forward (localhost:8431). Reproducibility cross-check vs the
    # drift-prone :free hosted endpoint. Requires the serve_g31b.sbatch job + tunnel up.
    "gemma4-31b-o2": _o2_llama_builder("gemma-4-31b", 8431),
    # Self-hosted Qwen3.5-122B-A10B (newest-gen Qwen flagship-tier MoE, 10B active) at Q4 on O2 L40S,
    # reached at localhost:8441. New reproducible primary + new Qwen-3.5 lineage (Apache-2.0).
    "qwen35-122b-o2": _o2_llama_builder("qwen3.5-122b", 8441, extra_body=_QWEN_NOTHINK),
    # Qwen3.5 size-ladder (self-hosted O2): same generation, vary size -> steerability-vs-size curve.
    # Thinking DISABLED: Qwen3.5 reasoning runs away on ward prompts (3k+ tok/step -> truncates, ~8min/episode).
    "qwen35-27b-o2": _o2_llama_builder("qwen3.5-27b", 8442, extra_body=_QWEN_NOTHINK),          # 27B dense
    "qwen35-35b-a3b-o2": _o2_llama_builder("qwen3.5-35b-a3b", 8443, extra_body=_QWEN_NOTHINK),  # 35B-A3B MoE
    # Mistral Medium 3.5 (128B dense, open Modified-MIT) — new vendor + the thinking ON/OFF experiment.
    "mistral-medium35-o2": _o2_llama_builder("mistral-medium-3.5", 8451),                          # reasoning OFF
    "mistral-medium35-think-o2": _o2_llama_builder("mistral-medium-3.5", 8452,
                                                   extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}}),  # reasoning ON
    # Llama-4-Scout (109B/17B MoE, MIT) — new Meta vendor, self-hosted Q4. Not a reasoning model -> default.
    "llama4-scout-o2": _o2_llama_builder("llama-4-scout", 8461),
    # Qwen3.6-35B-A3B (newest-gen Qwen MoE) — thinking disabled (Qwen reasoning runs away).
    "qwen36-35b-o2": _o2_llama_builder("qwen3.6-35b", 8462, extra_body=_QWEN_NOTHINK),
    # Microsoft Phi-4 (14B, MIT) — new Microsoft vendor (synthetic-data training). Default (not reasoning).
    "phi4-o2": _o2_llama_builder("phi-4", 8463),
    # IBM Granite-4.1-30B (dense, Apache) — new IBM enterprise vendor. Default.
    "granite41-30b-o2": _o2_llama_builder("granite-4.1-30b", 8464),
    # Gemma-4-12B (Q8) — gemma-4 size point (vs 31B) + same-size generation contrast vs gemma-3-12B. Thinking on.
    "gemma4-12b-o2": _o2_llama_builder("gemma-4-12b", 8432),
    # Gemma-4-E4B (Q8) — smallest gemma-4 (efficiency tier); completes gemma-4 size ladder E4B<12B<31B. Thinking on.
    "gemma4-e4b-o2": _o2_llama_builder("gemma-4-e4b", 8433),
    # GLM-4.7-Flash (30B-A3B MoE, Z.ai/Zhipu) — NEW vendor lineage. Default (validate thinking on first rows).
    "glm47-flash-o2": _o2_llama_builder("glm-4.7-flash", 8465),
    # Mistral-Small-3.2-24B (dense, instruct) — Mistral size point vs Medium-3.5. Default.
    "mistral-small32-o2": _o2_llama_builder("mistral-small-3.2", 8466),
    # ---- New vendor lineages (2026-06-30) to balance the pool (less Qwen/Gemma-heavy) ----
    # Cohere Command A+ (218B-A25B MoE, Apache 2.0) — NEW vendor. THINKING model -> run reasoning-off (G_RB=0).
    "cohere-command-a-o2": _o2_llama_builder("cohere-command-a-plus", 8467),
    # AllenAI OLMo-2-32B (fully-open, Apache 2.0) — NEW vendor (academic). Standard instruct.
    "olmo2-32b-o2": _o2_llama_builder("olmo-2-32b", 8468),
    # 01.AI Yi-1.5-34B (Apache 2.0) — NEW vendor. Standard chat.
    "yi-34b-o2": _o2_llama_builder("yi-1.5-34b", 8469),
    # TII Falcon3-10B (UAE) — NEW vendor. Standard instruct.
    "falcon3-10b-o2": _o2_llama_builder("falcon3-10b", 8470),
    # ---- Phi ladder + InternLM (2026-07-01) ----
    "phi4-mini-o2": _o2_llama_builder("phi-4-mini", 8471),            # 3.8B, standard
    "phi4-reasoning-o2": _o2_llama_builder("phi-4-reasoning", 8472),  # 14B, THINKING -> run reasoning-on, big slot
    "internlm3-8b-o2": _o2_llama_builder("internlm3-8b", 8473),       # Shanghai AI Lab, standard
    "medgemma27b-o2": _o2_llama_builder("medgemma-27b-text-it", 8474),    # MedGemma 27B text-only (Gemma-3 medical fine-tune), domain-specialized
    "medgemma-1.5-4b-o2": _o2_llama_builder("medgemma-1.5-4b-it", 8475),  # MedGemma 1.5 4B (newest medical Gemma-3), small domain-specialized
    # ---- New free-OR descriptive models (2026-07-01) ----
    "qwen3-coder-480b": _openrouter_builder("qwen/qwen3-coder:free"),
    "deepseek-r1": _openrouter_builder("deepseek/deepseek-r1:free"),
    "north-mini-code": _openrouter_builder("cohere/north-mini-code:free"),
    "qwen3-next-80b": _openrouter_builder("qwen/qwen3-next-80b-a3b-instruct:free"),
    "llama3.2-3b": _openrouter_builder("meta-llama/llama-3.2-3b-instruct:free"),

    "llama3.3-70b": _openrouter_builder("meta-llama/llama-3.3-70b-instruct:free"),
    "hermes3-405b": _openrouter_builder("nousresearch/hermes-3-llama-3.1-405b:free"),
    "glm4.5-air": _openrouter_builder("z-ai/glm-4.5-air:free"),
    "gpt-oss-120b": _openrouter_builder("openai/gpt-oss-120b:free"),
    "nemotron-nano-30b": _openrouter_builder("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    "laguna-xs2": _openrouter_builder("poolside/laguna-xs.2:free"),
    "lfm-2.5-1.2b": _openrouter_builder("liquid/lfm-2.5-1.2b-instruct:free"),
    "kimi-k2.6": _openrouter_builder("moonshotai/kimi-k2.6:free"),
    # Free descriptive-panel additions (2026-06-19; ":free" daily-quota, rate-capped) — new vendors/sizes
    "gpt-oss-20b": _openrouter_builder("openai/gpt-oss-20b:free"),                 # OpenAI open, 20B (size contrast vs 120B)
    "nex-n2-pro": _openrouter_builder("nex-agi/nex-n2-pro:free"),                  # Nex AGI (new vendor lineage)
    "cohere-north": _openrouter_builder("cohere/north-mini-code:free"),           # Cohere (new vendor lineage)
    "qwen3-coder": _openrouter_builder("qwen/qwen3-coder:free"),                   # Alibaba Qwen3-Coder 480B (large)
    "deepseek-v4-flash": _openrouter_builder("deepseek/deepseek-v4-flash"),
    "deepseek-v4-pro": _openrouter_builder("deepseek/deepseek-v4-pro"),         # same family, large tier
    "gemini-2.5-flash": _gemini_flash_builder(),
    "qwen3-max": _openrouter_builder("qwen/qwen3-max"),
    "qwen3.7-max": _openrouter_builder("qwen/qwen3.7-max"),       # frontier Qwen (50% off currently) -> 8th primary
    "minimax-m3": _openrouter_builder("minimax/minimax-m3"),
    "mistral-medium": _openrouter_builder("mistralai/mistral-medium-3.1"),
    "mistral-small": _openrouter_builder("mistralai/mistral-small-3.2-24b-instruct"),
    "glm-4.6": _openrouter_builder("z-ai/glm-4.6"),
    "claude-haiku45": _openrouter_builder("anthropic/claude-haiku-4.5"),
    "llama-4-maverick": _openrouter_builder("meta-llama/llama-4-maverick"),  # Meta T1 panel slot
}


def build_adapter(model_id: str) -> ModelAdapter:
    if model_id in BUILDERS:
        return BUILDERS[model_id]()
    e2_dir = _rev_entries()
    if model_id in e2_dir:
        return e2_dir[model_id]()
    raise KeyError(f"unknown model id {model_id}; known: {list(BUILDERS) + sorted(e2_dir)}")
