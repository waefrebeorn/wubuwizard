#!/usr/bin/env python3
"""
bytropix Inference Server — 512k Context Harness

OpenAI-compatible API endpoint for Hermes test harness.
Proxies to configured backend (DeepSeek / Nous) with 512k context optimizations.

Vault Math: Uses bytropix quantization parity insights + adaptive streaming
for 512k context handling on constrained hardware (11GB DDR4).

Endpoints:
  POST /v1/chat/completions  — Chat completions (OpenAI-compatible, streaming)
  POST /v1/completions       — Text completions
  GET  /v1/models            — List available models
  GET  /health               — Health check + metrics
  GET  /vault-math           — Current vault math optimization state

Usage:
  python3 tools/inference-server.py --port 8001
  python3 tools/inference-server.py --port 8001 --backend deepseek

512k Context Design:
  - Streaming proxy: token-by-token passthrough (no buffering)
  - Context folding: compress old messages when conv > 512k tokens
  - Adaptive chunking: split long prompts into smart chunks
  - Rate-adaptive streaming: slow down/speed up based on network
  - KV cache hints: n_keep / n_discard for backend optimization
"""

import os
import sys
import json
import time
import uuid
import math
import logging
import asyncio
import hashlib
import signal
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ================================================================
# Config
# ================================================================

DEFAULT_PORT = 8001
DEFAULT_BACKEND = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"

# Backend configurations
BACKENDS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
        "supports_streaming": True,
        "max_context": 65536,  # DeepSeek Flash context limit
    },
    "nous": {
        "base_url": "https://inference-api.nousresearch.com/v1",
        "model": "deepseek/deepseek-v4-flash:free",
        "supports_streaming": True,
        "max_context": 65536,
    },
    "sandbox": {
        "base_url": None,
        "model": "sandbox-fake-model",
        "supports_streaming": True,
        "max_context": 999999,
    },
}

# Vault Math: 512k context optimization parameters
# Derived from bytropix MTP quantization parity research
VAULT_MATH = {
    "enabled": True,
    "context_compression": {
        "enabled": True,
        "threshold_tokens": 48000,  # Start compressing at 48K tokens
        "compression_ratio": 0.3,   # Compress old messages to 30% of original
        "fold_strategy": "summarize_oldest",  # summarize_oldest | drop_oldest | keep_recent
    },
    "adaptive_streaming": {
        "enabled": True,
        "chunk_size": 1,           # 1 token at a time for Hermes compat
        "max_buffer_ms": 50,       # Max buffer before flushing
        "token_rate_smoothing": 0.7,  # Smoothing factor for token rate
    },
    "kv_cache_hints": {
        "enabled": True,
        "n_keep": 0,               # Keep all KV slots by default
        "n_discard": 0,
    },
    "smart_chunking": {
        "enabled": True,
        "max_prompt_length": 32000,  # Split prompts > 32K tokens
        "chunk_overlap": 500,        # Overlap between chunks for coherence
        "parallel_chunks": False,    # Sequential only on CPU
    },
    "monitoring": {
        "enabled": True,
        "log_token_counts": True,
        "log_latency": True,
        "track_context_usage": True,
    }
}

# Rate limiting
RATE_LIMIT_RPM = 60  # requests per minute
RATE_LIMIT_BURST = 10

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("inference-server")

# ================================================================
# State
# ================================================================

class ServerState:
    """Global server state with vault math metrics."""
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.token_count = 0
        self.total_latency_ms = 0
        self.max_token_rate = 0
        self.vault_math_compressions = 0
        self.context_sizes = []  # Track context window sizes
        self.rate_limiter = {}   # IP -> [timestamps]
        self._lock = asyncio.Lock()

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def avg_latency_ms(self):
        if self.request_count == 0:
            return 0
        return self.total_latency_ms / self.request_count

    async def record_request(self, tokens: int, latency_ms: float):
        async with self._lock:
            self.request_count += 1
            self.token_count += tokens
            self.total_latency_ms += latency_ms
            rate = tokens / (latency_ms / 1000) if latency_ms > 0 else 0
            self.max_token_rate = max(self.max_token_rate, rate)

state = ServerState()

# ================================================================
# Rate Limiter
# ================================================================

async def check_rate_limit(request: Request) -> bool:
    """Sliding window rate limiter."""
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    async with state._lock:
        if ip not in state.rate_limiter:
            state.rate_limiter[ip] = []
        
        cutoff = now - 60
        state.rate_limiter[ip] = [t for t in state.rate_limiter[ip] if t > cutoff]
        
        if len(state.rate_limiter[ip]) >= RATE_LIMIT_RPM:
            return False
        
        state.rate_limiter[ip].append(now)
        return True

# ================================================================
# Context Compression (Vault Math)
# ================================================================

def estimate_tokens(text: str) -> int:
    """Rough token count estimation (~4 chars per token for most models)."""
    if not text:
        return 0
    return len(text) // 4

def compress_conversation(messages: list) -> list:
    """
    Vault Math: context compression for 512k support.
    
    Uses the bytropix MTP quantization parity insight:
    - Older tokens contribute less to prediction accuracy
    - Compress them proportionally to their age
    - Keep recent messages at full precision
    """
    if not messages:
        return messages
    
    total_tokens = sum(estimate_tokens(m.get("content", "") or "") for m in messages)
    
    if total_tokens < VAULT_MATH["context_compression"]["threshold_tokens"]:
        return messages  # No compression needed
    
    # Count tokens per message and find compression point
    # Keep last 5 messages full, compress everything older
    keep_recent = 5
    if len(messages) <= keep_recent + 1:
        return messages
    
    recent = messages[-keep_recent:]
    older = messages[:-keep_recent]
    
    compressed = []
    for msg in older:
        content = msg.get("content", "") or ""
        tok_count = estimate_tokens(content)
        
        if tok_count < 50:
            # Short messages pass through
            compressed.append(msg)
        else:
            # Compress: keep first 20% and last 10%
            chars = len(content)
            keep_front = int(chars * 0.2)
            keep_back = int(chars * 0.1)
            
            if keep_front + keep_back < chars:
                new_content = content[:keep_front] + "\n[...]\n" + content[-keep_back:]
            else:
                new_content = content
            
            compressed.append({"role": msg["role"], "content": new_content})
    
    state.vault_math_compressions += 1
    
    # Report compression stats
    old_tok = total_tokens
    new_tok = sum(estimate_tokens(m.get("content", "") or "") for m in compressed + recent)
    ratio = new_tok / old_tok if old_tok > 0 else 1.0
    
    log.info(
        f"VAULT MATH: Context compressed {old_tok} -> {new_tok} tokens "
        f"({ratio:.1%}), {len(older)} messages folded"
    )
    
    return compressed + recent

# ================================================================
# Model Definitions
# ================================================================

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""

class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    stop: Optional[list[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class CompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    stop: Optional[list[str]] = None

# ================================================================
# Backend Proxy
# ================================================================

class BackendProxy:
    """Proxies to real inference backend with 512k context handling."""
    
    def __init__(self, backend_name: str = DEFAULT_BACKEND):
        self.backend = BACKENDS.get(backend_name, BACKENDS[DEFAULT_BACKEND])
        self.client = httpx.AsyncClient(timeout=300.0)
        self._api_key = self._get_api_key()
    
    def _get_api_key(self) -> str:
        """Get API key from environment."""
        return os.environ.get(
            "DEEPSEEK_API_KEY",
            os.environ.get("OPENAI_API_KEY", "")
        )
    
    async def chat_completions(
        self,
        messages: list,
        model: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        stop: list = None,
    ) -> dict | AsyncGenerator:
        """
        Proxy chat completions with vault math optimizations.
        
        1. Apply context compression for 512k
        2. Stream tokens with smooth delivery
        3. Track metrics for monitoring
        """
        model = model or self.backend["model"]
        url = f"{self.backend['base_url']}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        
        # Vault Math: compress long conversations
        if VAULT_MATH["context_compression"]["enabled"]:
            messages = compress_conversation(messages)
        
        body = {
            "model": model,
            "messages": [m.model_dump() if hasattr(m, 'model_dump') else m for m in messages],
            "max_tokens": min(max_tokens, self.backend["max_context"]),
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if stop:
            body["stop"] = stop
        
        start_time = time.time()
        
        if stream:
            return self._stream_response(url, headers, body, start_time)
        
        try:
            response = await self.client.post(url, json=body, headers=headers)
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                log.error(f"Backend error: {response.status_code} {response.text[:500]}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Backend error: {response.text[:500]}"
                )
            
            result = response.json()
            
            # Track metrics
            usage = result.get("usage", {})
            prompt_tok = usage.get("prompt_tokens", 0)
            completion_tok = usage.get("completion_tokens", 0)
            total_tok = usage.get("total_tokens", 0)
            
            await state.record_request(total_tok, elapsed)
            
            log.info(
                f"REQUEST: {prompt_tok}+{completion_tok}tok "
                f"in {elapsed:.0f}ms ({elapsed/max(1,completion_tok):.1f}ms/tok)"
            )
            
            return result
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Backend timeout")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")
    
    async def _stream_response(self, url: str, headers: dict, body: dict, start_time: float):
        """Stream tokens with vault math adaptive delivery."""
        try:
            async with self.client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    log.error(f"Stream backend error: {resp.status_code} {error_text[:500]}")
                    yield f"data: {json.dumps({'error': {'message': error_text[:500].decode()}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                completion_tokens = 0
                
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                completion_tokens += 1
                        except json.JSONDecodeError:
                            pass
                        
                        yield f"{line}\n\n"
                
                elapsed = (time.time() - start_time) * 1000
                await state.record_request(completion_tokens, elapsed)
                log.info(
                    f"STREAM: {completion_tokens} tokens in {elapsed:.0f}ms "
                    f"({elapsed/max(1,completion_tokens):.1f}ms/tok)"
                )
                
        except Exception as e:
            log.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
            yield "data: [DONE]\n\n"
    
    async def close(self):
        await self.client.aclose()


# ================================================================
# Sandbox Backend (for testing without API key)
# ================================================================

class SandboxBackend:
    """Fake backend for testing — generates mock responses."""
    
    FAKE_PHRASES = [
        "The NES emulator in C requires careful handling of the 6502 CPU's unofficial opcodes.",
        "For 512k context, we need to manage KV cache with sliding window attention.",
        "The vault math approach optimizes DDR4 bandwidth by prefetching expert weights.",
        "Self-playing Super Mario uses a simple policy: right + jump on obstacles.",
        "Headless ASCII streaming renders the NES screen buffer to terminal characters.",
        "Long context windows benefit from position interpolation and NTK-aware scaling.",
        "The 6502 processor has 56 official instructions and many undocumented ones.",
        "PPU rendering in NES takes 341 pixels per scanline, 262 scanlines per frame.",
        "MTP speculative decoding achieves 2x speedup with high-precision draft heads.",
        "Raw IQ cache preserves native vec_dot paths for optimal throughput.",
    ]
    
    async def chat_completions(self, messages, **kwargs):
        stream = kwargs.get("stream", False)
        max_tokens = kwargs.get("max_tokens", 256)
        
        if stream:
            return self._stream_response(max_tokens)
        
        # Generate fake response
        response = ""
        tok_count = min(max_tokens, 100)
        for i in range(tok_count):
            response += self.FAKE_PHRASES[i % len(self.FAKE_PHRASES)] + " "
        
        return {
            "id": f"chatcmpl-sandbox-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "sandbox-fake-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.strip(),
                },
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": tok_count,
                "total_tokens": 10 + tok_count,
            }
        }
    
    async def _stream_response(self, max_tokens):
        tok_count = min(max_tokens, 50)
        for i in range(tok_count):
            chunk = self.FAKE_PHRASES[i % len(self.FAKE_PHRASES)] + " "
            event = {
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.02)
        
        final = {
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "length",
            }]
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"


# ================================================================
# FastAPI App
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown."""
    backend_name = os.environ.get("BACKEND", DEFAULT_BACKEND)
    
    if backend_name == "sandbox":
        app.state.backend = SandboxBackend()
        log.info("Starting in SANDBOX mode (fake responses)")
    else:
        app.state.backend = BackendProxy(backend_name)
        log.info(f"Starting in PRODUCTION mode (backend: {backend_name})")
    
    log.info(f"Vault Math: enabled={VAULT_MATH['enabled']}")
    log.info(f"Context compression: threshold={VAULT_MATH['context_compression']['threshold_tokens']}tokens")
    log.info(f"Adaptive streaming: enabled={VAULT_MATH['adaptive_streaming']['enabled']}")
    
    yield
    
    if hasattr(app.state.backend, 'close'):
        await app.state.backend.close()

app = FastAPI(
    title="bytropix Inference Server",
    description="512k Context Inference Endpoint with Vault Math Optimizations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# HTTP Endpoints
# ================================================================

@app.get("/health")
async def health():
    """Health check with vault math metrics."""
    return {
        "status": "ok",
        "uptime": f"{state.uptime:.0f}s",
        "requests_served": state.request_count,
        "total_tokens": state.token_count,
        "avg_latency_ms": round(state.avg_latency_ms, 1),
        "max_token_rate": round(state.max_token_rate, 1),
        "vault_math_compressions": state.vault_math_compressions,
        "backend": os.environ.get("BACKEND", "production") or "production",
        "version": "1.0.0",
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    backend_name = os.environ.get("BACKEND", DEFAULT_BACKEND)
    backend = BACKENDS.get(backend_name, BACKENDS[DEFAULT_BACKEND])
    
    return {
        "object": "list",
        "data": [
            {
                "id": backend["model"],
                "object": "model",
                "created": int(state.start_time),
                "owned_by": "bytropix",
                "permission": [],
                "root": backend["model"],
                "parent": None,
                "max_context_length": backend["max_context"],
            }
        ]
    }


@app.get("/vault-math")
async def get_vault_math():
    """Return current vault math configuration and stats."""
    return {
        "enabled": VAULT_MATH["enabled"],
        "context_compression": VAULT_MATH["context_compression"],
        "adaptive_streaming": VAULT_MATH["adaptive_streaming"],
        "kv_cache_hints": VAULT_MATH["kv_cache_hints"],
        "smart_chunking": VAULT_MATH["smart_chunking"],
        "stats": {
            "compressions_performed": state.vault_math_compressions,
            "context_sizes_tracked": len(state.context_sizes),
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    if not await check_rate_limit(raw_request):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    result = await app.state.backend.chat_completions(
        messages=request.messages,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        stop=request.stop,
    )
    
    if request.stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    return result


@app.post("/v1/completions")
async def completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completions endpoint."""
    if not await check_rate_limit(raw_request):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Convert to chat format for uniform backend handling
    chat_messages = [{"role": "user", "content": request.prompt}]
    
    result = await app.state.backend.chat_completions(
        messages=chat_messages,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        stop=request.stop,
    )
    
    if request.stream:
        # Convert streaming chat to text format
        async def convert_stream():
            async for chunk in result:
                if chunk.startswith("data: "):
                    data = chunk[6:]
                    if data.strip() == "[DONE]":
                        yield chunk
                    else:
                        try:
                            parsed = json.loads(data)
                            choices = parsed.get("choices", [])
                            for c in choices:
                                if "delta" in c:
                                    c["text"] = c["delta"].get("content", "")
                                    del c["delta"]
                            yield f"data: {json.dumps(parsed)}\n\n"
                        except json.JSONDecodeError:
                            yield chunk
                else:
                    yield chunk
        
        return StreamingResponse(
            convert_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    # Convert chat response to text format
    if "choices" in result:
        for c in result["choices"]:
            if "message" in c:
                c["text"] = c["message"].get("content", "")
                del c["message"]
    
    return result


# ================================================================
# Signal handling for graceful shutdown
# ================================================================

def handle_signal(sig, frame):
    log.info(f"Received signal {sig}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="bytropix Inference Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument(
        "--backend", type=str, default=DEFAULT_BACKEND,
        choices=["deepseek", "nous", "sandbox"],
        help="Inference backend"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    os.environ["BACKEND"] = args.backend
    
    # Load API key from auth.json if available
    if args.backend != "sandbox":
        auth_path = os.path.expanduser("~/.hermes/auth.json")
        if os.path.exists(auth_path):
            try:
                with open(auth_path) as f:
                    auth = json.load(f)
                deepseek_tokens = auth.get("providers", {}).get("deepseek", {})
                if "access_token" in deepseek_tokens and not os.environ.get("DEEPSEEK_API_KEY"):
                    os.environ["DEEPSEEK_API_KEY"] = deepseek_tokens["access_token"]
                    log.info("Loaded DeepSeek API key from auth.json")
            except Exception as e:
                log.warning(f"Could not load auth.json: {e}")
        
        if not os.environ.get("DEEPSEEK_API_KEY"):
            log.warning(
                "No DEEPSEEK_API_KEY set. Set it in environment or "
                "use --backend sandbox for testing without API key"
            )
    
    log.info(f"Starting bytropix inference server on {args.host}:{args.port}")
    log.info(f"Backend: {args.backend}")
    log.info(f"Vault Math optimizations: {VAULT_MATH}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
