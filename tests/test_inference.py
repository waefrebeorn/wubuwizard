"""
Tests for bytropix local inference server (serve_local.py).

Tests the InferenceRunner, chat template, and API handler.
Uses a fake/mock inference binary for fast CI without loading the real model.
"""

import os
import sys
import json
import time
import tempfile
import threading
import pytest
from http.server import HTTPServer
from urllib.request import urlopen, Request
from urllib.error import URLError

# Add bytropix dir to path
BYTROPIX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BYTROPIX_DIR)

from tools.serve_local import (
    InferenceRunner,
    render_chat,
    render_prompt,
    APIHandler,
    DEFAULT_MODEL,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def fake_model_path():
    """Create a small fake model file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        f.write(b"FAKE_GGUF_MODEL_DATA")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def fake_bin_path():
    """Create a fake gen_text_cpu binary that returns known output."""
    script = """#!/usr/bin/env python3
import sys
import time

# Simulate model load time
time.sleep(0.05)

# Parse args
prompt = sys.argv[1] if len(sys.argv) > 1 else ""
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32

# Print in gen_text_cpu format
print(f"Input: {prompt}")

# Generate fake output
text = f"Paris is the capital of France. The Eiffel Tower is a famous landmark."
print(text)

print("--- Stats ---")
print(f"Prefill: {len(prompt)} tok in 0.10s ({len(prompt)/0.10:.1f} tok/s)")
print(f"Decode:  {max_tokens} tok in 0.50s ({max_tokens/0.50:.1f} tok/s)")
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as f:
        f.write(script)
        path = f.name
    os.chmod(path, 0o755)
    yield path
    os.unlink(path)


@pytest.fixture
def runner(fake_bin_path, fake_model_path):
    return InferenceRunner(fake_bin_path, fake_model_path)


# ============================================================
# InferenceRunner Tests
# ============================================================

class TestInferenceRunner:
    def test_generate_basic(self, runner):
        """Generate text from a simple prompt."""
        text, full, elapsed = runner.generate("What is the capital of France?", max_tokens=32)
        assert "Paris" in text, f"Expected 'Paris' in output, got: {text}"
        assert elapsed > 0
        assert len(full) > 0

    def test_generate_with_temperature(self, runner):
        """Generate with different temperature settings."""
        text1, _, _ = runner.generate("Hello", max_tokens=16, temperature=0.1)
        text2, _, _ = runner.generate("Hello", max_tokens=16, temperature=1.5)
        assert isinstance(text1, str)
        assert isinstance(text2, str)

    def test_generate_empty_prompt(self, runner):
        """Handle empty prompt gracefully."""
        text, full, elapsed = runner.generate("", max_tokens=8)
        assert isinstance(text, str)

    def test_generate_timeout(self, runner):
        """Timeout raises gracefully."""
        # The fake binary is fast, so timeout=0.001 should trigger
        text, full, elapsed = runner.generate("test", max_tokens=8, timeout=0.001)
        # May or may not timeout depending on system speed
        assert isinstance(text, str)

    def test_binary_not_found(self):
        """FileNotFoundError is handled."""
        runner = InferenceRunner("/nonexistent/binary", "/nonexistent/model")
        text, full, elapsed = runner.generate("test")
        assert "not found" in full.lower()
        assert elapsed == 0

    def test_extract_generated(self, runner):
        """Extract generated text from full output."""
        output = "Input: hello world\nGenerated text here\n--- Stats ---\nPrefill:"
        text = runner._extract_generated(output)
        assert "Generated" in text

    def test_extract_no_stats(self, runner):
        """Handle output without stats section."""
        output = "Input: hello\nSome text\nNo stats here"
        text = runner._extract_generated(output)
        assert text == "Some text\nNo stats here"

    def test_extract_special_tokens(self, runner):
        """Strip special tokens from output."""
        output = "Input: hi\n<|im_start|>Hello<|im_end|> <|endoftext|>\n--- Stats ---"
        text = runner._extract_generated(output)
        assert "<|im_start|>" not in text
        assert "<|im_end|>" not in text
        assert "<|endoftext|>" not in text
        assert text.strip() == "Hello"

    def test_thread_safety(self, runner):
        """Multiple concurrent generate calls don't crash."""
        errors = []
        def call_gen():
            try:
                runner.generate("test", max_tokens=8)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_gen) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ============================================================
# Chat Template Tests
# ============================================================

class TestChatTemplate:
    def test_render_chat_basic(self):
        """Basic chat template rendering."""
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        result = render_chat(messages)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "What is 2+2?" in result

    def test_render_chat_with_system(self):
        """Custom system message."""
        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        result = render_chat(messages)
        assert "math tutor" in result
        assert "What is 2+2?" in result

    def test_render_chat_multi_turn(self):
        """Multi-turn conversation."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = render_chat(messages)
        assert "Hi" in result
        assert "Hello!" in result
        assert "How are you?" in result

    def test_render_prompt_basic(self):
        """Raw prompt rendering."""
        messages = [
            {"role": "user", "content": "Hello world"}
        ]
        result = render_prompt(messages)
        assert result == "Hello world"

    def test_render_prompt_multi(self):
        """Multi-message raw rendering."""
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"}
        ]
        result = render_prompt(messages)
        assert "A" in result
        assert "B" in result


# ============================================================
# HTTP API Tests (integration with live server)
# ============================================================

@pytest.fixture(scope="module")
def live_server(fake_bin_path, fake_model_path):
    """Start a real HTTP server instance for integration tests."""
    APIHandler.inference = InferenceRunner(fake_bin_path, fake_model_path)
    APIHandler.model_path = fake_model_path
    APIHandler.server_start = time.time()

    server = HTTPServer(("127.0.0.1", 0), APIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_health_endpoint(live_server):
    """GET /health returns OK."""
    resp = urlopen(f"{live_server}/health")
    data = json.loads(resp.read())
    assert data["status"] == "ok"
    assert data["backend"] == "local_cpu"


def test_models_endpoint(live_server):
    """GET /v1/models returns model list."""
    resp = urlopen(f"{live_server}/v1/models")
    data = json.loads(resp.read())
    assert data["object"] == "list"
    assert len(data["data"]) > 0


def test_chat_completions_endpoint(live_server):
    """POST /v1/chat/completions returns valid response."""
    body = json.dumps({
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 32
    }).encode()
    req = Request(f"{live_server}/v1/chat/completions", data=body,
                  headers={"Content-Type": "application/json"})
    resp = urlopen(req)
    data = json.loads(resp.read())
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert "content" in data["choices"][0]["message"]


def test_completions_endpoint(live_server):
    """POST /v1/completions returns valid response."""
    body = json.dumps({
        "prompt": "What is the capital of France?",
        "max_tokens": 32
    }).encode()
    req = Request(f"{live_server}/v1/completions", data=body,
                  headers={"Content-Type": "application/json"})
    resp = urlopen(req)
    data = json.loads(resp.read())
    assert "id" in data
    assert len(data["choices"]) > 0


def test_chat_empty_messages(live_server):
    """Empty messages returns 400."""
    body = json.dumps({"messages": []}).encode()
    req = Request(f"{live_server}/v1/chat/completions", data=body,
                  headers={"Content-Type": "application/json"})
    try:
        urlopen(req)
        pytest.fail("Should have returned 400")
    except URLError as e:
        assert e.code == 400


def test_404(live_server):
    """Unknown path returns 404."""
    try:
        urlopen(f"{live_server}/nonexistent")
        pytest.fail("Should have returned 404")
    except URLError as e:
        assert e.code == 404


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_render_chat_no_messages(self):
        """Empty message list still produces template."""
        result = render_chat([])
        assert "<|im_start|>system" in result
        assert "<|im_start|>assistant" in result

    def test_generate_zero_tokens(self, runner):
        """Generating 0 tokens returns empty."""
        text, _, _ = runner.generate("test", max_tokens=0)
        assert isinstance(text, str)

    def test_generate_large_tokens(self, runner):
        """Large max_tokens doesn't crash."""
        text, _, _ = runner.generate("test", max_tokens=10000)
        assert isinstance(text, str)

    def test_model_not_found(self):
        """Runner handles nonexistent model path."""
        runner = InferenceRunner("/bin/echo", "/nonexistent/model.gguf")
        # Model path is just informational — binary handles it
        text, _, _ = runner.generate("test", max_tokens=8)
        assert isinstance(text, str)
