"""Unit tests for chronos_traffic_bench.py."""
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "chronos_traffic_bench",
    Path(__file__).with_name("chronos_traffic_bench.py"),
)
bench = importlib.util.module_from_spec(SPEC)
with patch.dict(os.environ, {"API_KEY": "", "API_KEY_FILE": "/dev/null"}):
    SPEC.loader.exec_module(bench)


class MockSSEHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Silence HTTP logs in test output
        pass

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self.send_error(404, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        req_data = json.loads(body_bytes.decode("utf-8"))

        if hasattr(self.server, "recorded_requests"):
            self.server.recorded_requests.append({
                "headers": dict(self.headers),
                "body": req_data,
            })

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # If custom generator is registered, use it
        if hasattr(self.server, "custom_stream_fn") and self.server.custom_stream_fn:
            for raw_chunk in self.server.custom_stream_fn(req_data):
                self.wfile.write(raw_chunk.encode("utf-8"))
                self.wfile.flush()
            return

        # Default fast simulated response
        is_seed = req_data.get("max_tokens") == 1
        is_cached = "[agent turn" in str(req_data) or "[cached turn]" in str(req_data)
        prompt_len = sum(len(m.get("content", "").split()) for m in req_data.get("messages", []))

        # Chunk 1: first token
        c1 = {"choices": [{"delta": {"content": "Hello "}, "index": 0}]}
        self.wfile.write(f"data: {json.dumps(c1)}\n\n".encode("utf-8"))
        self.wfile.flush()

        # Chunk 2: second token
        c2 = {"choices": [{"delta": {"content": "world!"}, "index": 0, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(c2)}\n\n".encode("utf-8"))
        self.wfile.flush()

        # Usage chunk
        c_usage = {
            "usage": {
                "prompt_tokens": max(10, prompt_len),
                "completion_tokens": 1 if is_seed else 5,
                "prompt_tokens_details": {
                    "cached_tokens": 32000 if is_cached else 0
                },
            },
            "tensorfold": {
                "token_sha": "fixedmocksha123456789",
            },
            "choices": [],
        }
        self.wfile.write(f"data: {json.dumps(c_usage)}\n\n".encode("utf-8"))
        self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class ChronosTrafficBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockSSEHandler)
        cls.server.recorded_requests = []
        cls.server.custom_stream_fn = None
        cls.port = cls.server.server_address[1]
        cls.url = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()

    def setUp(self):
        self.server.recorded_requests.clear()
        self.server.custom_stream_fn = None

    def test_01_endpoint_and_auth_resolution(self):
        """Test 1: Endpoint/PORT parsing from a temporary local.sh and auth header resolution."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            scripts = tmp / "scripts"
            scripts.mkdir()
            local_sh = scripts / "local.sh"

            # Default fallback when no local.sh and no env
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(bench.default_base(str(local_sh)), "http://127.0.0.1:8888")

            # Reading PORT from local.sh
            local_sh.write_text('PORT=18888\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(bench.default_base(str(local_sh)), "http://127.0.0.1:18888")

            local_sh.write_text('export PORT="8007" # note\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(bench.default_base(str(local_sh)), "http://127.0.0.1:8007")

            # PORT env var overrides local.sh
            with patch.dict(os.environ, {"PORT": "9090"}, clear=True):
                self.assertEqual(bench.default_base(str(local_sh)), "http://127.0.0.1:9090")

            # API_URL env var overrides both
            with patch.dict(os.environ, {"API_URL": "http://remote-cluster:7777/v1", "PORT": "9090"}, clear=True):
                self.assertEqual(bench.default_base(str(local_sh)), "http://remote-cluster:7777/v1")

            # Auth key resolution: API_KEY
            with patch.dict(os.environ, {"API_KEY": "sk-secret-env"}, clear=True):
                self.assertEqual(bench.api_key(), "sk-secret-env")

            # Auth key resolution: API_KEY_FILE
            key_file = tmp / "api-key"
            key_file.write_text("# header comment\nsk-from-file-123\n")
            with patch.dict(os.environ, {"API_KEY": "", "API_KEY_FILE": str(key_file)}, clear=True):
                self.assertEqual(bench.api_key(), "sk-from-file-123")

            key_file.write_text("Authorization: Bearer sk-from-header-line\n")
            with patch.dict(os.environ, {"API_KEY": "", "API_KEY_FILE": str(key_file)}, clear=True):
                self.assertEqual(bench.api_key(), "Bearer sk-from-header-line")

            # Verify Authorization header is sent in requests when key is set
            with patch.dict(os.environ, {"API_KEY": "sk-test-token-bearer"}, clear=True):
                bench.stream_chat(self.url, "test-model", [{"role": "user", "content": "hi"}], max_tokens=1)
                self.assertTrue(len(self.server.recorded_requests) > 0)
                auth_sent = self.server.recorded_requests[-1]["headers"].get("Authorization")
                self.assertEqual(auth_sent, "Bearer sk-test-token-bearer")

            # Verify no Authorization header when key is empty
            with patch.dict(os.environ, {"API_KEY": "", "API_KEY_FILE": "/dev/null"}, clear=True):
                bench.stream_chat(self.url, "test-model", [{"role": "user", "content": "hi"}], max_tokens=1)
                auth_sent = self.server.recorded_requests[-1]["headers"].get("Authorization")
                self.assertIsNone(auth_sent)

    def test_02_sse_streaming_metric_extraction(self):
        """Test 2: SSE streaming metric extraction (ttft_s, total_s, decode_tok_s, cached_tokens, token_sha)."""
        def custom_stream(req):
            time.sleep(0.03)
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Alpha '}}]})}\n\n"
            time.sleep(0.03)
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Beta'}}]})}\n\n"
            usage = {
                "prompt_tokens": 128,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 64},
            }
            yield f"data: {json.dumps({'usage': usage, 'tensorfold': {'token_sha': 'custom-sha-val'}})}\n\n"
            yield "data: [DONE]\n\n"

        self.server.custom_stream_fn = custom_stream

        suite_start = time.monotonic()
        res = bench.stream_chat(
            self.url,
            "DeepSeek-V4.1-Flash-1017",
            [{"role": "user", "content": "hello"}],
            max_tokens=10,
            seed=7,
            suite_start=suite_start,
        )

        self.assertNotIn("error", res)
        self.assertEqual(res["prompt_tokens"], 128)
        self.assertEqual(res["completion_tokens"], 10)
        self.assertEqual(res["cached_tokens"], 64)
        self.assertEqual(res["token_sha"], "custom-sha-val")
        self.assertEqual(res["text"], "Alpha Beta")
        self.assertGreaterEqual(res["ttft_s"], 0.02)
        self.assertGreaterEqual(res["total_s"], 0.05)
        self.assertIsNotNone(res["decode_tok_s"])
        # (10 - 1) / (total_s - ttft_s)
        expected_decode = round(9.0 / (res["total_s"] - res["ttft_s"]), 2)
        self.assertAlmostEqual(res["decode_tok_s"], expected_decode, delta=0.5)
        self.assertIsNotNone(res["started_at_rel_s"])
        self.assertIsNotNone(res["first_token_at_rel_s"])
        self.assertIsNotNone(res["finished_at_rel_s"])

        # Test fallback token_sha when tensorfold block is absent
        def no_tf_stream(req):
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'plain text response'}}]})}\n\n"
            yield f"data: {json.dumps({'usage': {'prompt_tokens': 10, 'completion_tokens': 3}})}\n\n"
            yield "data: [DONE]\n\n"

        self.server.custom_stream_fn = no_tf_stream
        res2 = bench.stream_chat(self.url, "DeepSeek-V4.1-Flash-1017", [{"role": "user", "content": "test"}], max_tokens=5)
        expected_sha = hashlib.sha256("plain text response".encode("utf-8")).hexdigest()
        self.assertEqual(res2["token_sha"], expected_sha)

    def test_03_all_suites_execution_and_json_output(self):
        """Test 3: Running all 3 suites against mock HTTP server, checking natural prompts and lane_reserve 8/4 layout."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "runs" / "test_out.json"

            # Execute via main() CLI flags with minimal delays for fast testing
            cli_args = [
                "chronos_traffic_bench.py",
                "--url", self.url,
                "--suite", "burst,cut_in,lane_reserve",
                "--label", "test-run",
                "--salt", "test1234",
                "--seed", "7",
                "--delay", "0.01",
                "--burst-delay", "0.01",
                "--lane-delay", "0.01",
                "--lane-stagger", "0.001",
                "--long-tokens", "100",
                "--cutin-long-tokens", "100",
                "--out", str(out_file),
            ]

            buf = io.StringIO()
            with patch.object(bench.sys, "argv", cli_args), contextlib.redirect_stdout(buf):
                exit_code = bench.main()

            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            with open(out_file, encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["schema"], "chronos_traffic_bench.v1")
            self.assertEqual(data["label"], "test-run")
            self.assertEqual(data["salt"], "test1234")
            self.assertEqual(data["reps"], 1)
            self.assertEqual(data["url"], self.url)

            # Check burst suite
            self.assertIn("burst", data["suites"])
            burst = data["suites"]["burst"]
            self.assertEqual(len(burst["agent_turns"]), 4)
            self.assertEqual(len(burst["short_utility"]), 4)
            self.assertIn("ttft_s", burst["summary"]["agent_turns"])
            self.assertIn("median", burst["summary"]["agent_turns"]["ttft_s"])
            self.assertIn("p50", burst["summary"]["agent_turns"]["ttft_s"])
            self.assertIn("p95", burst["summary"]["agent_turns"]["ttft_s"])
            self.assertEqual(burst["summary"]["agent_turns"]["count"], 4)
            self.assertEqual(burst["summary"]["short_utility"]["count"], 4)
            self.assertIsNotNone(burst["summary"]["aggregate_decode_tok_s"])

            # Check cut_in suite
            self.assertIn("cut_in", data["suites"])
            cut_in = data["suites"]["cut_in"]
            self.assertIn("long_request", cut_in["summary"])
            self.assertIn("token_sha", cut_in["summary"]["long_request"])
            self.assertIn("short_utility", cut_in["summary"])
            self.assertIn("cached_turn", cut_in["summary"])
            self.assertEqual(cut_in["summary"]["cached_turn"]["cached_tokens"], 32000)
            self.assertEqual(cut_in["summary"]["long_request"]["count"], 1)

            # Check lane_reserve suite: 8 long requests + 4 short utility
            self.assertIn("lane_reserve", data["suites"])
            lane = data["suites"]["lane_reserve"]
            self.assertEqual(len(lane["long_requests"]), 8)
            self.assertEqual(len(lane["short_utility"]), 4)
            self.assertEqual(lane["summary"]["long_requests"]["count"], 8)
            self.assertEqual(lane["summary"]["short_utility"]["count"], 4)
            self.assertIn("p50", lane["summary"]["long_requests"]["ttft_s"])
            self.assertIn("p50", lane["summary"]["short_utility"]["ttft_s"])
            self.assertIsNotNone(lane["summary"]["aggregate_decode_tok_s"])

            # Top-level summary
            self.assertIn("burst", data["summary"])
            self.assertIn("cut_in", data["summary"])
            self.assertIn("lane_reserve", data["summary"])

            # Verify prompt contents and max_tokens in recorded requests
            bodies = [r["body"] for r in self.server.recorded_requests]

            # Lane reserve long requests: max_tokens=1024, contains natural multi-section essay prompt
            lane_long_reqs = [b for b in bodies if b.get("max_tokens") == 1024]
            self.assertEqual(len(lane_long_reqs), 8)
            for req in lane_long_reqs:
                content = req["messages"][0]["content"]
                self.assertIn("Write a comprehensive, multi-section technical essay", content)
                self.assertIn("[ref lane-long-", content)

            # Lane reserve short utility requests: max_tokens=64, contains "Explain " structured topic
            lane_short_reqs = [b for b in bodies if "[ref lane-short-" in b["messages"][0]["content"]]
            self.assertEqual(len(lane_short_reqs), 4)
            for req in lane_short_reqs:
                self.assertEqual(req["max_tokens"], 64)
                content = req["messages"][0]["content"]
                self.assertIn("Explain ", content)
                self.assertIn("Provide a concise structured explanation", content)

            # Burst agent turns: max_tokens=256, contains natural essay prompt
            burst_agent_reqs = [b for b in bodies if "[agent turn" in b["messages"][0]["content"]]
            self.assertEqual(len(burst_agent_reqs), 4)
            for req in burst_agent_reqs:
                self.assertEqual(req["max_tokens"], 256)
                content = req["messages"][0]["content"]
                self.assertIn("Write a comprehensive, multi-section technical essay", content)

            # Cut-in long request: max_tokens=16, summarize in one sentence
            cutin_long_reqs = [b for b in bodies if "[ref cold-long-" in b["messages"][0]["content"]]
            self.assertEqual(len(cutin_long_reqs), 1)
            self.assertEqual(cutin_long_reqs[0]["max_tokens"], 16)
            self.assertIn("Summarize the text above in one sentence.", cutin_long_reqs[0]["messages"][0]["content"])

    def test_04_compare_markdown_table_and_token_sha_verification(self):
        """Test 4: --compare Markdown table rendering and cut_in token_sha verification on synthetic fixtures."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            fixture_a = {
                "schema": "chronos_traffic_bench.v1",
                "label": "baseline",
                "url": "http://127.0.0.1:18888",
                "model": "DeepSeek-V4.1-Flash-1017",
                "salt": "commonsalt1",
                "suites": {
                    "burst": {
                        "summary": {
                            "agent_turns": {
                                "ttft_s": {"median": 1.0, "mean": 1.05, "p95": 1.2},
                                "total_s": {"median": 5.0, "mean": 5.1, "p95": 5.3},
                                "decode_tok_s": {"median": 50.0, "mean": 50.0, "p95": 52.0},
                            },
                            "short_utility": {
                                "ttft_s": {"median": 0.2, "mean": 0.22, "p95": 0.25},
                                "total_s": {"median": 1.0, "mean": 1.05, "p95": 1.1},
                                "decode_tok_s": {"median": 60.0, "mean": 60.0, "p95": 62.0},
                            },
                            "aggregate_decode_tok_s": 80.0,
                        }
                    },
                    "cut_in": {
                        "summary": {
                            "long_request": {"ttft_s": 8.0, "total_s": 9.0, "token_sha": "matchingsha123456"},
                            "short_utility": {"ttft_s": 0.3, "total_s": 0.6},
                            "cached_turn": {"ttft_s": 0.2, "total_s": 0.5},
                        }
                    },
                    "lane_reserve": {
                        "summary": {
                            "short_utility": {
                                "ttft_s": {"median": 0.5, "p95": 0.6},
                                "total_s": {"median": 1.2},
                            },
                            "long_requests": {
                                "ttft_s": {"median": 7.0},
                                "total_s": {"median": 12.0},
                            },
                            "aggregate_decode_tok_s": 100.0,
                        }
                    },
                },
            }

            # Fixture B: Improved metrics, matching salt, matching token_sha
            fixture_b = {
                "schema": "chronos_traffic_bench.v1",
                "label": "optimized",
                "url": "http://127.0.0.1:18888",
                "model": "DeepSeek-V4.1-Flash-1017",
                "salt": "commonsalt1",
                "suites": {
                    "burst": {
                        "summary": {
                            "agent_turns": {
                                "ttft_s": {"median": 0.5, "mean": 0.52, "p95": 0.6},
                                "total_s": {"median": 2.5, "mean": 2.55, "p95": 2.65},
                                "decode_tok_s": {"median": 100.0, "mean": 100.0, "p95": 104.0},
                            },
                            "short_utility": {
                                "ttft_s": {"median": 0.1, "mean": 0.11, "p95": 0.12},
                                "total_s": {"median": 0.5, "mean": 0.52, "p95": 0.55},
                                "decode_tok_s": {"median": 120.0, "mean": 120.0, "p95": 124.0},
                            },
                            "aggregate_decode_tok_s": 160.0,
                        }
                    },
                    "cut_in": {
                        "summary": {
                            "long_request": {"ttft_s": 4.0, "total_s": 4.5, "token_sha": "matchingsha123456"},
                            "short_utility": {"ttft_s": 0.15, "total_s": 0.3},
                            "cached_turn": {"ttft_s": 0.1, "total_s": 0.25},
                        }
                    },
                    "lane_reserve": {
                        "summary": {
                            "short_utility": {
                                "ttft_s": {"median": 0.25, "p95": 0.3},
                                "total_s": {"median": 0.6},
                            },
                            "long_requests": {
                                "ttft_s": {"median": 3.5},
                                "total_s": {"median": 6.0},
                            },
                            "aggregate_decode_tok_s": 200.0,
                        }
                    },
                },
            }

            path_a = tmp / "run_a.json"
            path_b = tmp / "run_b.json"
            path_a.write_text(json.dumps(fixture_a))
            path_b.write_text(json.dumps(fixture_b))

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res_code = bench.compare(str(path_a), str(path_b))

            self.assertEqual(res_code, 0)
            out = buf.getvalue()

            # Verify table header and markdown structure
            self.assertIn("| Suite / Metric | Before (A) | After (B) | Delta (%) | Speedup |", out)
            self.assertIn("| agent_turns ttft_s (median) | 1.000s | 0.500s | -50.0% | 2.00x |", out)
            self.assertIn("| aggregate_decode_tok_s | 80.000 tok/s | 160.000 tok/s | +100.0% | 2.00x |", out)
            self.assertIn("MATCH (matchingsha123456)", out)

            # Fixture C: matching salt but mismatched token_sha
            fixture_c = dict(fixture_b)
            fixture_c["suites"]["cut_in"]["summary"]["long_request"]["token_sha"] = "mismatchedsha999999"
            path_c = tmp / "run_c.json"
            path_c.write_text(json.dumps(fixture_c))

            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                bench.compare(str(path_a), str(path_c))
            out2 = buf2.getvalue()
            self.assertIn("MISMATCH", out2)

            # Fixture D: different salts
            fixture_d = dict(fixture_b)
            fixture_d["salt"] = "differentsalt2"
            path_d = tmp / "run_d.json"
            path_d.write_text(json.dumps(fixture_d))

            buf3 = io.StringIO()
            with contextlib.redirect_stdout(buf3):
                bench.compare(str(path_a), str(path_d))
            out3 = buf3.getvalue()
            self.assertIn("SALTS DIFFER", out3)

    def test_05_multi_reps_and_distinct_salts(self):
        """Test 5: --reps 2 execution, distinct per-rep salts, and multi-rep metric aggregation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "runs" / "reps_out.json"

            cli_args = [
                "chronos_traffic_bench.py",
                "--url", self.url,
                "--suite", "burst,cut_in,lane_reserve",
                "--label", "multirep-run",
                "--salt", "multisalt",
                "--reps", "2",
                "--seed", "7",
                "--delay", "0.01",
                "--burst-delay", "0.01",
                "--lane-delay", "0.01",
                "--lane-stagger", "0.001",
                "--long-tokens", "100",
                "--cutin-long-tokens", "100",
                "--out", str(out_file),
            ]

            buf = io.StringIO()
            with patch.object(bench.sys, "argv", cli_args), contextlib.redirect_stdout(buf):
                exit_code = bench.main()

            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())

            with open(out_file, encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["schema"], "chronos_traffic_bench.v1")
            self.assertEqual(data["reps"], 2)
            self.assertEqual(data["salt"], "multisalt")

            # Verify distinct salts were used in HTTP requests for rep 0 and rep 1
            all_sent_texts = [
                str(r["body"]) for r in self.server.recorded_requests
            ]
            has_r0 = any("multisalt-r0" in t for t in all_sent_texts)
            has_r1 = any("multisalt-r1" in t for t in all_sent_texts)
            self.assertTrue(has_r0, "Expected requests containing multisalt-r0")
            self.assertTrue(has_r1, "Expected requests containing multisalt-r1")

            # Check burst suite with reps=2
            burst = data["suites"]["burst"]
            self.assertEqual(len(burst["reps"]), 2)
            self.assertEqual(len(burst["agent_turns"]), 8)  # 4 * 2 reps
            self.assertEqual(len(burst["short_utility"]), 8)  # 4 * 2 reps
            self.assertEqual(burst["summary"]["agent_turns"]["count"], 8)
            self.assertEqual(burst["summary"]["short_utility"]["count"], 8)
            self.assertIn("p50", burst["summary"]["agent_turns"]["ttft_s"])
            self.assertIn("mean", burst["summary"]["agent_turns"]["ttft_s"])
            self.assertIn("p95", burst["summary"]["agent_turns"]["ttft_s"])
            self.assertEqual(
                burst["summary"]["agent_turns"]["ttft_s"]["p50"],
                burst["summary"]["agent_turns"]["ttft_s"]["median"],
            )

            # Check lane_reserve suite with reps=2
            lane = data["suites"]["lane_reserve"]
            self.assertEqual(len(lane["reps"]), 2)
            self.assertEqual(len(lane["long_requests"]), 16)  # 8 * 2 reps
            self.assertEqual(len(lane["short_utility"]), 8)  # 4 * 2 reps
            self.assertEqual(lane["summary"]["long_requests"]["count"], 16)
            self.assertEqual(lane["summary"]["short_utility"]["count"], 8)
            self.assertIn("p50", lane["summary"]["long_requests"]["ttft_s"])
            self.assertIn("p50", lane["summary"]["short_utility"]["ttft_s"])

            # Check cut_in suite with reps=2
            cut_in = data["suites"]["cut_in"]
            self.assertEqual(len(cut_in["reps"]), 2)
            self.assertEqual(cut_in["summary"]["long_request"]["count"], 2)
            self.assertEqual(cut_in["summary"]["short_utility"]["count"], 2)
            self.assertEqual(cut_in["summary"]["cached_turn"]["count"], 2)
            self.assertIn("p50", cut_in["summary"]["long_request"])

            # Verify --compare works smoothly between reps=1 and reps=2
            out_file_1 = Path(tmp_dir) / "runs" / "reps1_out.json"
            cli_args_1 = [
                "chronos_traffic_bench.py",
                "--url", self.url,
                "--suite", "burst,cut_in,lane_reserve",
                "--label", "single-rep",
                "--salt", "multisalt",
                "--reps", "1",
                "--seed", "7",
                "--delay", "0.01",
                "--burst-delay", "0.01",
                "--lane-delay", "0.01",
                "--lane-stagger", "0.001",
                "--long-tokens", "100",
                "--cutin-long-tokens", "100",
                "--out", str(out_file_1),
            ]
            with patch.object(bench.sys, "argv", cli_args_1), contextlib.redirect_stdout(io.StringIO()):
                bench.main()

            compare_buf = io.StringIO()
            with contextlib.redirect_stdout(compare_buf):
                cmp_code = bench.compare(str(out_file_1), str(out_file))
            self.assertEqual(cmp_code, 0)
            cmp_out = compare_buf.getvalue()
            self.assertIn("| Suite / Metric | Before (A) | After (B) | Delta (%) | Speedup |", cmp_out)
            self.assertIn("- **Reps**: `1` vs `2`", cmp_out)
            self.assertIn("| agent_turns ttft_s (median) |", cmp_out)
            self.assertIn("| long_requests ttft_s (median) |", cmp_out)

            # Create another reps=2 run file with identical salt to verify MATCH
            out_file_2 = Path(tmp_dir) / "runs" / "reps2_match.json"
            cli_args_2 = [
                "chronos_traffic_bench.py",
                "--url", self.url,
                "--suite", "burst,cut_in,lane_reserve",
                "--label", "reps-2-match",
                "--salt", "multisalt",
                "--reps", "2",
                "--seed", "7",
                "--delay", "0.01",
                "--burst-delay", "0.01",
                "--lane-delay", "0.01",
                "--lane-stagger", "0.001",
                "--long-tokens", "100",
                "--cutin-long-tokens", "100",
                "--out", str(out_file_2),
            ]
            with patch.object(bench.sys, "argv", cli_args_2), contextlib.redirect_stdout(io.StringIO()):
                bench.main()

            match_buf = io.StringIO()
            with contextlib.redirect_stdout(match_buf):
                cmp_code2 = bench.compare(str(out_file), str(out_file_2))
            self.assertEqual(cmp_code2, 0)
            match_out = match_buf.getvalue()
            self.assertIn("MATCH (fixedmocksha123456789)", match_out)
            self.assertIn("- **Reps**: `2` vs `2`", match_out)


if __name__ == "__main__":
    unittest.main()
