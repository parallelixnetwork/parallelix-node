#!/usr/bin/env node
// =============================================================================
// parallelix-node · ParalleliX Node operator CLI · v2.0.0 (real GPU compute)
// =============================================================================
//
// The off-chain half of being a ParalleliX operator. Generates a separate node
// key (never your staking wallet), attaches a machine to a node you registered
// on-chain, sends signed liveness heartbeats (your uptime), and serves REAL
// ParalleliX AI inference requests on your local GPU via Ollama, returning a
// SHA-256 Proof-of-Execution the coordinator verifies.
//
//   parallelix-node setup                  One command: hardware, model, identity, stake auto-detect
//   parallelix-node probe                  Detect hardware + Ollama; print the tier
//   parallelix-node init [--wallet 0x..]   Generate node key + config; print nodeKeyHash
//   parallelix-node models [pull <name> | catalog | recommend]   List / pull local Ollama models
//   parallelix-node start --node-id N      Run the daemon: attach, heartbeat, serve
//             [--gpu|--cpu] [--model m]
//   parallelix-node service --node-id N    Install as a background service (systemd/launchd)
//   parallelix-node update                 Self-update to the latest release (sha256-verified)
//   parallelix-node verify                 Run diagnostics
//   parallelix-node status                 Print local node state
//   parallelix-node logs                   Tail the node log
//   parallelix-node version
//   parallelix-node help
//
// Real inference runs on Ollama (http://127.0.0.1:11434). No simulator.
// =============================================================================

import { secp256k1 } from "@noble/curves/secp256k1";
import { keccak_256 } from "@noble/hashes/sha3";
import { sha256 } from "@noble/hashes/sha256";
import { bytesToHex as toHex, hexToBytes } from "@noble/hashes/utils";
import os from "node:os";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

const VERSION = "2.0.0";

// Canonical on-chain references (Ethereum mainnet).
const PRLX = "0x93FF39f65cC1D21067939961993ADF3f36BBF893";
const STAKE_CONTRACT = "0x706851273c3f5892e2d68ff48dd80bea02a382b6"; // NodeRegistryLocker
const REWARDS_CONTRACT = "0x266939a8baa29344c7687ce2b5074af6dec984e3"; // OperatorStakeRewardsV2

const API_URL = (argFlag("--coordinator-url") || process.env.PARALLELIX_API_URL || "https://parallelix.io/api").replace(/\/$/, "");
const OLLAMA = (argFlag("--ollama-url") || process.env.OLLAMA_HOST || "http://127.0.0.1:11434").replace(/\/$/, "");
const DEFAULT_MODEL = argFlag("--model") || process.env.PARALLELIX_MODEL || "llama3.2";

// Curated open-source model catalog. fitGb = VRAM needed (GPU) or RAM budget
// (CPU mode), q4 quantization. Ascending quality order; recommend() picks the
// last entry that fits. Advisory only: any Ollama model name works everywhere.
const MODEL_CATALOG = [
  { id: "llama3.2",       params: "3B",  fitGb: 4,  note: "default; fastest small model" },
  { id: "mistral:7b",     params: "7B",  fitGb: 5,  note: "strong general 7B" },
  { id: "qwen2.5:7b",     params: "7B",  fitGb: 6,  note: "multilingual + code" },
  { id: "deepseek-r1:8b", params: "8B",  fitGb: 6,  note: "reasoning-tuned" },
  { id: "gemma2:9b",      params: "9B",  fitGb: 7,  note: "Google open model" },
  { id: "phi4",           params: "14B", fitGb: 10, note: "best quality under 12 GB" },
  { id: "qwen2.5:14b",    params: "14B", fitGb: 11, note: "step up if VRAM allows" },
  { id: "qwen2.5:32b",    params: "32B", fitGb: 20, note: "tier 3+ cards" },
  { id: "llama3.3:70b",   params: "70B", fitGb: 40, note: "tier 4 / datacenter" },
];

function recommendModel(hw) {
  const budget = hw.gpu ? hw.vramGb * 0.9 : hw.ramGb * 0.5;
  let pick = MODEL_CATALOG[0];
  for (const m of MODEL_CATALOG) if (m.fitGb <= budget) pick = m;
  return { pick, budget: Math.floor(budget) };
}

function printCatalog(installed, markId) {
  console.log(C.b("// model catalog") + C.d("  (any Ollama model also works)"));
  console.log(C.d("    model            params  needs   status"));
  for (const m of MODEL_CATALOG) {
    const mark = m.id === markId ? C.g(">") : " ";
    const status = installed.includes(m.id) ? C.g("installed") : C.d("-");
    console.log(`  ${mark} ${m.id.padEnd(16)} ${m.params.padEnd(7)} ${(m.fitGb + " GB").padEnd(7)} ${status}  ${C.d(m.note)}`);
  }
}

const DIR = path.join(os.homedir(), ".parallelix");
const KEY_PATH = path.join(DIR, "node.key");
const CFG_PATH = path.join(DIR, "config.json");
const LOG_PATH = path.join(DIR, "node.log");
const HEARTBEAT_MS = 10_000;
const POLL_MS = 2_000;
// Consecutive coordinator 403s (with no terminal reason) before giving up. At
// one beat / 10s this is ~5 min, long enough to ride out a transient upstream
// RPC blip (the coordinator's verifyNode can 403 on a flaky Infura rpc_error)
// rather than exiting after ~30s.
const REJECT_LIMIT = 30;
const RESULT_POST_RETRIES = 3;

const C = process.stdout.isTTY ? {
  g: (s) => `\x1b[38;2;215;255;1m${s}\x1b[0m`, d: (s) => `\x1b[2m${s}\x1b[0m`,
  b: (s) => `\x1b[1m${s}\x1b[0m`, r: (s) => `\x1b[38;2;255;90;90m${s}\x1b[0m`,
} : { g: (s) => s, d: (s) => s, b: (s) => s, r: (s) => s };
const ok = (m) => console.log(`${C.g("✓")} ${m}`);
const info = (m) => console.log(`${C.d("·")} ${m}`);
const warn = (m) => console.log(`${C.r("!")} ${m}`);
const die = (m) => { console.error(`${C.r("parallelix-node:")} ${m}`); process.exit(1); };

const fmtDur = (ms) => {
  const s = Math.floor(ms / 1000), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
};
const fmtNum = (n) => Number(n).toLocaleString("en-US", { maximumFractionDigits: 1 });
// Strict x.y.z compare; the release manifest must use plain numeric semver (no prerelease tags).
function semverGt(a, b) {
  const pa = String(a).split(".").map(Number), pb = String(b).split(".").map(Number);
  for (let i = 0; i < 3; i++) { const d = (pa[i] || 0) - (pb[i] || 0); if (d) return d > 0; }
  return false;
}
const tstamp = () => new Date().toISOString().slice(11, 19);

function argFlag(name) {
  const a = process.argv.find((x) => x.startsWith(`${name}=`));
  if (a) return a.slice(name.length + 1);
  const i = process.argv.indexOf(name);
  if (i !== -1 && process.argv[i + 1] && !process.argv[i + 1].startsWith("-")) return process.argv[i + 1];
  return null;
}
const hasFlag = (n) => process.argv.includes(n);
const timeout = (ms) => {
  if (AbortSignal.timeout) return AbortSignal.timeout(ms);
  // Fallback for Node < 17.3: an `undefined` signal would silently disable the
  // timeout on every fetch, so build a real one from an AbortController.
  const ac = new AbortController();
  setTimeout(() => ac.abort(), ms).unref?.();
  return ac.signal;
};

// ── keys ──────────────────────────────────────────────────────────────────────
function genKeypair() {
  let priv;
  for (let i = 0; i < 64; i++) { const p = secp256k1.utils.randomPrivateKey(); if (secp256k1.utils.isValidPrivateKey(p)) { priv = p; break; } }
  if (!priv) die("could not generate a valid node key");
  const pubXY = secp256k1.getPublicKey(priv, false).slice(1);
  const hash = keccak_256(pubXY);
  return { priv: toHex(priv), pubXY: toHex(pubXY), address: "0x" + toHex(hash.slice(12)), nodeKeyHash: "0x" + toHex(hash) };
}
function personalSign(message, privHex) {
  const m = new TextEncoder().encode(message);
  const p = new TextEncoder().encode(`\x19Ethereum Signed Message:\n${m.length}`);
  const composed = new Uint8Array(p.length + m.length); composed.set(p); composed.set(m, p.length);
  const sig = secp256k1.sign(keccak_256(composed), hexToBytes(privHex));
  return "0x" + sig.toCompactHex() + (27 + sig.recovery).toString(16).padStart(2, "0");
}
const loadConfig = () => { try { return JSON.parse(fs.readFileSync(CFG_PATH, "utf8")); } catch { return null; } };
const loadKey = () => { try { return fs.readFileSync(KEY_PATH, "utf8").trim(); } catch { return null; } };
const appendLog = (l) => { try { fs.appendFileSync(LOG_PATH, `${new Date().toISOString()} ${l}\n`); } catch {} };
const saveConfig = (cfg) => fs.writeFileSync(CFG_PATH, JSON.stringify(cfg, null, 2), { mode: 0o600 });

// ── hardware ──────────────────────────────────────────────────────────────────
function detectHardware() {
  const cores = os.cpus()?.length || 0, ramGb = Math.round(os.totalmem() / 1e9);
  let gpu = null, vramGb = 0;

  // 1) NVIDIA (Linux / Windows) — exact VRAM.
  try {
    const out = execFileSync("nvidia-smi", ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
      { stdio: ["ignore", "pipe", "ignore"], timeout: 4000 }).toString().trim();
    if (out) { const [n, mem] = out.split("\n")[0].split(",").map((s) => s.trim()); gpu = n; vramGb = Math.round(Number(mem) / 1024); }
  } catch {}

  // 2) Apple Silicon (Metal) — a real GPU for Ollama; uses unified memory.
  if (!gpu && process.platform === "darwin" && process.arch === "arm64") {
    let name = "Apple Silicon GPU";
    try {
      const sp = execFileSync("system_profiler", ["SPDisplaysDataType"], { stdio: ["ignore", "pipe", "ignore"], timeout: 5000 }).toString();
      const m = sp.match(/Chipset Model:\s*(.+)/);
      if (m) name = m[1].trim();
    } catch {}
    gpu = name; vramGb = ramGb; // unified memory is available to the GPU
  }

  // 3) AMD (Linux) via ROCm, best-effort.
  if (!gpu && process.platform === "linux") {
    try {
      const out = execFileSync("rocm-smi", ["--showproductname", "--showmeminfo", "vram"], { stdio: ["ignore", "pipe", "ignore"], timeout: 4000 }).toString();
      if (/gfx|Radeon|AMD/i.test(out)) {
        gpu = "AMD GPU (ROCm)";
        const mm = out.match(/(\d{9,})/); // VRAM total in bytes
        if (mm) vramGb = Math.round(Number(mm[1]) / 1e9);
      }
    } catch {}
  }

  let tier = 1, label = "CPU";
  if (gpu) {
    if (vramGb >= 40) { tier = 4; label = "GPU · datacenter"; }
    else if (vramGb >= 16) { tier = 3; label = "GPU · high"; }
    else { tier = 2; label = "GPU · entry"; }
    // Apple Silicon shares system memory; cap at "high", not datacenter.
    if (process.platform === "darwin" && tier === 4) { tier = 3; label = "GPU · high"; }
  }
  return { cores, ramGb, gpu, vramGb, tier, label };
}

// ── Ollama (the real inference runtime) ────────────────────────────────────────
async function ollamaVersion() {
  try { const r = await fetch(`${OLLAMA}/api/version`, { signal: timeout(4000) }); if (r.ok) return (await r.json()).version; } catch {}
  return null;
}
async function ollamaModels() {
  try { const r = await fetch(`${OLLAMA}/api/tags`, { signal: timeout(5000) }); if (r.ok) return ((await r.json()).models || []).map((m) => m.name); } catch {}
  return [];
}
async function ollamaGenerate(model, prompt, system) {
  const reqBody = { model, prompt, stream: false };
  if (system && typeof system === "string" && system.length) reqBody.system = system;
  const r = await fetch(`${OLLAMA}/api/generate`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify(reqBody), signal: timeout(120_000),
  });
  if (!r.ok) throw new Error(`ollama ${r.status}`);
  const d = await r.json();
  // prompt_eval_count = tokens in the prompt; eval_count = tokens generated.
  // The coordinator meters paid requests by these counts, so report them.
  return {
    text: d.response ?? "",
    inputTokens: Number.isFinite(d.prompt_eval_count) ? d.prompt_eval_count : null,
    outputTokens: Number.isFinite(d.eval_count) ? d.eval_count : null,
  };
}
async function ollamaPull(name) {
  process.stdout.write(`${C.d("·")} pulling ${name} (first run downloads the model) ...\n`);
  let r;
  try {
    r = await fetch(`${OLLAMA}/api/pull`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, stream: true }),
    });
  } catch (e) { die(`ollama is not running at ${OLLAMA} - install (https://ollama.com/download) and start it`); }
  if (!r.ok || !r.body) die(`ollama pull failed: HTTP ${r.status}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "", lastPct = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
      if (!line.trim()) continue;
      let p; try { p = JSON.parse(line); } catch { continue; }
      if (p.error) die(`ollama pull failed: ${p.error}`);
      if (p.total && p.completed != null) {
        const pct = Math.floor((p.completed / p.total) * 100);
        if (pct !== lastPct) { lastPct = pct; process.stdout.write(`\r${C.d("·")} ${(p.status || "downloading").slice(0, 40)} ${pct}%   `); }
      }
    }
  }
  process.stdout.write("\r\x1b[K");
  ok(`pulled ${name}`);
}
function poe(requestId, result, nodeId) { return "0x" + toHex(sha256(new TextEncoder().encode(`${requestId}|${result}|${nodeId}`))); }

// ── renderer interface ────────────────────────────────────────────────────────

/**
 * Dashboard renderer: full-screen ANSI on the alternate screen buffer.
 * Display-only (no raw-mode stdin). ASCII borders. One write per frame at
 * 1 fps + event-driven repaints. Always restores the terminal on stop().
 */
function makeDashboardRenderer(state) {
  const out = process.stdout;
  const feed = []; // newest first, capped
  let timer = null;
  let onResize = null;

  const GREEN = "\x1b[38;2;215;255;1m", DIM = "\x1b[2m", RESET = "\x1b[0m";
  const W = () => Math.max(60, out.columns || 80);
  // Truncate/pad to a visible width. Inputs here are always plain text (no ANSI),
  // so JS string length equals visible width.
  const pad = (s, n) => (s.length > n ? s.slice(0, Math.max(0, n - 1)) + "…" : s.padEnd(n));
  // Panel header bar: "+ LABEL ------...---+", total visible width w.
  const bar = (label, w) => {
    const head = `+ ${label} `;
    const dashes = Math.max(0, w - head.length - 1);
    return head + "-".repeat(dashes) + "+";
  };
  // Content row: "| <text padded to w-4> |", total visible width w.
  const row = (s, w) => `| ${pad(s, w - 4)} |`;
  const shortAddr = (a) => (a && a.length >= 10 ? a.slice(0, 6) + ".." + a.slice(-4) : (a || "?"));

  function frame() {
    const s = state();
    const w = W();
    const rows = out.rows || 24;
    const lines = [];

    // Header bar: PARALLELIX NODE left, vX.Y.Z right, dashes between.
    const left = "+ PARALLELIX NODE ";
    const right = ` v${VERSION} +`;
    const headDashes = Math.max(0, w - left.length - right.length);
    lines.push(left + "-".repeat(headDashes) + right);

    // Status row (green/dim dot composed plain, colored after padding).
    const statusWord = s.online ? "ACTIVE" : "CONNECTING";
    const statusText = `* ${statusWord}  node ${s.nodeId} . ${shortAddr(s.addr)} . tier ${s.tier} . ${s.mode}`;
    let statusRow = row(statusText, w);
    // Color just the leading "*": replace the first "| *" produced by row().
    const dot = s.online ? GREEN + "*" + RESET : DIM + "*" + RESET;
    statusRow = statusRow.replace("| *", "| " + dot);
    lines.push(statusRow);

    // Second status row: uptime / heartbeats / served (+ inference running).
    let line2 = `uptime ${fmtDur(Date.now() - s.startedAt)} . heartbeats ${fmtNum(s.beats)} . served ${fmtNum(s.served)}`;
    if (s.busy) line2 += " . inference running";
    lines.push(row(line2, w));

    // MODELS panel: up to 4, active marked.
    lines.push(bar("MODELS", w));
    const models = (s.models || []).slice(0, 4);
    if (!models.length) {
      lines.push(row("no models loaded", w));
    } else {
      for (const m of models) {
        if (m === s.model) {
          let r = row(`> ${pad(m, 22)} active`, w);
          r = r.replace("| >", "| " + GREEN + ">" + RESET);
          lines.push(r);
        } else {
          lines.push(DIM + row(`  ${pad(m, 22)} ready`, w) + RESET);
        }
      }
    }

    // EARNINGS panel.
    lines.push(bar("EARNINGS", w));
    const e = s.earnings || {};
    if (e.lifetimePrlx == null) {
      lines.push(row(e.stale ? "stats unavailable (stale)" : `served this session: ${fmtNum(s.served)} requests`, w));
    } else {
      // Session row only when per-request cost data exists; a permanent "+0"
      // above a growing lifetime number would read as a contradiction.
      if (e.sessionPrlx > 0) lines.push(row(`session    +${fmtNum(e.sessionPrlx)} $PRLX`, w));
      lines.push(row(`lifetime   ${fmtNum(e.lifetimePrlx)} $PRLX${e.stale ? " (stale)" : ""}`, w));
      if (e.claimablePrlx != null) lines.push(row(`claimable  ${fmtNum(e.claimablePrlx)} $PRLX`, w));
    }

    // REQUESTS panel: fill remaining rows with the feed (newest first).
    lines.push(bar("REQUESTS", w));
    // Fixed chrome = current lines (everything above incl. REQUESTS bar)
    // + 1 footer line. Reserve at least 1 request row.
    const reqRows = Math.max(1, rows - lines.length - 1);
    if (!feed.length) {
      lines.push(row("waiting for requests ...", w));
      for (let i = 1; i < reqRows; i++) lines.push(row("", w));
    } else {
      for (let i = 0; i < reqRows; i++) {
        lines.push(row(i < feed.length ? feed[i] : "", w));
      }
    }

    // Footer bar.
    const footText = s.updateVersion
      ? `v${s.updateVersion} available: parallelix-node update . ctrl-c drain`
      : `ctrl-c drain . docs.parallelix.io/node-cli`;
    const footHead = `+ ${footText} `;
    const footDashes = Math.max(0, w - footHead.length - 1);
    lines.push(footHead + "-".repeat(footDashes) + "+");

    out.write("\x1b[H" + lines.map((l) => l + "\x1b[K").join("\n") + "\x1b[J");
  }

  function push(line) {
    feed.unshift(line);
    if (feed.length > 50) feed.length = 50;
  }

  return {
    start() {
      out.write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H");
      timer = setInterval(() => {
        try { frame(); } catch (e) { appendLog("dashboard frame error " + String(e?.message || e)); }
      }, 1000);
      onResize = () => { try { frame(); } catch (e) { appendLog("dashboard frame error " + String(e?.message || e)); } };
      out.on("resize", onResize);
      frame();
    },
    event(type, d) {
      switch (type) {
        case "served":
          push(`${tstamp()} ${d.reqId} ${pad(d.model || "", 14)} ${String(d.ms).padStart(5)}ms poe OK${d.costPrlx ? ` +${fmtNum(d.costPrlx)}` : ""}`);
          break;
        case "serve-fail":
          push(`${tstamp()} ${d.reqId} FAILED ${String(d.err || "").slice(0, 30)}`);
          break;
        case "result-lost":
          push(`${tstamp()} ${d.reqId} result post failed (will not retry)`);
          break;
        case "hb-fail":
          push(`${tstamp()} heartbeat failed . retrying`);
          break;
        // "active" and "update" add no feed line but must repaint.
      }
      frame();
    },
    tick() { /* timer-driven; no-op */ },
    stop(reason) {
      if (timer) clearInterval(timer);
      if (onResize) out.off("resize", onResize);
      out.write("\x1b[?25h\x1b[?1049l");
      const graceful = reason === "SIGINT" || reason === "SIGTERM";
      console.log(graceful
        ? `${C.d("·")} draining (${reason}) · node going idle · stake stays locked, no slashing`
        : `${C.d("·")} stopped (${reason})`);
    },
  };
}

/**
 * Line renderer: journalctl-clean output for services, pipes, and --plain.
 * One timestamped line per event + a status line every 60s.
 */
function makeLineRenderer() {
  let lastStatusAt = 0;
  return {
    start(s) {
      console.log(`${tstamp()} daemon up · node ${s.nodeId} · ${s.mode} · model ${s.model} · ctrl-c to stop`);
    },
    event(type, d) {
      switch (type) {
        case "active":        console.log(`${tstamp()} coordinator acknowledged · node ACTIVE · uptime accruing`); break;
        case "served":        console.log(`${tstamp()} served ${d.reqId} · ${d.model} · ${d.ms}ms · poe ${d.poePrefix}`); break;
        case "serve-fail":    console.log(`${tstamp()} serve failed ${d.reqId} · ${d.err}`); break;
        case "result-lost":   console.log(`${tstamp()} result post failed ${d.reqId} (inference done, coordinator unreachable)`); break;
        case "hb-fail":       console.log(`${tstamp()} heartbeat failed (${d.detail}) · retrying`); break;
        case "update":        console.log(`${tstamp()} update available: v${d.version} · run: parallelix-node update`); break;
        case "earnings":      break; // line mode stays quiet; numbers are in status lines
      }
    },
    tick(s) {
      if (Date.now() - lastStatusAt < 60_000) return;
      lastStatusAt = Date.now();
      console.log(`${tstamp()} status ${s.online ? "active" : "connecting"} · heartbeats ${s.beats} · served ${s.served}${s.busy ? " · inference running" : ""}`);
    },
    stop(reason) {
      // Graceful signals drain; forced exits (not_staked / in_cooldown /
      // rejected) already printed their explanation, so keep this line neutral.
      const graceful = reason === "SIGINT" || reason === "SIGTERM";
      console.log(graceful
        ? `${tstamp()} draining (${reason}) · node going idle · stake stays locked, no slashing`
        : `${tstamp()} stopped (${reason})`);
    },
  };
}

// ── commands ──────────────────────────────────────────────────────────────────
async function cmdProbe() {
  const hw = detectHardware();
  console.log(C.b("// hardware probe"));
  info(`cpu       ${hw.cores} cores`);
  info(`memory    ${hw.ramGb} GB`);
  info(`gpu       ${hw.gpu ? `${hw.gpu} · ${hw.vramGb} GB VRAM` : "none detected (CPU mode)"}`);
  const ov = await ollamaVersion();
  info(`ollama    ${ov ? `running (v${ov})` : C.r("not running - install: https://ollama.com/download")}`);
  console.log("");
  ok(`maps to ${C.g(`tier ${hw.tier}`)} (${hw.label})`);
  if (!ov) info(`then start Ollama, and: ${C.b("parallelix-node models pull " + DEFAULT_MODEL)}`);
  info(`next: ${C.b("parallelix-node init")}`);
}

function cmdInit() {
  if (fs.existsSync(KEY_PATH) && !hasFlag("--force")) die(`a node key already exists at ${KEY_PATH}. Re-run with --force to overwrite (orphans the old node).`);
  // --force: regenerate. initIdentity never overwrites an existing key, so drop it first.
  if (fs.existsSync(KEY_PATH)) { try { fs.unlinkSync(KEY_PATH); } catch {} }
  const hw = detectHardware();
  const cfg = initIdentity(hw, argFlag("--wallet"));
  ok(`node key generated at ${KEY_PATH} ${C.d("(mode 0600)")}`);
  ok(`config written to ${CFG_PATH}`);
  console.log("\n" + C.b("// register this node on-chain (in the Console), using:"));
  console.log(`   ${C.d("tier")}         ${cfg.tier} (${hw.label})`);
  console.log(`   ${C.d("nodeKeyHash")}  ${C.g(cfg.nodeKeyHash)}\n`);
  info(`This key signs liveness + results only. It is NOT your staking wallet.`);
  info(`Stake + registerNode at ${C.b("https://app.parallelix.io")} with the nodeKeyHash above,`);
  info(`then: ${C.b(`parallelix-node start --node-id <id> ${hw.gpu ? "--gpu" : "--cpu"}`)}`);
}

/** Create key+config if absent; returns the loaded config. Idempotent. Never overwrites an existing key. */
function initIdentity(hw, wallet) {
  fs.mkdirSync(DIR, { recursive: true, mode: 0o700 });
  // A config without its key is unusable (the printed nodeKeyHash could never
  // be served from this machine). Drop the orphaned config and start fresh.
  if (!fs.existsSync(KEY_PATH) && fs.existsSync(CFG_PATH)) {
    warn(`config.json exists but the node key is missing; regenerating both`);
    try { fs.unlinkSync(CFG_PATH); } catch {}
  }
  if (!fs.existsSync(KEY_PATH)) {
    const kp = genKeypair();
    fs.writeFileSync(KEY_PATH, kp.priv, { mode: 0o600 });
    const cfg = { version: VERSION, nodeAddress: kp.address, nodeKeyHash: kp.nodeKeyHash, pubKey: "0x" + kp.pubXY,
      stakingWallet: wallet || null, tier: hw.tier, createdAt: new Date().toISOString(),
      coordinator: API_URL, ollama: OLLAMA, contracts: { staking: STAKE_CONTRACT, rewards: REWARDS_CONTRACT, prlx: PRLX } };
    saveConfig(cfg);
    return cfg;
  }
  return loadConfig();
}

async function cmdModels() {
  const sub = process.argv[3];
  if (sub === "pull") { const name = process.argv[4] || DEFAULT_MODEL; await ollamaPull(name); return; }
  if (sub === "catalog") { printCatalog(await ollamaModels(), null); return; }
  if (sub === "recommend") {
    const hw = detectHardware();
    const { pick, budget } = recommendModel(hw);
    printCatalog(await ollamaModels(), pick.id);
    console.log("");
    ok(`recommended: ${C.b(pick.id)} (${pick.params}, needs ~${pick.fitGb} GB; your budget ~${budget} GB ${hw.gpu ? "VRAM" : "RAM, CPU mode"})`);
    info(`pull it: ${C.b("parallelix-node models pull " + pick.id)}`);
    return;
  }
  const ov = await ollamaVersion();
  if (!ov) die("Ollama is not running. Install it (https://ollama.com/download) and start it.");
  const models = await ollamaModels();
  console.log(C.b(`// ollama models (${OLLAMA})`));
  if (!models.length) { warn("no models installed."); info(`pull one: ${C.b("parallelix-node models pull " + DEFAULT_MODEL)}`); return; }
  models.forEach((m) => info(m));
  info(`see the curated list: ${C.b("parallelix-node models catalog")}`);
}

async function cmdVerify() {
  let pass = true;
  console.log(C.b("// verify"));
  Number(process.versions.node.split(".")[0]) >= 18 ? ok(`node ${process.versions.node}`) : (warn(`node ${process.versions.node} (need 18+)`), pass = false);
  if (fs.existsSync(KEY_PATH)) { const mode = (fs.statSync(KEY_PATH).mode & 0o777).toString(8); mode === "600" ? ok(`node key present, perms ${mode}`) : (warn(`node key perms ${mode} (chmod 600 ${KEY_PATH})`), pass = false); }
  else { warn("no node key - run: parallelix-node init"); pass = false; }
  const cfg = loadConfig(); cfg ? ok(`config valid · node ${cfg.nodeAddress}`) : warn("no config - run: parallelix-node init");
  const hw = detectHardware(); ok(`hardware tier ${hw.tier} (${hw.label})`);
  const ov = await ollamaVersion(); ov ? ok(`ollama running (v${ov})`) : (warn(`ollama not running (${OLLAMA}) - install https://ollama.com/download`), pass = false);
  if (ov) { const m = await ollamaModels(); m.length ? ok(`models: ${m.join(", ")}`) : (warn(`no models - parallelix-node models pull ${DEFAULT_MODEL}`), pass = false); }
  try { const r = await fetch(`${API_URL}/health`, { signal: timeout(5000) }).catch(() => fetch(`${API_URL}/`, { signal: timeout(5000) })); (r && r.status < 500) ? ok(`coordinator reachable · ${API_URL}`) : (warn(`coordinator unreachable · ${API_URL}`), pass = false); } catch { warn(`coordinator unreachable · ${API_URL}`); pass = false; }
  console.log(""); pass ? ok("all checks passed · ready to start") : warn("some checks failed (see above)");
  process.exit(pass ? 0 : 1);
}

function cmdStatus() {
  const cfg = loadConfig(); if (!cfg) die("not initialised - run: parallelix-node init");
  console.log(C.b("// node status"));
  info(`node address   ${cfg.nodeAddress}`);
  info(`nodeKeyHash    ${cfg.nodeKeyHash}`);
  info(`tier           ${cfg.tier}`);
  info(`node id        ${cfg.nodeId || "(not set: run setup or start --node-id <id>)"}`);
  info(`coordinator    ${cfg.coordinator}`);
  info(`ollama         ${cfg.ollama || OLLAMA}`);
  info(`staking CA     ${cfg.contracts?.staking}`);
}

function cmdLogs() {
  if (!fs.existsSync(LOG_PATH)) die(`no log yet at ${LOG_PATH}`);
  console.log(fs.readFileSync(LOG_PATH, "utf8").trim().split("\n").slice(-50).join("\n"));
}

async function cmdStart() {
  const cfg0 = loadConfig();
  const nodeId = argFlag("--node-id") || cfg0?.nodeId || null;
  if (!nodeId) die("missing --node-id (none saved in config). Easiest: parallelix-node setup. Manual: parallelix-node start --node-id <id>");
  const priv = loadKey(), cfg = loadConfig();
  if (!priv || !cfg) die("not initialised - run: parallelix-node init");
  const hw = detectHardware();
  const mode = hasFlag("--gpu") ? "gpu" : hasFlag("--cpu") ? "cpu" : (hw.tier > 1 ? "gpu" : "cpu");

  // Real inference runtime must be up with a model, or we don't claim to serve.
  const ov = await ollamaVersion();
  if (!ov) die(`Ollama is not running at ${OLLAMA}. Install (https://ollama.com/download), start it, then: parallelix-node models pull ${DEFAULT_MODEL}`);
  let models = await ollamaModels();
  if (!models.length) { warn(`no model installed; pulling ${DEFAULT_MODEL} ...`); await ollamaPull(DEFAULT_MODEL); models = await ollamaModels(); }
  const model = models.includes(DEFAULT_MODEL) ? DEFAULT_MODEL : (models.find((m) => m.startsWith(DEFAULT_MODEL)) || models[0]);

  console.log(C.b(`// parallelix-node ${VERSION}`));
  info(`node ${nodeId} · ${cfg.nodeAddress} · ${mode} · model ${model}`);
  info(`coordinator ${API_URL} · ollama ${OLLAMA} (v${ov})`);
  appendLog(`start node=${nodeId} mode=${mode} model=${model} gpu=${hw.gpu || "none"}`);
  if (cfg.nodeId !== String(nodeId)) { cfg.nodeId = String(nodeId); saveConfig(cfg); }

  // A long-running daemon must survive a stray rejection/throw outside the
  // heartbeat/poll try-blocks (e.g. a write to a closed stdout) rather than
  // dying opaquely. Log and keep running so the failure is diagnosable.
  process.on("unhandledRejection", (e) => appendLog(`unhandledRejection ${String(e?.message || e)}`));
  process.on("uncaughtException", (e) => appendLog(`uncaughtException ${String(e?.message || e)}`));

  let beats = 0, served = 0, online = false, busy = false, rejects = 0;
  const capabilities = { version: VERSION, tier: hw.tier, mode, model, models, gpu: hw.gpu, vramGb: hw.vramGb, cores: hw.cores, ramGb: hw.ramGb };

  const startedAt = Date.now();
  const earnings = { lifetimePrlx: null, sessionPrlx: 0, claimablePrlx: null, stale: false };
  let updateVersion = null;
  const state = () => ({
    nodeId, addr: cfg.nodeAddress, tier: hw.tier, mode, model, models,
    online, beats, served, busy, startedAt, earnings, updateVersion,
  });
  const wantDash = process.stdout.isTTY && (process.stdout.columns || 80) >= 60 && !hasFlag("--plain");
  let renderer = wantDash ? makeDashboardRenderer(state) : makeLineRenderer();
  // A renderer bug must never kill a serving daemon: fall back to lines.
  const safeRender = (fn) => {
    try { fn(); } catch (e) {
      appendLog(`renderer error ${String(e?.message || e)} (falling back to plain)`);
      try { process.stdout.write("\x1b[?25h\x1b[?1049l"); } catch {}
      renderer = makeLineRenderer();
      try { renderer.start(state()); } catch {} // stdout dead = nothing left to render to
    }
  };
  safeRender(() => renderer.start(state()));

  const heartbeat = async () => {
    if (beats > 0 && beats % 10 === 0) {
      const fresh = await ollamaModels();
      if (fresh.length) { models = fresh; capabilities.models = fresh; }
    }
    const ts = Date.now();
    const message = `parallelix-node:heartbeat:${nodeId}:${ts}`;
    try {
      const r = await fetch(`${API_URL}/operator/heartbeat`, { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ nodeId, nodeAddress: cfg.nodeAddress, message, signature: personalSign(message, priv), capabilities, served, available: !busy }),
        signal: timeout(8000) });
      beats++;
      if (r.ok) {
        rejects = 0;
        if (!online) { online = true; safeRender(() => renderer.event("active", {})); }
      } else {
        online = false;
        appendLog(`heartbeat ${beats} http ${r.status}`);
        // The coordinator rejects (403) a node no longer backed on-chain.
        // Unstaked / in-cooldown is terminal — stop instead of spinning forever.
        if (r.status === 403) {
          let reason = "";
          try { const b = await r.json(); reason = b?.reason || b?.error || ""; } catch {}
          if (reason === "not_staked" || reason === "in_cooldown") {
            console.log(`\n${C.r("✗")} node ${nodeId} is ${reason === "in_cooldown" ? "in unstake cooldown" : "no longer staked"} on-chain. Nothing to serve, stopping.`);
            appendLog(`exit ${reason}`);
            try { renderer.stop(reason); } catch { process.stdout.write("\x1b[?25h\x1b[?1049l"); }
            process.exit(0);
          }
          if (++rejects >= REJECT_LIMIT) {
            console.log(`\n${C.r("✗")} coordinator kept rejecting node ${nodeId} for ${REJECT_LIMIT} beats (${reason || r.status}). If the --node-id is right and you registered this machine's nodeKeyHash, this is most likely a transient coordinator/RPC issue. Exiting; the service will restart and retry.`);
            appendLog(`exit rejected ${reason || r.status} after ${rejects} beats`);
            try { renderer.stop("rejected"); } catch { process.stdout.write("\x1b[?25h\x1b[?1049l"); }
            process.exit(1);
          }
        }
      }
    } catch (e) { online = false; appendLog(`heartbeat err ${String(e?.message || e)}`); safeRender(() => renderer.event("hb-fail", { detail: String(e?.message || e) })); }
  };

  const poll = async () => {
    if (busy) return;
    let job = null;
    try {
      const r = await fetch(`${API_URL}/operator/inbox?nodeId=${encodeURIComponent(nodeId)}`, { signal: timeout(6000) });
      if (r.ok) {
        const d = await r.json();
        const arr = Array.isArray(d.requests) ? d.requests : Array.isArray(d.jobs) ? d.jobs : [];
        const first = arr[0];
        job = first && typeof first === "object" ? first : null;
      }
    } catch {}
    if (!job) return;
    const reqId = job.id || job.requestId;
    if (!reqId) return; // malformed inbox entry — ignore rather than run inference on garbage
    busy = true;
    const prompt = job.prompt ?? job.input ?? "";
    const sys = job.system ?? null;
    const jm = job.model && models.includes(job.model) ? job.model : model;
    try {
      const t0 = Date.now();
      const gen = await ollamaGenerate(jm, prompt, sys);
      const result = gen.text;
      const ms = Date.now() - t0;
      const proof = poe(reqId, result, nodeId);
      const message = `parallelix-node:result:${reqId}`;
      const body = JSON.stringify({ nodeId, requestId: reqId, result, poe: proof, model: jm, ms, inputTokens: gen.inputTokens, outputTokens: gen.outputTokens, message, signature: personalSign(message, priv) });
      // The GPU work is already done and metered; a single dropped POST loses it
      // (no PoE recorded, no credit). Retry a few times before giving up.
      let posted = false;
      for (let attempt = 1; attempt <= RESULT_POST_RETRIES; attempt++) {
        try {
          const rr = await fetch(`${API_URL}/operator/result`, { method: "POST", headers: { "content-type": "application/json" }, body, signal: timeout(15000) });
          if (rr.ok) { posted = true; break; }
        } catch {}
        if (attempt < RESULT_POST_RETRIES) await new Promise((res) => setTimeout(res, 1000 * attempt));
      }
      if (posted) { served++; appendLog(`served ${reqId} model=${jm} ms=${ms} poe=${proof.slice(0, 14)}`); safeRender(() => renderer.event("served", { reqId, model: jm, ms, poePrefix: proof.slice(0, 14), costPrlx: 0 })); }
      else { appendLog(`result-post-failed ${reqId} model=${jm} ms=${ms} (inference done, coordinator unreachable after ${RESULT_POST_RETRIES} tries)`); safeRender(() => renderer.event("result-lost", { reqId })); }
    } catch (e) {
      appendLog(`serve-fail ${reqId} ${String(e?.message || e)}`);
      safeRender(() => renderer.event("serve-fail", { reqId, err: String(e?.message || e) }));
      // Signed failure report so the coordinator can verify it's really this
      // node giving up the request, and requeue it for another node at once
      // (an unsigned report is ignored and only requeues on the reaper TTL).
      const message = `parallelix-node:result:${reqId}`;
      try { await fetch(`${API_URL}/operator/result`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ nodeId, requestId: reqId, error: String(e?.message || e), message, signature: personalSign(message, priv) }), signal: timeout(8000) }); } catch {}
    } finally { busy = false; }
  };

  await heartbeat();
  const hb = setInterval(heartbeat, HEARTBEAT_MS);
  const pl = setInterval(poll, POLL_MS);
  const tk = setInterval(() => safeRender(() => renderer.tick(state())), 1000);

  // Earnings panel data. Owner address is saved to config by setup (or set
  // via init --wallet). Absent owner = panel shows served-count only.
  const owner = cfg.owner || cfg.stakingWallet || null;
  const pollEarnings = async () => {
    if (!owner) return;
    try {
      const r = await fetch(`${API_URL}/operator/stats/${owner}`, { signal: timeout(8000) });
      if (!r.ok) { earnings.stale = true; return; }
      const d = await r.json();
      // Defensive mapping; lifetimeEarningsPrlx is the verified key from the coordinator.
      const lifetime = Number(d.lifetimeEarningsPrlx ?? NaN);
      if (Number.isFinite(lifetime)) earnings.lifetimePrlx = lifetime;
      // claimablePrlx is not returned by this endpoint; leave null.
      earnings.claimablePrlx = null;
      earnings.stale = false;
    } catch { earnings.stale = true; }
  };
  pollEarnings();
  const ek = setInterval(pollEarnings, 30_000);

  // One non-blocking manifest check; failure-silent (3s budget).
  (async () => {
    try {
      const r = await fetch((process.env.PARALLELIX_MANIFEST_URL || "https://parallelix.io/cli/manifest.json"), { signal: timeout(3000) });
      if (!r.ok) return;
      const man = await r.json();
      if (man?.version && semverGt(man.version, VERSION)) {
        updateVersion = man.version;
        safeRender(() => renderer.event("update", { version: man.version }));
      }
    } catch {}
  })();

  // Drain cleanly on Ctrl-C AND on SIGTERM (systemctl stop / launchd / kill).
  const stop = (sig) => {
    clearInterval(hb); clearInterval(pl); clearInterval(tk); clearInterval(ek);
    try { renderer.stop(sig); } catch { process.stdout.write("\x1b[?25h\x1b[?1049l"); }
    appendLog(`stop ${sig}`);
    process.exit(0);
  };
  process.on("SIGINT", () => stop("SIGINT"));
  process.on("SIGTERM", () => stop("SIGTERM"));
}

// Install the node as a persistent background service so it survives logout /
// reboot and auto-restarts on crash. systemd (Linux) or launchd (macOS).
function cmdService() {
  const nodeId = argFlag("--node-id") || loadConfig()?.nodeId || null;
  if (!nodeId) die("missing --node-id (none saved in config). Run parallelix-node setup, or: parallelix-node service --node-id <id>");
  const modeFlag = hasFlag("--gpu") ? "--gpu" : hasFlag("--cpu") ? "--cpu" : "";
  installService(String(nodeId), modeFlag);
}

// Write + load the platform service unit (systemd/launchd). Callable from
// cmdService and from setup. modeFlag is "--gpu" | "--cpu" | "".
function installService(nodeId, modeFlag) {
  const bin = "/usr/local/bin/parallelix-node";

  if (process.platform === "linux") {
    const user = process.env.SUDO_USER || process.env.USER || "root";
    const unit = `[Unit]
Description=ParalleliX Node
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=${user}
ExecStart=${bin} start --node-id ${nodeId}${modeFlag ? " " + modeFlag : ""}
# on-failure (not always): a clean stop when the node is intentionally
# unstaked/in-cooldown exits 0 and must NOT be restarted into a tight loop.
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
`;
    const dst = "/etc/systemd/system/parallelix-node.service";
    try {
      fs.writeFileSync(dst, unit);
      execFileSync("systemctl", ["daemon-reload"]);
      execFileSync("systemctl", ["enable", "--now", "parallelix-node"]);
      ok("installed + started as systemd service 'parallelix-node'");
      info("logs: journalctl -u parallelix-node -f   ·   stop: sudo systemctl stop parallelix-node");
    } catch {
      fs.mkdirSync(DIR, { recursive: true });
      const tmp = path.join(DIR, "parallelix-node.service");
      fs.writeFileSync(tmp, unit);
      console.log(`\n${C.b("// systemd unit written to")} ${tmp}`);
      info("install it (needs sudo):");
      console.log(`  sudo cp ${tmp} ${dst}`);
      console.log(`  sudo systemctl daemon-reload && sudo systemctl enable --now parallelix-node`);
      console.log(`  ${C.d("logs:")} journalctl -u parallelix-node -f`);
    }
    return;
  }

  if (process.platform === "darwin") {
    const dir = path.join(os.homedir(), "Library", "LaunchAgents");
    const dst = path.join(dir, "io.parallelix.node.plist");
    const args = [bin, "start", "--node-id", nodeId, ...(modeFlag ? [modeFlag] : [])];
    const argXml = args.map((a) => `    <string>${a}</string>`).join("\n");
    const plist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>io.parallelix.node</string>
  <key>ProgramArguments</key>
  <array>
${argXml}
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>${path.join(DIR, "node.out.log")}</string>
  <key>StandardErrorPath</key><string>${path.join(DIR, "node.err.log")}</string>
</dict>
</plist>
`;
    fs.mkdirSync(dir, { recursive: true });
    fs.mkdirSync(DIR, { recursive: true });
    fs.writeFileSync(dst, plist);
    try { execFileSync("launchctl", ["unload", dst], { stdio: "ignore" }); } catch {}
    try {
      execFileSync("launchctl", ["load", "-w", dst]);
      ok("installed + started as launchd agent 'io.parallelix.node'");
      info(`logs: tail -f ${path.join(DIR, "node.out.log")}   ·   stop: launchctl unload ${dst}`);
    } catch (e) {
      warn(`wrote ${dst} but could not load: ${String(e?.message || e)}`);
      info(`run: launchctl load -w ${dst}`);
    }
    return;
  }

  die("service install supports Linux (systemd) and macOS (launchd). On Windows, run the node inside WSL2.");
}

async function cmdUpdate() {
  const MANIFEST_URL = (process.env.PARALLELIX_MANIFEST_URL || "https://parallelix.io/cli/manifest.json");
  info(`checking ${MANIFEST_URL} ...`);
  let man;
  try {
    const r = await fetch(MANIFEST_URL, { signal: timeout(5000) });
    if (!r.ok) die(`manifest fetch failed: HTTP ${r.status}`);
    man = await r.json();
  } catch (e) { die(`manifest fetch failed: ${String(e?.message || e)}`); }
  if (!man?.version || !man?.sha256 || !man?.url) die("manifest malformed; aborting, nothing changed");
  if (!semverGt(man.version, VERSION)) { ok(`already up to date (v${VERSION})`); return; }
  info(`v${man.version} available (current v${VERSION})${man.changelog ? ` · ${man.changelog}` : ""}`);

  const r = await fetch(man.url, { signal: timeout(60_000) });
  if (!r.ok) die(`download failed: HTTP ${r.status}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  const digest = toHex(sha256(bytes));
  const expected = String(man.sha256).replace(/^0x/, "").toLowerCase();
  if (digest !== expected) die(`checksum mismatch (expected ${expected.slice(0, 12)}.., got ${digest.slice(0, 12)}..); aborting, nothing changed`);

  const tmp = path.join(os.tmpdir(), `parallelix-node-${man.version}-${process.pid}.mjs`);
  fs.writeFileSync(tmp, bytes);
  try { execFileSync(process.execPath, ["--check", tmp], { stdio: "ignore" }); }
  catch { fs.unlinkSync(tmp); die("downloaded file failed node --check; aborting, nothing changed"); }

  // argv[1] is this .mjs file: the install shim execs `node <dir>/parallelix-node.mjs`.
  // If the shim ever changes to a wrapper script, this self-locate breaks: keep it exec-style.
  const self = fs.realpathSync(process.argv[1]);
  try {
    fs.copyFileSync(self, self + ".bak");
    try { fs.renameSync(tmp, self); }
    catch (e) {
      if (e.code === "EXDEV") { fs.copyFileSync(tmp, self); try { fs.unlinkSync(tmp); } catch {} }
      else throw e;
    }
  } catch (e) {
    try { fs.unlinkSync(tmp); } catch {} // never leave the download behind on a failed swap
    if (e.code === "EACCES" || e.code === "EPERM") die(`no write permission for ${self}. Re-run: sudo parallelix-node update`);
    throw e;
  }
  ok(`updated to v${man.version} (backup at ${self}.bak)`);
  if (process.platform === "linux" && fs.existsSync("/etc/systemd/system/parallelix-node.service"))
    info("restart the service: sudo systemctl restart parallelix-node");
  if (process.platform === "darwin" && fs.existsSync(path.join(os.homedir(), "Library/LaunchAgents/io.parallelix.node.plist")))
    info("restart the agent: launchctl kickstart -k gui/$(id -u)/io.parallelix.node");
}

// ── setup (the centerpiece) ─────────────────────────────────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function promptYesNo(q) {
  if (!process.stdin.isTTY) return false;
  process.stdout.write(q);
  return await new Promise((res) => {
    process.stdin.resume(); process.stdin.setEncoding("utf8");
    process.stdin.once("data", (d) => { process.stdin.pause(); res(!/^n/i.test(String(d).trim())); });
  });
}

/** Poll the coordinator until this keyHash appears on-chain. 30 min budget. */
async function pollRegistration(keyHash) {
  const deadline = Date.now() + 30 * 60_000;
  let backoff = 5_000, dots = 0;
  // Ctrl-C mid-poll: clear the spinner line so the shell prompt lands clean,
  // and remind that setup resumes where it left off.
  const onInt = () => {
    process.stdout.write("\r\x1b[K");
    console.log(`${C.d("·")} stopped. Re-run ${C.b("parallelix-node setup")} after staking; it resumes here.`);
    process.exit(0);
  };
  process.on("SIGINT", onInt);
  try {
    while (Date.now() < deadline) {
      try {
        const r = await fetch(`${API_URL}/operator/node-by-keyhash/${keyHash}`, { signal: timeout(8000) });
        if (r.ok) { process.stdout.write("\r\x1b[K"); return await r.json(); }
        backoff = r.status === 404 ? 5_000 : Math.min(backoff * 2, 30_000);
      } catch { backoff = Math.min(backoff * 2, 30_000); }
      process.stdout.write(`\r${C.d("·")} waiting for on-chain registration ${".".repeat((dots++ % 3) + 1)}   `);
      await sleep(backoff);
    }
    process.stdout.write("\r\x1b[K");
    return null;
  } finally {
    process.off("SIGINT", onInt);
  }
}

async function cmdSetup() {
  console.log(C.b(`// parallelix-node setup`) + C.d("  one command from this machine to a serving node"));
  // [1/5] hardware
  const hw = detectHardware();
  ok(`[1/5] hardware: ${hw.cores} cores · ${hw.ramGb} GB RAM · ${hw.gpu ? `${hw.gpu} (${hw.vramGb} GB VRAM)` : "no GPU (CPU mode)"} -> tier ${hw.tier} (${hw.label})`);

  // [2/5] inference runtime
  let ov = await ollamaVersion();
  if (!ov) {
    warn(`[2/5] Ollama is not running at ${OLLAMA}`);
    info(`install + start it: ${C.b("curl -fsSL https://ollama.com/install.sh | sh")}  (macOS: https://ollama.com/download)`);
    info("waiting for Ollama (ctrl-c to stop; setup is resumable, just re-run it) ...");
    while (!ov) { await sleep(3000); ov = await ollamaVersion(); }
  }
  ok(`[2/5] ollama running (v${ov})`);

  // [3/5] model
  let models = await ollamaModels();
  const haveCatalog = MODEL_CATALOG.filter((m) => models.includes(m.id));
  if (haveCatalog.length) {
    ok(`[3/5] model ready: ${haveCatalog.map((m) => m.id).join(", ")}`);
  } else {
    const { pick, budget } = recommendModel(hw);
    info(`[3/5] recommended for this machine: ${C.b(pick.id)} (${pick.params}, needs ~${pick.fitGb} GB of your ~${budget} GB)`);
    await ollamaPull(pick.id);
    models = await ollamaModels();
  }

  // [4/5] identity (idempotent; never overwrites an existing key)
  const cfg = initIdentity(hw, argFlag("--wallet"));
  ok(`[4/5] node identity ready · nodeKeyHash ${C.g(cfg.nodeKeyHash)}`);
  info(`this key signs liveness + results only. It is NOT your staking wallet.`);

  // [5/5] stake + auto-detect
  if (cfg.nodeId) {
    ok(`[5/5] node ${cfg.nodeId} already configured · nothing to register`);
    info(`run it: ${C.b("parallelix-node start")}`);
    return;
  }
  const link = `https://app.parallelix.io/operate/nodes/register?keyHash=${cfg.nodeKeyHash}&tier=${cfg.tier}`;
  console.log(`\n${C.b("// stake + register this node in the Console (form arrives prefilled):")}`);
  console.log(`   ${C.g(link)}\n`);
  const node = await pollRegistration(cfg.nodeKeyHash);
  if (!node) {
    info("not registered yet after 30 minutes. Stake in the Console, then re-run: parallelix-node setup");
    return;
  }
  cfg.nodeId = String(node.nodeId);
  cfg.owner = node.owner;
  saveConfig(cfg);
  ok(`[5/5] node ${node.nodeId} detected on-chain · tier ${node.tier} · ${fmtNum(node.stakePrlx)} $PRLX staked · saved`);

  const yes = await promptYesNo(`${C.d("·")} install as a 24/7 background service now? [Y/n] `);
  if (yes) installService(cfg.nodeId, hw.gpu ? "--gpu" : "--cpu");
  else info(`run it in the foreground: ${C.b("parallelix-node start")}  (node id is saved)`);
}

function cmdHelp() {
  console.log(`${C.g("ParalleliX")} node CLI ${C.d("v" + VERSION)} ${C.d("· real GPU compute via Ollama")}

${C.b("USAGE")}  parallelix-node <command> [options]

${C.b("COMMANDS")}
  setup                          one command: detect hardware, get a model,
                                 create identity, then auto-detect your stake
  probe                          detect hardware + Ollama, print the tier
  init [--wallet 0x..]           generate the node key + config; print nodeKeyHash
  models [pull <name> | catalog | recommend]   list / pull local Ollama models
  start --node-id <id>           run the daemon: attach, heartbeat (uptime),
        [--gpu|--cpu] [--model m]   serve real inference, return Proof-of-Execution
  service --node-id <id>         install as a background service (systemd/launchd):
        [--gpu|--cpu]               survives logout + reboot, auto-restarts
  update                         self-update to the latest release (sha256-verified)
  verify                         run diagnostics
  status / logs / version

${C.b("FLOW")}
  new operator:  parallelix-node setup        ${C.d("does everything below, in order")}
  manual:        probe -> models pull -> init -> stake in Console -> start --node-id <id>
  24/7:          parallelix-node service      ${C.d("(setup offers this automatically)")}

${C.b("WINDOWS")}  native Windows is not supported. Install WSL2
  (https://learn.microsoft.com/windows/wsl/install), open Ubuntu, then run the
  one-line installer + these steps inside it.

  docs: https://docs.parallelix.io/node-cli`);
}

const cmd = process.argv[2];
(async () => {
  switch (cmd) {
    case "setup": return cmdSetup();
    case "probe": return cmdProbe();
    case "init": return cmdInit();
    case "models": return cmdModels();
    case "update": return cmdUpdate();
    case "start": return cmdStart();
    case "service": return cmdService();
    case "verify": return cmdVerify();
    case "status": return cmdStatus();
    case "logs": return cmdLogs();
    case "version": case "--version": case "-v": return console.log(VERSION);
    case "help": case "--help": case "-h": case undefined: return cmdHelp();
    default: die(`unknown command: ${cmd}. Run: parallelix-node help`);
  }
})();
