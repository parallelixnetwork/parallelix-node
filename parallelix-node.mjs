#!/usr/bin/env node
// =============================================================================
// parallelix-node · ParalleliX Node operator CLI · v1.1 (real GPU compute)
// =============================================================================
//
// The off-chain half of being a ParalleliX operator. Generates a separate node
// key (never your staking wallet), attaches a machine to a node you registered
// on-chain, sends signed liveness heartbeats (your uptime), and serves REAL
// ParalleliX AI inference requests on your local GPU via Ollama, returning a
// SHA-256 Proof-of-Execution the coordinator verifies.
//
//   parallelix-node probe                  Detect hardware + Ollama; print the tier
//   parallelix-node init [--wallet 0x..]   Generate node key + config; print nodeKeyHash
//   parallelix-node models [pull <name>]   List / pull local Ollama models
//   parallelix-node start --node-id N      Run the daemon: attach, heartbeat, serve
//             [--gpu|--cpu] [--model m]
//   parallelix-node verify                 Run diagnostics
//   parallelix-node status                 Print local node state
//   parallelix-node logs                   Tail the node log
//   parallelix-node version
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

const VERSION = "1.4.1";

// Canonical on-chain references (Ethereum mainnet).
const PRLX = "0x93FF39f65cC1D21067939961993ADF3f36BBF893";
const STAKE_CONTRACT = "0x706851273c3f5892e2d68ff48dd80bea02a382b6"; // NodeRegistryLocker
const REWARDS_CONTRACT = "0x266939a8baa29344c7687ce2b5074af6dec984e3"; // OperatorStakeRewards

const API_URL = (argFlag("--coordinator-url") || process.env.PARALLELIX_API_URL || "https://parallelix.io/api").replace(/\/$/, "");
const OLLAMA = (argFlag("--ollama-url") || process.env.OLLAMA_HOST || "http://127.0.0.1:11434").replace(/\/$/, "");
const DEFAULT_MODEL = argFlag("--model") || process.env.PARALLELIX_MODEL || "llama3.2";

const DIR = path.join(os.homedir(), ".parallelix");
const KEY_PATH = path.join(DIR, "node.key");
const CFG_PATH = path.join(DIR, "config.json");
const LOG_PATH = path.join(DIR, "node.log");
const HEARTBEAT_MS = 10_000;
const POLL_MS = 2_000;

const C = process.stdout.isTTY ? {
  g: (s) => `\x1b[38;2;215;255;1m${s}\x1b[0m`, d: (s) => `\x1b[2m${s}\x1b[0m`,
  b: (s) => `\x1b[1m${s}\x1b[0m`, r: (s) => `\x1b[38;2;255;90;90m${s}\x1b[0m`,
} : { g: (s) => s, d: (s) => s, b: (s) => s, r: (s) => s };
const ok = (m) => console.log(`${C.g("✓")} ${m}`);
const info = (m) => console.log(`${C.d("·")} ${m}`);
const warn = (m) => console.log(`${C.r("!")} ${m}`);
const die = (m) => { console.error(`${C.r("parallelix-node:")} ${m}`); process.exit(1); };

function argFlag(name) {
  const a = process.argv.find((x) => x.startsWith(`${name}=`));
  if (a) return a.slice(name.length + 1);
  const i = process.argv.indexOf(name);
  if (i !== -1 && process.argv[i + 1] && !process.argv[i + 1].startsWith("-")) return process.argv[i + 1];
  return null;
}
const hasFlag = (n) => process.argv.includes(n);
const timeout = (ms) => (AbortSignal.timeout ? AbortSignal.timeout(ms) : undefined);

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
  const r = await fetch(`${OLLAMA}/api/pull`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, stream: false }) });
  if (!r.ok) die(`ollama pull failed: HTTP ${r.status}`);
  ok(`pulled ${name}`);
}
function poe(requestId, result, nodeId) { return "0x" + toHex(sha256(new TextEncoder().encode(`${requestId}|${result}|${nodeId}`))); }

// ── commands ──────────────────────────────────────────────────────────────────
async function cmdProbe() {
  const hw = detectHardware();
  console.log(C.b("// hardware probe"));
  info(`cpu       ${hw.cores} cores`);
  info(`memory    ${hw.ramGb} GB`);
  info(`gpu       ${hw.gpu ? `${hw.gpu} · ${hw.vramGb} GB VRAM` : "none detected (CPU mode)"}`);
  const ov = await ollamaVersion();
  info(`ollama    ${ov ? `running (v${ov})` : C.r("not running — install: https://ollama.com/download")}`);
  console.log("");
  ok(`maps to ${C.g(`tier ${hw.tier}`)} (${hw.label})`);
  if (!ov) info(`then start Ollama, and: ${C.b("parallelix-node models pull " + DEFAULT_MODEL)}`);
  info(`next: ${C.b("parallelix-node init")}`);
}

function cmdInit() {
  if (fs.existsSync(KEY_PATH) && !hasFlag("--force")) die(`a node key already exists at ${KEY_PATH}. Re-run with --force to overwrite (orphans the old node).`);
  fs.mkdirSync(DIR, { recursive: true, mode: 0o700 });
  const hw = detectHardware(), kp = genKeypair();
  fs.writeFileSync(KEY_PATH, kp.priv, { mode: 0o600 });
  const cfg = { version: VERSION, nodeAddress: kp.address, nodeKeyHash: kp.nodeKeyHash, pubKey: "0x" + kp.pubXY,
    stakingWallet: argFlag("--wallet") || null, tier: hw.tier, createdAt: new Date().toISOString(),
    coordinator: API_URL, ollama: OLLAMA, contracts: { staking: STAKE_CONTRACT, rewards: REWARDS_CONTRACT, prlx: PRLX } };
  fs.writeFileSync(CFG_PATH, JSON.stringify(cfg, null, 2), { mode: 0o600 });
  ok(`node key generated at ${KEY_PATH} ${C.d("(mode 0600)")}`);
  ok(`config written to ${CFG_PATH}`);
  console.log("\n" + C.b("// register this node on-chain (in the Console), using:"));
  console.log(`   ${C.d("tier")}         ${hw.tier} (${hw.label})`);
  console.log(`   ${C.d("nodeKeyHash")}  ${C.g(kp.nodeKeyHash)}\n`);
  info(`This key signs liveness + results only. It is NOT your staking wallet.`);
  info(`Stake + registerNode at ${C.b("https://app.parallelix.io")} with the nodeKeyHash above,`);
  info(`then: ${C.b(`parallelix-node start --node-id <id> ${hw.gpu ? "--gpu" : "--cpu"}`)}`);
}

async function cmdModels() {
  const sub = process.argv[3];
  if (sub === "pull") { const name = process.argv[4] || DEFAULT_MODEL; await ollamaPull(name); return; }
  const ov = await ollamaVersion();
  if (!ov) die("Ollama is not running. Install it (https://ollama.com/download) and start it.");
  const models = await ollamaModels();
  console.log(C.b(`// ollama models (${OLLAMA})`));
  if (!models.length) { warn("no models installed."); info(`pull one: ${C.b("parallelix-node models pull " + DEFAULT_MODEL)}`); return; }
  models.forEach((m) => info(m));
}

async function cmdVerify() {
  let pass = true;
  console.log(C.b("// verify"));
  Number(process.versions.node.split(".")[0]) >= 18 ? ok(`node ${process.versions.node}`) : (warn(`node ${process.versions.node} (need 18+)`), pass = false);
  if (fs.existsSync(KEY_PATH)) { const mode = (fs.statSync(KEY_PATH).mode & 0o777).toString(8); mode === "600" ? ok(`node key present, perms ${mode}`) : (warn(`node key perms ${mode} (chmod 600 ${KEY_PATH})`), pass = false); }
  else { warn("no node key — run: parallelix-node init"); pass = false; }
  const cfg = loadConfig(); cfg ? ok(`config valid · node ${cfg.nodeAddress}`) : warn("no config — run: parallelix-node init");
  const hw = detectHardware(); ok(`hardware tier ${hw.tier} (${hw.label})`);
  const ov = await ollamaVersion(); ov ? ok(`ollama running (v${ov})`) : (warn(`ollama not running (${OLLAMA}) — install https://ollama.com/download`), pass = false);
  if (ov) { const m = await ollamaModels(); m.length ? ok(`models: ${m.join(", ")}`) : (warn(`no models — parallelix-node models pull ${DEFAULT_MODEL}`), pass = false); }
  try { const r = await fetch(`${API_URL}/health`, { signal: timeout(5000) }).catch(() => fetch(`${API_URL}/`, { signal: timeout(5000) })); (r && r.status < 500) ? ok(`coordinator reachable · ${API_URL}`) : (warn(`coordinator unreachable · ${API_URL}`), pass = false); } catch { warn(`coordinator unreachable · ${API_URL}`); pass = false; }
  console.log(""); pass ? ok("all checks passed · ready to start") : warn("some checks failed (see above)");
  process.exit(pass ? 0 : 1);
}

function cmdStatus() {
  const cfg = loadConfig(); if (!cfg) die("not initialised — run: parallelix-node init");
  console.log(C.b("// node status"));
  info(`node address   ${cfg.nodeAddress}`);
  info(`nodeKeyHash    ${cfg.nodeKeyHash}`);
  info(`tier           ${cfg.tier}`);
  info(`coordinator    ${cfg.coordinator}`);
  info(`ollama         ${cfg.ollama || OLLAMA}`);
  info(`staking CA     ${cfg.contracts?.staking}`);
}

function cmdLogs() {
  if (!fs.existsSync(LOG_PATH)) die(`no log yet at ${LOG_PATH}`);
  console.log(fs.readFileSync(LOG_PATH, "utf8").trim().split("\n").slice(-50).join("\n"));
}

async function cmdStart() {
  const nodeId = argFlag("--node-id");
  if (!nodeId) die("missing --node-id. Register on-chain first, then: parallelix-node start --node-id <id>");
  const priv = loadKey(), cfg = loadConfig();
  if (!priv || !cfg) die("not initialised — run: parallelix-node init");
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
  ok(`daemon up · heartbeating + serving · ctrl-c to stop`);
  appendLog(`start node=${nodeId} mode=${mode} model=${model} gpu=${hw.gpu || "none"}`);

  let beats = 0, served = 0, online = false, busy = false, rejects = 0;
  const capabilities = { version: VERSION, tier: hw.tier, mode, model, gpu: hw.gpu, vramGb: hw.vramGb, cores: hw.cores, ramGb: hw.ramGb };

  const heartbeat = async () => {
    const ts = Date.now();
    const message = `parallelix-node:heartbeat:${nodeId}:${ts}`;
    try {
      const r = await fetch(`${API_URL}/operator/heartbeat`, { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ nodeId, nodeAddress: cfg.nodeAddress, message, signature: personalSign(message, priv), capabilities, served }),
        signal: timeout(8000) });
      beats++;
      if (r.ok) {
        rejects = 0;
        if (!online) { online = true; ok("coordinator acknowledged · node ACTIVE · uptime accruing"); }
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
            process.exit(0);
          }
          if (++rejects >= 3) {
            console.log(`\n${C.r("✗")} coordinator keeps rejecting node ${nodeId} (${reason || r.status}). Check the --node-id and that you registered this machine's nodeKeyHash.`);
            appendLog(`exit rejected ${reason || r.status}`);
            process.exit(1);
          }
        }
      }
    } catch (e) { online = false; appendLog(`heartbeat err ${String(e?.message || e)}`); }
    render();
  };

  const poll = async () => {
    if (busy) return;
    let job = null;
    try {
      const r = await fetch(`${API_URL}/operator/inbox?nodeId=${encodeURIComponent(nodeId)}`, { signal: timeout(6000) });
      if (r.ok) { const d = await r.json(); job = (d.requests || d.jobs || [])[0] || null; }
    } catch {}
    if (!job) return;
    busy = true;
    const reqId = job.id || job.requestId, prompt = job.prompt ?? job.input ?? "";
    const sys = job.system ?? null;
    const jm = job.model && models.includes(job.model) ? job.model : model;
    try {
      const t0 = Date.now();
      const gen = await ollamaGenerate(jm, prompt, sys);
      const result = gen.text;
      const ms = Date.now() - t0;
      const proof = poe(reqId, result, nodeId);
      const message = `parallelix-node:result:${reqId}`;
      await fetch(`${API_URL}/operator/result`, { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ nodeId, requestId: reqId, result, poe: proof, model: jm, ms, inputTokens: gen.inputTokens, outputTokens: gen.outputTokens, message, signature: personalSign(message, priv) }), signal: timeout(15000) });
      served++; appendLog(`served ${reqId} model=${jm} ms=${ms} poe=${proof.slice(0, 14)}`);
    } catch (e) {
      appendLog(`serve-fail ${reqId} ${String(e?.message || e)}`);
      // Signed failure report so the coordinator can verify it's really this
      // node giving up the request, and requeue it for another node at once
      // (an unsigned report is ignored and only requeues on the reaper TTL).
      const message = `parallelix-node:result:${reqId}`;
      try { await fetch(`${API_URL}/operator/result`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ nodeId, requestId: reqId, error: String(e?.message || e), message, signature: personalSign(message, priv) }), signal: timeout(8000) }); } catch {}
    } finally { busy = false; render(); }
  };

  const render = () => process.stdout.write(`\r${online ? C.g("●") : C.d("○")} ${online ? "active" : "connecting"} · heartbeats ${beats} · served ${served} ${busy ? "· " + C.g("running inference") : ""}      `);

  await heartbeat();
  const hb = setInterval(heartbeat, HEARTBEAT_MS);
  const pl = setInterval(poll, POLL_MS);
  // Drain cleanly on Ctrl-C AND on SIGTERM (systemctl stop / launchd / kill).
  const stop = (sig) => {
    clearInterval(hb); clearInterval(pl);
    console.log(`\n${C.d("·")} draining (${sig}) · node going idle · stake stays locked, no slashing`);
    appendLog(`stop ${sig}`);
    process.exit(0);
  };
  process.on("SIGINT", () => stop("SIGINT"));
  process.on("SIGTERM", () => stop("SIGTERM"));
}

// Install the node as a persistent background service so it survives logout /
// reboot and auto-restarts on crash. systemd (Linux) or launchd (macOS).
function cmdService() {
  const nodeId = argFlag("--node-id");
  if (!nodeId) die("missing --node-id. Usage: parallelix-node service --node-id <id> [--gpu|--cpu]");
  const modeFlag = hasFlag("--gpu") ? "--gpu" : hasFlag("--cpu") ? "--cpu" : "";
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
Restart=always
RestartSec=5

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

function cmdHelp() {
  console.log(`${C.g("ParalleliX")} node CLI ${C.d("v" + VERSION)} ${C.d("· real GPU compute via Ollama")}

${C.b("USAGE")}  parallelix-node <command> [options]

${C.b("COMMANDS")}
  probe                          detect hardware + Ollama, print the tier
  init [--wallet 0x..]           generate the node key + config; print nodeKeyHash
  models [pull <name>]           list / pull local Ollama models
  start --node-id <id>           run the daemon: attach, heartbeat (uptime),
        [--gpu|--cpu] [--model m]   serve real inference, return Proof-of-Execution
  service --node-id <id>         install as a background service (systemd/launchd):
        [--gpu|--cpu]               survives logout + reboot, auto-restarts
  verify                         run diagnostics
  status / logs / version

${C.b("FLOW")}
  1) install Ollama (https://ollama.com) + start it
  2) parallelix-node probe
  3) parallelix-node models pull ${DEFAULT_MODEL}
  4) parallelix-node init                ${C.d("→ prints your nodeKeyHash")}
  5) stake + registerNode at https://app.parallelix.io  ${C.d("→ get your node id")}
  6) parallelix-node start --node-id <id> --gpu       ${C.d("→ test it runs")}
  7) parallelix-node service --node-id <id> --gpu     ${C.d("→ run 24/7 in the background")}

${C.b("WINDOWS")}  native Windows is not supported. Install WSL2
  (https://learn.microsoft.com/windows/wsl/install), open Ubuntu, then run the
  one-line installer + these steps inside it.

  docs: https://docs.parallelix.io/node-cli`);
}

const cmd = process.argv[2];
(async () => {
  switch (cmd) {
    case "probe": return cmdProbe();
    case "init": return cmdInit();
    case "models": return cmdModels();
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
