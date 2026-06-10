# parallelix-node

The ParalleliX node operator client, v2. It attaches a CPU or GPU machine to a node registered on-chain, sends signed liveness heartbeats (uptime), and serves dispatched inference through a local [Ollama](https://ollama.com) runtime, returning a SHA-256 Proof-of-Execution the coordinator verifies.

This is the off-chain half of being an operator. The on-chain half (stake, register, claim) happens in the [Console](https://app.parallelix.io). In v2 the two halves connect themselves: `setup` prints a prefilled staking link and auto-detects your registration the moment the stake lands.

## Install

```bash
curl -fsSL https://parallelix.io/install.sh | sh
```

This installs Node, Ollama, a default open-source model, and the CLI, then offers to run setup.

## One command

```bash
parallelix-node setup
```

Setup runs the whole flow: detects your hardware and tier, checks Ollama, recommends and pulls the right open model for your VRAM, creates your node identity (a separate low-value key, never your staking wallet), prints a prefilled staking link for the Console, then polls the chain. When your stake lands it saves your node id and offers to install itself as a 24/7 service (systemd/launchd). Re-running setup is always safe; it resumes where it left off.

## Run

```bash
parallelix-node start            # node id saved by setup; live dashboard on a terminal
parallelix-node service          # run 24/7 in the background, journalctl-clean logs
```

The dashboard shows requests as your machine serves them, a Proof-of-Execution tick per result, and $PRLX accruing in real time.

## Models

```bash
parallelix-node models catalog    # Mistral, Qwen 2.5, DeepSeek-R1, Gemma 2, Phi-4, Llama 3, sized 3B-70B
parallelix-node models recommend  # picks the largest model your hardware runs with headroom
parallelix-node models pull <id>  # any Ollama model works
```

Installed models are advertised in heartbeats; the ParalleliX AI model picker lists what online nodes serve.

## Stay current

```bash
parallelix-node update
```

Fetches the latest release, verifies its sha256 against the published manifest, syntax-checks it, and swaps atomically with a backup. A bad download can never replace a working node.

## Manual path

The pre-v2 flow still works if you prefer it step by step:

```bash
parallelix-node probe                        # detect GPU/CPU + Ollama, print your tier
parallelix-node init                         # generate the node key, print the nodeKeyHash
# register on-chain in the Console with that nodeKeyHash, get a nodeId
parallelix-node start --node-id <id> --gpu   # attach, heartbeat, serve inference
parallelix-node verify | status | logs       # diagnostics
```

## Economics, plainly

Stake 50,000 $PRLX per node (no allowlist), get paid for uptime. Rewards stream per second, weighted by stake x hardware tier x uptime. No slashing: principal returns in full after a 7-day unstake cooldown. Earning requires a real online machine; holding alone earns nothing.

## Security model

- The node key signs liveness and results only. It is generated locally, stored at `~/.parallelix/node.key` (mode 0600), and is never your staking wallet key.
- Proof-of-Execution is a SHA-256 commitment over `requestId | result | nodeId`, verified by the coordinator. It is not a zero-knowledge proof.
- Self-update verifies the manifest sha256 and runs `node --check` before swapping; any mismatch aborts with the running binary untouched.

Docs: [docs.parallelix.io/node-cli](https://docs.parallelix.io/node-cli)

## License

MIT
