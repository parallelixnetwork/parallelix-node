# parallelix-node

The ParalleliX node operator client. It attaches a CPU or GPU machine to a node you registered on-chain, sends signed liveness heartbeats (uptime), and serves dispatched inference through a local [Ollama](https://ollama.com) runtime, returning a SHA-256 Proof-of-Execution the coordinator verifies.

This is the off-chain half of being an operator. The on-chain half (stake, register, claim) happens in the [Console](https://app.parallelix.io).

## Install

```bash
curl -fsSL https://parallelix.io/install.sh | sh
```

This installs Node, Ollama, a default open-source model, and the CLI.

## Use

```bash
parallelix-node probe                       # detect GPU/CPU + Ollama, print your tier
parallelix-node init                        # generate the node key locally, print the nodeKeyHash
# register on-chain in the Console with that nodeKeyHash, get a nodeId
parallelix-node start --node-id <id> --gpu  # attach, heartbeat, serve inference
parallelix-node service --node-id <id> --gpu # run as a 24/7 service
parallelix-node verify | status | logs       # diagnostics
```

The node key is a separate low-value keypair generated on your machine. It is used only for liveness and result signatures, never your staking wallet's key.

## Security

The client reads its coordinator URL (`https://parallelix.io/api` by default) and talks to a local Ollama at `127.0.0.1:11434`. It generates its own keypair locally and never transmits a private key. Read the source before you run it.

## License

MIT
