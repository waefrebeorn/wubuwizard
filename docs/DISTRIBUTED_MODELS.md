# Distributed models: the composition

One API for a model that lives on the swarm, not on a host. No single protocol
covers it, so the legs are composed. Status is honest: what is built and
tested, what is next.

| leg | protocol | status |
|---|---|---|
| address | BitTorrent magnet (BEP 9) | **built** — `src/wubu_magnet.c`, `make test_magnet` |
| hash | SHA-1 of the bencoded info dict (the btih) | next — needs a SHA-1 that is not here yet |
| swarm | DHT, BEP 5 (node IDs, KRPC, `get_peers`) | next |
| p2p | libp2p (multiaddr, bitswap) | next |
| permanence | Pirate Face / swarm retention | next |
| compute | DePIN: Akash, io.net, Gensyn, Bittensor | next |
| inference | OpenAI-compatible API | next |

## What exists

`include/wubu_magnet.h` / `src/wubu_magnet.c` build and parse a BEP 9 magnet
URI. A model is addressed by its info-hash and carries its trackers, so the
bytes can come from the swarm instead of one host.

```
magnet:?xt=urn:btih:<40 hex>&dn=<name>&tr=<tracker>&xl=<bytes>
```

The module computes no hash. A btih must be the SHA-1 of a real bencoded info
dict; inventing one would make the address a lie. `parse()` refuses a URI with
no `xt`, a malformed info-hash, a bad percent escape, or an over-long part —
every refusal the same `-1`, and the caller's struct is left untouched.

## What is next, in order

1. **SHA-1 + bencode** (`wubu_infohash.c`): hash a `.torrent` info dict into a
   real btih, and read the `index.json` manifests in `manifests/` so a model
   already tracked here can be addressed by magnet.
2. **DHT (BEP 5)**: node IDs and the XOR distance metric, KRPC
   `get_peers`/`announce_peer` messages, and a routing table that buckets by
   prefix distance.
3. **libp2p fallback**: multiaddr parse, bitswap block exchange.
4. **OpenAI-compatible API**: one endpoint that serves a swarm-resolved model.
5. **DePIN compute**: attach a job to Akash / io.net / Gensyn / Bittensor.
