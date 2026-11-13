# Distributed models: the composition

One API for a model that lives on the swarm, not on a host. No single protocol
covers it, so the legs are composed. Status is honest: what is built and
tested, what is next.

| leg | protocol | status |
|---|---|---|
| address | BitTorrent magnet (BEP 9) | **built** — `src/wubu_magnet.c`, `make test_magnet` |
| hash | SHA-1 of the bencoded info dict (the btih) | **built** — `src/wubu_sha1.c` + `src/wubu_torrent.c`, `make test_sha1` / `test_torrent` |
| swarm | DHT, BEP 5 (node IDs, KRPC, `get_peers`) | **built** — `src/wubu_dht.c`, `make test_dht` |
| p2p | libp2p (multiaddr, bitswap) | next |
| permanence | Pirate Face / swarm retention | next |
| compute | DePIN: Akash, io.net, Gensyn, Bittensor | next |
| inference | OpenAI-compatible API | next |

## What exists

### Magnet (BEP 9)

`include/wubu_magnet.h` / `src/wubu_magnet.c` build and parse:

```
magnet:?xt=urn:btih:<40 hex>&dn=<name>&tr=<tracker>&xl=<bytes>
```

Fail-closed: refuses a non-magnet, a URI with no `xt`, an unknown exact topic,
a short or non-hex info-hash, a bad percent escape, a parameter with no value,
and an over-long display name (a truncated name is a *different* model's
address). Every refusal is the same `-1` and leaves the caller's struct
untouched. `dn` is percent-encoded out and decoded in; unknown parameters are
ignored, as BEP 9 lets senders add them.

The magnet module computes no hash. A btih must be the SHA-1 of a real
bencoded info dict; inventing one would make the address a lie.

### SHA-1 (FIPS 180-4)

`include/wubu_sha1.h` / `src/wubu_sha1.c` — one-shot plus streaming, with the
schedule built per block so the caller's opaque context stays 96 bytes.

SHA-1 is broken for collision resistance and must not be used where a collision
matters. An info-hash is not such a use: it is an address, its job is to be
derived identically by every peer, and the swarm verifies the bytes against the
piece hashes anyway. Do not carry this primitive anywhere security-critical
without saying so out loud.

Verified against the published FIPS 180-4 vectors (`""`, `"abc"`, the 56-byte
message, the 1,000,000-`a` message) and the property the info-hash depends on:
splitting the input at *every* byte boundary streams to the same digest.

## What is next, in order

1. **The DHT routing table**: the identity/distance/KRPC leg exists; the table
   that buckets by shared-prefix length, answers `find_node`, and picks which
   peers to ask for an info-hash is next.
2. **libp2p fallback**: multiaddr parse, bitswap block exchange.
3. **OpenAI-compatible API**: one endpoint that serves a swarm-resolved model.
4. **DePIN compute**: attach a job to Akash / io.net / Gensyn / Bittensor.

### The DHT addressing core (BEP 5)

`include/wubu_dht.h` / `src/wubu_dht.c`:

- A node id is **derived** (SHA-1 of the node's seed), never typed in — two
  nodes cannot claim one id by both choosing it, and an id cannot be placed
  somewhere convenient.
- Distance is XOR over the 160-bit id, compared big-endian. Bucketing is by
  shared prefix length, so the closest buckets are known exactly.
- The four KRPC messages (`ping`, `find_node`, `get_peers`, `announce_peer`)
  encode and decode. Requests must carry a transaction id — without one a
  response cannot be matched — so encoding refuses to omit it.
- Errors carry their code under `e` as `[code, message]`, not under `r`.
- Fail-closed on non-bencode bodies, truncated datagrams, trailing bytes, a
  wrong-shaped id or token, and a request with no id. Every refusal is `-1`
  and leaves the caller's struct untouched.

48 checks, including the metric properties (symmetric, zero only when equal),
ordering against a known anchor, and a single differing bit reported at its
exact position across all 160 bits.

### Reading a `.torrent`

`include/wubu_torrent.h` / `src/wubu_torrent.c` locate the `info` value's byte
span and SHA-1 it. It does **not** re-encode the dictionary: hashing a
re-encoded copy gives a different hash unless the original was already
canonical, so a hash of anything other than the file's own bytes is a lie.

Fail-closed on a truncated file, a non-dictionary top level, a bad string
length, a leading-zero length prefix, trailing bytes, a missing `info`, or an
empty one (an empty info dict addresses nothing usable). Every refusal is the
same `-1`.

Tested against two independently constructed torrents whose spans and digests
were computed with Python's `hashlib`: a minimal single-file torrent, and a
multi-file one with keys before `info` and a nested `files` list — so the span
is proved to be exactly the info dict and not a fixed offset.
