# Distributed models: the composition

One API for a model that lives on the swarm, not on a host. No single protocol
covers it, so the legs are composed. Status is honest: what is built and
tested, what is next.

| leg | protocol | status |
|---|---|---|
| address | BitTorrent magnet (BEP 9) | **built** — `src/wubu_magnet.c`, `make test_magnet` |
| hash | SHA-1 of the bencoded info dict (the btih) | **built** — `src/wubu_sha1.c` + `src/wubu_torrent.c`, `make test_sha1` / `test_torrent` |
| swarm | DHT, BEP 5 (node IDs, KRPC, `get_peers`) | **built** — `src/wubu_dht.c`, `make test_dht` |
| swarm-state | routing table + peer selection | **built** — `src/wubu_dht_table.c`, `make test_dht_table` |
| p2p-address | libp2p multiaddr | **built** — `src/wubu_multiaddr.c`, `make test_multiaddr` |
| p2p-id | CID v0/v1 | **built** — `src/wubu_cid.c`, `make test_cid` |
| block-exch | bitswap Message | **built** — `src/wubu_bitswap.c`, `make test_bitswap` |
| mux-address | multistream-select | **built** — `src/wubu_multistream.c`, `make test_multistream` |
| stream-mux | yamux framing | **built** — `src/wubu_yamux.c`, `make test_yamux` |
| crypto | noise handshake | next |
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

1. **libp2p fallback**: multiaddr parse, bitswap block exchange.
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

### The routing table

`include/wubu_dht_table.h` / `src/wubu_dht_table.c` is the state half: buckets
keyed by shared-prefix length against **our own** id, so a bucket holds peers
at roughly one distance and the closest neighbourhood is known exactly.

- A full bucket (8) evicts its **stalest** member — a live peer beats a
  remembered one, and a dead slot otherwise stays occupied forever.
- Our own id and the zero id are refused: asking ourselves is a loop, and a
  zero id is what an uninitialised peer looks like, which poisons every
  distance calculation downstream.
- `closest()` orders by distance to a target using a bounded insertion into a
  fixed output (no heap, no allocation). `find_node()` caps at k=8.
- `reap()` drops entries silent past a window, the same discipline as the SDR
  session layer: a slot becomes reusable only after its peer is reaped.

41 checks, including eviction of the stalest rather than the freshest, exact
ordering by distance, the k cap, and reap keeping the recent while dropping
the silent.

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

### libp2p multiaddr

`include/wubu_multiaddr.h` / `src/wubu_multiaddr.c`. A multiaddr is a
self-describing path to a peer — `/ip4/1.2.3.4/tcp/6881/p2p/<peer-id>` — so a
peer can be named without knowing yet how to reach it. Same property as the
magnet: an address that survives the host.

- ipv4, ipv6 (including `::` compression and the IPv4-mapped tail), tcp, udp,
  dns/dns4/dns6, p2p, and the flag protocols (`quic`, `tls`, `ws`, ...).
- Fail-closed: unknown protocol, trailing or doubled slash, leading-zero octet
  or port, out-of-range values, malformed IPv6. Every refusal is `-1` and the
  caller's struct is untouched.
- The IPv6 vectors were checked against Python's `ipaddress`: 16 literals where
  the parser and the reference implementation agree on accept/reject AND on the
  resulting bytes — including `::ffff:1.2.3.4`, which builds back as the
  equivalent hex groups `::ffff:102:304`.

80 checks.

Two bugs my own tests caught, both worth remembering: `strchr(p, '.')` for the
IPv4-tail test matched a dot belonging to a LATER group and rejected valid
addresses (scope the search with `strcspn(p, ":")` first), and `::` followed by
eight explicit groups was accepted even though RFC 4291 requires `::` to stand
for at least one zero group.

### Content identifiers

`include/wubu_cid.h` / `src/wubu_cid.c`. A CID names WHAT something is rather
than where it lives: `<version><codec><multihash>`. The codec names the
encoding (raw, dag-pb, dag-cbor); the multihash names the hash and carries the
digest. Splitting them is what lets a peer verify bytes before knowing how to
interpret them.

- v0 (the 46-character legacy form) parses and builds as base58btc only — it
  has no other legal shape, and one that is not dag-pb / sha2-256 / 32 bytes
  does not build at all.
- v1 in base32 (`b`), base58btc (`z`) or base16 (`f`). The same CID in two
  bases decodes to identical fields and re-encodes byte-identically.
- Varints are exposed publicly because bitswap reuses them.

Every accepted literal was decoded and re-encoded independently with Python's
stdlib, including two CIDs that differ only in their last base58 digit — both
legal, with different digests. 92 checks.

Two findings worth keeping: an unknown encoding must be validated BEFORE any
branching, because a builder ending in `else { hex }` happily emits a valid
string for a nonsense base; and a negative test that assumes a one-digit
variant should be rejected will pass a broken parser and fail a correct one.

### Block exchange

`include/wubu_bitswap.h` / `src/wubu_bitswap.c`. A peer that knows WHAT it
wants still needs a way to ask for it:

- **Wantlist** — a list of CIDs plus, per entry, a priority, whether it is a
  cancellation, whether the peer wants a block or just confirmation of
  possession, and whether a "don't have" answer is acceptable. `full` marks
  whether this is the entire list rather than an incremental update.
- **Block** — a multihash prefix plus the bytes.
- **BlockPresence** — a CID plus HAVE or DONT_HAVE.

The protobuf primitives (`tag`, `varint`, length-delimited fields) are public
because the other legs reuse them. A negative int32 priority is written
sign-extended to 64 bits, which is what protobuf requires — the golden vector
in the test carries nine `0xff` bytes plus `0x01` for `-1`.

The deprecated `Message.block` repeated-bytes form is refused rather than
half-read: it carries raw blocks with no structure to validate. The module
encodes and decodes; it does not carry — producing bytes and validating
structure is its whole job.

45 checks, including a byte-for-byte golden message checked against the
go-bitswap schema by hand.

One finding worth keeping: an empty buffer is a valid, empty protobuf message,
and refusing it makes an absent payload indistinguishable from a corrupt one.
But a NULL buffer WITH a non-zero length is still a lie and is refused.

### Protocol negotiation

`include/wubu_multistream.h` / `src/wubu_multistream.c`. Before any bitswap
message travels, both peers must agree on which protocol the connection speaks:

- `<varint length><bytes>` frames, where the length counts the bytes on the
  wire -- including the trailing newline for strings.
- `/multistream/1.0.0\n` is the header every side sends first. It is 19 bytes,
  not 20 or 22: counting it wrong puts two bytes of the following frame inside
  the header and the read fails.
- A proposal such as `/ipfs/bitswap/1.2.0\n`; `na\n` when it is unknown.
- Anything printable ending in a newline is a string message; anything else is
  an opaque DATA frame, which this layer reports without interpreting.

`wubu_ms_select` compares bare protocol names against the wire form, so the
wanted table reads `/ipfs/id/1.0.0` rather than carrying its own newline.

48 checks, including the header asserted byte-for-byte against the known wire
form.

Three findings worth keeping:

- **The trailing newline is required but is not printable (0x0A).** A scan that
  treats the whole body as printable rejects the header itself; the final byte
  sits outside that scan.
- **A leading slash is not a repeated slash.** Segment validation must start
  with "no previous character", or every legal protocol fails as `//`.
- **A newline-free body is DATA, not corruption.** Refusing it would make a
  negotiated binary protocol indistinguishable from a broken frame.

### Stream multiplexing

`include/wubu_yamux.h` / `src/wubu_yamux.c`. One connection carries many logical
streams; each frame is a fixed 12-byte header plus an optional payload:

```
version    1 byte    always 0
type       1 byte    DATA / WINDOW_UPDATE / PING / GOAWAY
flags      2 bytes   BE: SYN, ACK, FIN, RST
streamID   4 bytes   BE: odd = client-initiated, even = server
length     4 bytes   BE: payload length, or the window delta
```

This module frames and parses headers. It does not open sockets, track windows,
or reassemble streams -- framing is its whole job.

The two golden vectors (a DATA/SYN frame carrying "hello" and a PING/SYN with a
7-byte code) are asserted byte-for-byte against hand-checked layouts.

One API decision worth recording: **a DATA frame's wire length is DERIVED from
the payload.** The first version trusted a caller-supplied `h->length`, which
meant a caller could claim a length that did not match the bytes actually handed
over -- the exact class of bug this whole stack exists to prevent. Control
frames keep their own length field, because there it means a window delta or a
ping code rather than a byte count.

51 checks, including consecutive frames walked forward until every byte is
consumed.

### Standards conformance notes

Checked against the published specs rather than memory:

- **yamux frame types are 0x0 Data, 0x1 Window Update, 0x2 Ping, 0x3 Go Away.**
  My first enum started at 1 -- an off-by-one that would put a Window Update on
  the wire where every libp2p peer reads Data. The test now asserts the
  spec's own numbers explicitly.
- **Multiformats unsigned varints have a practical maximum of NINE bytes
  (63 bits)**, and non-minimal encodings must be rejected: `{0x81, 0x00}` is 1
  with a wasted byte and is refused. Values at or above 2^63 are refused by
  `wubu_varint_encode` rather than silently growing to ten bytes.
- **protobuf varints differ in exactly one way**: a negative int32 is
  sign-extended to 64 bits, so `-1` is ten bytes (`ff` x9, `01`). That is why
  `wubu_pb_varint` exists separately from `wubu_varint_encode`. Refusing it
  would emit protobuf a peer cannot read.
- **multistream-select messages are UTF-8 strings always followed by `\n`**,
  length-prefixed including the newline; the responder ECHOES the protocol id
  on agreement or answers `na`. Both peers send `/multistream/1.0.0\n` first.

