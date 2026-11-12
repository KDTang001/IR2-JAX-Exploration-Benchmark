# Environment Presets Implementation Plan

**Goal:** Configurable rewards, propagation, addressing, accounting and sending with explicit custom functions, without starting training.

**Architecture:** Keep existing event ordering and state. Resolve named functions at the host/tracing boundary, retain the byte protocol and local-information boundary. Include implementation identities in configuration binding.

**Tech Stack:** Python dataclasses, JAX pure functions, existing JSON configuration and pytest.

- [x] Add a small versioned preset registry and validated HostConfig fields. Preserve legacy defaults; reject duplicate registrations and invalid combinations. Bind implementation digests in configuration identity.
- [x] Extract reward statistics/calculation; implement coverage, distance, sparse and source-aligned IR2 individual shaping. Document the original round-wide team bonus separately: it cannot silently become an event-local bonus.
- [x] Implement ideal, range, LOS, configurable attenuation and IR2 discrete-ray propagation. IR2 noise is sampled once per episode, independently of event/dropout keys.
- [x] Extend packet addressing, one-charge broadcast/subset delivery and success charging. Add local-only send scheduling and custom messages without bypassing protocol validation.
- [x] Integrate reward terms and physical/external link diagnostics into active and frozen transitions. Preserve default random streams and event ordering.
- [x] Add focused preset integration tests, original signal/reward comparisons, and a runnable custom example. Run relevant existing packet/channel/event and training-schema checks together, not after every edit.
- [x] Document supported combinations, hashing, independent fault conditions and metric comparability. Deploy the verified source to the authorized server without manipulating training processes.

Verification uses the existing server conda ex environment; local Python lacks JAX. No automatic full experiment campaign or new dependencies beyond necessary validation tooling.
