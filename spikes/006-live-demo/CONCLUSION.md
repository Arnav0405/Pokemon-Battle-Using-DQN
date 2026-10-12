# Spike 006 — live-watching one battle

**Question.** The plan's demo pass leaves "live spectate path verified" unchecked. Can a battle be
watched as it happens on this machine, and by which route?

## Measured

| Route | Verdict |
|---|---|
| Browser spectate at `http://localhost:8000/<battle-tag>` | **Dead in this checkout.** `showdown/` has no client directory (verified: only `config/ data/ databases/ dist/ lib/ logs/ node_modules/ server/ sim/ test/ tools/ translations/`). The page served at the root is `showdown/server/static/index.html`, whose entire body is `document.location.replace(protocol + '//' + host + suffix + '.psim.us' + pathname)` — it hands the browser to the public client, which has no record of a local battle. |
| Local replay HTML, opened from disk | **Renders but needs the network.** `runs/p3_400k/demo/*.html` embeds the full protocol log, but the replay page's renderer fetches `play.pokemonshowdown.com/js/replay-embed.js`. |
| Standalone transcript page | **Works offline.** `evaluate.py --demo` also writes `<tag>-transcript.html` from the embedded log; a static page, no network. Post-hoc, not live. |
| Live terminal view (this spike) | **Works, verified.** `live_battle.py --delay 0.4` prints the battle turn by turn off `AbstractBattle._replay_data`, which poke-env appends to as protocol messages arrive. Real output: 27 turns, move/damage/faint lines, `[live] done: win=True outcome=win turns=27`. |
| Local browser client (`smogon/pokemon-showdown-client`) | **Not attempted.** Built client connects to an arbitrary server via `testclient-new.html?~~localhost:8000`. Untested caveat: `showdown/config/config.js` still sets `exports.loginserver = 'http://play.pokemonshowdown.com/'`, so a client login may expect network access. |

## Notes

- `_replay_data` holds every protocol line including `|request|` (a full battle-state JSON blob per
  turn), so the ticker filters by line type — print only `turn/move/switch/-damage/-heal/faint/...`.
- The opponent side is **not** a websocket participant: `make_opponent` builds it with
  `start_listening=False` and `SingleAgentWrapper` drives it in-process. A websocket spectator would
  still see both sides, because the env's own connection receives the whole battle room.
- `evaluate.py --demo-delay <s>` already exists and is the knob that makes a live view followable.
- Cost of a live demo in wall-clock: one battle at `--delay 0.4` ≈ 12 s for ~27 turns.
