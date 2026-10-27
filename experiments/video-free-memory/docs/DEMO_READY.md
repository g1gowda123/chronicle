# Demo ready — 20-minute checklist

**Branch UI:** http://127.0.0.1:8082 (zine + VFM memory)
**Fallback UI:** http://127.0.0.1:8080 (main, no VFM)

## Run / stop

```bash
cd /home/g1/projects/chronicle-vfm
./chronicle.sh status
./chronicle.sh start        # :8082; safe if already running
./chronicle.sh restart
./chronicle.sh stop         # only this worktree
```

The worktree uses the sibling main venv and imports this tree; no local `.venv`
needed. Main's `:8080` is untouched.

## Precompiled clips (do not rebuild before judging)

| Clip | What is ready |
|---|---|
| Synthetic demo | graph + captions + evidence + count bounds |
| Aria walk 60 s | **20 s caption windows**, frames at 2/10/18/… |
| London walk 15 min | caption-first vault, 95 KB vault / 2.4 MB zip |

Query does **not** need Ollama or a GPU. Models can be stopped:
`ollama stop qwen2.5vl:3b` — answers still work.

## Exact prompts to demo

```text
what happened at time 10s?
→ Frame at t=10.0s: the person walking… + object: person holding cup
```

```text
what is the person holding?
→ A cup.
```

```text
how many bowls?          (cook, if loaded)
→ 4 bowl tracks … physical bowl count is 0
```

```text
what happened around 10 minutes?   (London walk)
→ Frame at t=598.0s: busy London street…
```

## UI beats

1. Play the 15-minute walk; show 249 MB source vs 95 KB vault.
2. Ask “what happened at 10 minutes?” — answer in ms, no video.
3. Ask “what is the person holding?” — direct answer.
4. Click a citation / `frame @ t`; screenshot (`S`) and fullscreen (`F`).
5. Export graphs ZIP; import it back.
6. If asked something the vault misses, graph fallback answers; provider shows `graph-fallback`.

## If something breaks

- Memory card says **not compiled**: click **Compile memory** only if there is time; otherwise switch to synthetic/walking.
- VLM/GPU unavailable: queries from precompiled vaults still work.
- Branch UI fails: open **http://127.0.0.1:8080** and pitch the live graph + memory roadmap.
- Do not run `./chronicle.sh test` during the demo (it touches the test app/routes, not the live page, but it is noise).

## What not to say

- Do not call it “an experiment”; say “main + the memory layer”.
- Do not claim 30/30 is a public benchmark; it is an internal regression suite.
- Do not claim physical counts without the caveat.
