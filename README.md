# Optimizarr

[![CI](https://github.com/patrix87/optimizarr/actions/workflows/ci.yml/badge.svg)](https://github.com/patrix87/optimizarr/actions/workflows/ci.yml)
[![GitHub Release](https://img.shields.io/github/v/release/patrix87/optimizarr)](https://github.com/patrix87/optimizarr/releases/latest)
[![GitHub Issues](https://img.shields.io/github/issues/patrix87/optimizarr)](https://github.com/patrix87/optimizarr/issues)
[![GitHub PRs](https://img.shields.io/github/issues-pr/patrix87/optimizarr)](https://github.com/patrix87/optimizarr/pulls)
[![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue)](LICENSE)

**Keep your Radarr/Sonarr library aligned with your quality profiles, automatically and
efficiently.** Optimizarr continuously re-evaluates the releases available for every movie and
episode you already have, and grabs a *better* one when it exists: a genuine quality upgrade, or
a leaner file at the same quality. It picks the way a careful human would, weighing score,
resolution, and size together against the profile you chose, instead of assuming bigger is
better.

It does two jobs, both optional and independent:

1. **Optimizer**: a continuous worker that improves the files you already have.
2. **Unmonitor**: a cron job that stops the \*arr apps from chasing pointless upgrades forever.

## Why you might want it

- **Realign an existing library to a profile, including downgrades.** Change a Radarr/Sonarr
  quality profile and the app will not touch the files you already have. Optimizarr will: point a
  movie at a leaner profile and it will **shrink the file, lower the bitrate, even drop
  resolution** to match what you now want. Radarr/Sonarr only ever upgrade; Optimizarr realigns
  in *both* directions. This is the most-requested thing it does.
- **Reclaim disk.** The size-leaning profiles (Efficient, Compact) swap bloated files for
  good-but-smaller encodes, without ever making a file *bigger* unless it is a real upgrade.
- **Pick like a human, per profile.** Five shipped profiles (**Remux, Quality, Balanced,
  Efficient, Compact**) each have their own taste, from "max quality, size be damned" to
  "smallest file that is still good." A multi-objective ranking (TOPSIS) plus hard size and
  resolution gates chooses among the releases your indexers actually offer.
- **Stop the endless upgrade chase.** The **Unmonitor** job unmonitors movies/episodes a set
  number of days after release, so Radarr/Sonarr stop grabbing "new" releases off RSS just
  because they appeared.

It is safe by design: it **never inflates a file** to hit a target, it **cannot oscillate**
(every swap strictly raises a single quality score, so a file is never revisited), and
"optimized" means *the algorithm can no longer find anything better*, never merely "we triggered a
grab" (grabs fail to download all the time, and it handles that).

> Want the details: the per-preset size tables, the swap rule, the TOPSIS formulas, the config
> model, and the worker loop? See **[ALGORITHM.md](ALGORITHM.md)**.

## Transparency

This tool was developed entirely with AI (Claude Opus). I am an experienced developer, I have
reviewed all of the code, and it is thoroughly tested with good unit-test coverage. I would not
have released it if I were not satisfied with its quality. That said, I think it is important to
disclose when AI is used to build tools.

## Designed for Profilarr quality profiles

Optimizarr pairs with **[Profilarr](https://github.com/Dictionarry-Hub/profilarr)**. The
defaults assume Profilarr's custom-format scoring convention, where a "perfect" release scores
around **1,000,000**, and the shipped profiles auto-attach to a Radarr/Sonarr quality profile
whose name contains a keyword, e.g. `2160p Quality` to Quality, `1080p Efficient` to Efficient,
`2160p Remux` to Remux.

It is **not strictly required**. Both the score scale and the profiles are configurable, so
Optimizarr works against any custom-format scoring scheme. Profilarr just saves you the tuning.

## Quick start (Docker Compose)

```yaml
services:
  optimizarr:
    image: ghcr.io/patrix87/optimizarr:latest
    container_name: optimizarr
    restart: unless-stopped
    environment:
      TZ: America/Toronto
      LOG_LEVEL: INFO
      RADARR_URL: http://radarr:7878
      RADARR_API_KEY: ${RADARR_API_KEY}
      SONARR_URL: http://sonarr:8989
      SONARR_API_KEY: ${SONARR_API_KEY}
    volumes:
      - ./config:/config   # put config.toml here
      - ./data:/data       # state.json lives here
```

Drop a `.env` next to it with `RADARR_API_KEY=...` / `SONARR_API_KEY=...`, put your
`config.toml` in `./config/`, then `docker compose up -d`. Pin a tag (e.g. `:v0.2.0`) instead of
`:latest` for reproducible deploys; to build from a clone, swap `image:` for `build: .`.

**Try it dry first:** set `dry_run = true` in `config.toml` to log every would-be grab/unmonitor
without changing anything.

> **Enable Failed Download Handling** in Radarr/Sonarr (it is on by default). The optimizer relies
> on it to blocklist dead releases so it can move to the next-best one.

## Configuration

Two parts:

- **Environment** holds only secrets, URLs, and paths.
- **`config.toml`** holds all behavior, layered on top of the bundled
  [`defaults.toml`](optimizarr/defaults.toml), so you only set what you want to change.
  `defaults.toml` documents every option inline: read it, then copy just the keys you want to
  change into your `config.toml`. With no `config.toml` at all, the defaults are used as-is.

### Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `RADARR_URL` / `RADARR_API_KEY` | unset | Set both to enable Radarr. |
| `SONARR_URL` / `SONARR_API_KEY` | unset | Set both to enable Sonarr. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `TZ` | container default | Affects cron evaluation and log timestamps. |

If neither Radarr nor Sonarr is configured, the container exits 1.

### The bits you will actually touch

The defaults are sensible; most setups only adjust a handful of keys (every key is documented
inline in [`defaults.toml`](optimizarr/defaults.toml)):

| Key | What it does |
| --- | --- |
| `[optimizer] enabled` / `[unmonitor] enabled` | Turn each feature on/off independently. |
| `[optimizer.<app>] min_age_days` + `release_type` | Only touch items this many days past every listed date, so freshly released/imported items are left alone. |
| `[optimizer.<app>] allow_size_increase` | `false` blocks any bigger file (also blocks resolution upgrades). |
| `[optimizer.<app>] allow_quality_downgrade` | `false` blocks lower-score releases. **Turn this off if you only ever want upgrades.** Leaving it on is what lets Efficient/Compact realign downward. |
| `[unmonitor.<app>] days` / `release_type` / `require_cutoff_met` | When to unmonitor after release, and whether to wait for the quality cutoff first. |
| `[optimizer.topsis]` | The selection engine: per-preset size **tables** (floor/target/bloat), **presets**, and tuning. You rarely need this. See [ALGORITHM.md](ALGORITHM.md). |

The optimizer selects items by **`hasFile`**, regardless of monitored state. It improves the
existing library, and the unmonitor job deliberately strips monitoring once a file exists.

## Local development

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .

# Run against a real Radarr/Sonarr without Docker:
cp .env.example .env             # fill in URLs + API keys
# Optionally create a config.toml with any overrides (it layers on optimizarr/defaults.toml,
# which documents every option). With no config.toml, the built-in defaults are used.
# optimizarr reads /config/config.toml and writes /data/state.json by default;
# to run outside Docker, edit CONFIG_PATH / STATE_PATH at the top of optimizarr/config.py.
uv run --env-file .env python -m optimizarr
```

Set `dry_run = true` in `config.toml` first to log would-be actions without writing anything (no
grabs, no unmonitors, no state changes). `tools/weight_lab.py` renders how each profile scores and
picks on sample releases.

## Releasing

Two workflows live in `.github/workflows/`:

- **ci.yml** runs `ruff check`, `ruff format --check`, and `pytest` on every push to `main` and
  every pull request.
- **release.yml** builds a `linux/amd64` image and pushes it to `ghcr.io/patrix87/optimizarr`
  (the image name follows the repo via `${{ github.repository }}`), then publishes a **GitHub
  Release** with auto-generated notes for the tag.

| Trigger | Image tags pushed | GitHub Release |
| --- | --- | --- |
| Push tag `v1.2.3` | `1.2.3`, `1.2`, `1`, `latest` | created (auto notes) |
| Manual `workflow_dispatch` | (none, no semver tag applies) | not created |

```sh
git tag v0.2.0
git push --tags
```

One-time repo setup before the first release run:

- **Settings, Actions, General, Workflow permissions**: *Read and write permissions*.
- After the first push, open the package at `https://github.com/patrix87?tab=packages`, set
  visibility to **Public**, and confirm it linked to the repo.

## Maintenance

```sh
uv lock --upgrade
```

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). You may use, modify, and share it for any
noncommercial purpose, and you must keep the required attribution notice. Commercial use is not
permitted without a separate agreement. Copyright (c) 2026 Patrick Veilleux (Patrix87).
