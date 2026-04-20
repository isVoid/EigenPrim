#!/usr/bin/env python3
"""Post-install configuration for numbast-optix devcontainer.

Runs on container creation to set up:
- Onboarding bypass (when CLAUDE_CODE_OAUTH_TOKEN is set)
- Claude settings (bypassPermissions mode)
- GPG keyring (copied from read-only host mount)
- GitHub CLI authentication (via GH_TOKEN)
- Tmux configuration (200k history, mouse support)
- Directory ownership fixes for mounted volumes
"""

import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def setup_onboarding_bypass():
    """Bypass the interactive onboarding wizard when CLAUDE_CODE_OAUTH_TOKEN is set.

    Runs `claude -p` to seed ~/.claude.json with auth state. The subprocess
    writes the config file during startup before the API call completes, so
    a timeout is expected and acceptable. After the subprocess finishes (or
    times out), we check whether ~/.claude.json was populated and only then
    set hasCompletedOnboarding.

    Workaround for https://github.com/anthropics/claude-code/issues/8938.
    """
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if not token:
        print(
            "[post_install] No CLAUDE_CODE_OAUTH_TOKEN set, skipping onboarding bypass",
            file=sys.stderr,
        )
        return

    # When `CLAUDE_CONFIG_DIR` is set, as is done in `devcontainer.json`, `claude` unexpectedly
    # looks for `.claude.json` in *that* folder, instead of in `~`, contradicting the documentation.
    #  See https://github.com/anthropics/claude-code/issues/3833#issuecomment-3694918874
    claude_json_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home()))
    claude_json = claude_json_dir / ".claude.json"

    print("[post_install] Running claude -p to populate auth state...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["claude", "-p", "ok"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(
                f"[post_install] claude -p exited {result.returncode}: "
                f"{result.stderr.strip()}",
                file=sys.stderr,
            )
    except subprocess.TimeoutExpired:
        print(
            "[post_install] claude -p timed out (expected on cold start)",
            file=sys.stderr,
        )
    except (FileNotFoundError, OSError) as e:
        print(
            f"[post_install] Warning: could not run claude ({e}) — "
            "onboarding bypass skipped",
            file=sys.stderr,
        )
        return

    if not claude_json.exists():
        print(
            f"[post_install] Warning: {claude_json} not created by claude -p — "
            "onboarding bypass skipped",
            file=sys.stderr,
        )
        return

    config: dict = {}
    try:
        config = json.loads(claude_json.read_text())
    except json.JSONDecodeError as e:
        print(
            f"[post_install] Warning: {claude_json} has invalid JSON ({e}), "
            "starting fresh",
            file=sys.stderr,
        )

    config["hasCompletedOnboarding"] = True

    claude_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        f"[post_install] Onboarding bypass configured: {claude_json}", file=sys.stderr
    )


def setup_claude_settings():
    """Configure Claude Code with bypassPermissions enabled."""
    claude_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
    claude_dir.mkdir(parents=True, exist_ok=True)

    settings_file = claude_dir / "settings.json"

    # Load existing settings or start fresh
    settings = {}
    if settings_file.exists():
        with contextlib.suppress(json.JSONDecodeError):
            settings = json.loads(settings_file.read_text())

    # Set bypassPermissions mode
    if "permissions" not in settings:
        settings["permissions"] = {}
    settings["permissions"]["defaultMode"] = "bypassPermissions"

    settings_file.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(
        f"[post_install] Claude settings configured: {settings_file}", file=sys.stderr
    )


def setup_gpg():
    """Copy GPG keyring from read-only host mount into a writable directory.

    The host's ~/.gnupg is mounted read-only at ~/.gnupg-host. GPG requires a
    writable homedir for lock files and sockets, so we copy the key material
    into ~/.gnupg and set strict permissions (GPG enforces 700).
    """
    host_gnupg = Path.home() / ".gnupg-host"
    gnupg_dir = Path.home() / ".gnupg"

    if not host_gnupg.exists():
        print("[post_install] No ~/.gnupg-host mount found, skipping GPG setup", file=sys.stderr)
        return

    if gnupg_dir.exists() and any(gnupg_dir.iterdir()):
        print("[post_install] ~/.gnupg already populated, skipping GPG copy", file=sys.stderr)
        return

    gnupg_dir.mkdir(mode=0o700, exist_ok=True)

    # Copy key material and config (skip sockets and lock files)
    key_files = [
        "pubring.kbx", "trustdb.gpg", "gpg.conf", "gpg-agent.conf",
        "pubring.kbx~",  # backup
    ]
    # Also copy private-keys-v1.d/ directory
    for item in host_gnupg.iterdir():
        if item.name.startswith("S.") or item.suffix == ".lock":
            continue  # skip sockets and lock files
        dest = gnupg_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
            dest.chmod(0o700)
        elif item.is_file():
            shutil.copy2(item, dest)
            dest.chmod(0o600)

    gnupg_dir.chmod(0o700)
    print(f"[post_install] GPG keyring copied to {gnupg_dir}", file=sys.stderr)


def setup_gh_auth():
    """Configure GitHub CLI authentication via GH_TOKEN.

    When GH_TOKEN is set, gh commands authenticate automatically. We run
    `gh auth setup-git` to configure the git credential helper so that
    git push/pull also work with GitHub repos over HTTPS.
    """
    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        print("[post_install] No GH_TOKEN set, skipping gh auth setup", file=sys.stderr)
        return

    try:
        subprocess.run(
            ["gh", "auth", "setup-git"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        print("[post_install] GitHub CLI git credential helper configured", file=sys.stderr)
    except FileNotFoundError:
        print("[post_install] Warning: gh not found, skipping auth setup", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"[post_install] Warning: gh auth setup-git failed: {e.stderr.strip()}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("[post_install] Warning: gh auth setup-git timed out", file=sys.stderr)


def setup_tmux_config():
    """Configure tmux with 200k history, mouse support, and vi keys."""
    tmux_conf = Path.home() / ".tmux.conf"

    if tmux_conf.exists():
        print("[post_install] Tmux config exists, skipping", file=sys.stderr)
        return

    config = """\
# 200k line scrollback history
set-option -g history-limit 200000

# Enable mouse support
set -g mouse on

# Use vi keys in copy mode
setw -g mode-keys vi

# Start windows and panes at 1, not 0
set -g base-index 1
setw -g pane-base-index 1

# Renumber windows when one is closed
set -g renumber-windows on

# Faster escape time for vim
set -sg escape-time 10

# True color support
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB"

# Terminal features (ghostty, cursor shape in vim)
set -as terminal-features ",xterm-ghostty:RGB"
set -as terminal-features ",xterm*:RGB"
set -ga terminal-overrides ",xterm*:colors=256"
set -ga terminal-overrides '*:Ss=\\E[%p1%d q:Se=\\E[ q'

# Status bar
set -g status-style 'bg=#333333 fg=#ffffff'
set -g status-left '[#S] '
set -g status-right '%Y-%m-%d %H:%M'
"""
    tmux_conf.write_text(config, encoding="utf-8")
    print(f"[post_install] Tmux configured: {tmux_conf}", file=sys.stderr)


def fix_directory_ownership():
    """Fix ownership of mounted volumes that may have root ownership."""
    uid = os.getuid()
    gid = os.getgid()

    dirs_to_fix = [
        Path.home() / ".claude",
        Path("/commandhistory"),
        Path.home() / ".config" / "gh",
    ]

    for dir_path in dirs_to_fix:
        if dir_path.exists():
            try:
                # Use sudo to fix ownership if needed
                stat_info = dir_path.stat()
                if stat_info.st_uid != uid:
                    subprocess.run(
                        ["sudo", "chown", "-R", f"{uid}:{gid}", str(dir_path)],
                        check=True,
                        capture_output=True,
                    )
                    print(
                        f"[post_install] Fixed ownership: {dir_path}", file=sys.stderr
                    )
            except (PermissionError, subprocess.CalledProcessError) as e:
                print(
                    f"[post_install] Warning: Could not fix ownership of {dir_path}: {e}",
                    file=sys.stderr,
                )


def setup_global_gitignore():
    """Set up global gitignore and local git config.

    Since ~/.gitconfig is mounted read-only from host, we create a local
    config file that includes the host config and adds container-specific
    settings like core.excludesfile and delta configuration.

    GIT_CONFIG_GLOBAL env var (set in devcontainer.json) points git to this
    local config as the "global" config.
    """
    home = Path.home()
    gitignore = home / ".gitignore_global"
    local_gitconfig = home / ".gitconfig.local"
    host_gitconfig = home / ".gitconfig"

    # Create global gitignore with common patterns
    patterns = """\
# Claude Code
.claude/

# macOS
.DS_Store
.AppleDouble
.LSOverride
._*

# Python
*.pyc
*.pyo
__pycache__/
*.egg-info/
.eggs/
*.egg
.venv/
venv/
.mypy_cache/
.ruff_cache/

# Node
node_modules/
.npm/

# Editors
*.swp
*.swo
*~
.idea/
.vscode/
*.sublime-*

# Misc
*.log
.env.local
.env.*.local
"""
    gitignore.write_text(patterns, encoding="utf-8")
    print(f"[post_install] Global gitignore created: {gitignore}", file=sys.stderr)

    # Create local git config that includes host config and sets excludesfile + delta
    local_config = f"""\
# Container-local git config
# Includes host config (mounted read-only) and adds container settings

[include]
    path = {host_gitconfig}

[core]
    excludesfile = {gitignore}
    pager = delta

[interactive]
    diffFilter = delta --color-only

[delta]
    navigate = true
    light = false
    line-numbers = true
    side-by-side = false

[merge]
    conflictstyle = diff3

[diff]
    colorMoved = default

[gpg "ssh"]
    program = /usr/bin/ssh-keygen
"""
    local_gitconfig.write_text(local_config, encoding="utf-8")
    print(
        f"[post_install] Local git config created: {local_gitconfig}", file=sys.stderr
    )


def main():
    """Run all post-install configuration."""
    print("[post_install] Starting post-install configuration...", file=sys.stderr)

    setup_onboarding_bypass()
    setup_claude_settings()
    setup_gpg()
    setup_gh_auth()
    setup_tmux_config()
    fix_directory_ownership()
    setup_global_gitignore()

    print("[post_install] Configuration complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
