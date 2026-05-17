.PHONY: help smoke test lint format check status loop stop preflight backup stats health clean-runs

# Default target: print a short usage banner.
help:
	@echo 'rl-research — common operations'
	@echo
	@echo '  make smoke      — verify the install (CUDA, gymnasium, dm_control, TB)'
	@echo '  make test       — pytest (full suite, no GPU runs)'
	@echo '  make lint       — ruff check'
	@echo '  make format     — ruff format'
	@echo '  make check      — lint + format-check + test (the CLAUDE.md quality gate)'
	@echo
	@echo '  make status     — one-screen ops dashboard'
	@echo '  make preflight  — health check (used between iterations)'
	@echo '  make health     — deep readiness check (preflight + auth probe + smoke)'
	@echo '  make stats      — refresh lab/CORPUS_STATS.md from the ledger'
	@echo '  make backup     — snapshot ledger / lessons / threads to lab/.backups'
	@echo
	@echo '  make loop       — start the headless loop in a tmux session named "loop"'
	@echo '  make stop       — request halt (writes lab/HALT_REQUESTED.md)'
	@echo
	@echo 'See docs/operations.md for the full ops runbook.'

smoke:
	uv run python tests/smoke_test.py

test:
	uv run pytest

lint:
	uv run ruff check src tests lab/templates scripts

format:
	uv run ruff format src tests lab/templates scripts

check:
	uv run ruff check src tests lab/templates scripts
	uv run ruff format --check src tests lab/templates scripts
	uv run pytest

status:
	@bash scripts/status.sh

preflight:
	@bash scripts/preflight.sh

stats:
	uv run python scripts/corpus_stats.py

backup:
	@bash scripts/ledger_backup.sh

health:
	@bash scripts/preflight.sh
	@command -v claude >/dev/null 2>&1 || { echo 'claude CLI not on PATH'; exit 1; }
	@echo 'claude CLI: present (auth state will be probed by first iteration)'
	@uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print(f"CUDA: {torch.cuda.get_device_name(0)}")'
	@echo 'health: OK'

loop:
	@if tmux has-session -t loop 2>/dev/null; then \
	  echo 'tmux session "loop" already exists; attach with: tmux attach -t loop'; \
	  exit 1; \
	fi
	tmux new-session -d -s loop -c "$(CURDIR)" 'scripts/loop.sh'
	@echo 'started loop in tmux session "loop"; attach with: tmux attach -t loop'

stop:
	@if [ -f lab/HALT_REQUESTED.md ]; then \
	  echo 'lab/HALT_REQUESTED.md already exists:'; \
	  cat lab/HALT_REQUESTED.md; \
	else \
	  echo 'halt requested by `make stop` at '"$$(date -Iseconds)" > lab/HALT_REQUESTED.md; \
	  echo 'wrote lab/HALT_REQUESTED.md; the loop will exit after the current iteration'; \
	fi
