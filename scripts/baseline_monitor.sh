#!/usr/bin/env bash
# Print one-line status for each long PPO baseline:
#   <env>  <pid?>  <tb-events-size>  <tb-events-age>  <result.json status>
# stdout from the worker is buffered, so log-mtime is unreliable; the tfevents
# file mtime is the authoritative progress signal.

set -u
cd /home/michael/projects/rl-research

now=$(date +%s)
declare -A ENVS=(
  [pendulum]=Pendulum-v1
  [atari]=MontezumaRevenge-v5
  [breakout]=Breakout-v5
  [humanoid]=humanoid.run
)
declare -A MATCH=(
  [pendulum]="rl_research.baselines.ppo.*--env Pendulum-v1"
  [atari]="rl_research.baselines.ppo.*--env ALE/MontezumaRevenge-v5"
  [breakout]="rl_research.baselines.ppo.*--env ALE/Breakout-v5"
  [humanoid]="rl_research.baselines.ppo.*--env humanoid.run"
)

for tag in pendulum atari breakout humanoid; do
  d=${ENVS[$tag]}
  pid=$(pgrep -f "${MATCH[$tag]}" | tail -1)
  ev=$(ls -t lab/baselines/ppo/$d/seed-0/events.out.tfevents.* 2>/dev/null | head -1)
  if [[ -n $ev && -f $ev ]]; then
    sz=$(stat -c %s "$ev")
    age=$(($now - $(stat -c %Y "$ev")))
  else
    sz="?"
    age="?"
  fi
  printf "%-9s pid=%-7s tb_size=%9s  tb_age=%5ss\n" "$tag" "${pid:-DEAD}" "$sz" "$age"
done

echo "--- result.json status ---"
for d in Pendulum-v1 MontezumaRevenge-v5 Breakout-v5 humanoid.run; do
  rj=lab/baselines/ppo/$d/seed-0/result.json
  if [[ -f $rj ]]; then
    age=$(($now - $(stat -c %Y "$rj")))
    printf "%-22s result.json age=%ss\n" "$d" "$age"
  else
    printf "%-22s result.json missing (run still in flight)\n" "$d"
  fi
done

echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
