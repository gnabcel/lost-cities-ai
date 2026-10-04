#!/usr/bin/env bash
# Keeps training running unattended: when the expert-iteration loop ends or crashes,
# it restarts it from the last accepted iteration (runs/ei/history.jsonl). It also
# revives the tracker, the dashboard and the play server.
# The loop's arguments are read from ei_args.txt on every restart.
# Start:  nohup ./supervisor.sh >> runs_supervisor.log 2>&1 < /dev/null &
# Stop:   touch STOP   (kills the running iteration and exits; delete STOP before restarting)
cd "$(dirname "$0")"
PY=.venv/bin/python
MAX_FAILS=3
fails=0

log() { echo "$(date '+%F %T') $*"; }

revive() {  # revive <pgrep pattern> <command...>
    local pat=$1; shift
    if ! pgrep -f "$pat" >/dev/null; then
        log "reviving: $*"
        nohup "$@" < /dev/null &
    fi
}

revive_all() {
    revive "lost_cities[.]tracker" sh -c "exec $PY -u -m lost_cities.tracker --interval 60 >> runs_tracker.log 2>&1"
    revive "http[.]server 8765" sh -c "exec python3 -m http.server 8765 --bind 127.0.0.1 > /dev/null 2>&1"
    revive "lost_cities[.]play_server" sh -c "exec $PY -u -m lost_cities.play_server >> runs_play.log 2>&1"
}

log "supervisor started (pid $$)"
while [ ! -e STOP ]; do
    revive_all
    if pgrep -f "lost_cities[.]expert_iter" >/dev/null; then  # started by someone else
        sleep 60
        continue
    fi

    read -r base next < <(python3 -c '
import json
last = [json.loads(l) for l in open("runs/ei/history.jsonl") if l.strip()][-1]
base = "runs/ei/u%05d.pt" % last["iter"] if last["accepted"] else last["base"]
print(base, last["iter"] + 1)')
    read -r -a extra < <(grep -v '^#' ei_args.txt | tr '\n' ' ')
    before=$(wc -l < runs/ei/history.jsonl)
    log "starting loop from $base, iteration $next: ${extra[*]}"
    echo -e "\n=== supervisor $(date '+%F %T'): restarting from $base, iteration $next · ${extra[*]} ===" >> runs_ei.log
    $PY -u -m lost_cities.expert_iter --init "$base" --run ei --start-iter "$next" --iters 1000 "${extra[@]}" \
        >> runs_ei.log 2>&1 < /dev/null &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        if [ -e STOP ]; then
            log "found STOP: stopping the loop"
            pkill -P "$pid"  # the loop's workers
            kill "$pid"
            break
        fi
        revive_all
        sleep 60
    done
    wait "$pid"
    code=$?
    [ -e STOP ] && break
    after=$(wc -l < runs/ei/history.jsonl)
    if [ "$after" -gt "$before" ]; then
        fails=0
    else
        fails=$((fails + 1))
        log "the loop exited with code $code without completing an iteration ($fails/$MAX_FAILS)"
        if [ "$fails" -ge "$MAX_FAILS" ]; then
            log "giving up: $MAX_FAILS failures in a row"
            exit 1
        fi
    fi
    sleep 60
done
log "found STOP, exiting"
