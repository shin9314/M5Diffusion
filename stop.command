#!/bin/zsh
cd "${0:A:h}"
if [[ -f work/server.pid ]]; then
  server_pid=$(cat work/server.pid)
  if [[ "$server_pid" == <-> ]] && /bin/ps -p "$server_pid" -o command= | /usr/bin/grep -Fq "$PWD/serve.py"; then
    kill "$server_pid"
    for attempt in {1..30}; do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 0.1
    done
  fi
  rm -f work/server.pid
fi
