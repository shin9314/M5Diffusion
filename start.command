#!/bin/zsh
cd "${0:A:h}"
mkdir -p work
if ! /usr/bin/curl --max-time 2 -fsS http://127.0.0.1:7861/api/health 2>/dev/null | /usr/bin/grep -Fq '"app": "M5Diffusion"'; then
  nohup "$PWD/.venv/bin/python" "$PWD/serve.py" > "$PWD/work/server.log" 2>&1 </dev/null &
  echo $! > work/server.pid
fi
for attempt in {1..40}; do
  if /usr/bin/curl --max-time 2 -fsS http://127.0.0.1:7861/api/health 2>/dev/null | /usr/bin/grep -Fq '"app": "M5Diffusion"'; then
    /usr/bin/open http://127.0.0.1:7861
    exit 0
  fi
  sleep 0.25
done
/usr/bin/osascript -e 'display alert "M5Diffusion を起動できませんでした" message "M5Diffusion/work/server.log を確認してください。"'
exit 1
