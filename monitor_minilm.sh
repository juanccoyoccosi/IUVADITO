#!/bin/bash
if ! curl -s --max-time 10 -X POST https://127.0.0.1:5000/bot -k -H "Content-Type: application/json" -d '{"mensaje": "ping"}' > /dev/null; then
  echo "MiniLM no responde. Reiniciando..."
  sudo systemctl restart MiniLM.service
fi

#agragado a crontab 