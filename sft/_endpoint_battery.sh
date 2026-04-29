#!/bin/bash
# 10-prompt battery through /api/chat/stream to shake out the full web stack.
# Prints pass/fail per prompt based on whether a sensible tool was called
# (or for chitchat/no_tool cases, whether no tool was called).

declare -a CASES=(
  "hi there|no_tool"
  "what time is it|get_time"
  "what's the weather in Tokyo|get_weather"
  "convert 100 usd to eur|convert"
  "who is Ada Lovelace|wiki"
  "any news today|get_news"
  "remind me to feed the cat tomorrow|schedule_reminder"
  "ohm's law for 12V across 500 ohms|circuit"
  "let's convert this argument into a conversation|no_tool"
  "call me when you're free|no_tool"
)

pass=0; fail=0
for case in "${CASES[@]}"; do
  prompt="${case%|*}"
  expected="${case##*|}"
  tool_line=$(curl -s --max-time 60 -N -X POST http://127.0.0.1/api/chat/stream \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "import json,sys; print(json.dumps({'text': sys.argv[1]}))" "$prompt")" \
    | grep '"type": "tool_call"' | head -1)
  if [ "$expected" = "no_tool" ]; then
    if [ -z "$tool_line" ]; then
      echo "PASS  [$expected] $prompt"
      ((pass++))
    else
      name=$(echo "$tool_line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('name','?'))")
      echo "FAIL  [$expected] $prompt  (got: $name)"
      ((fail++))
    fi
  else
    if [ -z "$tool_line" ]; then
      echo "FAIL  [$expected] $prompt  (no tool call)"
      ((fail++))
    else
      name=$(echo "$tool_line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('name','?'))")
      if [ "$name" = "$expected" ]; then
        echo "PASS  [$expected] $prompt"
        ((pass++))
      else
        echo "MISM  [$expected] $prompt  (got: $name)"
        ((fail++))
      fi
    fi
  fi
done
echo "---"
echo "result: $pass/$(($pass + $fail))"
