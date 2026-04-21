#!/bin/bash
SESSION="proc_0d7597745ca2"
LAST_OUTPUT=""
INTERVAL=900  # 15 minutes

while true; do
    # Capture current output
    CURRENT=$(hermes process poll "$SESSION" 2>/dev/null || echo "PROCESS_GONE")
    
    # Check if process finished
    if echo "$CURRENT" | grep -q "PROCESS_GONE\|exited\|exit_code"; then
        echo "[$(date -Iseconds)] PROCESS FINISHED"
        echo "$CURRENT"
        break
    fi
    
    # Check if output changed
    SNIPPET=$(echo "$CURRENT" | tail -5 | md5sum)
    if [ "$SNIPPET" != "$LAST_OUTPUT" ]; then
        echo "[$(date -Iseconds)] NEW OUTPUT DETECTED"
        echo "$CURRENT"
        LAST_OUTPUT="$SNIPPET"
    else
        echo "[$(date -Iseconds)] No change, sleeping ${INTERVAL}s..."
    fi
    
    sleep $INTERVAL
done
