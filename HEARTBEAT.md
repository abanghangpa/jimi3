# HEARTBEAT.md

# STAMPEDE Alert Forwarding
# Check for pending STAMPEDE alerts and forward to WhatsApp
# The signal bot writes alerts to pending_whatsapp_alerts.txt
# This heartbeat picks them up and sends via WhatsApp

## Check STAMPEDE Alerts
- Read `/root/.openclaw/workspace/stampede/pending_whatsapp_alerts.txt`
- If it has content, send each alert to WhatsApp +601112827947 using the message tool
- After sending, clear the file (write empty string)
- If file is empty or doesn't exist, skip silently
