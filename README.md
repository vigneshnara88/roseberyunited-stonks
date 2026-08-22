# Time Travelling Stonks Man

Deployable Flask service for UBS GCC 2026.

Endpoint:

```text
POST /stonks
```

Render start command:

```text
gunicorn app:app --workers 1 --threads 8 --timeout 30 --bind 0.0.0.0:$PORT
```
