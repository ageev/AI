# food-widget - server-rendered Android widget for the food log

Android home-screen widget showing today's calories vs budget, macros and the
meal list. The phone app is a dumb image viewer; everything is rendered
server-side by the agent host as PNG cards.

## Why this shape

- **Server-side rendering** (PIL, Inter fonts, dark + light, several size
  buckets): the layout lives in Python next to the data; the app never parses
  data and never changes.
- **Static publishing:** PNGs + `meta.json` go to an SMB share served by the
  web server under a capability URL (`https://<host>/hw-<long-random-token>/`).
  The random path token is the only access control - the URL itself is treated
  as a secret.
- **FCM data-message** (topic `hw-<slot>`) pokes the phone after each publish;
  the widget re-fetches `meta.json` and the right PNG. Pull-to-refresh works
  too.
- **No secrets on disk:** SMB credentials, the capability URL and the Firebase
  service account are vault items (`secret` CLI); the FCM key is loaded
  in-memory at send time only.

## Flow

```
foodlog.py add                      (meal logged)
  -> hooks/food_widget_push.py      aggregate today's journal -> JSON payload
     -> widget_push.py              render every bucket x theme -> PNG + meta.json + manifest.json
        -> upload_smb.py            atomic publish to the share
        -> fcm_send.py              data-message to topic hw-<slot>
```

Each person has their own slot (`calories_<key>`) with their own budget and
widget; `foodlog.py` fires the hook with the right key automatically after
every `add`/`correct`. Every push appends one status line to `push.log`
(status only, never payload data) - the first place to look when "the widget
did not update".

## Files

- [`hooks/food_widget_push.py`](hooks/food_widget_push.py) - food-log → payload glue
- [`widget_push.py`](widget_push.py) - render + publish CLI (`widget_push.py <slot> [--local-only]`)
- [`render/`](render/) - `card.py` primitives, `theme.py` palettes/fonts, `slots/calories_today.py` the calories card
- [`upload_smb.py`](upload_smb.py) - atomic SMB publisher (tmp + rename, autoindex-safe markers)
- [`fcm_send.py`](fcm_send.py) - FCM HTTP v1 sender (vault-held service account)
- [`config.example.toml`](config.example.toml) - share/publish/render/slot config, sanitized

## Omitted

The Android app source (a thin image-widget client with a slot picker,
polling + FCM), the other slots (weight, workout, kids allowance) and
`push.log`. Person keys sanitized to `alice` / `bob`, hosts and URLs to
placeholders.
