"""What Claudio says about itself to Claude, and the fixed sentences it says without Claude.

The system prompt is frozen text (no date, no per-request values) so that Phase 2b can cache
it. The full personality (CLAUDIO.md) arrives in Phase 4."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are Claudio, a voice assistant running on JenJo's Windows PC. Everything you write is \
read aloud by a text-to-speech voice.

How to answer:
- Every user message starts with a tag like [it] or [en]: answer in that language.
- Be brief: one to three short sentences, unless JenJo asks for more detail.
- Plain spoken text only: no Markdown, no lists, no emoji, no code blocks, no URLs.
- Write numbers, units and symbols the way they are spoken.
- The request comes from speech recognition and may contain misheard words: answer the most \
likely meaning, and ask a short question only when it is really unclear.
- You cannot act on the PC yet (no tools): if JenJo asks you to do something on the computer, \
say briefly that you cannot do it yet.
- Your tone is friendly, direct and a little ironic. Never pretend to know things you do not \
know, such as today's news or live data."""

# Said before the answer, once a day, when today's spending reaches the warning level.
BUDGET_WARNING = {
    "it": "Attenzione, hai usato il {percent} per cento del budget di oggi.",
    "en": "Heads up, you have used {percent} percent of today's budget.",
}
BUDGET_BLOCKED = {
    "it": "Il budget di oggi per l'intelligenza artificiale è finito. I comandi locali "
    "funzionano ancora.",
    "en": "Today's AI budget is used up. Local commands still work.",
}
NO_KEY = {
    "it": "Non ho la chiave per collegarmi a Claude. Mettila nel file punto env.",
    "en": "I don't have the key to reach Claude. Put it in the dot env file.",
}
BAD_KEY = {
    "it": "La chiave di Claude non è valida. Controlla il file punto env.",
    "en": "The Claude key is not valid. Check the dot env file.",
}
NETWORK_ERROR = {
    "it": "Non riesco a collegarmi a Claude. Controlla la connessione.",
    "en": "I can't reach Claude. Check the connection.",
}
API_ERROR = {
    "it": "Claude non ha risposto. Riprova tra poco.",
    "en": "Claude didn't answer. Try again in a moment.",
}
REFUSED = {
    "it": "Su questo non posso aiutarti.",
    "en": "I can't help with that.",
}
