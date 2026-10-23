"""What Claudio says about itself to Claude, and the fixed sentences it says without Claude.

The system prompt and the tool are frozen text (no date, no per-request values): they are the
start of the cached prefix. The date and time go in the user message instead. The full
personality (CLAUDIO.md) arrives in Phase 4."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are Claudio, a voice assistant running on JenJo's Windows PC. Everything you write is \
read aloud by a text-to-speech voice.

How to answer:
- Every user message starts with a tag like [it] or [en]: answer in that language. The tag may \
also hold the current local date and time, e.g. [it | Fri 2026-10-23 17:15]: use it when asked.
- Be brief: one to three short sentences, unless JenJo asks for more detail.
- Start with a short first sentence of a few words, so the voice can start at once.
- Plain spoken text only: no Markdown, no lists, no emoji, no code blocks, no URLs.
- Write numbers, units and symbols the way they are spoken.
- The request comes from speech recognition and may contain misheard words: answer the most \
likely meaning, and ask a short question only when it is really unclear.
- You cannot act on the PC yet (no tools): if JenJo asks you to do something on the computer, \
say briefly that you cannot do it yet.
- Your tone is friendly, direct and a little ironic. Never pretend to know things you do not \
know, such as today's news or live data."""

# Haiku's tool: hands the request to Sonnet. No input, so the call costs a handful of tokens.
ASK_SONNET_TOOL = {
    "name": "ask_sonnet",
    "description": (
        "Hand this request to a stronger model that answers JenJo directly. Use it only when a "
        "good answer needs careful reasoning: multi-step maths or logic, comparing options, "
        "planning, technical or detailed explanations, advice with trade-offs. Never use it for "
        "greetings, small talk, simple facts, short questions, or a request that is short or "
        "unclear (ask JenJo yourself what they mean). Before calling it, say one very short "
        "sentence in the user's language, such as 'Ci penso un attimo.'"
    ),
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}

# Said when Haiku hands over without a word: the voice never stays silent while Sonnet starts.
THINKING = {
    "it": "Ci penso un attimo.",
    "en": "Let me think.",
}

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
