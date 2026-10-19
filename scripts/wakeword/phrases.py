"""What the wake word models learn: the phrases to fire on, the ones to ignore, and the
prompts for JenJo's own recordings (scripts/record_samples.py)."""

from __future__ import annotations

# (text, weight) per espeak-ng language. Italian pronunciation first; the English one
# ("Hey Clow-dee-oh") because Claudio understands English too. Light mispronunciations are
# positives on purpose: a missed letter must still wake Claudio.
POSITIVES: dict[str, dict[str, list[tuple[str, int]]]] = {
    "hey_claudio": {
        "it": [
            ("hey claudio", 6),
            ("ehi claudio", 5),
            ("ei claudio", 2),
            ("hei claudio", 1),
            ("ehi, claudio!", 2),
            ("hey claudio?", 1),
            ("ehi clàudio", 1),
            ("hey cladio", 1),
            ("hey clodio", 1),
        ],
        "en-us": [("hey claudio", 3), ("hey clowdio", 1)],
    },
    "claudio": {
        "it": [
            ("claudio", 8),
            ("claudio!", 3),
            ("claudio?", 3),
            ("clàudio", 1),
            ("cladio", 1),
            ("clodio", 1),
        ],
        "en-us": [("claudio", 3)],
    },
}

# Close to the wake words but not them: the models must stay silent. No "Claudia" here (nor
# in the sentences below): JenJo is fine with it waking CLAUDIO, and "Claudia" differs from
# "Claudio" only by the last vowel, so teaching the models to refuse it taught the
# single-word model to refuse his own voice saying "Claudio" too.
ADVERSARIAL: dict[str, list[str]] = {
    "it": [
        "clara",
        "ehi clara",
        "audio",
        "l'audio",
        "alza l'audio",
        "radio",
        "la radio",
        "chiudi",
        "chiudilo",
        "chiudi tutto",
        "caldo",
        "che caldo",
        "laudi",
        "lodi",
        "lode",
        "plaudo",
        "applaudo",
        "gaudio",
        "studio",
        "lo studio",
        "dio",
        "addio",
        "clausola",
        "claudicante",
        "claus",
        "ehi paolo",
        "ehi giulio",
        "ehi mario",
        "ehi aldo",
        "ehi carlo",
        "ehi giorgio",
        "ehi antonio",
        "ehi luca",
        "ehi marco",
        "ehi tu",
        "ehi ciao",
        "eh già",
        "ehi guarda",
        "ok google",
        "hey google",
        "ehi siri",
        "alexa",
        "hey jarvis",
        "clown",
        "cloud",
        "ehi cloud",
        "ciao",
        "ehi",
        "hey",
        "eh",
        "aldo",
        "laura",
        "ehi laura",
        "cla",
        "clau",
        "dio mio",
        "ehi dico",
        "hey studio",
        "chi è",
        "qualcuno",
    ],
    "en-us": [
        "hey cloud",
        "hey claude",
        "claude",
        "cloudy",
        "hey clyde",
        "hey google",
        "hey siri",
        "audio",
        "studio",
        "radio",
        "hey buddy",
        "hey dude",
        "hey lady",
        "hey jarvis",
        "hello",
        "hey there",
        "okay",
        "rodeo",
        "patio",
        "cardio",
        "hey cardio",
    ],
}

# Ordinary speech, including Claudio's own commands said without the wake word.
SENTENCES: dict[str, list[str]] = {
    "it": [
        "apri chrome",
        "apri la cartella download",
        "che ore sono",
        "che giorno è oggi",
        "alza il volume",
        "abbassa il volume",
        "metti in pausa la musica",
        "oggi pomeriggio vado a fare la spesa",
        "hai visto le chiavi della macchina",
        "domani mattina devo alzarmi presto",
        "stasera guardiamo un film",
        "la riunione è spostata alle tre",
        "mi passi il caricabatterie",
        "fa freddo qui dentro",
        "il mio amico paolo arriva domani",
        "chiudi la finestra per favore",
        "ascolto la radio in macchina",
        "lo studio è al secondo piano",
        "che caldo che fa oggi",
        "ci vediamo dopo cena",
        "non trovo più il telecomando",
        "accendi la luce in cucina",
        "quanto costa questo",
        "devo finire il progetto entro venerdì",
        "andiamo a fare una passeggiata",
        "il treno è in ritardo di dieci minuti",
        "ho fame, cosa mangiamo",
        "la partita inizia alle nove",
        "ricordami di chiamare la mamma",
        "questo computer è un po' lento",
        "metti la canzone di prima",
        "buongiorno a tutti",
        "grazie mille, sei gentilissimo",
        "a che ora chiude il negozio",
        "scrivi un messaggio a giulia",
        "il gatto dorme sul divano",
        "stamattina ho bevuto due caffè",
        "piove da tre giorni",
        "dove hai messo gli occhiali",
    ],
    "en-us": [
        "open chrome",
        "what time is it",
        "turn up the volume",
        "what day is it today",
        "how are you doing",
        "see you tomorrow",
        "I am going to the store",
        "can you call me later",
        "the meeting starts at noon",
        "where are my keys",
        "let's watch a movie tonight",
        "thank you very much",
        "good morning everyone",
        "the weather is nice today",
        "play some music",
        "I forgot my password",
    ],
}

# Recording sessions (record_samples.py): the takes go through the styles in order, a few
# takes each. A style ending in "_cmd" has a command glued after the wake word: used only
# to measure, never to train.
RECORDING_STYLES: dict[str, list[tuple[str, str, str]]] = {  # (slug, how, what to say)
    "hey_claudio": [
        ("normale", "normale", "Hey Claudio"),
        ("veloce", "veloce, tutto attaccato", "Hey Claudio"),
        ("lento", "lento e chiaro", "Hey Claudio"),
        ("piano", "a bassa voce", "Hey Claudio"),
        ("forte", "forte, come da un'altra stanza", "Hey Claudio"),
        ("lontano", "a 2 metri dal microfono", "Hey Claudio"),
        ("ehi", 'con "Ehi" al posto di "Hey"', "Ehi Claudio"),
        ("storpiato", "mangiandoti una lettera", "Hey Cladio (o Hey Clodio)"),
        ("distratto", "distratto, come se pensassi ad altro", "Hey Claudio"),
        ("comando_cmd", "seguito subito da un comando", "Hey Claudio che ore sono"),
    ],
    "claudio": [
        ("normale", "normale", "Claudio"),
        ("veloce", "veloce", "Claudio"),
        ("lento", "lento e chiaro", "Claudio"),
        ("piano", "a bassa voce", "Claudio"),
        ("forte", "chiamandolo forte, come da un'altra stanza", "Claudio!"),
        ("lontano", "a 2 metri dal microfono", "Claudio"),
        ("domanda", "come una domanda", "Claudio?"),
        ("storpiato", "mangiandoti una lettera", "Cladio (o Clodio)"),
        ("distratto", "distratto, come se pensassi ad altro", "Claudio"),
        ("comando_cmd", "seguito subito da un comando", "Claudio apri Chrome"),
    ],
}

# JenJo's own voice saying everything but the wake word (record_samples.py negatives).
NEGATIVE_PROMPTS: list[str] = [
    *SENTENCES["it"][7:36],
    "Alza l'audio della televisione",
    "Hey, ciao, come stai?",
    "Ehi, guarda qui",
    "Chiudi tutto e andiamo",
    "Apri Chrome",
    "Che ore sono",
    "Hey, how are you doing?",
    "Turn up the volume, please",
    "What time is it?",
]


def weighted(phrases: list[tuple[str, int]]) -> list[str]:
    """Each phrase repeated by its weight, so a uniform pick follows the weights."""
    return [text for text, weight in phrases for _ in range(weight)]


def recording_plan(model: str, count: int) -> list[tuple[str, str, str]]:
    """(slug, how, what to say) for each of `count` takes: the styles in order, the same number
    of takes each (the first styles get one more when count does not divide), so JenJo
    changes the way of speaking, or walks 2 m away, only once per style."""
    styles = RECORDING_STYLES[model]
    per, extra = divmod(count, len(styles))
    return [style for i, style in enumerate(styles) for _ in range(per + (i < extra))]


def is_eval_only(path_stem: str) -> bool:
    """Takes with a command glued after the wake word measure the model, never train it."""
    return path_stem.endswith("_cmd")
