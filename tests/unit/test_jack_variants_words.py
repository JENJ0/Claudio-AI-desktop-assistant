import sys

from claudio.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import jack_variants


def test_words_from_the_command_line_get_the_table_and_the_proposal() -> None:
    spellings = jack_variants.spellings_for(["File", "computer", "subprime"], {"file": "fàiil"})
    assert spellings["File"][:2] == ["File", "fàiil"]  # as written, then JenJo's table
    assert spellings["computer"] == ["computer", "compiùter"]  # from /komˈpju.ter/
    assert spellings["subprime"] == ["subprime"]  # nothing known: only as written
