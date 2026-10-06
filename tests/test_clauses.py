from src.clauses import split_clauses


def test_numbered_items_are_separate_clauses():
    text = (
        "1. The Tenant shall pay rent on the first day of every month.\n"
        "2. The Landlord may terminate this agreement at any time without notice."
    )
    clauses = split_clauses(text)
    assert len(clauses) == 2
    assert clauses[1].startswith("2. The Landlord may terminate")


def test_sentences_are_split_and_short_fragments_dropped():
    text = "Welcome. We may change these terms at any time without notifying you. Thanks!"
    clauses = split_clauses(text)
    assert clauses == ["We may change these terms at any time without notifying you."]


def test_abbreviations_do_not_break_sentences():
    text = "Disputes are governed by the laws of the U.S. and will be resolved by arbitration."
    assert len(split_clauses(text)) == 1


def test_wrapped_lines_are_joined():
    text = "You agree that we may remove any content\nat our sole discretion without prior notice."
    assert split_clauses(text) == [
        "You agree that we may remove any content at our sole discretion without prior notice."
    ]


def test_empty_text_gives_no_clauses():
    assert split_clauses("   \n  ") == []
