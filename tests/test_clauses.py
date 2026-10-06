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


def test_heading_lines_are_not_merged_into_the_clause():
    text = (
        "4. Changes to These Terms\n"
        "We may modify these Terms at any time and at our sole discretion.\n"
        "Termination\n"
        "We may suspend your account at any time without notice."
    )
    assert split_clauses(text) == [
        "We may modify these Terms at any time and at our sole discretion.",
        "We may suspend your account at any time without notice.",
    ]


def test_empty_text_gives_no_clauses():
    assert split_clauses("   \n  ") == []


def test_bullet_list_without_full_stops_is_kept():
    text = (
        "Fees\n"
        "- Non-refundable deposit of Rs 5000 is kept\n"
        "- We may share your data with advertisers\n"
        "• Late payment penalty of 2% per month applies"
    )
    assert split_clauses(text) == [
        "- Non-refundable deposit of Rs 5000 is kept",
        "- We may share your data with advertisers",
        "• Late payment penalty of 2% per month applies",
    ]


def test_wrapped_line_before_capitalised_word_is_joined():
    text = (
        "Any dispute arising out of this Agreement shall go to the\n"
        "Courts of Delaware exclusively and you waive any class action."
    )
    assert split_clauses(text) == [
        "Any dispute arising out of this Agreement shall go to the Courts of Delaware exclusively "
        "and you waive any class action."
    ]


def test_wrapped_line_ending_in_preposition_is_joined():
    text = "All claims will be settled by binding arbitration under\nThe rules of the ICC.\nYou waive your right to a jury trial."
    assert split_clauses(text) == [
        "All claims will be settled by binding arbitration under The rules of the ICC.",
        "You waive your right to a jury trial.",
    ]


def test_abbreviation_inside_a_word_still_ends_the_sentence():
    text = "You must pass the eligibility test. We may cancel your booking at any time."
    assert len(split_clauses(text)) == 2


def test_abbreviation_before_capital_does_not_split():
    text = "Payments are processed by Stripe Inc. Ltd. and Rs. 500 is charged as a convenience fee per order."
    assert len(split_clauses(text)) == 1
