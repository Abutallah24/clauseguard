from src.rules import detect_rules


def categories(text):
    return {hit.category for hit in detect_rules(text)}


def test_auto_renewal_detected():
    text = "Your subscription will automatically renew each month unless cancelled."
    assert "Auto-renewal" in categories(text)


def test_bond_detected():
    text = "The intern shall serve a bond of 12 months and pay Rs. 50,000 if they leave early."
    assert "Bond / lock-in penalty" in categories(text)


def test_data_sharing_detected():
    text = "We may share your personal data with third parties for advertising purposes."
    assert "Data sharing" in categories(text)


def test_deposit_forfeiture_detected():
    text = "The security deposit shall be forfeited if the tenant vacates before 11 months."
    assert "Deposit forfeiture" in categories(text)


def test_fees_detected():
    text = "A late fee of 5% per day will be charged on any unpaid amount."
    assert "Hidden fees / penalties" in categories(text)


def test_benign_sentence_has_no_hits():
    assert detect_rules("The office is open from 9 am to 6 pm on weekdays.") == []
