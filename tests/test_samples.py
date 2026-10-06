"""Every demo file in samples/ must extract and produce findings (guards the live demo)."""

from pathlib import Path

import pytest

from src import extract, models
from src.pipeline import analyze

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _read(name: str) -> bytes:
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} missing; run python -m samples.make_samples")
    return path.read_bytes()


@pytest.mark.parametrize(
    "name, extractor, phrase",
    [
        ("streamly_terms.pdf", extract.from_pdf, "arbitration"),
        ("rental_agreement_scanned.pdf", extract.from_pdf, "security deposit"),
        ("internship_offer.docx", extract.from_docx, "bond"),
    ],
)
def test_sample_files_extract(name, extractor, phrase):
    assert phrase in extractor(_read(name)).lower()


def test_page_images_in_order():
    text = extract.from_images([_read("internship_offer_page1.png"), _read("internship_offer_page2.png")]).lower()
    assert text.index("stipend") < text.index("confidentiality")


def test_camera_photo_ocr():
    # Same path the in-app camera uses: a tilted, blurred phone-style photo.
    text = extract.from_images([_read("camera_photo.jpg")]).lower()
    assert "terms of service" in text
    assert "automatically renew" in text


@pytest.mark.skipif(not models.available_models(), reason="no trained model")
def test_scanned_rental_agreement_is_high_risk():
    result = analyze(extract.from_pdf(_read("rental_agreement_scanned.pdf")), "tfidf_logreg")
    assert result.label in ("Medium", "High")
    assert "Deposit forfeiture" in result.counts
