from vlm_uncertainty.datasets.base import VLExample
from vlm_uncertainty.datasets.textvqa import textvqa_record


def test_textvqa_record_keeps_only_required_fields():
    record = textvqa_record(
        {
            "question_id": 7,
            "question": "What word is on the sign?",
            "answers": ["open", "closed", "open"],
            "ocr_tokens": ["ignored"],
        },
        image_path="textvqa_images/validation/7.jpg",
        prompt_template="Q: {question} A:",
    )

    assert record == {
        "question_id": "7",
        "question": "Q: What word is on the sign? A:",
        "image": "textvqa_images/validation/7.jpg",
    }


def test_vl_example_accepts_textvqa_columns():
    example = VLExample.from_record(
        {
            "question_id": "7",
            "question": "What word is on the sign?",
            "image": "file:///tmp/textvqa.jpg",
        }
    )

    assert example.id == "7"
    assert example.prompt == "What word is on the sign?"
