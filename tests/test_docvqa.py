from vlm_uncertainty.datasets.base import VLExample
from vlm_uncertainty.datasets.docvqa import docvqa_record


def test_docvqa_record_uses_only_required_fields():
    record = docvqa_record(
        {
            "questionId": "123",
            "question": "What is the invoice number?",
            "answers": ["ignored"],
            "docId": 99,
        },
        image_path="docvqa_images/validation/123.jpg",
        prompt_template="Answer: {question}",
    )

    assert record == {
        "questionId": "123",
        "image": "docvqa_images/validation/123.jpg",
        "question": "Answer: What is the invoice number?",
    }


def test_vl_example_accepts_docvqa_columns():
    example = VLExample.from_record(
        {
            "questionId": "123",
            "question": "What is the invoice number?",
            "image": "file:///tmp/doc.jpg",
        }
    )

    assert example.id == "123"
    assert example.prompt == "What is the invoice number?"
    assert example.images == ["file:///tmp/doc.jpg"]
