from vlm_uncertainty.datasets.vizwiz import majority_answer, vizwiz_record


def test_majority_answer_prefers_first_answer_on_tie():
    assert majority_answer(["red", "blue", "blue", "red"]) == "red"


def test_vizwiz_record_keeps_answers_in_metadata():
    record = vizwiz_record(
        {
            "question_id": "abc123",
            "question": "What color is it?",
            "answers": ["white", "white", "gray"],
            "category": "other",
        },
        image_path="vizwiz_images/val/abc123.jpg",
        prompt_template="Answer briefly: {question}",
    )

    assert record["id"] == "abc123"
    assert record["image"] == "vizwiz_images/val/abc123.jpg"
    assert record["prompt"] == "Answer briefly: What color is it?"
    assert record["target"] == "white"
    assert record["metadata"]["answers"] == ["white", "white", "gray"]
    assert record["metadata"]["category"] == "other"
