from vlm_uncertainty.datasets.arrow import is_arrow_dataset, load_vl_arrow, save_vl_arrow


def test_arrow_dataset_roundtrip(tmp_path):
    dataset_path = tmp_path / "dataset"
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake")

    save_vl_arrow(
        dataset_path,
        [
            {
                "id": "one",
                "image": "image.jpg",
                "prompt": "What is shown?",
                "target": "test",
                "metadata": {"source": "unit"},
            }
        ],
    )

    assert is_arrow_dataset(dataset_path)
    dataset = load_vl_arrow(dataset_path)

    assert len(dataset) == 1
    example = dataset[0]
    assert example.id == "one"
    assert example.prompt == "What is shown?"
    assert example.target == "test"
    assert example.metadata == {"source": "unit"}
    assert example.images == [image_path.resolve().as_uri()]
