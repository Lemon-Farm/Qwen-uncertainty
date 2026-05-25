from vlm_uncertainty.datasets.preprocessing import build_messages, is_url


def test_is_url():
    assert is_url("https://example.com/image.jpg")
    assert not is_url("images/example.jpg")


def test_build_messages_with_system_prompt():
    messages = build_messages(
        prompt="describe it",
        images=["file:///tmp/image.jpg"],
        system_prompt="be concise",
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["content"][0]["type"] == "image"
    assert messages[1]["content"][-1]["text"] == "describe it"
