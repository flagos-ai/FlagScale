import pytest
import requests


@pytest.mark.usefixtures("test_path", "test_type", "test_task", "test_case")
def test_equal(test_path, test_type, test_task, test_case, monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    url = "http://127.0.0.1:6701/generate"
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    test_data = {"prompt": "Introduce BAAI."}

    response = requests.post(url, headers=headers, json=test_data)

    # Assert that the response status code is 200
    assert response.status_code == 200

    # Assert that the generated text is not empty
    assert len(response.text) > 0, "Generated text should not be empty"
