import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False,
        help="Run integration tests that hit the real Benzinga API",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests that call the real Benzinga API")
