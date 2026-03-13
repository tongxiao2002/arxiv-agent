"""Tests that docs and examples stay aligned with the supported MVP."""

from pathlib import Path

from arxiv_agent.config import Config


def test_config_example_loads_cleanly():
    """Test the example config parses and passes static validation."""
    config = Config.from_yaml(Path("config.yaml.example"))
    assert config.validate() is True


def test_repo_config_loads_cleanly():
    """Test the checked-in config also matches the current schema."""
    config = Config.from_yaml(Path("config.yaml"))
    assert config.validate() is True


def test_env_example_only_mentions_supported_provider_keys():
    """Test .env.example does not advertise unsupported provider secrets."""
    env_example = Path(".env.example").read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY" not in env_example
    assert "OPENAI_API_KEY" in env_example
    assert "ANTHROPIC_API_KEY" in env_example
    assert "REQUEST_TIMEOUT" in env_example


def test_readme_calls_out_supported_and_unsupported_features():
    """Test README stays honest about current MVP support."""
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## Supported Today" in readme
    assert "## Not Yet Implemented" in readme
    assert "`sources.primary: papers_cool`" in readme
    assert "`llm.provider: local`" in readme
    assert "DEEPSEEK" not in readme
    assert "sendgrid" not in readme.lower()
    assert "mailgun" not in readme.lower()


def test_pyproject_dependencies_match_runtime_install_path():
    """Test package metadata stays aligned with requirements.txt."""
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"deepagents"' in pyproject
    assert requirements[0] == "deepagents"
    assert "sendgrid" not in pyproject
    assert "mailgun" not in pyproject
