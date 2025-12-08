import warnings
from pathlib import Path

from dotenv import load_dotenv

from tests import patch_agent

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"You should use `Logger` instead\.",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"You should use `LoggerProvider` instead\.",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"You should use `ProxyLoggerProvider` instead\.",
)

patch_agent.install_usage_collector()

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


def pytest_sessionfinish(session, exitstatus):
    patch_agent.print_report_usage()
