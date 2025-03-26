"""This module contains pytest configuration and hooks.

It sets up a global logger and defines hooks for pytest to log events
such as the start and finish of a test session.

Functions:
    pytest_configure(config): Applies global test configuration.
    pytest_sessionstart(session): Logs the start of a test session.
    pytest_sessionfinish(session, exitstatus): Logs the end of a test session.
"""

import logging

import pytest

# Configure a global logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Adjust level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)


def pytest_configure(config):  # pylint: disable=unused-argument
    """Hook function for pytest that is called after command line options have been parsed
    and all plugins and initial configuration are set up.

    This function is typically used to perform global test configuration or setup
    tasks before any tests are executed.

    Args:
        config (pytest.Config): The pytest configuration object containing command-line
            options and plugin configurations.
    """
    logger.info("=== Global Test Configuration Applied ===")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """Pytest hook implementation that is executed at the start of a test session.

    This function logs a message indicating that the test session has started.

    Args:
        session: The pytest session object (not used in this implementation).
    """
    logger.info("=== Test Session Started ===")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """Hook function called after the test session ends.

    This function is executed after all tests have been run and the test session
    is about to finish. It can be used to perform cleanup or logging tasks.

    Args:
        session (Session): The pytest session object containing information
            about the test session.
        exitstatus (int): The exit status code of the test session. This indicates
            whether the tests passed, failed, or were interrupted.

    Note:
        The `pylint: disable=unused-argument` directive is used to suppress
        warnings for unused arguments in this function.
    """
    logger.info("=== Test Session Finished ===")
