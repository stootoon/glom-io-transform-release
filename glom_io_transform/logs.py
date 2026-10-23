"""Package logging: one place to turn the narration up or down.

The data-loading and fitting code says what it is doing as it goes, which is
useful on a single call and unreadable in a loop over seeds. Those messages are
log.info, so a notebook can quieten them before making its calls:

    from glom_io_transform import logs
    logs.set_level("WARNING")    # quiet
    logs.set_level("INFO")       # the default: narrate, as the prints used to

A handler is attached here rather than left to the caller so that the default
behaviour matches what the code did when these were prints. Modules get their
logger with logging.getLogger(__name__); those are children of this one, so
they carry no level of their own and follow whatever is set here.
"""
import logging
import sys

LOGGER_NAME = "glom_io_transform"

log = logging.getLogger(LOGGER_NAME)


def _setup():
    # Idempotent: a module reload must not stack up duplicate handlers.
    if log.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    # No prefix, so the output reads exactly as the prints did.
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    # The root logger is the caller's to configure; don't emit through it too.
    log.propagate = False


_setup()


def set_level(level):
    """Set the level for every glom_io_transform logger. Accepts "INFO" or logging.INFO."""
    log.setLevel(level.upper() if isinstance(level, str) else level)
    return logging.getLevelName(log.level)


def quiet():
    """Silence the narration; WARNING and above still show."""
    return set_level(logging.WARNING)


def verbose():
    """Narrate again, as the code did when these were prints."""
    return set_level(logging.INFO)
