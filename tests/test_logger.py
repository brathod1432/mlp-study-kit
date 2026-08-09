"""Tests for nn_core.logger -- ObjLogger and title_message."""
import io
import sys
import pytest
from nn_core.logger import ObjLogger, title_message


class TestObjLogger:
    def test_logger_creates_with_default_name(self):
        logger = ObjLogger()
        assert logger.name == "Logger"

    def test_logger_creates_with_custom_name(self):
        logger = ObjLogger("MyTest")
        assert logger.name == "MyTest"

    def test_call_produces_output(self, capsys):
        logger = ObjLogger("Test")
        logger("hello world", color="white")
        out = capsys.readouterr().out
        assert "hello world" in out
        assert "[Test]" in out

    def test_info_produces_output(self, capsys):
        logger = ObjLogger("Test")
        logger.info("info message")
        assert "info message" in capsys.readouterr().out

    def test_debug_produces_output(self, capsys):
        logger = ObjLogger("Test")
        logger.debug("debug message")
        assert "debug message" in capsys.readouterr().out

    def test_warning_produces_output(self, capsys):
        logger = ObjLogger("Test")
        logger.warning("warn message")
        assert "warn message" in capsys.readouterr().out

    def test_error_produces_output(self, capsys):
        logger = ObjLogger("Test")
        logger.error("error message")
        assert "error message" in capsys.readouterr().out

    def test_success_produces_output(self, capsys):
        logger = ObjLogger("Test")
        logger.success("done")
        assert "done" in capsys.readouterr().out

    def test_invalid_color_falls_back_gracefully(self, capsys):
        logger = ObjLogger("Test")
        logger("message", color="not_a_color")
        assert "message" in capsys.readouterr().out


class TestTitleMessage:
    def test_title_prints_border_and_message(self, capsys):
        title_message("Hello", color="blue")
        out = capsys.readouterr().out
        assert "Hello" in out
        assert "#" in out
