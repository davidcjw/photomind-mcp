from unittest.mock import MagicMock, patch
import pytest
from photomind.device_indexer import _check_tool, _list_devices, _get_device_name


class TestCheckTool:
    def test_existing_tool_returns_true(self):
        with patch("shutil.which", return_value="/usr/bin/ifuse"):
            assert _check_tool("ifuse") is True

    def test_missing_tool_returns_false(self):
        with patch("shutil.which", return_value=None):
            assert _check_tool("ifuse") is False


class TestListDevices:
    def test_returns_udids_when_devices_connected(self):
        mock_result = MagicMock()
        mock_result.stdout = "abc123\ndef456\n"
        with patch("subprocess.run", return_value=mock_result):
            assert _list_devices() == ["abc123", "def456"]

    def test_returns_empty_list_when_no_devices(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _list_devices() == []

    def test_strips_whitespace_from_udids(self):
        mock_result = MagicMock()
        mock_result.stdout = "  abc123  \n  def456  \n"
        with patch("subprocess.run", return_value=mock_result):
            assert _list_devices() == ["abc123", "def456"]


class TestGetDeviceName:
    def test_returns_device_name(self):
        mock_result = MagicMock()
        mock_result.stdout = "David's iPhone\n"
        with patch("subprocess.run", return_value=mock_result):
            assert _get_device_name("abc123") == "David's iPhone"

    def test_returns_unknown_when_empty(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _get_device_name("abc123") == "Unknown Device"
