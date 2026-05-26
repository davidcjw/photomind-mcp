import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from photomind.device_indexer import _check_tool, _list_devices, _get_device_name
from photomind.device_indexer import DeviceIndexer
from photomind.database import DatabaseManager


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


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    db.connect()
    yield db
    db.close()


class TestDeviceIndexer:
    def test_raises_when_idevice_id_missing(self, db, tmp_path):
        indexer = DeviceIndexer(db)
        with patch("photomind.device_indexer._check_tool", return_value=False):
            with pytest.raises(RuntimeError, match="idevice_id.*not found"):
                indexer.sync(str(tmp_path / "dest"))

    def test_raises_when_no_device_connected(self, db, tmp_path):
        indexer = DeviceIndexer(db)
        with patch("photomind.device_indexer._check_tool", return_value=True):
            with patch("photomind.device_indexer._list_devices", return_value=[]):
                with pytest.raises(RuntimeError, match="No iPhone detected"):
                    indexer.sync(str(tmp_path / "dest"))

    def test_raises_when_specific_device_not_found(self, db, tmp_path):
        indexer = DeviceIndexer(db)
        with patch("photomind.device_indexer._check_tool", return_value=True):
            with patch("photomind.device_indexer._list_devices", return_value=["abc123"]):
                with pytest.raises(RuntimeError, match="xyz999.*not found"):
                    indexer.sync(str(tmp_path / "dest"), device_id="xyz999")

    def test_raises_when_mount_fails(self, db, tmp_path):
        indexer = DeviceIndexer(db)
        (tmp_path / "mnt").mkdir()
        with patch("photomind.device_indexer._check_tool", return_value=True), \
             patch("photomind.device_indexer._list_devices", return_value=["abc123"]), \
             patch("photomind.device_indexer._get_device_name", return_value="iPhone"), \
             patch("subprocess.run") as mock_run, \
             patch("tempfile.mkdtemp", return_value=str(tmp_path / "mnt")):
            mock_run.return_value = MagicMock(returncode=1, stderr="ifuse: failed")
            with pytest.raises(RuntimeError, match="ifuse mount failed"):
                indexer.sync(str(tmp_path / "dest"))

    def test_copies_supported_files_and_indexes(self, db, tmp_path):
        mnt = tmp_path / "mnt"
        dcim = mnt / "DCIM" / "100APPLE"
        dcim.mkdir(parents=True)
        (dcim / "IMG_0001.JPG").write_bytes(b"fake-jpeg-1")
        (dcim / "IMG_0002.HEIC").write_bytes(b"fake-heic")
        (dcim / "VID_0001.MOV").write_bytes(b"fake-video")  # should be skipped

        dest = tmp_path / "dest"
        indexer = DeviceIndexer(db)

        with patch("photomind.device_indexer._check_tool", return_value=True), \
             patch("photomind.device_indexer._list_devices", return_value=["abc123"]), \
             patch("photomind.device_indexer._get_device_name", return_value="David's iPhone"), \
             patch("subprocess.run") as mock_run, \
             patch("tempfile.mkdtemp", return_value=str(mnt)):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = indexer.sync(str(dest))

        assert result["device"] == "David's iPhone"
        assert result["udid"] == "abc123"
        assert result["copied"] == 2          # jpg + heic, not mov
        assert result["skipped_existing"] == 0
        assert result["copy_errors"] == 0
        assert (dest / "100APPLE" / "IMG_0001.JPG").exists()
        assert (dest / "100APPLE" / "IMG_0002.HEIC").exists()
        assert not (dest / "100APPLE" / "VID_0001.MOV").exists()

    def test_skips_already_copied_files(self, db, tmp_path):
        mnt = tmp_path / "mnt"
        dcim = mnt / "DCIM" / "100APPLE"
        dcim.mkdir(parents=True)
        (dcim / "IMG_0001.JPG").write_bytes(b"fake-jpeg")

        dest = tmp_path / "dest"
        dest_sub = dest / "100APPLE"
        dest_sub.mkdir(parents=True)
        (dest_sub / "IMG_0001.JPG").write_bytes(b"already-there")

        indexer = DeviceIndexer(db)
        with patch("photomind.device_indexer._check_tool", return_value=True), \
             patch("photomind.device_indexer._list_devices", return_value=["abc123"]), \
             patch("photomind.device_indexer._get_device_name", return_value="iPhone"), \
             patch("subprocess.run") as mock_run, \
             patch("tempfile.mkdtemp", return_value=str(mnt)):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = indexer.sync(str(dest))

        assert result["copied"] == 0
        assert result["skipped_existing"] == 1
