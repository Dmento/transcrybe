import unittest

from audio_utils import get_supported_audio_extensions, is_supported_audio_file


class AudioUploadTests(unittest.TestCase):
    def test_supported_audio_extensions_include_common_formats(self):
        exts = get_supported_audio_extensions()
        self.assertIn("wav", exts)
        self.assertIn("mp3", exts)
        self.assertIn("ogg", exts)
        self.assertIn("m4a", exts)
        self.assertIn("flac", exts)

    def test_supported_audio_file_detection(self):
        self.assertTrue(is_supported_audio_file("demo.wav"))
        self.assertTrue(is_supported_audio_file("demo.MP3"))
        self.assertFalse(is_supported_audio_file("demo.txt"))


if __name__ == "__main__":
    unittest.main()
