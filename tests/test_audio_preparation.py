import unittest

from audio_utils import should_convert_to_wav


class AudioPreparationTests(unittest.TestCase):
    def test_wav_files_do_not_need_conversion(self):
        self.assertFalse(should_convert_to_wav("lecture.wav"))
        self.assertFalse(should_convert_to_wav("lecture.WAV"))

    def test_non_wav_files_need_conversion(self):
        self.assertTrue(should_convert_to_wav("lecture.mp3"))
        self.assertTrue(should_convert_to_wav("lecture.ogg"))


if __name__ == "__main__":
    unittest.main()
