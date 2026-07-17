import unittest

from freeconvert_utils import build_freeconvert_payload, infer_input_format


class FreeConvertUtilsTests(unittest.TestCase):
    def test_build_freeconvert_payload_uses_expected_formats(self):
        payload = build_freeconvert_payload("ogg", "mp3")

        self.assertEqual(payload["tasks"]["convert"]["input_format"], "ogg")
        self.assertEqual(payload["tasks"]["convert"]["output_format"], "mp3")
        self.assertEqual(payload["tasks"]["export-url"]["operation"], "export/url")

    def test_infer_input_format_uses_file_extension(self):
        self.assertEqual(infer_input_format("demo.MP3"), "mp3")
        self.assertEqual(infer_input_format("demo.wav"), "wav")
        self.assertEqual(infer_input_format("demo"), "")


if __name__ == "__main__":
    unittest.main()
