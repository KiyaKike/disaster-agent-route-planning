import tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from probe_crasar_region import RemoteTIFF

class RangeCache(unittest.TestCase):
    def reader(self,folder):
        r=RemoteTIFF.__new__(RemoteTIFF);r.size=7;r.blocksize=4;r.bytes=0;r.cap=8;r.blocks={};r.calls=[]
        r.folder=Path(folder);r.path='public-test.tif';r.lock=threading.Lock();return r
    def response(self,data,header='bytes 0-6/7'):
        r=MagicMock();r.__enter__.return_value=r;r.status=206;r.headers={'Content-Range':header};r.read.return_value=data;return r
    def test_group_preserves_exact_blocks_and_eof(self):
        with tempfile.TemporaryDirectory() as folder:
            r=self.reader(folder)
            with patch('probe_crasar_region.open_request',return_value=self.response(b'abcdefg')):r.get_group([0,1])
            self.assertEqual(r.blocks,{0:b'abcd',1:b'efg'});self.assertEqual(r.bytes,7)
            self.assertEqual((r.folder/'000000000001.bin').read_bytes(),b'efg')
    def test_short_or_wrong_range_never_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            for response in [self.response(b'abc'),self.response(b'abcdefg','bytes 1-7/8')]:
                r=self.reader(folder)
                with patch('probe_crasar_region.open_request',return_value=response):
                    with self.assertRaises(RuntimeError):r.get_group([0,1])
                self.assertEqual(r.blocks,{})

if __name__=='__main__':unittest.main()
