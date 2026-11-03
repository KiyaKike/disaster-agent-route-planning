import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from extract_native_replication36 import SelectedRangeCache
from external_footprint_core import NATIVE128,NATIVE256
from native_replication36_core import decide_replication
from test_external_rgb_extraction import synthetic_member


class Response:
    def __init__(self,member,offset,count,body,status=206,content_range=None):
        self.status=status;self.headers={"Content-Range":content_range or f"bytes {offset}-{offset+count-1}/{member['archive_size']}","Content-Length":str(count)}
        self.body=body;self.reads=0
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,count):
        self.reads+=1;part=self.body[:count];self.body=self.body[count:];return part


class ReplicationRangeTests(unittest.TestCase):
    def setUp(self):
        _,_,self.header,self.member=synthetic_member();self.member["mode"]="pre"
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        self.cache=SelectedRangeCache(self.root,[self.member],cap=1024)
    def tearDown(self):self.directory.cleanup()

    def test_exact_whitelisted_header_has_bounded_receipt(self):
        response=Response(self.member,0,30,self.header)
        with patch('extract_native_replication36.urlopen',return_value=response):data,row=self.cache.read(self.member,0,30,'header30')
        self.assertEqual(data,self.header);self.assertEqual(row['http_status'],206);self.assertEqual(self.cache.reserved,30)

    def test_unselected_member_is_rejected_before_network(self):
        member=copy.deepcopy(self.member);member['path']='E/images/another.png'
        with patch('extract_native_replication36.urlopen') as request:
            with self.assertRaises(ValueError):self.cache.read(member,0,30,'header30')
            request.assert_not_called()

    def test_changed_whitelist_metadata_is_rejected(self):
        member=copy.deepcopy(self.member);member['archive_size']+=1
        with self.assertRaises(ValueError):self.cache.read(member,0,30,'header30')

    def test_interval_cannot_cross_next_member(self):
        with self.assertRaises(ValueError):self.cache.read(self.member,self.member['next_header_or_directory']-1,2,'name_extra')

    def test_fixed_header_cannot_expand_or_shift(self):
        for offset,count in [(0,31),(1,30)]:
            with self.assertRaises(ValueError):self.cache.read(self.member,offset,count,'header30')

    def test_rgb_cannot_be_requested_with_mask_payload_role(self):
        with self.assertRaises(ValueError):self.cache.read(self.member,50,self.member['compressed_bytes'],'mask_compressed')

    def test_cap_is_persistent_and_precedes_body_read(self):
        cache=SelectedRangeCache(self.root,[self.member],cap=29)
        with patch('extract_native_replication36.urlopen') as request:
            with self.assertRaises(ValueError):cache.read(self.member,0,30,'header30')
            request.assert_not_called()

    def test_full_http200_body_is_never_read(self):
        response=Response(self.member,0,30,self.header,status=200)
        with patch('extract_native_replication36.urlopen',return_value=response):
            with self.assertRaises(ValueError):self.cache.read(self.member,0,30,'header30')
        self.assertEqual(response.reads,0);self.assertEqual(self.cache.reserved,30)

    def test_wrong_content_range_stops_before_read(self):
        response=Response(self.member,0,30,self.header,content_range='bytes 1-30/999')
        with patch('extract_native_replication36.urlopen',return_value=response):
            with self.assertRaises(ValueError):self.cache.read(self.member,0,30,'header30')
        self.assertEqual(response.reads,0)

    def test_truncated_body_remains_reserved_and_halts(self):
        response=Response(self.member,0,30,self.header[:5])
        with patch('extract_native_replication36.urlopen',return_value=response):
            with self.assertRaises(ValueError):self.cache.read(self.member,0,30,'header30')
        self.assertEqual(self.cache.reserved,30);self.assertTrue(self.cache.halted.is_set());self.assertEqual(self.cache.ledger[0]['received_bytes'],5)

    def test_cached_exact_bytes_replay_without_network_or_new_reservation(self):
        response=Response(self.member,0,30,self.header)
        with patch('extract_native_replication36.urlopen',return_value=response):self.cache.read(self.member,0,30,'header30')
        cache=SelectedRangeCache(self.root,[self.member],cap=30)
        with patch('extract_native_replication36.urlopen') as request:
            data,_=cache.read(self.member,0,30,'header30');request.assert_not_called()
        self.assertEqual(data,self.header);self.assertEqual(cache.reserved,30)

    def test_failure_blocks_other_new_requests(self):
        self.cache.halted.set()
        with patch('extract_native_replication36.urlopen') as request:
            with self.assertRaises(ValueError):self.cache.read(self.member,0,30,'header30')
            request.assert_not_called()


def fixture():
    events=[];totals=[];opportunities=[]
    for domain in ['all_published_pixels','both_dates_nonzero']:
        for mode in [NATIVE128,NATIVE256]:
            for event in ['A','B','zero']:
                events.append({'domain':domain,'budget':8,'policy':'random_exact','input_mode':mode,'event':event,
                    'reference_pixels':100 if event!='zero' else 0,'incremental_covered_reference_pixels':10 if mode==NATIVE128 else 20})
            totals.append({'domain':domain,'budget':8,'policy':'random_exact','input_mode':mode,
                'event_macro_incremental_covered_pixels_per_scene':10 if mode==NATIVE128 else 20,'reference_precision':.5 if mode==NATIVE128 else .6})
        for event in ['A','B']:
            for i in range(3):opportunities.append({'domain':domain,'event':event,'sample_id':f'{event}{i}','input_mode':NATIVE256,'incremental_covered_reference_pixels':1})
    return {'event_rows':events,'totals':totals},opportunities


class ReplicationScreenTests(unittest.TestCase):
    def test_all_predeclared_new36_screens_can_pass(self):
        summary,opportunity=fixture();self.assertTrue(decide_replication(summary,opportunity)['primary_screen_passed'])

    def test_fewer_than_six_detectable_new_scenes_is_insufficient(self):
        summary,opportunity=fixture();opportunity=[r for r in opportunity if r['sample_id']!='A0']
        result=decide_replication(summary,opportunity);self.assertFalse(result['primary_screen_passed']);self.assertFalse(result['evidence_sufficient'])

    def test_pooled_gain_cannot_hide_a_reference_positive_event_regression(self):
        summary,opportunity=fixture()
        for row in summary['event_rows']:
            if row['input_mode']==NATIVE256 and row['event']=='B':row['incremental_covered_reference_pixels']=9
        self.assertFalse(decide_replication(summary,opportunity)['primary_screen_passed'])

    def test_reference_precision_drop_stops_even_with_coverage_gain(self):
        summary,opportunity=fixture()
        for row in summary['totals']:
            if row['input_mode']==NATIVE256:row['reference_precision']=.49
        self.assertFalse(decide_replication(summary,opportunity)['primary_screen_passed'])


if __name__=='__main__':unittest.main()
