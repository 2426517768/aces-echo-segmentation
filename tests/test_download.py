"""Local HTTP fixtures reproduce broken chunked transfers; no clinical data."""
from __future__ import annotations
from collections import Counter
import errno
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import ssl
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
import download_public as downloader


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = bytes(range(256))*2000
        owner = cls
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass
            def do_GET(self):
                identity = self.path.split("/")[-2]
                owner.calls[identity] += 1
                owner.ranges[identity].append(self.headers.get("Range"))
                if identity == "forbidden" and owner.deny:
                    self.send_error(403)
                    return
                payload = owner.bad_gzip if identity == "gzip" else owner.payload
                offset = int(self.headers["Range"].split("=")[1].split("-")[0]) if self.headers.get("Range") else 0
                first = owner.calls[identity] == 1
                broken = identity == "always" or identity in {"chunk", "ignored"} and first
                if identity == "ignored":
                    offset = 0
                if identity == "wrong_range" and offset:
                    offset = 0
                self.send_response(206 if self.headers.get("Range") and identity != "ignored" else 200)
                if self.headers.get("Range") and identity != "ignored":
                    self.send_header("Content-Range",f"bytes {offset}-{len(payload)-1}/{len(payload)}")
                self.send_header("Content-Type","application/octet-stream")
                if broken:
                    self.send_header("Transfer-Encoding","chunked")
                    self.send_header("Connection","close")
                else:
                    self.send_header("Content-Length",str(len(payload)-offset))
                self.end_headers()
                try:
                    if broken:
                        prefix=payload[offset:offset+10013]
                        self.wfile.write(f"{len(prefix):X}\r\n".encode()+prefix+b"\r\n")
                        self.wfile.flush()
                        self.close_connection=True  # no zero chunk: reproduces IncompleteRead
                    elif identity == "short" and first:
                        self.wfile.write(payload[offset:offset+10013])
                        self.wfile.flush()
                        self.close_connection=True
                    else:
                        self.wfile.write(payload[offset:])
                except (BrokenPipeError, ConnectionResetError):
                    pass
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.base=f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()

    def setUp(self):
        type(self).calls=Counter()
        type(self).ranges={k:[] for k in ["good","chunk","ignored","wrong_range","short","always","forbidden","gzip"]}
        type(self).deny=True
        a=bytearray(gzip.compress(b"software-test-not-clinical"*50));a[-8]^=1
        type(self).bad_gzip=bytes(a)
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def entry(self,identity):
        payload=self.bad_gzip if identity=="gzip" else self.payload
        return {"id":identity,"relative_path":"database_nifti/"+("fixture_ED.nii.gz" if identity=="gzip" else identity+".bin"),
                "size":len(payload),"sha256":hashlib.sha256(payload).hexdigest()}

    def fetch(self,identity,**kwargs):
        return downloader.download_file(self.entry(identity),self.root,api_base=self.base,sleeper=lambda _:None,timeout=2,**kwargs)

    def assert_payload(self,identity):
        self.assertEqual(downloader.target_path(self.root,self.entry(identity)).read_bytes(),self.payload)

    def test_chunked_disconnect_preserves_partial_and_resumes(self):
        self.fetch("chunk")
        self.assert_payload("chunk")
        self.assertEqual(self.ranges["chunk"],[None,"bytes=10013-"])

    def test_server_ignoring_range_does_not_duplicate_prefix(self):
        self.fetch("ignored")
        self.assert_payload("ignored")
        self.assertEqual(self.calls["ignored"],2)

    def test_incorrect_content_range_restarts_single_file(self):
        p=self.root/downloader.CACHE_NAME/"parts/wrong_range.part";p.parent.mkdir(parents=True)
        p.write_bytes(self.payload[:50])
        self.fetch("wrong_range")
        self.assert_payload("wrong_range")
        self.assertEqual(self.ranges["wrong_range"],["bytes=50-",None])

    def test_short_content_length_read_is_retried(self):
        self.fetch("short")
        self.assert_payload("short")
        self.assertEqual(self.ranges["short"],[None,"bytes=10013-"])

    def test_verified_file_is_skipped_without_a_new_request(self):
        self.fetch("good")
        self.fetch("good")
        self.assertEqual(self.calls["good"],1)

    def test_tls_eof_during_connect_retries_without_losing_prefix(self):
        entry=self.entry("good")
        part=self.root/downloader.CACHE_NAME/"parts/good.part"
        part.parent.mkdir(parents=True);part.write_bytes(self.payload[:10013])
        attempts,delays=[],[]
        def opener(req,**kw):
            attempts.append(req.get_header("Range"))
            if len(attempts)<3:
                raise urllib.error.URLError(ssl.SSLEOFError(8,"UNEXPECTED_EOF_WHILE_READING"))
            return urllib.request.urlopen(req,**kw)
        downloader.download_file(entry,self.root,api_base=self.base,opener=opener,sleeper=delays.append,timeout=2)
        self.assert_payload("good")
        self.assertEqual(attempts,["bytes=10013-"]*3)
        self.assertEqual(delays,[2,4])

    def test_tls_eof_during_body_resumes_from_saved_blocks(self):
        owner=self
        class Response:
            status=200
            headers={"Content-Type":"application/octet-stream"}
            reads=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,size):
                self.reads+=1
                if self.reads==1:return owner.payload[:10013]
                raise ssl.SSLEOFError(8,"UNEXPECTED_EOF_WHILE_READING")
        tries=[]
        def opener(req,**kw):
            tries.append(req.get_header("Range"))
            return Response() if len(tries)==1 else urllib.request.urlopen(req,**kw)
        downloader.download_file(self.entry("good"),self.root,api_base=self.base,opener=opener,sleeper=lambda _:None,timeout=2)
        self.assert_payload("good")
        self.assertEqual(tries,[None,"bytes=10013-"])

    def test_complete_validated_body_is_accepted_after_final_tls_eof(self):
        owner=self
        class Response:
            status=200
            headers={"Content-Type":"application/octet-stream"}
            reads=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,size):
                self.reads+=1
                if self.reads==1:return owner.payload
                raise ssl.SSLEOFError(8,"closed after complete body")
        result=downloader.download_file(self.entry("good"),self.root,attempts=1,
                                       opener=lambda *a,**k:Response(),sleeper=lambda _:None)
        self.assertEqual(result,"downloaded")
        self.assert_payload("good")

    def test_certificate_and_local_disk_failures_are_not_retried(self):
        for cause in [ssl.SSLCertVerificationError(1,"certificate verify failed"),
                      OSError(errno.ENOSPC,"disk full"),PermissionError(errno.EACCES,"access denied")]:
            calls=[]
            def opener(*args,**kwargs):
                calls.append(1)
                raise urllib.error.URLError(cause)
            with self.assertRaises(downloader.DownloadError) as caught:
                downloader.download_file(self.entry("good"),self.root,opener=opener,sleeper=lambda _:None)
            self.assertEqual(calls,[1])
            self.assertFalse(caught.exception.transient)
        self.assertTrue(downloader.retryable(ssl.SSLError(1,"SSL protocol error")))

    def test_persistent_disconnect_keeps_resumable_partial(self):
        with self.assertRaises(RuntimeError):self.fetch("always",attempts=2)
        p=self.root/downloader.CACHE_NAME/"parts/always.part"
        self.assertGreater(p.stat().st_size,10013)
        self.assertFalse(downloader.target_path(self.root,self.entry("always")).exists())

    def test_403_is_not_retried(self):
        with self.assertRaises(RuntimeError):self.fetch("forbidden")
        self.assertEqual(self.calls["forbidden"],1)

    def test_incorrect_supplied_hash_never_becomes_a_completed_file(self):
        entry=self.entry("good");entry["sha256"]="0"*64
        with self.assertRaises(RuntimeError):
            downloader.download_file(entry,self.root,api_base=self.base,attempts=2,sleeper=lambda _:None)
        self.assertFalse(downloader.target_path(self.root,entry).exists())

    def test_gzip_crc_failure_never_becomes_a_completed_nifti(self):
        with self.assertRaises(RuntimeError):self.fetch("gzip",attempts=2)
        self.assertFalse(downloader.target_path(self.root,self.entry("gzip")).exists())

    def test_source_path_traversal_rejected(self):
        entry=self.entry("good");entry["relative_path"]="../escape.bin"
        with self.assertRaises(ValueError):
            downloader.download_file(entry,self.root,api_base=self.base)

    def test_partial_dataset_marker_blocks_completion_and_resumes(self):
        entries=[self.entry("good"),self.entry("forbidden")]
        downloader.write_json(self.root/downloader.STATE_NAME,{"schema":1,"collection_id":downloader.COLLECTION_ID,"status":"downloading","files":entries})
        class Client: api_base=self.base
        def fetch(entry,root,**kw):
            return downloader.download_file(entry,root,attempts=2,sleeper=lambda _:None,timeout=2,**kw)
        with self.assertRaises(RuntimeError):
            downloader.download_dataset(self.root,workers=1,client=Client(),file_downloader=fetch)
        self.assertTrue(downloader.incomplete_dataset(self.root))
        self.assertEqual(downloader.read_json(self.root/downloader.STATE_NAME)["status"],"incomplete")
        type(self).deny=False
        downloader.download_dataset(self.root,workers=1,client=Client(),file_downloader=fetch)
        self.assertFalse(downloader.incomplete_dataset(self.root))
        self.assertEqual(self.calls["good"],1)

    def seeded_entries(self,identities):
        entries=[self.entry(i) for i in identities]
        downloader.write_json(self.root/downloader.STATE_NAME,{"schema":1,"collection_id":downloader.COLLECTION_ID,"status":"incomplete","files":entries})
        return entries

    def test_failed_file_is_deferred_while_other_files_finish(self):
        self.seeded_entries(["chunk","good"])
        order,delays=[],[]
        class Client:api_base=self.base
        def fetch(entry,root,**kw):
            order.append(entry["id"])
            if order==["chunk"]:
                raise downloader.DownloadError(entry,urllib.error.URLError(ssl.SSLEOFError(8,"EOF")))
            return downloader.download_file(entry,root,sleeper=lambda _:None,timeout=2,**kw)
        state=downloader.download_dataset(self.root,client=Client(),file_downloader=fetch,rounds=2,sleeper=delays.append)
        self.assertEqual(order,["chunk","good","chunk"])
        self.assertEqual(delays,[30])
        self.assertEqual(state["verified_files"],2)
        self.assertFalse(downloader.incomplete_dataset(self.root))

    def test_bounded_recovery_exhaustion_is_incomplete_and_restartable(self):
        self.seeded_entries(["good","always"])
        calls,delays=Counter(),[]
        class Client:api_base=self.base
        def fetch(entry,root,**kw):
            calls[entry["id"]]+=1
            if entry["id"]=="always":
                raise downloader.DownloadError(entry,urllib.error.URLError("offline"))
            return downloader.download_file(entry,root,sleeper=lambda _:None,timeout=2,**kw)
        with self.assertRaises(downloader.DatasetDownloadIncomplete):
            downloader.download_dataset(self.root,client=Client(),file_downloader=fetch,rounds=3,sleeper=delays.append)
        state=downloader.read_json(self.root/downloader.STATE_NAME)
        self.assertEqual(state["status"],"incomplete")
        self.assertEqual(state["verified_files"],1)
        self.assertEqual(state["pending_files"],[self.entry("always")["relative_path"]])
        self.assertEqual(calls,{"good":1,"always":3})
        self.assertEqual(delays,[30,60])
        self.assertTrue(downloader.incomplete_dataset(self.root))
        def recovered(entry,root,**kw):
            # Use the healthy fixture endpoint but retain the original source ID.
            return downloader.download_file(entry,root,opener=lambda req,**k:urllib.request.urlopen(self.base+"/file/good/download",**k),sleeper=lambda _:None,timeout=2,**kw)
        downloader.download_dataset(self.root,client=Client(),file_downloader=recovered,sleeper=lambda _:None)
        self.assertFalse(downloader.incomplete_dataset(self.root))
        self.assertEqual(self.calls["good"],2)  # existing good file was not requested again

    def test_general_outage_cools_down_before_requesting_all_files(self):
        self.seeded_entries(["a","b","c","d","e"])
        order,delays=[],[]
        class Client:api_base=self.base
        def fetch(entry,root,**kw):
            order.append(entry["id"])
            if len(order)<=3:
                raise downloader.DownloadError(entry,urllib.error.URLError("offline"))
            downloader.target_path(root,entry).parent.mkdir(parents=True,exist_ok=True)
            downloader.target_path(root,entry).write_bytes(self.payload)
        state=downloader.download_dataset(self.root,client=Client(),file_downloader=fetch,rounds=2,sleeper=delays.append)
        self.assertEqual(order,["a","b","c","d","e","a","b","c"])
        self.assertEqual(delays,[30])
        self.assertEqual(state["verified_files"],5)
        self.assertFalse(downloader.incomplete_dataset(self.root))

    def test_parallel_failure_recovery_uses_serial_round(self):
        self.seeded_entries(["a","b","c","d"])
        first=Counter()
        class Client:api_base=self.base
        def fetch(entry,root,**kw):
            first[entry["id"]]+=1
            if first[entry["id"]]==1:
                raise downloader.DownloadError(entry,urllib.error.URLError("offline"))
            downloader.target_path(root,entry).parent.mkdir(parents=True,exist_ok=True)
            downloader.target_path(root,entry).write_bytes(self.payload)
        from concurrent.futures import ThreadPoolExecutor
        workers=[]
        def pool(*args,**kwargs):
            workers.append(kwargs["max_workers"])
            return ThreadPoolExecutor(*args,**kwargs)
        with patch.object(downloader,"ThreadPoolExecutor",side_effect=pool):
            state=downloader.download_dataset(self.root,client=Client(),workers=2,file_downloader=fetch,rounds=3,sleeper=lambda _:None)
        self.assertEqual(workers[0],2)
        self.assertTrue(all(w==1 for w in workers[1:]))
        self.assertEqual(state["verified_files"],4)
        self.assertFalse(downloader.incomplete_dataset(self.root))

    def test_keyboard_interrupt_preserves_incomplete_state(self):
        self.seeded_entries(["good"])
        class Client:api_base=self.base
        def interrupted(*args,**kw):raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            downloader.download_dataset(self.root,client=Client(),file_downloader=interrupted,sleeper=lambda _:None)
        self.assertEqual(downloader.read_json(self.root/downloader.STATE_NAME)["status"],"incomplete")
        self.assertTrue(downloader.incomplete_dataset(self.root))

    def test_cli_defaults_to_one_worker(self):
        with patch.object(downloader,"download_dataset") as fetch,patch.object(downloader.shutil,"disk_usage") as disk:
            disk.return_value.free=100*2**30
            self.assertEqual(downloader.main(["--root",str(self.root)]),0)
            self.assertEqual(fetch.call_args.kwargs["workers"],1)

    def test_invalid_recovery_options_fail_immediately(self):
        for kwargs in [{"workers":0},{"workers":5},{"rounds":0}]:
            with self.assertRaises(ValueError):downloader.download_dataset(self.root,**kwargs)
        with self.assertRaises(ValueError):self.fetch("good",attempts=0)

    def test_index_selects_only_official_data_and_split_folders(self):
        class Client:
            def list(self,route,**params):
                if route=="folder" and params.get("parentType")=="collection":
                    return [{"_id":"n","name":"database_nifti"},{"_id":"s","name":"database_split"},{"_id":"j","name":"jupyter"}]
                if route=="folder":return []
                if route=="item":return [{"_id":params["folderId"]+"i","name":"file"}]
                return [{"_id":route.split("/")[1]+"f","name":"fixture.txt","size":100}]
        entries=downloader.build_index(Client(),workers=1)
        self.assertEqual(len(entries),2)
        self.assertEqual({e["relative_path"].split("/")[0] for e in entries},{"database_nifti","database_split"})

    def test_incomplete_dataset_cannot_enter_preparation_with_one_existing_image(self):
        import launcher
        from aces.utils import read_json,write_json
        settings=read_json(Path(launcher.__file__).parent/'settings.json')
        raw=self.root/'data/camus';raw.mkdir(parents=True)
        (raw/'patient0001_2CH_ED.nii.gz').touch()
        downloader.write_json(raw/downloader.STATE_NAME,{'status':'incomplete','files':[]})
        write_json(self.root/'settings.json',settings)
        with patch.object(launcher,'ROOT',self.root),patch.object(launcher,'unpack_if_present'),patch('download_public.main',side_effect=RuntimeError('network fixture')),patch('aces.data.prepare_camus') as prepare:
            with self.assertRaises(RuntimeError):launcher.ensure_data(settings,self.root/'settings.json')
            prepare.assert_not_called()


if __name__=="__main__":unittest.main(verbosity=2)
