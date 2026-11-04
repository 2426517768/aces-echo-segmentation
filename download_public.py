"""Official CAMUS static-file download with retries, Range resume and integrity checks."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import errno
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
COLLECTION_ID = "6373703d73e9f0047faa1bc8"
STATE_NAME = ".camus_download_state.json"
CACHE_NAME = ".camus_download_cache"
CHUNK_SIZE = 256 * 1024
DOWNLOAD_VERSION = "1.2"


class DownloadError(RuntimeError):
    """A file failed; the scheduler may defer transient network failures."""
    def __init__(self, entry, cause):
        self.entry, self.cause = entry, cause
        self.transient = retryable(cause)
        super().__init__(f"未完成 {entry['relative_path']}；进度已保留。原因：{cause}")


class DatasetDownloadIncomplete(RuntimeError):
    """All bounded recovery rounds ended with some files still missing."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def safe_name(name):
    if not isinstance(name, str) or name in {"", ".", ".."} or re.search(r'[\\/<>:"|?*\x00-\x1f]', name) or name.endswith((" ", ".")):
        raise ValueError(f"Invalid source name: {name!r}")
    return name


def safe_id(identifier):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(identifier)):
        raise ValueError("Invalid source ID")
    return str(identifier)


def target_path(root, entry):
    rel = entry["relative_path"]
    if not isinstance(rel, str) or "\\" in rel or rel.startswith("/"):
        raise ValueError("Invalid source path")
    for part in rel.split("/"):
        safe_name(part)
    path = (root/rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes destination")
    return path


def retryable(error):
    if isinstance(error, DownloadError):
        return error.transient
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504}
    if isinstance(error, urllib.error.URLError):
        return retryable(error.reason) if isinstance(error.reason, BaseException) else True
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, ssl.SSLError):
        return True
    # Local disk/permission failures cannot be repaired by network retries.
    if isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EACCES, errno.EPERM, errno.EROFS}:
        return False
    return isinstance(error, (http.client.IncompleteRead, http.client.RemoteDisconnected,
                             urllib.error.URLError, OSError, TimeoutError, ValueError))


class GirderClient:
    def __init__(self, cache, api_base=API_BASE, attempts=5, timeout=30, sleeper=time.sleep):
        self.cache, self.api_base = Path(cache), api_base.rstrip("/")
        self.attempts, self.timeout, self.sleeper = attempts, timeout, sleeper
        self.cancel = threading.Event()

    def get_json(self, route, params=None):
        url = self.api_base+"/"+route.lstrip("/")
        if params:
            url += "?"+urllib.parse.urlencode(params)
        cached = self.cache/"api"/(hashlib.sha256(url.encode()).hexdigest()+".json")
        if cached.exists():
            return read_json(cached)
        for attempt in range(self.attempts):
            if self.cancel.is_set():
                raise RuntimeError("目录索引已暂停，已缓存元数据保留。")
            try:
                req = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "ACES-CAMUS/"+DOWNLOAD_VERSION})
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.load(response)
                write_json(cached, data)
                return data
            except Exception as error:
                if not retryable(error) or attempt+1 == self.attempts:
                    raise
                print(f"官方目录连接中断，重试 {attempt+1}/{self.attempts}。", flush=True)
                self.sleeper(min(2**attempt, 8))

    def list(self, route, **params):
        result, seen, offset, limit = [], set(), 0, 200
        while True:
            page = self.get_json(route, {**params, "limit": limit, "offset": offset})
            if not isinstance(page, list):
                raise ValueError("Invalid metadata list")
            for item in page:
                identity = item.get("_id")
                if not identity or identity in seen:
                    raise ValueError("Metadata pagination overlap")
                seen.add(identity)
                result.append(item)
            if len(page) < limit:
                return result
            offset += len(page)


def bounded_map(pool, function, values, workers, cancel):
    cursor, pending = iter(values), set()
    def submit():
        value = next(cursor, None)
        if value is not None:
            pending.add(pool.submit(function, value))
    try:
        for _ in range(workers):submit()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.remove(future)
                yield future.result()
                submit()
    except BaseException:
        cancel.set()
        raise
    finally:
        for future in pending:future.cancel()


def build_index(client, workers=4):
    roots = client.list("folder", parentType="collection", parentId=COLLECTION_ID)
    roots = [f for f in roots if f.get("name") in {"database_nifti", "database_split"}]
    if {f["name"] for f in roots} != {"database_nifti", "database_split"}:
        raise ValueError("Official NIfTI/split folders not both available")
    todo = [(f["_id"], safe_name(f["name"])) for f in roots]
    items, visited = [], set()
    cancel = getattr(client, "cancel", threading.Event())
    folders_done = 0

    def folder_node(pair):
        identity, rel = pair
        children = client.list("folder", parentType="folder", parentId=identity)
        node_items = client.list("item", folderId=identity)
        return ([(f["_id"], rel+"/"+safe_name(f["name"])) for f in children],
                [(i, rel) for i in node_items])

    def item_files(pair):
        item, rel = pair
        files = client.list("item/"+safe_id(item["_id"])+"/files")
        if not files:
            raise ValueError("Source item has no files")
        out = []
        for file in files:
            entry = {"id": safe_id(file["_id"]), "relative_path": rel+"/"+safe_name(file["name"]), "size": int(file["size"])}
            if entry["size"] < 0:
                raise ValueError("Negative source length")
            for alg, length in [("sha256", 64), ("sha512", 128)]:
                val = file.get(alg)
                if isinstance(val, str) and re.fullmatch(r"[a-fA-F0-9]{"+str(length)+"}", val):
                    entry[alg] = val.lower()
            out.append(entry)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while todo:
            layer, todo = todo, []
            for identity, rel in layer:
                if identity in visited:
                    raise ValueError("Repeated source folder ID")
                visited.add(identity)
            for children, node_items in bounded_map(pool, folder_node, layer, workers, cancel):
                todo.extend(children)
                items.extend(node_items)
                folders_done += 1
                if folders_done % 25 == 0:
                    print(f"目录索引：已读取 {folders_done} 个目录。", flush=True)
            print(f"已读取 {len(visited)} 个目录，发现 {len(items)} 个条目。", flush=True)
        entries = []
        for n, files in enumerate(bounded_map(pool, item_files, items, workers, cancel), 1):
            entries.extend(files)
            if n == 1 or n % 200 == 0 or n == len(items):
                print(f"文件清单 {n}/{len(items)}。", flush=True)
    paths = [e["relative_path"].casefold() for e in entries]
    if not entries or len(set(paths)) != len(paths):
        raise ValueError("Empty index or conflicting destination paths")
    return sorted(entries, key=lambda e: e["relative_path"])


def check_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return None
    h = hashlib.sha256()
    h512 = hashlib.sha512() if entry.get("sha512") else None
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(block)
            if h512:
                h512.update(block)
    value = h.hexdigest()
    if entry.get("sha256") and value != entry["sha256"]:
        return None
    if h512 and h512.hexdigest() != entry["sha512"]:
        return None
    if entry["relative_path"].lower().endswith(".nii.gz"):
        try:
            with gzip.open(path, "rb") as f:
                while f.read(CHUNK_SIZE):
                    pass
        except (OSError, EOFError):
            return None
    return value


def download_file(entry, destination, api_base=API_BASE, attempts=6, timeout=60,
                  opener=urllib.request.urlopen, sleeper=time.sleep, stop=None):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    destination = Path(destination).resolve()
    path = target_path(destination, entry)
    identifier = safe_id(entry["id"])
    cache = destination/CACHE_NAME
    part = cache/"parts"/(identifier+".part")
    receipt = cache/"verified"/(identifier+".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    part.parent.mkdir(parents=True, exist_ok=True)

    def accept(value):
        stat = path.stat()
        write_json(receipt, {"expected": entry, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": value})

    if path.exists():
        if receipt.exists():
            old, stat = read_json(receipt), path.stat()
            if old.get("expected") == entry and old.get("size") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
                return "already_verified"
        value = check_file(path, entry)
        if value:
            accept(value)
            return "existing_verified"
    last_error = None
    for attempt in range(attempts):
        if stop is not None and stop.is_set():
            raise RuntimeError("Download stopped; partial file retained")
        offset = part.stat().st_size if part.exists() else 0
        if offset == entry["size"]:
            value = check_file(part, entry)
            if value:
                os.replace(part, path)
                accept(value)
                return "downloaded"
            part.write_bytes(b"")
            offset = 0
        elif offset > entry["size"]:
            part.write_bytes(b"")
            offset = 0
        headers = {"Accept-Encoding": "identity", "User-Agent": "ACES-CAMUS/"+DOWNLOAD_VERSION}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(api_base.rstrip("/")+"/file/"+identifier+"/download", headers=headers)
        try:
            with opener(req, timeout=timeout) as response:
                if response.status == 206:
                    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not m or int(m[1]) != offset or int(m[2]) != entry["size"]-1 or int(m[3]) != entry["size"]:
                        part.write_bytes(b"")
                        raise ValueError("Incorrect Content-Range; restarting individual file")
                elif response.status == 200:
                    offset = 0  # Server ignored Range: truncate, never append its full body.
                else:
                    raise ValueError(f"Unexpected HTTP status {response.status}")
                if "html" in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("Source returned a webpage instead of the original file")
                count = offset
                with part.open("ab" if offset else "wb") as f:
                    while True:
                        if stop is not None and stop.is_set():
                            raise RuntimeError("Download stopped; partial file retained")
                        try:
                            block = response.read(CHUNK_SIZE)
                        except http.client.IncompleteRead as error:
                            if error.partial and count+len(error.partial) <= entry["size"]:
                                f.write(error.partial)
                            raise
                        if not block:
                            break
                        if count+len(block) > entry["size"]:
                            raise ValueError("Body exceeds expected source length")
                        f.write(block)
                        count += len(block)
            value = check_file(part, entry)
            if not value:
                raise ValueError("Length, supplied hash, or gzip CRC verification failed")
            os.replace(part, path)
            accept(value)
            return "downloaded"
        except Exception as error:
            last_error = error
            # A peer may close TLS after the complete file arrived. Accept only
            # after the same length/hash/CRC checks, including on the final try.
            if part.exists() and part.stat().st_size == entry["size"]:
                value = check_file(part, entry)
                if value:
                    os.replace(part, path)
                    accept(value)
                    return "downloaded"
                part.write_bytes(b"")
            if not retryable(error) or attempt+1 == attempts:
                break
            delay = min(2**(attempt+1), 30)
            print(f"连接中断，{delay} 秒后重试 {attempt+2}/{attempts}：{entry['relative_path']}", flush=True)
            sleeper(delay)
    raise DownloadError(entry, last_error) from last_error


def incomplete_dataset(destination):
    destination = Path(destination)
    marker = destination/STATE_NAME
    if not marker.exists():
        return False
    state = read_json(marker)
    if state.get("status") != "complete":
        return True
    return any(not target_path(destination, e).is_file() or target_path(destination, e).stat().st_size != e["size"]
               for e in state.get("files", []))


def download_dataset(destination, workers=1, client=None, file_downloader=download_file,
                     rounds=4, sleeper=time.sleep):
    if workers not in {1, 2, 3, 4} or rounds < 1:
        raise ValueError("workers must be 1..4 and rounds must be positive")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination/STATE_NAME
    state = read_json(marker) if marker.exists() else {}
    client = client or GirderClient(destination/CACHE_NAME)
    if state.get("files") and state.get("schema") == 1 and state.get("collection_id") == COLLECTION_ID:
        entries = state["files"]
    else:
        state = {"schema": 1, "collection_id": COLLECTION_ID, "api_base": client.api_base, "status": "indexing"}
        write_json(marker, state)
        entries = build_index(client)
    state.update(status="downloading", files=entries, total_bytes=sum(e["size"] for e in entries))
    write_json(marker, state)
    print(f"CAMUS 下载器 v{DOWNLOAD_VERSION}：{len(entries)} 个文件，{state['total_bytes']/2**30:.2f} GiB；"
          f"并发 {workers}，最多 {rounds} 轮，可续传。", flush=True)
    stop = threading.Event()
    remaining, done, completed_bytes = list(entries), 0, 0
    failures = {}
    try:
        last = time.monotonic()
        for round_number in range(1, rounds+1):
            concurrency = workers if round_number == 1 else 1
            print(f"下载第 {round_number}/{rounds} 轮：待处理 {len(remaining)} 个文件，并发 {concurrency}。", flush=True)
            pending, cursor, deferred = {}, iter(remaining), []
            consecutive_failures, pause_round = 0, False
            pool = ThreadPoolExecutor(max_workers=concurrency)
            try:
                def submit_next():
                    entry = next(cursor, None)
                    if entry is not None:
                        pending[pool.submit(file_downloader, entry, destination, api_base=client.api_base, stop=stop)] = entry
                for _ in range(concurrency):
                    submit_next()
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        entry = pending.pop(future)
                        try:
                            future.result()
                        except DownloadError as error:
                            if not error.transient:
                                raise
                            deferred.append(entry)
                            failures[entry["relative_path"]] = str(error.cause)
                            consecutive_failures += 1
                            print(f"暂缓 {entry['relative_path']}；继续其他文件，本轮结束后补下载。", flush=True)
                            # During a general outage, do not burn six attempts
                            # on every remaining source file. Cool down first.
                            if consecutive_failures >= 3:
                                pause_round = True
                        else:
                            done += 1
                            completed_bytes += entry["size"]
                            consecutive_failures = 0
                            failures.pop(entry["relative_path"], None)
                            if done == 1 or done % 50 == 0 or done == len(entries) or time.monotonic()-last >= 5:
                                print(f"已校验 {done}/{len(entries)}，{completed_bytes/2**30:.2f}/{state['total_bytes']/2**30:.2f} GiB。", flush=True)
                                last = time.monotonic()
                    if not pause_round:
                        for _ in finished:
                            submit_next()
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            # Untouched files go first; only exhausted transient failures are
            # deferred. No clinical file is omitted from the completion gate.
            remaining = list(cursor)+deferred
            state.update(retry_round=round_number, verified_files=done, verified_bytes=completed_bytes,
                         pending_files=[e["relative_path"] for e in remaining], failed_files=failures)
            write_json(marker, state)
            if not remaining:
                break
            if round_number < rounds:
                delay = min(30*2**(round_number-1), 60)
                print(f"尚有 {len(remaining)} 个文件待完成；{delay} 秒后自动以单线程补下载，请保留窗口。", flush=True)
                sleeper(delay)
        if remaining:
            raise DatasetDownloadIncomplete(
                f"已校验 {done}/{len(entries)} 个文件，{len(remaining)} 个仍未完成。"
                f"已用完 {rounds} 轮自动恢复；请稍后重新双击 START.bat。"
                "无需删除 data 或重新安装环境，已校验文件会自动跳过；数据齐全前不会进入训练。")
        state.update(status="complete", verified_files=done)
        state.pop("error", None)
        state.pop("failed_files", None)
        state.pop("pending_files", None)
        write_json(marker, state)
    except BaseException as error:
        stop.set()
        state.update(status="incomplete", error=str(error))
        write_json(marker, state)
        raise
    return state


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Official resumable CAMUS downloader")
    parser.add_argument("--root")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=1)
    args = parser.parse_args([] if argv is None else argv)
    settings = read_json(root/"settings.json") if (root/"settings.json").exists() else {}
    destination = Path(args.root or settings.get("paths", {}).get("camus_root", "data/camus"))
    if not destination.is_absolute():
        destination = root/destination
    destination.mkdir(parents=True, exist_ok=True)
    print("CAMUS 官方入口：https://www.creatis.insa-lyon.fr/Challenge/camus/", flush=True)
    print("沿用你已确认的提供方使用条款；改为获取原始单文件。", flush=True)
    if shutil.disk_usage(destination).free < 15*2**30:
        raise RuntimeError("可用空间低于 15GiB，请释放空间或指定其他 data 目录。")
    download_dataset(destination, workers=args.workers)
    print("全部文件校验完成，可继续训练。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("已暂停。再次运行可复用已校验文件并续传。", flush=True)
        raise SystemExit(130)
    except (DownloadError, DatasetDownloadIncomplete) as error:
        print(str(error), flush=True)
        raise SystemExit(1)
