"""
Prepare 100-Sample Subset from Official DMOSpeech 2 Cross-Sentence Benchmark
Extracts the first 100 evaluation pairs from data/librispeech_pc_test_clean_cross_sentence.lst
and ensures the reference prompt audio files are available.
"""

import os
import sys
import json
import tarfile
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "BENCHMARK_100_SAMPLES")
DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
LST_PATH = os.path.join(BASE_DIR, "data", "librispeech_pc_test_clean_cross_sentence.lst")
MANIFEST_OUT = os.path.join(DATA_DIR, "manifest_100.json")

def parse_first_100():
    if not os.path.exists(LST_PATH):
        raise FileNotFoundError(f"Benchmark list not found at: {LST_PATH}")

    with open(LST_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    samples = []
    required_files = set()

    for idx, line in enumerate(lines[:100], 1):
        parts = line.split("\t")
        if len(parts) != 6:
            continue
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = parts
        
        # Audio file relative path in LibriSpeech/test-clean:
        # e.g. 4992-41806-0009 -> LibriSpeech/test-clean/4992/41806/4992-41806-0009.flac
        spk_id, ch_id, _ = ref_utt.split("-")
        rel_flac_path = f"LibriSpeech/test-clean/{spk_id}/{ch_id}/{ref_utt}.flac"
        required_files.add(rel_flac_path)

        samples.append({
            "sample_id": idx,
            "speaker_id": spk_id,
            "chapter_id": ch_id,
            "ref_utt": ref_utt,
            "ref_duration": float(ref_dur),
            "ref_text": ref_txt,
            "gen_utt": gen_utt,
            "gen_duration": float(gen_dur),
            "gen_text": gen_txt,
            "rel_flac_path": rel_flac_path
        })

    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"[OK] Parsed {len(samples)} benchmark pairs.")
    print(f"[INFO] Unique reference audio files needed: {len(required_files)}")
    return samples, required_files

def ensure_audio_files(required_files):
    # Check which files already exist
    missing_files = []
    for rel_path in required_files:
        full_path = os.path.join(DATA_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_files.append(rel_path)

    if not missing_files:
        print("[OK] All required audio files already exist locally.")
        return

    print(f"[INFO] {len(missing_files)} files need to be retrieved.")
    tar_path = os.path.join(DATA_DIR, "test-clean.tar.gz")
    
    if not os.path.exists(tar_path):
        url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
        print(f"[INFO] Downloading official LibriSpeech test-clean (346 MB) from OpenSLR...")
        print(f"       URL: {url}")
        
        def reporthook(blocknum, blocksize, totalsize):
            percent = min(100.0, blocknum * blocksize * 100.0 / totalsize)
            downloaded_mb = (blocknum * blocksize) / (1024 * 1024)
            total_mb = totalsize / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {downloaded_mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, tar_path, reporthook)
        print("\n[OK] Download complete.")

    print(f"[INFO] Extracting needed audio files from {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Extract only members in required_files
        members = [m for m in tar.getmembers() if any(m.name.endswith(rf) or rf in m.name for rf in required_files)]
        print(f"[INFO] Extracting {len(members)} relevant files...")
        tar.extractall(path=DATA_DIR, members=members)

    print("[OK] All reference audio files are ready in data/LibriSpeech/test-clean/.")

if __name__ == "__main__":
    samples, required = parse_first_100()
    ensure_audio_files(required)
