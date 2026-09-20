"""Build the 8-clip SDR pack deterministically (prereg_amendments.md §2).

Source: HF mirror roro128/musdb18-hq-flac (test split = the original
MUSDB18-HQ test set; lossless FLAC re-encode of danjacobellis/musdb18HQ;
CC BY-NC-SA 4.0 — research use only). Audio is NEVER committed; this script
plus configs/sdr_clips.json (track, start sample, sha256 of every derived
wav) make the pack reproducible.

Rule: the first 8 tracks, in shard order, whose mixture AND vocals rows are
present in the shards given; clip = 352800 samples starting at the first
candidate start >= 30 % of the track length, scanning forward in 1-second
steps from the largest multiple of 441 <= 30 % of the length, at which the
8-s VOCALS segment has RMS >= 0.01 (a silent-vocals clip makes SDR
degenerate); a track with no such start is skipped.

Usage (off-pod, in a venv with pyarrow + soundfile):
  python scripts/prepare_sdr_clips.py --shards test-00000.parquet test-00001.parquet \
      --revision <hf revision sha> --out-dir <pack dir> --manifest configs/sdr_clips.json
"""

import argparse
import hashlib
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

CLIP = 352800
HOP = 441
SR = 44100
N_CLIPS = 8
MIN_VOCAL_RMS = 0.01


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', nargs='+', required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--dataset', default='roro128/musdb18-hq-flac')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--manifest', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tracks = {}          # track path -> {'mixture': bytes, 'vocals': bytes}
    order = []
    shard_meta = []
    for shard in args.shards:
        pf = pq.ParquetFile(shard)
        shard_meta.append({'file': os.path.basename(shard),
                           'rows': pf.metadata.num_rows,
                           'size_bytes': os.path.getsize(shard)})
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            for row in t.to_pylist():
                path, inst = row['path'], row['instrument']
                track = os.path.dirname(path)
                if inst not in ('mixture', 'vocals'):
                    continue
                if track not in tracks:
                    tracks[track] = {}
                    order.append(track)
                tracks[track][inst] = row['audio']['bytes']

    complete = [t for t in order if {'mixture', 'vocals'} <= set(tracks[t])]
    assert len(complete) >= N_CLIPS, f'only {len(complete)} complete tracks'
    clips = []
    for track in complete:
        if len(clips) == N_CLIPS:
            break
        i = len(clips)
        stems = {}
        for kind in ('mixture', 'vocals'):
            audio, sr = sf.read(io.BytesIO(tracks[track][kind]), dtype='float32',
                                always_2d=True)
            assert sr == SR, (track, kind, sr)
            stems[kind] = audio
        n = min(len(stems['mixture']), len(stems['vocals']))
        start = (int(0.3 * n) // HOP) * HOP
        while start + CLIP <= n and np.sqrt(np.mean(
                stems['vocals'][start:start + CLIP] ** 2)) < MIN_VOCAL_RMS:
            start += SR                      # 1-second steps, deterministic
        if start + CLIP > n:
            print(f'skip {track}: no 8-s segment with vocals RMS >= {MIN_VOCAL_RMS}')
            continue
        entry = {'index': i, 'track': os.path.basename(track), 'source_path': track,
                 'start_sample': start, 'n_samples': CLIP, 'sample_rate': SR,
                 'track_length_samples': n, 'files': {}}
        for kind in ('mixture', 'vocals'):
            name = f'clip{i:02d}_{kind}.wav'
            dest = os.path.join(args.out_dir, name)
            seg = np.ascontiguousarray(stems[kind][start:start + CLIP])
            sf.write(dest, seg, SR, subtype='FLOAT')
            entry['files'][kind] = {'name': name, 'sha256': sha256_file(dest),
                                    'rms': float(np.sqrt(np.mean(seg ** 2)))}
        clips.append(entry)
        print(f"clip {i}: {entry['track']} start={start} "
              f"vocals_rms={entry['files']['vocals']['rms']:.4f}")

    manifest = {'dataset': args.dataset, 'revision': args.revision,
                'license': 'CC BY-NC-SA 4.0 (research use; audio not redistributed)',
                'derived_from': 'danjacobellis/musdb18HQ (MUSDB18-HQ test split)',
                'selection_rule': 'first 8 tracks in shard order with mixture+vocals '
                                  'and a non-silent vocals segment; start = largest '
                                  'multiple of 441 <= 30% of length, advanced in '
                                  '1-s steps until vocals RMS >= 0.01; 352800 '
                                  'samples; float32 wav',
                'min_vocal_rms': MIN_VOCAL_RMS,
                'shards': shard_meta, 'clips': clips}
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'-> {args.manifest} ({len(clips)} clips) pack: {args.out_dir}')


if __name__ == '__main__':
    main()
