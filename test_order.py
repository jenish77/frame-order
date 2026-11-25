import os
import shutil
import tempfile
import numpy as np
from PIL import Image
import cv2
import order

def test_digits():
    result = order.order_frames('examples/digits')
    expected = [
        '27cecfd2.png', # 1
        '4e8585ae.png', # 2
        '79e53b10.png', # 3
        'c93f39e4.png', # 4
        '914dc7bc.png', # 5
    ]
    assert result == expected or result == list(reversed(expected)), f"Digits mismatch: {result}"
    print("[PASS] test_digits passed!")

def test_single_and_two_frames():
    tmp_dir = tempfile.mkdtemp()
    try:
        # 1 frame
        im = Image.new('RGB', (64, 64), color='red')
        im.save(os.path.join(tmp_dir, 'f1.png'))
        res = order.order_frames(tmp_dir)
        assert res == ['f1.png'], f"Expected ['f1.png'], got {res}"
        
        # 2 frames
        im2 = Image.new('RGB', (64, 64), color='blue')
        im2.save(os.path.join(tmp_dir, 'f2.png'))
        res2 = order.order_frames(tmp_dir)
        assert set(res2) == {'f1.png', 'f2.png'} and len(res2) == 2, f"Failed 2 frames: {res2}"
        print("[PASS] test_single_and_two_frames passed!")
    finally:
        shutil.rmtree(tmp_dir)

def test_synthetic_video_sequence():
    tmp_dir = tempfile.mkdtemp()
    try:
        N = 30
        names = [f"frame_{i:04d}.png" for i in range(N)]
        
        for t in range(N):
            im_arr = np.full((80, 100, 3), 40, dtype=np.uint8)
            # Moving ball
            cx = int(15 + t * 2.2)
            cy = int(20 + t * 1.5)
            cv2.circle(im_arr, (cx, cy), 8, (0, 255, 100), -1)
            Image.fromarray(im_arr).save(os.path.join(tmp_dir, names[t]))
            
        res = order.order_frames(tmp_dir)
        assert len(res) == N
        assert set(res) == set(names)
        
        # Check if consecutive pairs are preserved
        res_idx = [names.index(x) for x in res]
        is_forward = (res_idx == list(range(N)))
        is_backward = (res_idx == list(reversed(range(N))))
        assert is_forward or is_backward, f"Sequence not properly ordered: {res_idx}"
        print("[PASS] test_synthetic_video_sequence passed!")
    finally:
        shutil.rmtree(tmp_dir)

def test_large_video_heuristic():
    tmp_dir = tempfile.mkdtemp()
    try:
        N = 50
        names = [f"f_{i:03d}.png" for i in range(N)]
        
        for t in range(N):
            im_arr = np.full((60, 60, 3), 30, dtype=np.uint8)
            cx = int(5 + t * 1.0)
            cy = int(10 + t * 1.0)
            cv2.rectangle(im_arr, (cx, cy), (cx + 10, cy + 10), (200, 100, 50), -1)
            Image.fromarray(im_arr).save(os.path.join(tmp_dir, names[t]))
            
        res = order.order_frames(tmp_dir)
        assert len(res) == N
        assert set(res) == set(names)
        
        res_idx = [names.index(x) for x in res]
        is_forward = (res_idx == list(range(N)))
        is_backward = (res_idx == list(reversed(range(N))))
        assert is_forward or is_backward, f"Large sequence not ordered: {res_idx}"
        print("[PASS] test_large_video_heuristic passed!")
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    test_digits()
    test_single_and_two_frames()
    test_synthetic_video_sequence()
    test_large_video_heuristic()
    print("ALL TESTS PASSED SUCCESSFULLY!")
