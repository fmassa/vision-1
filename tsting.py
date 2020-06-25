import time
import torch, torchvision

def interp(x, scale=2):
    o1, o2 = int(x.shape[-2] * scale), int(x.shape[-1] * scale)
    temp = torch.ops.torchvision.interpolate(x, o1, 2)
    return torch.ops.torchvision.interpolate(temp, o2, 3)


def timeit(f, n=5):
    t = time.time()
    for _ in range(n):
        f()
    return (time.time() - t) / n


shape = [32, 64, 64, 128]
shape = [1, 64, 64, 3]
s = 2

a = torch.rand(*shape).permute(0, 3, 1, 2)
a_c = a.contiguous()


for shape, n in [([1, 64, 64, 3], 100), ([1, 512, 512, 3], 100)]:
    a = torch.rand(*shape).permute(0, 3, 1, 2)
    a_c = a.contiguous()

    for s in [0.125, 0.25, 0.5, 1.0]:

        for cont in [False, True]:
            x = a_c if cont else a
            r1 = interp(x, s)
            r2 = torch.nn.functional.interpolate(
                x, scale_factor=s, mode='bilinear', align_corners=False, recompute_scale_factor=False)
            assert (r1 - r2).abs().max() < 1e-6
            t1 = timeit(lambda : interp(x, s), n=n)
            t2 = timeit(lambda : torch.nn.functional.interpolate(
                x, scale_factor=s, mode='bilinear', align_corners=False, recompute_scale_factor=False), n=n)
            print(f"shape={shape}, scale={s:02.2f}, contiguous={cont}, mine={t1:.5f}, orig={t2:.5f}, speedup={t2 / t1:.2f}")

#from IPython import embed; embed()
