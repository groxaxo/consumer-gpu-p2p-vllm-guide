/*
 * p2p_bandwidth_bench.cu
 *
 * Comprehensive GPU peer-to-peer and host-device bandwidth benchmark.
 * Tests all GPU pairs for:
 *   - P2P capability and transport type
 *   - Unidirectional bandwidth (multiple transfer sizes)
 *   - Bidirectional bandwidth (simultaneous transfers)
 *   - Round-trip latency
 *   - Host-to-device / device-to-host baseline for each GPU
 *
 * Compile:
 *   nvcc -O2 -arch=native -o p2p_bandwidth_bench p2p_bandwidth_bench.cu
 * Or via the Python wrapper:
 *   python3 p2p_bandwidth_bench.py
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

// ─── helpers ────────────────────────────────────────────────────────────────

#define CHK(x) do {                                                     \
    cudaError_t _e = (x);                                               \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d  %s: %s\n",               \
                __FILE__, __LINE__, #x, cudaGetErrorString(_e));        \
        exit(1);                                                        \
    }                                                                   \
} while(0)

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CHK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// ─── per-GPU baseline (HtoD / DtoH) ─────────────────────────────────────────

static void bench_host_device(int gpu_id, size_t bytes, int iters) {
    CHK(cudaSetDevice(gpu_id));

    void *d = nullptr, *h = nullptr;
    CHK(cudaMalloc(&d, bytes));
    CHK(cudaMallocHost(&h, bytes));
    memset(h, 0xCC, bytes);

    cudaEvent_t t0, t1;
    CHK(cudaEventCreate(&t0));
    CHK(cudaEventCreate(&t1));

    // warmup
    CHK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    CHK(cudaDeviceSynchronize());

    // HtoD
    CHK(cudaEventRecord(t0));
    for (int k = 0; k < iters; k++)
        CHK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    CHK(cudaEventRecord(t1));
    CHK(cudaEventSynchronize(t1));
    float htod_gbps = (float)(bytes * iters) / (elapsed_ms(t0,t1) / 1000.0f) / 1e9f;

    // DtoH
    CHK(cudaEventRecord(t0));
    for (int k = 0; k < iters; k++)
        CHK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    CHK(cudaEventRecord(t1));
    CHK(cudaEventSynchronize(t1));
    float dtoh_gbps = (float)(bytes * iters) / (elapsed_ms(t0,t1) / 1000.0f) / 1e9f;

    printf("  GPU%d  HtoD: %6.2f GB/s   DtoH: %6.2f GB/s\n",
           gpu_id, htod_gbps, dtoh_gbps);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    CHK(cudaFree(d));
    CHK(cudaFreeHost(h));
}

// ─── unidirectional P2P bandwidth for one pair ──────────────────────────────

static float bench_unidirectional(int src, int dst, size_t bytes, int iters,
                                  bool use_p2p) {
    void *d_src = nullptr, *d_dst = nullptr;
    CHK(cudaSetDevice(src)); CHK(cudaMalloc(&d_src, bytes));
    CHK(cudaSetDevice(dst)); CHK(cudaMalloc(&d_dst, bytes));
    CHK(cudaSetDevice(src));
    CHK(cudaMemset(d_src, 0xAB, bytes));

    cudaEvent_t t0, t1;
    CHK(cudaEventCreate(&t0));
    CHK(cudaEventCreate(&t1));

    // warmup
    if (use_p2p) {
        CHK(cudaMemcpyPeer(d_dst, dst, d_src, src, bytes));
    } else {
        void *tmp; CHK(cudaMallocHost(&tmp, bytes));
        CHK(cudaSetDevice(src)); CHK(cudaMemcpy(tmp, d_src, bytes, cudaMemcpyDeviceToHost));
        CHK(cudaSetDevice(dst)); CHK(cudaMemcpy(d_dst, tmp, bytes, cudaMemcpyHostToDevice));
        CHK(cudaFreeHost(tmp));
        CHK(cudaSetDevice(src));
    }
    CHK(cudaDeviceSynchronize());

    CHK(cudaEventRecord(t0));
    for (int k = 0; k < iters; k++) {
        if (use_p2p) {
            CHK(cudaMemcpyPeer(d_dst, dst, d_src, src, bytes));
        } else {
            void *tmp; CHK(cudaMallocHost(&tmp, bytes));
            CHK(cudaSetDevice(src)); CHK(cudaMemcpy(tmp, d_src, bytes, cudaMemcpyDeviceToHost));
            CHK(cudaSetDevice(dst)); CHK(cudaMemcpy(d_dst, tmp, bytes, cudaMemcpyHostToDevice));
            CHK(cudaFreeHost(tmp));
            CHK(cudaSetDevice(src));
        }
    }
    CHK(cudaEventRecord(t1));
    CHK(cudaEventSynchronize(t1));

    float gbps = (float)(bytes * iters) / (elapsed_ms(t0,t1) / 1000.0f) / 1e9f;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    CHK(cudaSetDevice(src)); CHK(cudaFree(d_src));
    CHK(cudaSetDevice(dst)); CHK(cudaFree(d_dst));
    return gbps;
}

// ─── bidirectional P2P bandwidth for one pair ───────────────────────────────

static float bench_bidirectional(int gpu0, int gpu1, size_t bytes, int iters,
                                 bool use_p2p) {
    void *d0a = nullptr, *d0b = nullptr, *d1a = nullptr, *d1b = nullptr;
    CHK(cudaSetDevice(gpu0)); CHK(cudaMalloc(&d0a, bytes)); CHK(cudaMalloc(&d0b, bytes));
    CHK(cudaSetDevice(gpu1)); CHK(cudaMalloc(&d1a, bytes)); CHK(cudaMalloc(&d1b, bytes));

    cudaStream_t s0, s1;
    CHK(cudaSetDevice(gpu0)); CHK(cudaStreamCreate(&s0));
    CHK(cudaSetDevice(gpu1)); CHK(cudaStreamCreate(&s1));

    cudaEvent_t t0, t1;
    CHK(cudaSetDevice(gpu0));
    CHK(cudaEventCreate(&t0));
    CHK(cudaEventCreate(&t1));

    auto do_transfer = [&]() {
        if (use_p2p) {
            CHK(cudaMemcpyPeerAsync(d1a, gpu1, d0a, gpu0, bytes, s0));
            CHK(cudaMemcpyPeerAsync(d0b, gpu0, d1b, gpu1, bytes, s1));
        } else {
            // sequential fallback — true async bidir on CPU bounce is complex
            void *tmp; CHK(cudaMallocHost(&tmp, bytes));
            CHK(cudaSetDevice(gpu0)); CHK(cudaMemcpy(tmp, d0a, bytes, cudaMemcpyDeviceToHost));
            CHK(cudaSetDevice(gpu1)); CHK(cudaMemcpy(d1a, tmp, bytes, cudaMemcpyHostToDevice));
            CHK(cudaSetDevice(gpu1)); CHK(cudaMemcpy(tmp, d1b, bytes, cudaMemcpyDeviceToHost));
            CHK(cudaSetDevice(gpu0)); CHK(cudaMemcpy(d0b, tmp, bytes, cudaMemcpyHostToDevice));
            CHK(cudaFreeHost(tmp));
        }
        CHK(cudaSetDevice(gpu0)); CHK(cudaStreamSynchronize(s0));
        CHK(cudaSetDevice(gpu1)); CHK(cudaStreamSynchronize(s1));
    };

    // warmup
    do_transfer();

    CHK(cudaSetDevice(gpu0));
    CHK(cudaEventRecord(t0));
    for (int k = 0; k < iters; k++) do_transfer();
    CHK(cudaSetDevice(gpu0));   // do_transfer() may leave device set to gpu1
    CHK(cudaEventRecord(t1));
    CHK(cudaEventSynchronize(t1));

    // bidir: count bytes in both directions
    float gbps = (float)(bytes * iters * 2) / (elapsed_ms(t0,t1) / 1000.0f) / 1e9f;

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaStreamDestroy(s0);
    CHK(cudaSetDevice(gpu1)); cudaStreamDestroy(s1);
    CHK(cudaSetDevice(gpu0)); CHK(cudaFree(d0a)); CHK(cudaFree(d0b));
    CHK(cudaSetDevice(gpu1)); CHK(cudaFree(d1a)); CHK(cudaFree(d1b));
    return gbps;
}

// ─── round-trip latency ──────────────────────────────────────────────────────

static float bench_latency_us(int src, int dst, bool use_p2p) {
    const size_t PING = 4;   // 4-byte ping
    void *d_src = nullptr, *d_dst = nullptr;
    CHK(cudaSetDevice(src)); CHK(cudaMalloc(&d_src, PING));
    CHK(cudaSetDevice(dst)); CHK(cudaMalloc(&d_dst, PING));
    CHK(cudaSetDevice(src));

    cudaEvent_t t0, t1;
    CHK(cudaEventCreate(&t0));
    CHK(cudaEventCreate(&t1));

    const int rounds = 200;

    auto do_ping = [&]() {
        if (use_p2p) {
            CHK(cudaMemcpyPeer(d_dst, dst, d_src, src, PING));
            CHK(cudaMemcpyPeer(d_src, src, d_dst, dst, PING));
        } else {
            void *tmp; CHK(cudaMallocHost(&tmp, PING));
            CHK(cudaSetDevice(src)); CHK(cudaMemcpy(tmp, d_src, PING, cudaMemcpyDeviceToHost));
            CHK(cudaSetDevice(dst)); CHK(cudaMemcpy(d_dst, tmp, PING, cudaMemcpyHostToDevice));
            CHK(cudaSetDevice(dst)); CHK(cudaMemcpy(tmp, d_dst, PING, cudaMemcpyDeviceToHost));
            CHK(cudaSetDevice(src)); CHK(cudaMemcpy(d_src, tmp, PING, cudaMemcpyHostToDevice));
            CHK(cudaFreeHost(tmp));
        }
    };

    // warmup
    for (int k = 0; k < 10; k++) do_ping();
    CHK(cudaDeviceSynchronize());

    CHK(cudaSetDevice(src));
    CHK(cudaEventRecord(t0));
    for (int k = 0; k < rounds; k++) do_ping();
    CHK(cudaEventRecord(t1));
    CHK(cudaEventSynchronize(t1));

    float us = elapsed_ms(t0, t1) * 1000.0f / rounds;

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    CHK(cudaSetDevice(src)); CHK(cudaFree(d_src));
    CHK(cudaSetDevice(dst)); CHK(cudaFree(d_dst));
    return us;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    int ngpu = 0;
    CHK(cudaGetDeviceCount(&ngpu));

    if (ngpu < 2) {
        fprintf(stderr, "Need at least 2 GPUs.\n");
        return 1;
    }

    // ── header ──────────────────────────────────────────────────────────────
    printf("================================================================================\n");
    printf("  GPU PEER-TO-PEER BANDWIDTH BENCHMARK\n");
    printf("================================================================================\n");

    // ── GPU inventory ────────────────────────────────────────────────────────
    printf("\n[GPU Inventory]\n");
    for (int i = 0; i < ngpu; i++) {
        cudaDeviceProp p;
        CHK(cudaGetDeviceProperties(&p, i));
        printf("  GPU%d  %-30s  %4d MiB  CUDA %d.%d  SM%d%d\n",
               i, p.name,
               (int)(p.totalGlobalMem / (1024*1024)),
               p.major, p.minor,
               p.major, p.minor);
    }

    // ── P2P capability matrix ────────────────────────────────────────────────
    printf("\n[P2P Capability Matrix]\n");
    printf("  (P=direct P2P DMA  C=CPU bounce fallback)\n\n");
    printf("         ");
    for (int j = 0; j < ngpu; j++) printf("  GPU%-2d", j);
    printf("\n");
    for (int i = 0; i < ngpu; i++) {
        printf("  GPU%-2d  ", i);
        for (int j = 0; j < ngpu; j++) {
            if (i == j) { printf("   --  "); continue; }
            int ok = 0;
            cudaDeviceCanAccessPeer(&ok, i, j);
            printf("   %s   ", ok ? "P" : "C");
        }
        printf("\n");
    }

    // enable P2P where supported
    for (int i = 0; i < ngpu; i++) {
        for (int j = 0; j < ngpu; j++) {
            if (i == j) continue;
            int ok = 0;
            cudaDeviceCanAccessPeer(&ok, i, j);
            if (ok) {
                CHK(cudaSetDevice(i));
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                    CHK(e);
            }
        }
    }

    // ── host-device baselines ────────────────────────────────────────────────
    printf("\n[Host <-> Device Baseline]  (256 MiB, 8 iters)\n");
    for (int i = 0; i < ngpu; i++)
        bench_host_device(i, 256UL * 1024 * 1024, 8);

    // ── pairwise bandwidth sweep ─────────────────────────────────────────────
    static const size_t sizes[]  = { 1, 16, 64, 256 };      // MiB
    static const int    niters[] = { 50, 20, 10,  5  };
    const int nsizes = 4;

    printf("\n[Unidirectional Bandwidth: GPU_src -> GPU_dst]\n");
    printf("  %-14s", "Pair/Method");
    for (int s = 0; s < nsizes; s++)
        printf("  %4zu MiB", sizes[s]);
    printf("\n");

    for (int src = 0; src < ngpu; src++) {
        for (int dst = 0; dst < ngpu; dst++) {
            if (src == dst) continue;
            int ok = 0;
            cudaDeviceCanAccessPeer(&ok, src, dst);

            // P2P (or CPU bounce)
            printf("  GPU%d->GPU%d  %-3s", src, dst, ok ? "P2P" : "CPU");
            for (int s = 0; s < nsizes; s++) {
                float gbps = bench_unidirectional(src, dst,
                                                  sizes[s] * 1024 * 1024,
                                                  niters[s], ok);
                printf("  %7.2f", gbps);
            }
            printf("  GB/s\n");

            // If P2P is available, also show what CPU bounce costs
            if (ok) {
                printf("  GPU%d->GPU%d  CPU", src, dst);
                for (int s = 0; s < nsizes; s++) {
                    float gbps = bench_unidirectional(src, dst,
                                                      sizes[s] * 1024 * 1024,
                                                      niters[s], false);
                    printf("  %7.2f", gbps);
                }
                printf("  GB/s\n");
            }
        }
    }

    // ── bidirectional bandwidth ──────────────────────────────────────────────
    printf("\n[Bidirectional Bandwidth: GPU_a <-> GPU_b]  (256 MiB, 5 iters)\n");
    for (int a = 0; a < ngpu; a++) {
        for (int b = a + 1; b < ngpu; b++) {
            int ok = 0;
            cudaDeviceCanAccessPeer(&ok, a, b);
            float gbps = bench_bidirectional(a, b, 256UL * 1024 * 1024, 5, ok);
            printf("  GPU%d<->GPU%d  %-3s  %7.2f GB/s\n",
                   a, b, ok ? "P2P" : "CPU", gbps);
        }
    }

    // ── round-trip latency ───────────────────────────────────────────────────
    printf("\n[Round-Trip Latency (4-byte ping)]  (200 rounds)\n");
    for (int a = 0; a < ngpu; a++) {
        for (int b = a + 1; b < ngpu; b++) {
            int ok = 0;
            cudaDeviceCanAccessPeer(&ok, a, b);
            float us = bench_latency_us(a, b, ok);
            printf("  GPU%d<->GPU%d  %-3s  %7.1f us\n",
                   a, b, ok ? "P2P" : "CPU", us);
        }
    }

    // ── summary & advice ─────────────────────────────────────────────────────
    printf("\n[Summary & Recommendations]\n");
    for (int a = 0; a < ngpu; a++) {
        for (int b = a + 1; b < ngpu; b++) {
            int ok = 0;
            cudaDeviceCanAccessPeer(&ok, a, b);
            float gbps = bench_unidirectional(a, b, 256UL * 1024 * 1024, 5, ok);
            const char *rating =
                gbps >= 20.0f ? "EXCELLENT (NVLink or Gen4 x16)" :
                gbps >= 10.0f ? "GOOD      (Gen4 x8 / Gen3 x16)" :
                gbps >=  4.0f ? "OK        (Gen4 x4 / Gen3 x8)"  :
                gbps >=  1.0f ? "POOR      (Gen3/4 x1-x2 or CPU bounce)" :
                                "VERY POOR (Gen1/2 x1, check slot!)";
            printf("  GPU%d<->GPU%d  %6.2f GB/s  %s\n", a, b, gbps, rating);
            if (gbps < 1.0f) {
                printf("    ^ Check PCIe slot: lspci -vv | grep -A3 'LnkSta'\n");
                printf("    ^ Confirm both GPUs are in x8+ physical slots\n");
                printf("    ^ If on PCH (Z-series chipset), consider moving to CPU PEG slots\n");
            }
        }
    }

    // disable P2P
    for (int i = 0; i < ngpu; i++) {
        for (int j = 0; j < ngpu; j++) {
            if (i == j) continue;
            int ok = 0;
            cudaDeviceCanAccessPeer(&ok, i, j);
            if (ok) {
                CHK(cudaSetDevice(i));
                cudaDeviceDisablePeerAccess(j);
            }
        }
    }

    printf("\n================================================================================\n");
    return 0;
}
