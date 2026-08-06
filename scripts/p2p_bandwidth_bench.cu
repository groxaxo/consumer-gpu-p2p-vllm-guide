/*
 * p2p_bandwidth_bench.cu
 *
 * Measures the CUDA peer-copy API and an explicit pinned-host bounce baseline.
 * This program does NOT infer the physical transport from cudaMemcpyPeer().
 * Run p2p_probe.cu first: only direct peer kernel reads/writes and CUDA IPC
 * integrity prove that mapped peer access is usable.
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

[[noreturn]] void fail(cudaError_t error, const char* expression, int line) {
  std::fprintf(stderr, "CUDA error line=%d expression=%s: %s\n", line,
               expression, cudaGetErrorString(error));
  std::exit(2);
}

#define CUDA_CHECK(expr)                                                   \
  do {                                                                     \
    const cudaError_t _error = (expr);                                     \
    if (_error != cudaSuccess) fail(_error, #expr, __LINE__);             \
  } while (0)

using Clock = std::chrono::steady_clock;

struct Allocation {
  int device;
  void* pointer;
};

void free_allocation(const Allocation& allocation) {
  CUDA_CHECK(cudaSetDevice(allocation.device));
  CUDA_CHECK(cudaFree(allocation.pointer));
}

double gib_per_second(std::size_t bytes, int iterations,
                      Clock::duration duration) {
  const double seconds = std::chrono::duration<double>(duration).count();
  return static_cast<double>(bytes) * iterations / seconds /
         (1024.0 * 1024.0 * 1024.0);
}

bool verify_copy(int device, const void* device_data,
                 const std::vector<unsigned char>& expected) {
  std::vector<unsigned char> observed(expected.size());
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMemcpy(observed.data(), device_data, observed.size(),
                        cudaMemcpyDeviceToHost));
  return observed == expected;
}

double benchmark_peer_copy(int source, int destination, std::size_t bytes,
                           int iterations, bool* correct) {
  void* source_data = nullptr;
  void* destination_data = nullptr;
  std::vector<unsigned char> pattern(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    pattern[i] = static_cast<unsigned char>((i * 131U + source * 17U) & 0xffU);
  }

  CUDA_CHECK(cudaSetDevice(source));
  CUDA_CHECK(cudaMalloc(&source_data, bytes));
  CUDA_CHECK(cudaMemcpy(source_data, pattern.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaSetDevice(destination));
  CUDA_CHECK(cudaMalloc(&destination_data, bytes));

  CUDA_CHECK(cudaMemcpyPeer(destination_data, destination, source_data, source,
                            bytes));
  CUDA_CHECK(cudaSetDevice(destination));
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto start = Clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    CUDA_CHECK(cudaMemcpyPeer(destination_data, destination, source_data, source,
                              bytes));
  }
  CUDA_CHECK(cudaSetDevice(destination));
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto stop = Clock::now();

  *correct = verify_copy(destination, destination_data, pattern);
  free_allocation({source, source_data});
  free_allocation({destination, destination_data});
  return gib_per_second(bytes, iterations, stop - start);
}

double benchmark_host_bounce(int source, int destination, std::size_t bytes,
                             int iterations, bool* correct) {
  void* source_data = nullptr;
  void* destination_data = nullptr;
  void* host_staging = nullptr;
  std::vector<unsigned char> pattern(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    pattern[i] = static_cast<unsigned char>((i * 193U + source * 29U) & 0xffU);
  }

  CUDA_CHECK(cudaSetDevice(source));
  CUDA_CHECK(cudaMalloc(&source_data, bytes));
  CUDA_CHECK(cudaMemcpy(source_data, pattern.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaSetDevice(destination));
  CUDA_CHECK(cudaMalloc(&destination_data, bytes));
  CUDA_CHECK(cudaMallocHost(&host_staging, bytes));

  CUDA_CHECK(cudaSetDevice(source));
  CUDA_CHECK(cudaMemcpy(host_staging, source_data, bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaSetDevice(destination));
  CUDA_CHECK(cudaMemcpy(destination_data, host_staging, bytes,
                        cudaMemcpyHostToDevice));

  const auto start = Clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    CUDA_CHECK(cudaSetDevice(source));
    CUDA_CHECK(cudaMemcpy(host_staging, source_data, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaSetDevice(destination));
    CUDA_CHECK(cudaMemcpy(destination_data, host_staging, bytes,
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaSetDevice(destination));
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto stop = Clock::now();

  *correct = verify_copy(destination, destination_data, pattern);
  CUDA_CHECK(cudaFreeHost(host_staging));
  free_allocation({source, source_data});
  free_allocation({destination, destination_data});
  return gib_per_second(bytes, iterations, stop - start);
}

void enable_reported_peer_access(int device_count) {
  for (int accessor = 0; accessor < device_count; ++accessor) {
    for (int owner = 0; owner < device_count; ++owner) {
      if (accessor == owner) continue;
      int reported = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&reported, accessor, owner));
      if (!reported) continue;
      CUDA_CHECK(cudaSetDevice(accessor));
      const cudaError_t error = cudaDeviceEnablePeerAccess(owner, 0);
      if (error == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
      } else if (error != cudaSuccess) {
        fail(error, "cudaDeviceEnablePeerAccess", __LINE__);
      }
    }
  }
}

void disable_peer_access(int device_count) {
  for (int accessor = 0; accessor < device_count; ++accessor) {
    for (int owner = 0; owner < device_count; ++owner) {
      if (accessor == owner) continue;
      int reported = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&reported, accessor, owner));
      if (!reported) continue;
      CUDA_CHECK(cudaSetDevice(accessor));
      const cudaError_t error = cudaDeviceDisablePeerAccess(owner);
      if (error == cudaErrorPeerAccessNotEnabled) {
        (void)cudaGetLastError();
      } else if (error != cudaSuccess) {
        fail(error, "cudaDeviceDisablePeerAccess", __LINE__);
      }
    }
  }
}

}  // namespace

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::fprintf(stderr, "At least two visible CUDA devices are required.\n");
    return 1;
  }

  std::printf("CUDA_PEER_COPY_BENCHMARK_VERSION=2\n");
  std::printf("NOTE=cudaMemcpyPeer throughput does not prove direct BAR1 transport\n");
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("GPU index=%d name=\"%s\" compute=%d.%d\n", device,
                properties.name, properties.major, properties.minor);
  }

  std::printf("\nReported capability matrix (query only; not an integrity result)\n");
  for (int source = 0; source < device_count; ++source) {
    for (int destination = 0; destination < device_count; ++destination) {
      if (source == destination) continue;
      int reported = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&reported, source, destination));
      std::printf("CAPABILITY source=%d destination=%d reported=%d\n", source,
                  destination, reported);
    }
  }

  enable_reported_peer_access(device_count);
  const std::size_t sizes_mib[] = {1, 16, 64, 256};
  const int iterations[] = {100, 30, 12, 5};
  bool all_correct = true;

  for (int source = 0; source < device_count; ++source) {
    for (int destination = 0; destination < device_count; ++destination) {
      if (source == destination) continue;
      int reported = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&reported, source, destination));
      if (!reported) {
        std::printf("PAIR source=%d destination=%d result=SKIP reason=capability_false\n",
                    source, destination);
        all_correct = false;
        continue;
      }
      for (std::size_t size_index = 0;
           size_index < sizeof(sizes_mib) / sizeof(sizes_mib[0]);
           ++size_index) {
        const std::size_t bytes = sizes_mib[size_index] * 1024ULL * 1024ULL;
        bool peer_correct = false;
        bool bounce_correct = false;
        const double peer_gib = benchmark_peer_copy(
            source, destination, bytes, iterations[size_index], &peer_correct);
        const double bounce_gib = benchmark_host_bounce(
            source, destination, bytes, iterations[size_index], &bounce_correct);
        all_correct = all_correct && peer_correct && bounce_correct;
        std::printf(
            "BANDWIDTH source=%d destination=%d size_mib=%zu "
            "cuda_peer_copy_gib_s=%.3f host_bounce_gib_s=%.3f "
            "peer_copy_correct=%d host_bounce_correct=%d\n",
            source, destination, sizes_mib[size_index], peer_gib, bounce_gib,
            peer_correct ? 1 : 0, bounce_correct ? 1 : 0);
      }
    }
  }

  disable_peer_access(device_count);
  std::printf("RESULT=%s\n", all_correct ? "PASS" : "FAIL");
  return all_correct ? 0 : 1;
}
