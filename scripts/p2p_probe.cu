/*
 * p2p_probe.cu
 *
 * Integrity-first CUDA peer-access probe.
 *
 * cudaDeviceCanAccessPeer() is only a capability report.  It can return true
 * on a broken or incorrectly patched driver.  This program enables each
 * directed peer mapping and then launches kernels on one GPU that directly
 * read from and write to memory owned by another GPU.  Exact uint64_t patterns
 * are verified on the host so silent corruption is treated as failure.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kReadSalt = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kWriteSalt = 0xd1b54a32d192ed03ULL;

[[noreturn]] void die_cuda(cudaError_t error, const char* expression,
                           const char* file, int line) {
  std::fprintf(stderr, "CUDA_ERROR file=%s line=%d expression=%s code=%d message=%s\n",
               file, line, expression, static_cast<int>(error),
               cudaGetErrorString(error));
  std::exit(2);
}

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t _error = (expr);                                       \
    if (_error != cudaSuccess) {                                             \
      die_cuda(_error, #expr, __FILE__, __LINE__);                           \
    }                                                                        \
  } while (0)

__host__ __device__ inline std::uint64_t mix64(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

__host__ __device__ inline std::uint64_t source_pattern(std::size_t index,
                                                         int owner) {
  return mix64(static_cast<std::uint64_t>(index) ^
               (static_cast<std::uint64_t>(owner + 1) << 48) ^ kReadSalt);
}

__host__ __device__ inline std::uint64_t read_pattern(std::size_t index,
                                                       int owner,
                                                       int accessor) {
  return source_pattern(index, owner) ^
         mix64(static_cast<std::uint64_t>(accessor + 1) ^ kReadSalt);
}

__host__ __device__ inline std::uint64_t write_pattern(std::size_t index,
                                                        int owner,
                                                        int accessor) {
  return mix64(static_cast<std::uint64_t>(index) ^
               (static_cast<std::uint64_t>(owner + 1) << 40) ^
               (static_cast<std::uint64_t>(accessor + 1) << 52) ^ kWriteSalt);
}

__global__ void peer_read_kernel(const std::uint64_t* remote_owner,
                                 std::uint64_t* local_result,
                                 std::size_t count,
                                 int owner,
                                 int accessor) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    local_result[index] =
        remote_owner[index] ^
        mix64(static_cast<std::uint64_t>(accessor + 1) ^ kReadSalt);
  }
}

__global__ void peer_write_kernel(std::uint64_t* remote_owner,
                                  std::size_t count,
                                  int owner,
                                  int accessor) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    remote_owner[index] = write_pattern(index, owner, accessor);
  }
}

struct Options {
  std::size_t mebibytes = 8;
  int iterations = 3;
  bool require_ampere = false;
};

void usage(const char* argv0) {
  std::fprintf(
      stderr,
      "Usage: %s [--size-mib N] [--iterations N] [--require-ampere]\n",
      argv0);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--size-mib" && i + 1 < argc) {
      options.mebibytes = std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = std::atoi(argv[++i]);
    } else if (arg == "--require-ampere") {
      options.require_ampere = true;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      usage(argv[0]);
      std::exit(64);
    }
  }
  if (options.mebibytes == 0 || options.mebibytes > 1024 ||
      options.iterations < 1 || options.iterations > 100) {
    std::fprintf(stderr, "Invalid probe size or iteration count.\n");
    std::exit(64);
  }
  return options;
}

bool verify_vector(const std::vector<std::uint64_t>& values,
                   int owner,
                   int accessor,
                   bool read_test,
                   std::size_t* bad_index,
                   std::uint64_t* expected,
                   std::uint64_t* actual) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint64_t wanted = read_test
                                     ? read_pattern(i, owner, accessor)
                                     : write_pattern(i, owner, accessor);
    if (values[i] != wanted) {
      *bad_index = i;
      *expected = wanted;
      *actual = values[i];
      return false;
    }
  }
  return true;
}

bool test_direction(int accessor, int owner, std::size_t count, int iterations) {
  int capability = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&capability, accessor, owner));
  if (!capability) {
    std::printf(
        "PAIR accessor=%d owner=%d capability=0 read=SKIP write=SKIP result=FAIL\n",
        accessor, owner);
    return false;
  }

  CUDA_CHECK(cudaSetDevice(accessor));
  cudaError_t enable_error = cudaDeviceEnablePeerAccess(owner, 0);
  if (enable_error == cudaErrorPeerAccessAlreadyEnabled) {
    (void)cudaGetLastError();
  } else if (enable_error != cudaSuccess) {
    std::printf(
        "PAIR accessor=%d owner=%d capability=1 enable=FAIL code=%d message=%s result=FAIL\n",
        accessor, owner, static_cast<int>(enable_error),
        cudaGetErrorString(enable_error));
    return false;
  }

  const std::size_t bytes = count * sizeof(std::uint64_t);
  std::vector<std::uint64_t> host_source(count);
  std::vector<std::uint64_t> host_result(count);
  for (std::size_t i = 0; i < count; ++i) {
    host_source[i] = source_pattern(i, owner);
  }

  std::uint64_t* owner_source = nullptr;
  std::uint64_t* owner_write = nullptr;
  std::uint64_t* accessor_result = nullptr;

  CUDA_CHECK(cudaSetDevice(owner));
  CUDA_CHECK(cudaMalloc(&owner_source, bytes));
  CUDA_CHECK(cudaMalloc(&owner_write, bytes));
  CUDA_CHECK(cudaMemcpy(owner_source, host_source.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(owner_write, 0, bytes));

  CUDA_CHECK(cudaSetDevice(accessor));
  CUDA_CHECK(cudaMalloc(&accessor_result, bytes));
  CUDA_CHECK(cudaMemset(accessor_result, 0, bytes));

  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  bool read_ok = true;
  bool write_ok = true;
  std::size_t bad_index = 0;
  std::uint64_t expected = 0;
  std::uint64_t actual = 0;

  for (int iteration = 0; iteration < iterations && read_ok; ++iteration) {
    CUDA_CHECK(cudaSetDevice(accessor));
    peer_read_kernel<<<blocks, threads>>>(owner_source, accessor_result, count,
                                          owner, accessor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_result.data(), accessor_result, bytes,
                          cudaMemcpyDeviceToHost));
    read_ok = verify_vector(host_result, owner, accessor, true, &bad_index,
                            &expected, &actual);
  }

  for (int iteration = 0; iteration < iterations && write_ok; ++iteration) {
    CUDA_CHECK(cudaSetDevice(owner));
    CUDA_CHECK(cudaMemset(owner_write, 0, bytes));
    CUDA_CHECK(cudaSetDevice(accessor));
    peer_write_kernel<<<blocks, threads>>>(owner_write, count, owner, accessor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(owner));
    CUDA_CHECK(cudaMemcpy(host_result.data(), owner_write, bytes,
                          cudaMemcpyDeviceToHost));
    write_ok = verify_vector(host_result, owner, accessor, false, &bad_index,
                             &expected, &actual);
  }

  CUDA_CHECK(cudaSetDevice(accessor));
  CUDA_CHECK(cudaFree(accessor_result));
  CUDA_CHECK(cudaSetDevice(owner));
  CUDA_CHECK(cudaFree(owner_source));
  CUDA_CHECK(cudaFree(owner_write));
  CUDA_CHECK(cudaSetDevice(accessor));
  const cudaError_t disable_error = cudaDeviceDisablePeerAccess(owner);
  if (disable_error != cudaSuccess &&
      disable_error != cudaErrorPeerAccessNotEnabled) {
    std::fprintf(stderr,
                 "WARNING accessor=%d owner=%d disable_peer=%s\n", accessor,
                 owner, cudaGetErrorString(disable_error));
  } else if (disable_error != cudaSuccess) {
    (void)cudaGetLastError();
  }

  std::printf(
      "PAIR accessor=%d owner=%d capability=1 read=%s write=%s result=%s",
      accessor, owner, read_ok ? "PASS" : "FAIL",
      write_ok ? "PASS" : "FAIL", (read_ok && write_ok) ? "PASS" : "FAIL");
  if (!read_ok || !write_ok) {
    std::printf(" bad_index=%zu expected=0x%016llx actual=0x%016llx",
                bad_index, static_cast<unsigned long long>(expected),
                static_cast<unsigned long long>(actual));
  }
  std::printf("\n");
  return read_ok && write_ok;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  std::printf("P2P_PROBE_VERSION=2\n");
  std::printf("GPU_COUNT=%d\n", device_count);
  std::printf("PROBE_SIZE_MIB=%zu\n", options.mebibytes);
  std::printf("PROBE_ITERATIONS=%d\n", options.iterations);

  if (device_count < 2) {
    std::fprintf(stderr, "At least two visible CUDA devices are required.\n");
    std::printf("RESULT=FAIL\n");
    return 1;
  }

  bool architecture_ok = true;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf(
        "GPU index=%d name=\"%s\" compute=%d.%d unified_addressing=%d\n",
        device, properties.name, properties.major, properties.minor,
        properties.unifiedAddressing);
    if (options.require_ampere && properties.major != 8) {
      architecture_ok = false;
      std::fprintf(stderr,
                   "GPU %d is compute capability %d.%d, not Ampere (8.x).\n",
                   device, properties.major, properties.minor);
    }
    if (!properties.unifiedAddressing) {
      architecture_ok = false;
      std::fprintf(stderr,
                   "GPU %d does not report unified virtual addressing.\n",
                   device);
    }
  }

  const std::size_t bytes = options.mebibytes * 1024ULL * 1024ULL;
  const std::size_t count = bytes / sizeof(std::uint64_t);
  bool all_ok = architecture_ok;

  for (int accessor = 0; accessor < device_count; ++accessor) {
    for (int owner = 0; owner < device_count; ++owner) {
      if (accessor == owner) {
        continue;
      }
      all_ok = test_direction(accessor, owner, count, options.iterations) &&
               all_ok;
    }
  }

  std::printf("RESULT=%s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
