# Corroded CUDA

This is a small collection of CUDA kernels written in the Rust programming language.
It utilizes the (`rust-cuda`)[https://github.com/Rust-GPU/rust-cuda/tree/main] repository's crates
to allow kernels to be created end-to-end in Rust.

Currently, the project has followed the guide in the aforementioned `rust-cuda` repository, so
it only has a simple vector sum kernel.

There is also some rudimentary Digital Signal Processing found here. This includes some kernels
written for sinusoidal signal generation, as well as a CPU-driven version to compare GPU algorithm
speeds to. There is also a Low-Pass Filter implementation (that doesn't seem to quite work yet...)

## Dependencies

The `rust-cuda` libraries have some strange dependencies. I've added some tooling to make things easier
to get it all set up.

The only requirement is that you have an Nvidia GPU and the `nvidia-container-toolkit` and Docker installed.

The only step to get the environment running is to run this script:

```
./start_env.sh
```

This will launch you into a new docker container after building it with the correct dependencies. All that's left to
do is run `cargo run`, and the rest should happen automatically.

## Kernels

There are 3 kernels included in this project. The first one, `add`, is a simple vector addition, collecting
the elements of two f32 vectors into a single vector.

The other two are sinusoidal signal generators. They both run considerably faster than the CPU-driven version,
just found in main.rs. The first, `generate_signal_1d`, is a straightforward GPU-driven signal generator, that
uses global memory to write to the output. The second, `generate_signal_1d_shared_mem`, is also GPU-driven, but
it utilizes shared_memory to share the memory across threads in a block. This is an optimization that can greatly
speed up CUDA kernels. In this specific case, it is a little faster, but not much.

