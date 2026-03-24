# Corroded CUDA

This is a small collection of CUDA kernels written in the Rust programming language.
It utilizes the (`rust-cuda`)[https://github.com/Rust-GPU/rust-cuda/tree/main] repository's crates
to allow kernels to be created end-to-end in Rust.

Currently, the project has only followed the guide in the aforementioned `rust-cuda` repository, so
it only has a simple vector sum kernel. More will come soon!

## Dependencies

The `rust-cuda` libraries have some strange dependencies. I've added some tooling to make things easier
to get it all set up.

The only requirement is that you have an Nvidia GPU and the `nvidia-container-toolkit` and Docker installed.

The only step to get the environment running is to run this script:

```
./start_env.sh
```

## Kernels

Currently, there is only one simple vector summation kernel in the repo. More will be coming soon
