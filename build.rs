use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main()
{
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");

    let out_dir = path::PathBuf::from(env::var("OUT_DIR").unwrap_or(String::from("./build")));
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or(String::from("./manifests")));

    CudaBuilder::new(manifest_dir.join("kernels"))
        .copy_to(out_dir.join("kernels.ptx"))
        .build()
        .unwrap();
}
