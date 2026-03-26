use cuda_std::prelude::*;
use cuda_std::GpuFloat;
use cuda_std::thread::sync_threads;

pub type Float = f32;
pub type Int = i32;
pub type Size = usize;

pub static TILE_SIZE : Size = 32;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn add(a: &[Float], b: &[Float], c: *mut Float)
{
    let i = thread::index_1d() as usize;

    if i < a.len() && i < b.len()
    {
        let elem = unsafe { &mut *c.add(i) };
        *elem = a[i] + b[i];
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn generate_signal_1d(frequencies: &[Float], sample_count: &Size, sampling_rate: &Int, out: *mut Float)
{
    let global_index = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as Size;

    if global_index < frequencies.len() * sample_count 
    {
        let index = global_index % *sample_count;
        let freq = global_index / *sample_count;

        const PI : f32 = 3.141529;

        let f = frequencies[freq];


        let mut res : f32 = 2.0 * PI * (index as f32) * f / (*sampling_rate as f32);
        res = res.sin();

        let global_elem = unsafe { &mut *out.add(global_index) };
        *global_elem = res;
    }
}

/**
 * Shared memory greatly reduces latency for IO operations to GPU memory, so by using shared memory
 * tiles as we do in this kernel, we can hope to see a nice increase in speed
 */
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn generate_signal_1d_shared_mem(frequencies: &[Float], sample_count: &Size, sampling_rate: &Int, out: *mut Float)
{
    let global_index = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as Size;
    
    let shared_tile : *mut Float = cuda_std::shared::dynamic_shared_mem::<Float>();

    if global_index < frequencies.len() * sample_count 
    {
        let index = global_index % *sample_count;
        let freq = global_index / *sample_count;

        const PI : f32 = 3.141529;

        let f = frequencies[freq];


        let mut res : f32 = 2.0 * PI * (index as f32) * f / (*sampling_rate as f32);
        res = res.sin();

        let tile_elem = unsafe { &mut *shared_tile.add(thread::thread_idx_x() as usize) };
        *tile_elem = res;
    }


    sync_threads();

    if global_index < frequencies.len() * sample_count
    {
        let tile_elem = unsafe { &mut *shared_tile.add(thread::thread_idx_x() as usize) };
        let global_elem = unsafe { &mut *out.add(global_index) };
        *global_elem = *tile_elem;
    }
}
