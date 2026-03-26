use cust::prelude::*;
use cust::function::{GridSize, BlockSize};
use kernels::{Float, Size, Int, TILE_SIZE};
use rustfft::FftPlanner;
use core::f32;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{error::Error, f32::consts::PI};
use rustfft::num_complex::Complex;
use plotters::prelude::*;

static PTX : &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn gen_signal(frequencies : &[f32], sample_count : usize, sampling_rate : i32) -> Vec<f32>
{
    let mut signal : Vec<f32> = vec![0.0; sample_count];

    for f in frequencies
    {
        for n in 0..sample_count
        {
             let mut res : f32 = 2.0 * PI * (n as f32) * (*f)/ (sampling_rate as f32);
             res = res.sin();
             signal[n] += res;
        }
    }

    signal
}

fn collect_signal(freq_signals: &[f32], sample_count : usize, frequency_count : usize) -> Vec<f32>
{
    let mut signal = vec![0.0; sample_count];

    for i in 0..sample_count
    {
        for f in 0..frequency_count
        {
            let index = f * sample_count + i;
            signal[i] += freq_signals[index];
        }
    }

    signal
}

fn plot_signal(signal: &[f32], sample_rate: usize, filename: &str) -> Result<(), Box<dyn Error>>
{
    let root = BitMapBackend::new(filename, (1024, 512)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_val = signal.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = signal.iter().cloned().fold(f32::INFINITY, f32::min);

    let mut chart = ChartBuilder::on(&root)
        .caption(filename, ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..(signal.len() as f32 / sample_rate as f32), min_val..max_val)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        signal.iter().enumerate().map(|(i, &s)| (i as f32 / sample_rate as f32, s)),
        &BLUE
    ))?;

    root.present()?;

    println!("Plot saved at {:?}", filename);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>>
{
    let _ctx = cust::quick_init();

    let module = Module::from_ptx(PTX, &[])?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // --- BEGIN SIGNAL GENERATION --- //

    let frequencies = vec![440.0, 880.0, 1000.0, 2000.0];
    let f_gpu = frequencies.as_slice().as_dbuf()?;
    
    let sample_count : Size = 512;
    let sc_gpu = sample_count.as_dbox()?;

    let sample_freq: Int = 5000;
    let sf_gpu = sample_freq.as_dbox()?;

    let total_samples = frequencies.len() * sample_count;
    let grid_size = GridSize::x((total_samples / TILE_SIZE) as u32);
    let block_size = BlockSize::x(TILE_SIZE as u32);

    println!("Generating signals with frequencies: {:?},\n\t{:?} samples,\n\tat sampling rate {:?}Hz", frequencies, sample_count, sample_freq);


    // --- CPU-GENERATED SIGNAL --- //
    let start_cpu = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let _cpu_signal = gen_signal(frequencies.as_slice(), sample_count, sample_freq);
    let end_cpu = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

    println!("\n\nCPU Signal took {:?}", (end_cpu - start_cpu));

    // --- GPU-GENERATED SIGNAL (GLOBAL MEMORY) --- //
    let mut output : Vec<Float> = vec![0.0 as Float; frequencies.len() * sample_count];
    let out_gpu = output.as_slice().as_dbuf()?;

    let sig_gen = module.get_function("generate_signal_1d")?;
    let start_global_mem = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    unsafe
    {
        launch!(
            sig_gen<<<grid_size, block_size, 0, stream>>>(
                f_gpu.as_device_ptr(),
                f_gpu.len(),
                sc_gpu.as_device_ptr(),
                sf_gpu.as_device_ptr(),
                out_gpu.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;
    let end_global_mem = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

    out_gpu.copy_to(&mut output)?;

    let _signal = collect_signal(output.as_slice(), sample_count, frequencies.len());

    println!("\n\nGPU Signal w/ Global Memory took: {:?}", (end_global_mem - start_global_mem));

    // --- GPU-GENERATED SIGNAL --- //
    let mut out_shared : Vec<Float> = vec![0.0 as Float; frequencies.len() * sample_count]; 
    let os_gpu = out_shared.as_slice().as_dbuf()?;

    let sig_shared = module.get_function("generate_signal_1d_shared_mem")?;
    let start_shared = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    unsafe
    {
        launch!(
            sig_shared<<<grid_size, block_size, 0, stream>>>(
                f_gpu.as_device_ptr(),
                f_gpu.len(),
                sc_gpu.as_device_ptr(),
                sf_gpu.as_device_ptr(),
                os_gpu.as_device_ptr()
            )
        )?;
    }
    stream.synchronize()?;
    let end_shared = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

    os_gpu.copy_to(&mut out_shared)?;

    let shared_signal = collect_signal(out_shared.as_slice(), sample_count, frequencies.len());
    
    println!("\n\nGPU Signal w/ Shared Memory took: {:?}", (end_shared - start_shared));

    plot_signal(&shared_signal, sample_freq as usize, "before_low_pass.png")?;

    // --- BEGIN FFT LOW PASS FILTER --- //

    let mut buffer : Vec<Complex<f32>> = shared_signal.iter().map(|&s| Complex { re: s, im: 0.0 }).collect();

    let start_fft = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(sample_count);

    fft.process(&mut buffer);

    let freq_cutoff = 500; // Hz - cut off the 1000 and 2000 Hz bands from the original signal
    let cutoff = (freq_cutoff * sample_count) / sample_freq as usize;

    // cut out all frequencies greater than the cutoff
    for i in 0..buffer.len()
    {
        let freq : Size;
        if i <= sample_count / 2
        {
            freq = i;
        } else
        {
            freq = sample_count - i;
        }

        if freq > cutoff
        {
            buffer[i] = Complex{ re: 0.0, im: 0.0 };
        }
    }

    let fft_inv = planner.plan_fft_inverse(sample_count);

    fft_inv.process(&mut buffer);

    for b in buffer.iter_mut()
    {
        b.re /= sample_count as f32;
        b.im /= sample_count as f32;
    }

    let end_fft = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

    println!("\n\nFFT-based Low Pass Filter took {:?}\n\n", (end_fft - start_fft));

    let final_signal : Vec<f32> = buffer.iter().map(|&c| c.re).collect();

    plot_signal(&final_signal, sample_freq as usize, "after_low_pass.png")?;
    Ok(())
}
