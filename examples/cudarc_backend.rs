use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::{Array, ArrayBuilder, AsArray, FixedSizeListBuilder, PrimitiveBuilder},
    datatypes::{Float64Type, Int64Type, UInt64Type},
};
use candle_core::cuda::CudaError;
use cudarc::{
    curand::{
        result::{create_generator, set_stream, NormalFill},
        sys::curandGenerator_t,
    },
    driver::{
        result::{
            memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async, memset_d8_async,
            stream::synchronize,
        },
        CudaDevice, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, DriverError,
        LaunchAsync, LaunchConfig, PinnedHostSlice,
    },
    nvrtc::CompileOptions,
};
use nuts_rs::{
    DiagGradNutsSettings, DrawStorage, LogpError, Math, Model, Sampler, SamplerWaitResult, Settings,
};
use rand::{rng, thread_rng, Rng};
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;

fn run_cudarc() -> anyhow::Result<()> {
    let device = cudarc::driver::CudaDevice::new_with_stream(0)?;

    let opts = CompileOptions {
        include_paths: vec!["/opt/cuda/include".into()],
        ..Default::default()
    };

    let ptx = cudarc::nvrtc::compile_ptx_with_opts(include_str!("kernels.cu"), opts)?;

    // and dynamically load it into the device
    device.load_ptx(ptx, "math_module", &["sum"])?;

    let N: usize = 10000;

    let sin_kernel = device.get_func("math_module", "sum").unwrap();
    //let cfg = LaunchConfig::for_num_elems(N as u32);
    let cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(8 * 128), 1, 1),
        block_dim: (8 * 128, 1, 1),
        shared_mem_bytes: 0,
    };

    let inp = device.htod_copy(vec![1.0f32; N])?;
    let mut out = device.alloc_zeros::<f32>(N)?;

    for i in 0..10 {
        device.memset_zeros(&mut out)?;
        unsafe { sin_kernel.clone().launch(cfg, (N, &inp, &mut out)) }?;
        let out_host: Vec<f32> = device.dtoh_sync_copy(&out)?;
        assert_eq!(out_host[0], N as f32);
    }
    Ok(())
}

struct CudarcMath {
    device: Arc<CudaDevice>,
    // TODO
    //rng: Arc<CudaRng>,
    rng: curandGenerator_t,
    stream: CudaStream,
    launch_config: LaunchConfig,
    tmp_scalar: CudaSlice<f32>,
    tmp_two: CudaSlice<f32>,
    host_tmp_scalar: PinnedHostSlice<f32>,
    host_tmp_two: PinnedHostSlice<f32>,
    host_tmp_vector: PinnedHostSlice<f32>,
    dim: usize,
}

#[non_exhaustive]
#[derive(Debug, Error)]
enum Error {
    #[error("Failed array operation with candle")]
    CandleError(#[from] candle_core::Error),
    #[error("Invalid logp value")]
    LogpError(f64),
    #[error("Failed to compute the gradient")]
    GradientError,
    #[error("Failed to run cuda function")]
    CudaError(#[from] CudaError),
    #[error("Cuda driver returned an error")]
    DriverError(#[from] DriverError),
}

impl LogpError for Error {
    fn is_recoverable(&self) -> bool {
        if let Error::LogpError(_) = self {
            true
        } else {
            false
        }
    }
}

const ITEMS_PER_THREAD: u32 = 1;
const BLOCK_THREADS: u32 = 256;

impl Math for CudarcMath {
    type Vector = CudaSlice<f32>;
    type EigVectors = ();
    type EigValues = ();
    type LogpErr = Error;
    type Err = Error;
    type TransformParams = ();

    fn new_array(&mut self) -> Self::Vector {
        self.device.alloc_zeros(self.dim).unwrap()
    }

    fn new_eig_vectors<'a>(
        &'a mut self,
        _vals: impl ExactSizeIterator<Item = &'a [f64]>,
    ) -> Self::EigVectors {
        todo!()
    }

    fn new_eig_values(&mut self, _vals: &[f64]) -> Self::EigValues {
        todo!()
    }

    fn logp_array(
        &mut self,
        position: &Self::Vector,
        gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr> {
        let kernel = self.device.get_func("math_module", "normal_logp").unwrap();

        unsafe {
            memset_d8_async(
                *self.tmp_scalar.device_ptr_mut(),
                0,
                size_of::<f32>(),
                self.stream.stream,
            )
            .unwrap();

            kernel
                .launch_on_stream(
                    &self.stream,
                    self.launch_config,
                    (self.dim, position, gradient, &mut self.tmp_scalar),
                )
                .unwrap();
            memcpy_dtoh_async(
                self.host_tmp_scalar.as_mut_slice(),
                *self.tmp_scalar.device_ptr(),
                self.stream.stream,
            )
            .unwrap();
            synchronize(self.stream.stream).unwrap();
        }

        let logp: f32 = self.host_tmp_scalar.as_slice()[0];

        if logp.is_finite() {
            Ok(logp as f64)
        } else {
            Err(Error::LogpError(logp as f64))
        }
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpErr> {
        todo!()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn scalar_prods3(
        &mut self,
        positive1: &Self::Vector,
        negative1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64) {
        let kernel = self
            .device
            .get_func("math_module", "scalar_prods3")
            .unwrap();

        unsafe {
            memset_d8_async(
                *self.tmp_two.device_ptr_mut(),
                0,
                2 * size_of::<f32>(),
                self.stream.stream,
            )
            .unwrap();

            kernel
                .launch_on_stream(
                    &self.stream,
                    self.launch_config,
                    (
                        self.dim,
                        positive1,
                        negative1,
                        positive2,
                        x,
                        y,
                        &mut self.tmp_two,
                    ),
                )
                .unwrap();
            memcpy_dtoh_async(
                self.host_tmp_two.as_mut_slice(),
                *self.tmp_two.device_ptr(),
                self.stream.stream,
            )
            .unwrap();
            synchronize(self.stream.stream).unwrap();
        }

        let vals = self.host_tmp_two.as_slice();
        (vals[0] as f64, vals[1] as f64)
    }

    fn scalar_prods2(
        &mut self,
        positive1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64) {
        let kernel = self
            .device
            .get_func("math_module", "scalar_prods2")
            .unwrap();

        unsafe {
            memset_d8_async(
                *self.tmp_two.device_ptr_mut(),
                0,
                2 * size_of::<f32>(),
                self.stream.stream,
            )
            .unwrap();

            kernel
                .launch_on_stream(
                    &self.stream,
                    self.launch_config,
                    (self.dim, positive1, positive2, x, y, &mut self.tmp_two),
                )
                .unwrap();
            memcpy_dtoh_async(
                self.host_tmp_two.as_mut_slice(),
                *self.tmp_two.device_ptr(),
                self.stream.stream,
            )
            .unwrap();
            synchronize(self.stream.stream).unwrap();
        }

        let vals = self.host_tmp_two.as_slice();
        (vals[0] as f64, vals[1] as f64)
    }

    fn sq_norm_sum(&mut self, x: &Self::Vector, y: &Self::Vector) -> f64 {
        let kernel = self.device.get_func("math_module", "sq_norm_sum").unwrap();

        unsafe {
            memset_d8_async(
                *self.tmp_scalar.device_ptr_mut(),
                0,
                size_of::<f32>(),
                self.stream.stream,
            )
            .unwrap();

            kernel
                .launch_on_stream(
                    &self.stream,
                    self.launch_config,
                    (self.dim, x, y, &mut self.tmp_scalar),
                )
                .unwrap();
            memcpy_dtoh_async(
                self.host_tmp_scalar.as_mut_slice(),
                *self.tmp_scalar.device_ptr(),
                self.stream.stream,
            )
            .unwrap();
            synchronize(self.stream.stream).unwrap();
        }

        self.host_tmp_scalar.as_slice()[0] as f64
    }

    fn read_from_slice(&mut self, dest: &mut Self::Vector, source: &[f64]) {
        source
            .iter()
            .zip(self.host_tmp_vector.as_mut_slice().iter_mut())
            .for_each(|(&src, dest)| {
                *dest = src as f32;
            });

        unsafe {
            memcpy_htod_async(
                *dest.device_ptr_mut(),
                self.host_tmp_vector.as_slice(),
                self.stream.stream,
            )
            .unwrap();
            synchronize(self.stream.stream).unwrap();
        }
    }

    fn write_to_slice(&mut self, source: &Self::Vector, dest: &mut [f64]) {
        unsafe {
            memcpy_dtoh_async(
                self.host_tmp_vector.as_mut_slice(),
                *source.device_ptr(),
                self.stream.stream,
            )
            .unwrap();
            synchronize(self.stream.stream).unwrap();
        }
        dest.iter_mut()
            .zip(self.host_tmp_vector.as_slice().iter())
            .for_each(|(dest, &val)| *dest = val as f64);
    }

    fn eigs_as_array(&mut self, source: &Self::EigValues) -> Box<[f64]> {
        todo!()
    }

    fn copy_into(&mut self, array: &Self::Vector, dest: &mut Self::Vector) {
        assert!(array.len() == dest.len());
        unsafe {
            memcpy_dtod_async(
                *dest.device_ptr_mut(),
                *array.device_ptr(),
                array.len() * size_of::<f32>(),
                self.stream.stream,
            )
            .unwrap();
        }
    }

    fn axpy_out(&mut self, x: &Self::Vector, y: &Self::Vector, a: f64, out: &mut Self::Vector) {
        let kernel = self.device.get_func("math_module", "axpy_out").unwrap();
        unsafe {
            kernel.launch_on_stream(
                &self.stream,
                self.launch_config,
                (self.dim, x, y, a as f32, out),
            )
        }
        .unwrap();
    }

    fn axpy(&mut self, x: &Self::Vector, y: &mut Self::Vector, a: f64) {
        let kernel = self.device.get_func("math_module", "axpy").unwrap();
        unsafe {
            kernel
                .launch_on_stream(&self.stream, self.launch_config, (self.dim, x, y, a as f32))
                .unwrap();
        }
    }

    fn fill_array(&mut self, array: &mut Self::Vector, val: f64) {
        let kernel = self.device.get_func("math_module", "fill_array").unwrap();
        unsafe {
            kernel.launch_on_stream(
                &self.stream,
                self.launch_config,
                (self.dim, array, val as f32),
            )
        }
        .unwrap();
    }

    fn array_all_finite(&mut self, array: &Self::Vector) -> bool {
        // TODO
        true
    }

    fn array_all_finite_and_nonzero(&mut self, array: &Self::Vector) -> bool {
        // TODO
        true
    }

    fn array_mult(
        &mut self,
        array1: &Self::Vector,
        array2: &Self::Vector,
        dest: &mut Self::Vector,
    ) {
        let kernel = self.device.get_func("math_module", "array_mult").unwrap();
        unsafe {
            kernel.launch_on_stream(
                &self.stream,
                self.launch_config,
                (self.dim, array1, array2, dest),
            )
        }
        .unwrap();
    }

    fn array_mult_eigs(
        &mut self,
        stds: &Self::Vector,
        rhs: &Self::Vector,
        dest: &mut Self::Vector,
        vecs: &Self::EigVectors,
        vals: &Self::EigValues,
    ) {
        todo!()
    }

    fn array_vector_dot(&mut self, array1: &Self::Vector, array2: &Self::Vector) -> f64 {
        let kernel = self
            .device
            .get_func("math_module", "array_vector_dot")
            .unwrap();

        unsafe {
            memset_d8_async(
                *self.tmp_scalar.device_ptr_mut(),
                0,
                size_of::<f32>(),
                self.stream.stream,
            )
            .unwrap();
            kernel
                .launch_on_stream(
                    &self.stream,
                    self.launch_config,
                    (self.dim, array1, array2, &mut self.tmp_scalar),
                )
                .unwrap();
            memcpy_dtoh_async(
                self.host_tmp_scalar.as_mut_slice(),
                *self.tmp_scalar.device_ptr(),
                self.stream.stream,
            )
            .unwrap();
            synchronize(self.stream.stream).unwrap();
        };

        self.host_tmp_scalar.as_slice()[0] as f64
    }

    fn array_gaussian<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        stds: &Self::Vector,
    ) {
        // TODO seed!
        unsafe {
            NormalFill::fill(
                self.rng,
                *dest.device_ptr_mut() as *mut f32,
                dest.len(),
                0.0f32,
                1.0f32,
            )
        }
        .unwrap();
    }

    fn array_gaussian_eigs<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        scale: &Self::Vector,
        vals: &Self::EigValues,
        vecs: &Self::EigVectors,
    ) {
        todo!()
    }

    fn array_update_variance(
        &mut self,
        mean: &mut Self::Vector,
        variance: &mut Self::Vector,
        value: &Self::Vector,
        diff_scale: f64,
    ) {
        let kernel = self
            .device
            .get_func("math_module", "array_update_variance")
            .unwrap();
        unsafe {
            kernel.launch_on_stream(
                &self.stream,
                self.launch_config,
                (self.dim, mean, variance, value, diff_scale as f32),
            )
        }
        .unwrap();
    }

    fn array_update_var_inv_std_draw(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        draw_var: &Self::Vector,
        scale: f64,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        /*
        // TODO if zero or not finite..
        // TODO clmap before or after scale?
        let draw_var = draw_var.as_detached_tensor();
        let draw_var = (draw_var * scale).unwrap().clamp(clamp.0, clamp.1).unwrap();
        variance_out.set(&draw_var).unwrap();
        inv_std
            .set(&draw_var.recip().unwrap().sqrt().unwrap())
            .unwrap();
        */
        todo!()
    }

    fn array_update_var_inv_std_draw_grad(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        draw_var: &Self::Vector,
        grad_var: &Self::Vector,
        fill_invalid: Option<f64>,
        clamp: (f64, f64),
    ) {
        let kernel = self
            .device
            .get_func("math_module", "array_update_var_inv_std_draw_grad")
            .unwrap();

        unsafe {
            kernel
                .launch_on_stream(
                    &self.stream,
                    self.launch_config,
                    (
                        self.dim,
                        variance_out,
                        inv_std,
                        draw_var,
                        grad_var,
                        fill_invalid.unwrap_or(-1.0) as f32,
                        clamp.0 as f32,
                        clamp.1 as f32,
                    ),
                )
                .unwrap();
        }
    }

    fn array_update_var_inv_std_grad(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        gradient: &Self::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        let kernel = self
            .device
            .get_func("math_module", "array_update_var_inv_std_grad")
            .unwrap();

        unsafe {
            kernel
                .launch_on_stream(
                    &self.stream,
                    self.launch_config,
                    (
                        self.dim,
                        variance_out,
                        inv_std,
                        gradient,
                        fill_invalid as f32,
                        clamp.0 as f32,
                        clamp.1 as f32,
                    ),
                )
                .unwrap();
        }
    }

    fn inv_transform_normalize(
        &mut self,
        params: &Self::TransformParams,
        untransformed_position: &Self::Vector,
        untransofrmed_gradient: &Self::Vector,
        transformed_position: &mut Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<f64, Self::LogpErr> {
        todo!()
    }

    fn init_from_untransformed_position(
        &mut self,
        params: &Self::TransformParams,
        untransformed_position: &Self::Vector,
        untransformed_gradient: &mut Self::Vector,
        transformed_position: &mut Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<(f64, f64), Self::LogpErr> {
        todo!()
    }

    fn init_from_transformed_position(
        &mut self,
        params: &Self::TransformParams,
        untransformed_position: &mut Self::Vector,
        untransformed_gradient: &mut Self::Vector,
        transformed_position: &Self::Vector,
        transformed_gradient: &mut Self::Vector,
    ) -> Result<(f64, f64), Self::LogpErr> {
        todo!()
    }

    fn update_transformation<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        rng: &mut R,
        untransformed_positions: impl ExactSizeIterator<Item = &'a Self::Vector>,
        untransformed_gradients: impl ExactSizeIterator<Item = &'a Self::Vector>,
        untransformed_logps: impl ExactSizeIterator<Item = &'a f64>,
        params: &'a mut Self::TransformParams,
    ) -> Result<(), Self::LogpErr> {
        todo!()
    }

    fn new_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        untransformed_position: &Self::Vector,
        untransfogmed_gradient: &Self::Vector,
        chain: u64,
    ) -> Result<Self::TransformParams, Self::LogpErr> {
        todo!()
    }

    fn transformation_id(&self, params: &Self::TransformParams) -> Result<i64, Self::LogpErr> {
        todo!()
    }
}

struct Storage {
    draws: FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>,
}

impl Storage {
    fn new(size: usize) -> Storage {
        let values = PrimitiveBuilder::new();
        let draws = FixedSizeListBuilder::new(values, size as i32);
        Storage { draws }
    }
}

impl DrawStorage for Storage {
    fn append_value(&mut self, point: &[f64]) -> anyhow::Result<()> {
        self.draws.values().append_slice(point);
        self.draws.append(true);
        Ok(())
    }

    fn finalize(mut self) -> anyhow::Result<Arc<dyn Array>> {
        Ok(ArrayBuilder::finish(&mut self.draws))
    }

    fn inspect(&self) -> anyhow::Result<Arc<dyn Array>> {
        Ok(ArrayBuilder::finish_cloned(&self.draws))
    }
}

struct NormalModel {
    dim: usize,
    device: Arc<CudaDevice>,
}

impl NormalModel {
    fn new(dim: usize, device: usize) -> anyhow::Result<Self> {
        let device = CudaDevice::new_with_stream(device)?;

        let opts = CompileOptions {
            include_paths: vec!["/opt/cuda/include".into()],
            ..Default::default()
        };

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(include_str!("kernels.cu"), opts)?;
        device.load_ptx(
            ptx,
            "math_module",
            &[
                "sum",
                "normal_logp",
                "scalar_prods3",
                "scalar_prods2",
                "sq_norm_sum",
                "axpy_out",
                "axpy",
                "fill_array",
                "array_mult",
                "array_vector_dot",
                "array_update_variance",
                "array_update_var_inv_std_draw_grad",
                "array_update_var_inv_std_grad",
            ],
        )?;

        Ok(NormalModel { dim, device })
    }
}

impl Model for NormalModel {
    type Math<'model> = CudarcMath;

    type DrawStorage<'model, S: Settings>
        = Storage
    where
        Self: 'model;

    fn new_trace<'model, S: Settings, R: Rng + ?Sized>(
        &'model self,
        _rng: &mut R,
        _chain_id: u64,
        _settings: &'model S,
    ) -> anyhow::Result<Self::DrawStorage<'model, S>> {
        Ok(Storage::new(self.dim))
    }

    fn math(&self) -> anyhow::Result<Self::Math<'_>> {
        let device = self.device.clone();
        let stream = device.fork_default_stream()?;

        let n: u32 = self.dim.try_into().unwrap();
        let launch_config = LaunchConfig {
            grid_dim: (n.div_ceil(ITEMS_PER_THREAD * BLOCK_THREADS), 1, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        //let rng = CudaRng::new(0, device.clone())?;

        let rng = create_generator().unwrap();

        unsafe { set_stream(rng, stream.stream as *mut _).unwrap() };

        let tmp_scalar = device.alloc_zeros::<f32>(1)?;
        let tmp_two = device.alloc_zeros::<f32>(2)?;

        let host_tmp_scalar = unsafe { device.alloc_pinned_noflags(1)? };
        let host_tmp_two = unsafe { device.alloc_pinned_noflags(2)? };
        let host_tmp_vector = unsafe { device.alloc_pinned_noflags(self.dim)? };

        Ok(CudarcMath {
            dim: self.dim,
            device,
            stream,
            rng,
            launch_config,
            tmp_scalar,
            tmp_two,
            host_tmp_scalar,
            host_tmp_two,
            host_tmp_vector,
        })
    }

    fn init_position<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> anyhow::Result<()> {
        let normal = StandardNormal;
        position.iter_mut().for_each(|x| *x = normal.sample(rng));
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let model = NormalModel::new(100000, 0)?;
    let settings = DiagGradNutsSettings {
        seed: 42,
        num_chains: 6,
        maxdepth: 10,
        ..Default::default()
    };

    let mut sampler = Sampler::new(model, settings, 6, None)?;

    let start = Instant::now();

    let trace = loop {
        match sampler.wait_timeout(Duration::from_secs(1)) {
            SamplerWaitResult::Trace(trace) => break trace,
            SamplerWaitResult::Timeout(new_sampler) => sampler = new_sampler,
            SamplerWaitResult::Err(err, _trace) => return Err(err),
        };
    };
    let elapsed = start.elapsed();
    println!("{elapsed:?}");

    let stats = trace.chains[0].stats.clone();
    let stats = stats.as_struct();
    dbg!(stats.column_names());
    let diverging = stats.column_by_name("diverging").unwrap();
    let diverging = diverging.as_boolean();
    dbg!(diverging);

    let n_steps = stats.column_by_name("n_steps").unwrap();
    let n_steps = n_steps.as_primitive::<UInt64Type>();
    dbg!(n_steps);

    let energy = stats.column_by_name("energy").unwrap();
    let energy = energy.as_primitive::<Float64Type>();
    dbg!(energy);

    let logp = stats.column_by_name("logp").unwrap();
    let logp = logp.as_primitive::<Float64Type>();
    dbg!(logp);

    let step_size = stats.column_by_name("step_size").unwrap();
    let step_size = step_size.as_primitive::<Float64Type>();
    dbg!(step_size);

    let draws = trace.chains[0].draws.clone();
    let draws = draws.as_fixed_size_list();
    //dbg!(draws);

    // Some tests

    let model = NormalModel::new(2, 0)?;

    let mut math = model.math()?;
    let mut array1 = math.new_array();
    let mut array2 = math.new_array();
    let mut array3 = math.new_array();

    math.fill_array(&mut array1, 3.0);
    math.fill_array(&mut array2, 5.0);
    math.fill_array(&mut array3, 7.0);

    let mut buffer = vec![1.0f64; math.dim()];
    let mut buffer2 = vec![0.0f64; math.dim()];
    math.read_from_slice(&mut array1, &buffer);
    math.write_to_slice(&array1, &mut buffer2);
    assert!(buffer2[0] == 1.0);

    math.copy_into(&array1, &mut array2);
    math.write_to_slice(&array2, &mut buffer2);
    assert!(buffer2[0] == 1.0);

    buffer2.fill(0.0);
    math.fill_array(&mut array2, 5.0);
    math.write_to_slice(&array2, &mut buffer2);
    assert!(buffer2[0] == 5.0);

    math.fill_array(&mut array1, 3.0);
    math.fill_array(&mut array2, 5.0);
    assert!(math.array_vector_dot(&array1, &array2) == 3.0 * 5.0 * 2.0);

    math.array_mult(&array1, &array2, &mut array3);
    math.write_to_slice(&array3, &mut buffer);
    assert!(buffer[0] == 3.0 * 5.0);

    let logp = math.logp_array(&array1, &mut array2).unwrap();
    assert!(logp == -9.0);
    math.write_to_slice(&array2, &mut buffer);
    assert!(buffer[0] == -3.0);

    let logp = math.logp_array(&array1, &mut array2).unwrap();
    assert!(logp == -9.0);
    math.write_to_slice(&array2, &mut buffer);
    assert!(buffer[0] == -3.0);

    math.fill_array(&mut array1, 3.0);
    math.fill_array(&mut array2, 5.0);
    math.fill_array(&mut array3, 7.0);
    math.axpy(&array1, &mut array2, 11.0);
    math.write_to_slice(&array2, &mut buffer);
    assert!(buffer[0] == 3.0 * 11.0 + 5.0);

    math.fill_array(&mut array1, 3.0);
    math.fill_array(&mut array2, 5.0);
    math.fill_array(&mut array3, 7.0);
    math.axpy_out(&array1, &array2, 11.0, &mut array3);
    math.write_to_slice(&array3, &mut buffer);
    assert!(buffer[0] == 3.0 * 11.0 + 5.0);

    let mut rng = rng();
    math.array_gaussian(&mut rng, &mut array1, &array2);
    math.write_to_slice(&array1, &mut buffer);
    Ok(())
}
