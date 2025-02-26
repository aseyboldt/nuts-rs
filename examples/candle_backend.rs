use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::{
    array::{Array, ArrayBuilder, FixedSizeListBuilder, PrimitiveBuilder},
    datatypes::Float64Type,
};
use candle_core::{DType, Device, Tensor, Var};
use nuts_rs::{
    DiagGradNutsSettings, DrawStorage, LogpError, Math, Model, Sampler, SamplerWaitResult, Settings,
};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;

struct CandleMath {
    device: Device,
    dim: usize,
    dtype: DType,
}

#[non_exhaustive]
#[derive(Debug, Error)]
enum CandleError {
    #[error("Failed array operation with candle")]
    CandleError(#[from] candle_core::Error),
    #[error("Invalid logp value")]
    LogpError(f64),
    #[error("Failed to compute the gradient")]
    GradientError,
}

impl LogpError for CandleError {
    fn is_recoverable(&self) -> bool {
        if let CandleError::LogpError(_) = self {
            true
        } else {
            false
        }
    }
}

impl Math for CandleMath {
    type Vector = Var;
    type EigVectors = ();
    type EigValues = ();
    type LogpErr = CandleError;
    type Err = CandleError;
    type TransformParams = ();

    fn new_array(&mut self) -> Self::Vector {
        Var::zeros((self.dim,), self.dtype, &self.device).unwrap()
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
        let pos = position.as_tensor();
        let logp = (pos.sqr()?.sum_all()? / 2.0)?.neg().unwrap();
        let grad = logp.backward()?;
        gradient.set(grad.get(&pos).ok_or_else(|| CandleError::GradientError)?)?;
        let logp: f32 = logp.to_scalar()?;
        if logp.is_finite() {
            Ok(logp as f64)
        } else {
            Err(CandleError::LogpError(logp as f64))
        }
    }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::LogpErr> {
        let mut grad = Var::from_slice(gradient, (self.dim,), &self.device)?;
        let pos = Var::from_slice(position, (self.dim,), &self.device)?;
        let logp = self.logp_array(&pos, &mut grad)?;
        gradient.copy_from_slice(&grad.to_vec1()?);
        Ok(logp)
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
        let p1 = positive1.as_detached_tensor();
        let p2 = positive2.as_detached_tensor();
        let p3 = negative1.as_detached_tensor().neg();
        let a = ((p1 + p2).unwrap() + p3).unwrap();
        let x = x.as_detached_tensor();
        let y = y.as_detached_tensor();
        let out1 = (&a * x).unwrap().sum_all().unwrap();
        let out2 = (&a * y).unwrap().sum_all().unwrap();

        let out1: f32 = out1.to_scalar().unwrap();
        let out2: f32 = out2.to_scalar().unwrap();
        (out1 as f64, out2 as f64)
    }

    fn scalar_prods2(
        &mut self,
        positive1: &Self::Vector,
        positive2: &Self::Vector,
        x: &Self::Vector,
        y: &Self::Vector,
    ) -> (f64, f64) {
        let p1 = positive1.as_detached_tensor();
        let p2 = positive2.as_detached_tensor();
        let a = (p1 + p2).unwrap();
        let x = x.as_detached_tensor();
        let y = y.as_detached_tensor();
        let out1 = (&a * x).unwrap().sum_all().unwrap();
        let out2 = (&a * y).unwrap().sum_all().unwrap();

        let out1: f32 = out1.to_scalar().unwrap();
        let out2: f32 = out2.to_scalar().unwrap();
        (out1 as f64, out2 as f64)
    }

    fn sq_norm_sum(&mut self, x: &Self::Vector, y: &Self::Vector) -> f64 {
        let x = x.as_detached_tensor();
        let y = y.as_detached_tensor();
        let out = (x * y).unwrap().sqr().unwrap().sum_all().unwrap();
        let out: f32 = out.to_scalar().unwrap();
        out as f64
    }

    fn read_from_slice(&mut self, dest: &mut Self::Vector, source: &[f64]) {
        let src = Tensor::from_iter(source.iter().map(|&val| val as f32), &self.device).unwrap();
        dest.set(&src).unwrap();
    }

    fn write_to_slice(&mut self, source: &Self::Vector, dest: &mut [f64]) {
        let vals: Vec<f32> = source.to_vec1().unwrap();
        dest.iter_mut()
            .zip(vals.iter())
            .for_each(|(dest, &val)| *dest = val as f64);
    }

    fn eigs_as_array(&mut self, source: &Self::EigValues) -> Box<[f64]> {
        todo!()
    }

    fn copy_into(&mut self, array: &Self::Vector, dest: &mut Self::Vector) {
        dest.set(&array.as_detached_tensor()).unwrap()
    }

    fn axpy_out(&mut self, x: &Self::Vector, y: &Self::Vector, a: f64, out: &mut Self::Vector) {
        let output = (a * x.as_detached_tensor()).unwrap() + y.as_detached_tensor();
        out.set(&output.unwrap()).unwrap();
    }

    fn axpy(&mut self, x: &Self::Vector, y: &mut Self::Vector, a: f64) {
        let output = (a * x.as_detached_tensor()).unwrap() + y.as_detached_tensor();
        y.set(&output.unwrap()).unwrap();
    }

    fn fill_array(&mut self, array: &mut Self::Vector, val: f64) {
        let vals = Tensor::full(val as f32, (self.dim,), &self.device).unwrap();
        array.set(&vals).unwrap();
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
        let out = array1.as_detached_tensor() * array2.as_detached_tensor();
        dest.set(&out.unwrap()).unwrap()
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
        let out = (array1.as_detached_tensor() * array2.as_detached_tensor())
            .unwrap()
            .sum_all()
            .unwrap();
        let out: f32 = out.to_scalar().unwrap();
        out as f64
    }

    fn array_gaussian<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        dest: &mut Self::Vector,
        stds: &Self::Vector,
    ) {
        // TODO
        //self.device.set_seed(rng.next_u64()).unwrap();
        let rand = Tensor::randn(0.0f32, 1.0f32, (self.dim,), &self.device).unwrap();
        let rand = (rand * stds.as_detached_tensor()).unwrap();
        dest.set(&rand).unwrap();
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
        let diff = (mean.as_detached_tensor() - value.as_detached_tensor()).unwrap();
        let scaled = (&diff * diff_scale).unwrap();
        let new_mean = (mean.as_detached_tensor() + scaled).unwrap();
        mean.set(&new_mean).unwrap();
        let new_var = variance.as_detached_tensor() + diff.sqr().unwrap();
        variance.set(&new_var.unwrap()).unwrap();
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
        // TODO if zero or not finite..
        // TODO clmap before or after scale?
        let draw_var = draw_var.as_detached_tensor();
        let draw_var = (draw_var * scale).unwrap().clamp(clamp.0, clamp.1).unwrap();
        variance_out.set(&draw_var).unwrap();
        inv_std
            .set(&draw_var.recip().unwrap().sqrt().unwrap())
            .unwrap();
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
        // TODO if zero or not finite..
        let draw_var = draw_var.as_detached_tensor();
        let grad_var = grad_var.as_detached_tensor();

        let draw_var = (draw_var * grad_var)
            .unwrap()
            .sqrt()
            .unwrap()
            .clamp(clamp.0, clamp.1)
            .unwrap();
        variance_out.set(&draw_var).unwrap();
        inv_std
            .set(&draw_var.recip().unwrap().sqrt().unwrap())
            .unwrap();
    }

    fn array_update_var_inv_std_grad(
        &mut self,
        variance_out: &mut Self::Vector,
        inv_std: &mut Self::Vector,
        gradient: &Self::Vector,
        fill_invalid: f64,
        clamp: (f64, f64),
    ) {
        // TODO if zero or not finite..
        let grad_var = gradient.as_detached_tensor().abs();

        let draw_var = grad_var
            .unwrap()
            .sqrt()
            .unwrap()
            .clamp(clamp.0, clamp.1)
            .unwrap();
        variance_out.set(&draw_var).unwrap();
        inv_std
            .set(&draw_var.recip().unwrap().sqrt().unwrap())
            .unwrap();
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
}

impl NormalModel {
    fn new(dim: usize) -> Self {
        NormalModel { dim }
    }
}

impl Model for NormalModel {
    type Math<'model> = CandleMath;

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
        let device = Device::new_cuda_with_stream(0)?;
        let dtype = DType::F32;
        Ok(CandleMath {
            dim: self.dim,
            device,
            dtype,
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
    let model = NormalModel::new(1000);
    let settings = DiagGradNutsSettings {
        seed: 42,
        num_chains: 6,
        maxdepth: 2,
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
    Ok(())
}
