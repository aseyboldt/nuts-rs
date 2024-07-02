use std::{
    collections::VecDeque,
    ops::{RangeBounds, RangeFrom, RangeTo},
};

use arrow::array::StructArray;
use faer::{Col, Mat};
use rand_distr::num_traits::{CheckedNeg, CheckedSub};

use crate::{
    mass_matrix::{DrawGradCollector, MassMatrix},
    mass_matrix_adapt::MassMatrixAdaptStrategy,
    nuts::{AdaptStats, AdaptStrategy, SamplerStats, StatTraceBuilder},
    potential::EuclideanPotential,
    state::State,
    Math,
};

#[derive(Debug)]
struct InnerMatrix<M: Math> {
    vecs: M::EigVectors,
    vals: M::EigValues,
    vals_inv: M::EigValues,
}

#[derive(Debug)]
pub struct LowRankMassMatrix<M: Math> {
    inv_stds: M::Vector,
    store_mass_matrix: bool,
    inner: Option<InnerMatrix<M>>,
}

impl<M: Math> LowRankMassMatrix<M> {
    pub fn new(math: &mut M, store_mass_matrix: bool) -> Self {
        todo!()
    }
}

#[derive(Clone, Debug, Copy, Default)]
pub struct LowRankSettings {
    pub store_mass_matrix: bool,
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrix<M> {
    type Stats = ();
    type Builder = ();

    fn new_builder(&self, settings: &impl crate::Settings, dim: usize) -> Self::Builder {}

    fn current_stats(&self, math: &mut M) -> Self::Stats {}
}

impl<M: Math> MassMatrix<M> for LowRankMassMatrix<M> {
    fn update_velocity(&self, math: &mut M, state: &mut crate::state::InnerState<M>) {
        todo!()
    }

    fn update_kinetic_energy(&self, math: &mut M, state: &mut crate::state::InnerState<M>) {
        todo!()
    }

    fn randomize_momentum<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut crate::state::InnerState<M>,
        rng: &mut R,
    ) {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Stats {}

#[derive(Debug)]
pub struct Builder {}

impl StatTraceBuilder<Stats> for Builder {
    fn append_value(&mut self, value: Stats) {
        let Stats {} = value;
    }

    fn finalize(self) -> Option<StructArray> {
        None
    }

    fn inspect(&self) -> Option<StructArray> {
        None
    }
}

#[derive(Debug)]
pub struct LowRankMassMatrixStrategy {
    draws: VecDeque<Vec<f64>>,
    grads: VecDeque<Vec<f64>>,
    ndim: usize,
}

impl LowRankMassMatrixStrategy {
    pub fn new(ndim: usize) -> Self {
        let draws = VecDeque::with_capacity(100);
        let grads = VecDeque::with_capacity(100);

        Self { draws, grads, ndim }
    }

    pub fn add_draw<M: Math>(&mut self, math: &mut M, state: &State<M>) {
        assert!(math.dim() == self.ndim);
        let mut draw = vec![0f64; self.ndim];
        math.write_to_slice(&state.q, &mut draw);
        let mut grad = vec![0f64; self.ndim];
        math.write_to_slice(&state.grad, &mut grad);

        self.draws.push_back(draw);
        self.grads.push_back(grad);
    }

    pub fn clear(&mut self) {
        self.draws.clear();
        self.grads.clear();
    }

    pub fn update<M: Math>(&mut self, math: &mut M, matrix: &mut LowRankMassMatrix<M>, gamma: f64) {
        let draws_vec = &self.draws;
        let grads_vec = &self.grads;

        let ndraws = draws_vec.len();
        assert!(grads_vec.len() == ndraws);

        let mut draws: Mat<f64> = Mat::zeros(self.ndim, ndraws);
        let mut grads: Mat<f64> = Mat::zeros(self.ndim, ndraws);

        for (i, (draw, grad)) in draws_vec.iter().zip(grads_vec.iter()).enumerate() {
            draws.col_as_slice_mut(i).copy_from_slice(&draw[..]);
            grads.col_as_slice_mut(i).copy_from_slice(&grad[..]);
        }

        // Compute diagonal approximation and transform draws and grads
        let stds = Col::from_fn(self.ndim, |col| {
            let draw_mean = draws.col(1).sum() / (self.ndim as f64);
            let grad_mean = grads.col(col).sum() / (self.ndim as f64);
            let draw_std: f64 = draws
                .col(col)
                .iter()
                .map(|&val| (val - draw_mean) * (val - draw_mean))
                .sum::<f64>()
                .sqrt();
            let grad_std: f64 = grads
                .col(col)
                .iter()
                .map(|&val| (val - grad_mean) * (val - grad_mean))
                .sum::<f64>()
                .sqrt();

            let std = (draw_std / grad_std).sqrt();

            let draw_scale = (std * (ndraws as f64)).recip();
            draws
                .col_mut(col)
                .iter_mut()
                .for_each(|val| *val = (*val - draw_mean) * draw_scale);

            let grad_scale = std * (ndraws as f64).recip();
            grads
                .col_mut(col)
                .iter_mut()
                .for_each(|val| *val = (*val - grad_mean) * grad_scale);

            std
        });

        let mut cov_draws = (&draws) * draws.transpose();
        let mut cov_grads = (&grads) * grads.transpose();

        cov_draws
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .for_each(|x| *x += gamma);
        cov_grads
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .for_each(|x| *x += gamma);

        let (vals, vecs) = compute_matrix_mean(cov_draws, cov_grads);

        todo!()
    }
}

fn compute_matrix_mean(A: Mat<f64>, B_inv: Mat<f64>) -> (Col<f64>, Mat<f64>) {
    todo!()
}

impl<M: Math> AdaptStats<M> for LowRankMassMatrixStrategy {
    fn num_grad_evals(stats: &Self::Stats) -> usize {
        todo!()
    }
}

impl<M: Math> SamplerStats<M> for LowRankMassMatrixStrategy {
    type Stats = Stats;

    type Builder = Builder;

    fn new_builder(&self, _settings: &impl crate::Settings, _dim: usize) -> Self::Builder {
        Builder {}
    }

    fn current_stats(&self, _math: &mut M) -> Self::Stats {
        Stats {}
    }
}

impl<M: Math> AdaptStrategy<M> for LowRankMassMatrixStrategy {
    type Potential = EuclideanPotential<M, LowRankMassMatrix<M>>;

    type Collector = DrawGradCollector<M>;

    type Options = LowRankSettings;

    fn new(math: &mut M, _options: Self::Options, _num_tune: u64) -> Self {
        Self::new(math.dim())
    }

    fn init<R: rand::Rng + ?Sized>(
        &mut self,
        math: &mut M,
        _options: &mut crate::nuts::NutsOptions,
        potential: &mut Self::Potential,
        state: &State<M>,
        _rng: &mut R,
    ) {
        self.add_draw(math, state);
        potential
            .mass_matrix
            .update_scale(math, &state.grad, 1f64, (1e-20, 1e20))
    }

    fn adapt<R: rand::Rng + ?Sized>(
        &mut self,
        _math: &mut M,
        _options: &mut crate::nuts::NutsOptions,
        _potential: &mut Self::Potential,
        _draw: u64,
        _collector: &Self::Collector,
        _state: &State<M>,
        _rng: &mut R,
    ) {
    }

    fn new_collector(&self, math: &mut M) -> Self::Collector {
        DrawGradCollector::new(math)
    }

    fn is_tuning(&self) -> bool {
        unreachable!()
    }
}

impl<M: Math> MassMatrixAdaptStrategy<M> for LowRankMassMatrixStrategy {
    type MassMatrix = LowRankMassMatrix<M>;

    fn update_estimators(&mut self, math: &mut M, collector: &Self::Collector) {
        if collector.is_good {
            let mut draw = vec![0f64; self.ndim];
            math.write_to_slice(&collector.draw, &mut draw);
            self.draws.push_back(draw);

            let mut grad = vec![0f64; self.ndim];
            math.write_to_slice(&collector.grad, &mut grad);
            self.grads.push_back(grad);
        }
    }

    fn switch(&mut self, math: &mut M) {
        todo!()
    }

    fn current_count(&self) -> u64 {
        self.draws.len()
    }

    fn background_count(&self) -> u64 {
        self.draws.len().checked_sub(self.background.start).unwrap() as u64
    }

    fn update_potential(&self, math: &mut M, potential: &mut Self::Potential) -> bool {
        todo!()
    }
}
