use std::{marker::PhantomData, sync::Arc};

use arrow::{
    array::{
        ArrayBuilder, FixedSizeListBuilder, Float64Builder, Int64Builder, PrimitiveBuilder,
        StructArray,
    },
    datatypes::{DataType, Field, Float64Type, Int64Type},
};

use crate::{
    hamiltonian::{Direction, Hamiltonian, LeapfrogResult, Point},
    sampler_stats::{SamplerStats, StatTraceBuilder},
    state::{State, StatePool},
    DivergenceInfo, LogpError, Math, NutsError, Settings,
};

pub struct TransformedPoint<M: Math> {
    untransformed_position: M::Vector,
    untransformed_gradient: M::Vector,
    transformed_position: M::Vector,
    transformed_gradient: M::Vector,
    velocity: M::Vector,
    index_in_trajectory: i64,
    logp: f64,
    logdet: f64,
    kinetic_energy: f64,
    initial_energy: f64,
    transform_id: i64,
}

pub struct TransformedPointStatsBuilder {
    fisher_distance: Float64Builder,
    transformed_position: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    transformed_gradient: Option<FixedSizeListBuilder<PrimitiveBuilder<Float64Type>>>,
    transformation_index: PrimitiveBuilder<Int64Type>,
}

impl<M: Math> StatTraceBuilder<M, TransformedPoint<M>> for TransformedPointStatsBuilder {
    fn append_value(&mut self, math: Option<&mut M>, value: &TransformedPoint<M>) {
        let math = math.expect("Transformed point stats need math instance");
        let Self {
            fisher_distance,
            transformed_position,
            transformed_gradient,
            transformation_index,
        } = self;

        fisher_distance.append_value(
            math.sq_norm_sum(&value.transformed_position, &value.transformed_gradient),
        );
        transformation_index.append_value(value.transform_id);

        if let Some(store) = transformed_position {
            store
                .values()
                .append_slice(&math.box_array(&value.transformed_position));
            store.append(true);
        }
        if let Some(store) = transformed_gradient {
            store
                .values()
                .append_slice(&math.box_array(&value.transformed_gradient));
            store.append(true);
        }
    }

    fn finalize(self) -> Option<StructArray> {
        let Self {
            mut fisher_distance,
            transformed_position,
            transformed_gradient,
            mut transformation_index,
        } = self;

        let mut fields = vec![
            Field::new("fisher_distance", DataType::Float64, false),
            Field::new("transformation_index", DataType::Int64, false),
        ];
        let mut arrays = vec![
            ArrayBuilder::finish(&mut fisher_distance),
            ArrayBuilder::finish(&mut transformation_index),
        ];

        if let Some(mut transformed_position) = transformed_position {
            let array = ArrayBuilder::finish(&mut transformed_position);
            fields.push(Field::new(
                "transformed_position",
                array.data_type().clone(),
                true,
            ));
            arrays.push(array);
        }

        if let Some(mut transformed_gradient) = transformed_gradient {
            let array = ArrayBuilder::finish(&mut transformed_gradient);
            fields.push(Field::new(
                "transformed_gradient",
                array.data_type().clone(),
                true,
            ));
            arrays.push(array);
        }

        Some(StructArray::new(fields.into(), arrays, None))
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self {
            fisher_distance,
            transformed_position,
            transformed_gradient,
            transformation_index,
        } = self;

        let mut fields = vec![
            Field::new("fisher_distance", DataType::Float64, false),
            Field::new("transformation_index", DataType::Int64, false),
        ];
        let mut arrays = vec![
            ArrayBuilder::finish_cloned(fisher_distance),
            ArrayBuilder::finish_cloned(transformation_index),
        ];

        if let Some(transformed_position) = transformed_position {
            let array = ArrayBuilder::finish_cloned(transformed_position);
            fields.push(Field::new(
                "transformed_position",
                array.data_type().clone(),
                true,
            ));
            arrays.push(array);
        }

        if let Some(transformed_gradient) = transformed_gradient {
            let array = ArrayBuilder::finish_cloned(transformed_gradient);
            fields.push(Field::new(
                "transformed_gradient",
                array.data_type().clone(),
                true,
            ));
            arrays.push(array);
        }

        Some(StructArray::new(fields.into(), arrays, None))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TransformedPointStatsOptions {
    pub store_transformed: bool,
}

impl<M: Math> SamplerStats<M> for TransformedPoint<M> {
    type Builder = TransformedPointStatsBuilder;
    type StatOptions = TransformedPointStatsOptions;

    fn new_builder(
        &self,
        stat_options: Self::StatOptions,
        settings: &impl Settings,
        dim: usize,
    ) -> Self::Builder {
        let count = settings.hint_num_tune() + settings.hint_num_draws();

        let mut transformed_position = None;
        let mut transformed_gradient = None;
        if stat_options.store_transformed {
            let items = PrimitiveBuilder::new();
            transformed_position = Some(FixedSizeListBuilder::new(items, dim as _));
            let items = PrimitiveBuilder::new();
            transformed_gradient = Some(FixedSizeListBuilder::new(items, dim as _));
        }
        TransformedPointStatsBuilder {
            fisher_distance: Float64Builder::with_capacity(count),
            transformation_index: Int64Builder::with_capacity(count),
            transformed_gradient,
            transformed_position,
        }
    }
}

impl<M: Math> TransformedPoint<M> {
    fn first_velocity_halfstep(&self, math: &mut M, out: &mut Self, epsilon: f64) {
        math.axpy_out(
            &self.transformed_gradient,
            &self.velocity,
            epsilon / 2.,
            &mut out.velocity,
        );
    }

    fn position_step(&self, math: &mut M, out: &mut Self, epsilon: f64) {
        math.axpy_out(
            &out.velocity,
            &self.transformed_position,
            epsilon,
            &mut out.transformed_position,
        );
    }

    fn second_velocity_halfstep(&mut self, math: &mut M, epsilon: f64) {
        math.axpy(&self.transformed_gradient, &mut self.velocity, epsilon / 2.);
    }

    fn update_kinetic_energy(&mut self, math: &mut M) {
        self.kinetic_energy = 0.5 * math.array_vector_dot(&self.velocity, &self.velocity);
    }

    fn init_from_untransformed_position(
        &mut self,
        hamiltonian: &TransformedHamiltonian<M>,
        math: &mut M,
    ) -> Result<(), M::LogpErr> {
        let (logp, logdet) = {
            math.init_from_untransformed_position(
                hamiltonian.params.as_ref().expect("No transformation set"),
                &self.untransformed_position,
                &mut self.untransformed_gradient,
                &mut self.transformed_position,
                &mut self.transformed_gradient,
            )
        }?;
        self.logp = logp;
        self.logdet = logdet;
        Ok(())
    }

    fn init_from_transformed_position(
        &mut self,
        hamiltonian: &TransformedHamiltonian<M>,
        math: &mut M,
    ) -> Result<(), M::LogpErr> {
        let (logp, logdet) = {
            math.init_from_transformed_position(
                hamiltonian.params.as_ref().expect("No transformation set"),
                &mut self.untransformed_position,
                &mut self.untransformed_gradient,
                &self.transformed_position,
                &mut self.transformed_gradient,
            )
        }?;
        self.logp = logp;
        self.logdet = logdet;
        Ok(())
    }

    fn is_valid(&self, math: &mut M) -> bool {
        if !math.array_all_finite(&self.transformed_position) {
            return false;
        }
        if !math.array_all_finite_and_nonzero(&self.transformed_gradient) {
            return false;
        }
        if !math.array_all_finite(&self.untransformed_gradient) {
            return false;
        }
        if !math.array_all_finite(&self.untransformed_position) {
            return false;
        }

        true
    }
}

impl<M: Math> Point<M> for TransformedPoint<M> {
    fn position(&self) -> &<M as Math>::Vector {
        &self.untransformed_position
    }

    fn gradient(&self) -> &<M as Math>::Vector {
        &self.untransformed_gradient
    }

    fn index_in_trajectory(&self) -> i64 {
        self.index_in_trajectory
    }

    fn energy(&self) -> f64 {
        self.kinetic_energy - (self.logp + self.logdet)
    }

    fn initial_energy(&self) -> f64 {
        self.initial_energy
    }

    fn logp(&self) -> f64 {
        self.logp
    }

    fn new(math: &mut M) -> Self {
        Self {
            untransformed_position: math.new_array(),
            untransformed_gradient: math.new_array(),
            transformed_position: math.new_array(),
            transformed_gradient: math.new_array(),
            velocity: math.new_array(),
            index_in_trajectory: 0,
            logp: 0f64,
            logdet: 0f64,
            kinetic_energy: 0f64,
            transform_id: -1,
            initial_energy: 0f64,
        }
    }

    fn copy_into(&self, math: &mut M, other: &mut Self) {
        let Self {
            untransformed_position,
            untransformed_gradient,
            transformed_position,
            transformed_gradient,
            velocity,
            index_in_trajectory,
            logp,
            logdet,
            kinetic_energy,
            transform_id,
            initial_energy,
        } = self;

        other.index_in_trajectory = *index_in_trajectory;
        other.logp = *logp;
        other.logdet = *logdet;
        other.kinetic_energy = *kinetic_energy;
        other.transform_id = *transform_id;
        other.initial_energy = *initial_energy;
        math.copy_into(untransformed_position, &mut other.untransformed_position);
        math.copy_into(untransformed_gradient, &mut other.untransformed_gradient);
        math.copy_into(transformed_position, &mut other.transformed_position);
        math.copy_into(transformed_gradient, &mut other.transformed_gradient);
        math.copy_into(velocity, &mut other.velocity);
    }
}

pub struct TransformedHamiltonian<M: Math> {
    ones: M::Vector,
    zeros: M::Vector,
    step_size: f64,
    params: Option<M::TransformParams>,
    max_energy_error: f64,
    _phantom: PhantomData<M>,
    pool: StatePool<M, TransformedPoint<M>>,
}

impl<M: Math> TransformedHamiltonian<M> {
    pub fn new(math: &mut M, max_energy_error: f64) -> Self {
        let mut ones = math.new_array();
        math.fill_array(&mut ones, 1f64);
        let mut zeros = math.new_array();
        math.fill_array(&mut zeros, 0f64);
        let pool = StatePool::new(math, 10);
        Self {
            step_size: 0f64,
            ones,
            zeros,
            params: None,
            max_energy_error,
            _phantom: Default::default(),
            pool,
        }
    }

    pub fn init_transformation<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        math: &mut M,
        position: &[f64],
        chain: u64,
    ) -> Result<(), NutsError> {
        let mut gradient_array = math.new_array();
        let mut position_array = math.new_array();
        math.read_from_slice(&mut position_array, position);
        let _ = math
            .logp_array(&position_array, &mut gradient_array)
            .map_err(|e| NutsError::BadInitGrad(Box::new(e)))?;
        let params = math
            .new_transformation(rng, &position_array, &gradient_array, chain)
            .map_err(|e| NutsError::BadInitGrad(Box::new(e)))?;
        self.params = Some(params);
        Ok(())
    }

    pub fn update_params<'a, R: rand::Rng + ?Sized>(
        &'a mut self,
        math: &'a mut M,
        rng: &mut R,
        draws: impl ExactSizeIterator<Item = &'a M::Vector>,
        grads: impl ExactSizeIterator<Item = &'a M::Vector>,
        logps: impl ExactSizeIterator<Item = &'a f64>,
    ) -> Result<(), NutsError> {
        math.update_transformation(
            rng,
            draws,
            grads,
            logps,
            self.params.as_mut().expect("Transformation was empty"),
        )
        .map_err(|e| NutsError::BadInitGrad(Box::new(e)))?;
        Ok(())
    }
}

pub struct Builder {
    step_size: Float64Builder,
}

impl<M: Math> StatTraceBuilder<M, TransformedHamiltonian<M>> for Builder {
    fn append_value(&mut self, _math: Option<&mut M>, value: &TransformedHamiltonian<M>) {
        let Self { step_size } = self;
        step_size.append_value(value.step_size);
    }

    fn finalize(self) -> Option<StructArray> {
        let Self { mut step_size } = self;

        let fields = vec![Field::new("step_size", DataType::Float64, false)];
        let arrays = vec![ArrayBuilder::finish(&mut step_size)];

        Some(StructArray::new(fields.into(), arrays, None))
    }

    fn inspect(&self) -> Option<StructArray> {
        let Self { step_size } = self;

        let fields = vec![Field::new("step_size", DataType::Float64, false)];
        let arrays = vec![ArrayBuilder::finish_cloned(step_size)];

        Some(StructArray::new(fields.into(), arrays, None))
    }
}

impl<M: Math> SamplerStats<M> for TransformedHamiltonian<M> {
    type Builder = Builder;
    type StatOptions = ();

    fn new_builder(
        &self,
        _stat_options: Self::StatOptions,
        settings: &impl Settings,
        _dim: usize,
    ) -> Self::Builder {
        Builder {
            step_size: Float64Builder::with_capacity(
                settings.hint_num_draws() + settings.hint_num_tune(),
            ),
        }
    }
}

impl<M: Math> Hamiltonian<M> for TransformedHamiltonian<M> {
    type Point = TransformedPoint<M>;

    fn leapfrog<C: crate::nuts::Collector<M, Self::Point>>(
        &mut self,
        math: &mut M,
        start: &State<M, Self::Point>,
        dir: Direction,
        collector: &mut C,
    ) -> LeapfrogResult<M, Self::Point> {
        let mut out = self.pool().new_state(math);
        let out_point = out.try_point_mut().expect("New point has other references");

        out_point.initial_energy = start.point().initial_energy();
        out_point.transform_id = start.point().transform_id;

        let sign = match dir {
            Direction::Forward => 1,
            Direction::Backward => -1,
        };

        let epsilon = (sign as f64) * self.step_size;

        start
            .point()
            .first_velocity_halfstep(math, out_point, epsilon);

        start.point().position_step(math, out_point, epsilon);
        if let Err(logp_error) = out_point.init_from_transformed_position(self, math) {
            if !logp_error.is_recoverable() {
                return LeapfrogResult::Err(logp_error);
            }
            let div_info = DivergenceInfo {
                logp_function_error: Some(Arc::new(Box::new(logp_error))),
                start_location: Some(math.box_array(start.point().position())),
                start_gradient: Some(math.box_array(start.point().gradient())),
                start_momentum: None,
                end_location: None,
                start_idx_in_trajectory: Some(start.point().index_in_trajectory()),
                end_idx_in_trajectory: None,
                energy_error: None,
            };
            collector.register_leapfrog(math, start, &out, Some(&div_info));
            return LeapfrogResult::Divergence(div_info);
        }

        out_point.second_velocity_halfstep(math, epsilon);

        out_point.update_kinetic_energy(math);
        out_point.index_in_trajectory = start.index_in_trajectory() + sign;

        let energy_error = out_point.energy_error();
        if (energy_error > self.max_energy_error) | !energy_error.is_finite() {
            let divergence_info = DivergenceInfo {
                logp_function_error: None,
                start_location: Some(math.box_array(start.point().position())),
                start_gradient: Some(math.box_array(start.point().gradient())),
                end_location: Some(math.box_array(out_point.position())),
                start_momentum: None,
                start_idx_in_trajectory: Some(start.index_in_trajectory()),
                end_idx_in_trajectory: Some(out.index_in_trajectory()),
                energy_error: Some(energy_error),
            };
            collector.register_leapfrog(math, start, &out, Some(&divergence_info));
            return LeapfrogResult::Divergence(divergence_info);
        }

        collector.register_leapfrog(math, start, &out, None);

        LeapfrogResult::Ok(out)
    }

    fn is_turning(
        &self,
        math: &mut M,
        state1: &State<M, Self::Point>,
        state2: &State<M, Self::Point>,
    ) -> bool {
        let (start, end) = if state1.index_in_trajectory() < state2.index_in_trajectory() {
            (state1, state2)
        } else {
            (state2, state1)
        };

        let (turn1, turn2) = math.scalar_prods3(
            &end.point().transformed_position,
            &start.point().transformed_position,
            &self.zeros,
            &start.point().velocity,
            &end.point().velocity,
        );

        (turn1 < 0f64) | (turn2 < 0f64)
    }

    fn init_state(
        &mut self,
        math: &mut M,
        init: &[f64],
    ) -> Result<State<M, Self::Point>, NutsError> {
        let mut state = self.pool().new_state(math);
        let point = state.try_point_mut().expect("State already in use");
        math.read_from_slice(&mut point.untransformed_position, init);

        point
            .init_from_untransformed_position(self, math)
            .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;

        if !point.is_valid(math) {
            Err(NutsError::BadInitGrad(
                anyhow::anyhow!("Invalid initial point").into(),
            ))
        } else {
            Ok(state)
        }
    }

    fn initialize_trajectory<R: rand::Rng + ?Sized>(
        &self,
        math: &mut M,
        state: &mut State<M, Self::Point>,
        rng: &mut R,
    ) -> Result<(), NutsError> {
        let point = state.try_point_mut().expect("State has other references");
        math.array_gaussian(rng, &mut point.velocity, &self.ones);
        let current_transform_id = math
            .transformation_id(self.params.as_ref().expect("No transformation set"))
            .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;
        if current_transform_id != point.transform_id {
            let logdet = math
                .inv_transform_normalize(
                    self.params.as_ref().expect("No transformation set"),
                    &point.untransformed_position,
                    &point.untransformed_gradient,
                    &mut point.transformed_position,
                    &mut point.transformed_gradient,
                )
                .map_err(|e| NutsError::LogpFailure(Box::new(e)))?;
            point.logdet = logdet;
            point.transform_id = current_transform_id;
        }
        point.update_kinetic_energy(math);
        point.index_in_trajectory = 0;
        point.initial_energy = point.energy();
        Ok(())
    }

    fn pool(&mut self) -> &mut StatePool<M, Self::Point> {
        &mut self.pool
    }

    fn copy_state(&mut self, math: &mut M, state: &State<M, Self::Point>) -> State<M, Self::Point> {
        let mut new_state = self.pool.new_state(math);
        state.point().copy_into(
            math,
            new_state
                .try_point_mut()
                .expect("New point should not have other references"),
        );
        new_state
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn step_size_mut(&mut self) -> &mut f64 {
        &mut self.step_size
    }
}
