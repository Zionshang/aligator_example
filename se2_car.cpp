#include <iostream>
#include <proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp>

#include <aligator/modelling/dynamics/ode-abstract.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>
#include <aligator/solvers/fddp/solver-fddp.hpp>
#include <aligator/modelling/autodiff/finite-difference.hpp>

using SE2 = proxsuite::nlp::SETpl<2, double>;
ALIGATOR_DYNAMIC_TYPEDEFS(double); // 让后面的Eigen矩阵自动使用double类型

/*
 The dynamics of the car are given by
 x = [x, y, cos(theta), sin(theta)]
 u = [v, omega]
 dot_x = [v * cos(theta), v * sin(theta), omega]
*/
using DynamicsFiniteDifference = aligator::autodiff::DynamicsFiniteDifferenceHelper<double>;

// ODEAbstractTpl is Base class for ODE dynamics dot_x = f(x,u)
struct CarDynamics : aligator::dynamics::ODEAbstractTpl<double>
{

    using Base = aligator::dynamics::ODEAbstractTpl<double>;
    using ODEData = aligator::dynamics::ContinuousDynamicsDataTpl<double>;

    // ODEAbstractTpl 继承自 ContinuousDynamicsAbstractTpl，需要传入 state space 和 control space 的维度
    CarDynamics() : Base(SE2(), 2) {}

    // Evaluate the ODE vector field: this returns the value of dot_x
    void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                 ODEData &data) const override
    {
        // actuation matrix
        double s, c;
        s = std::sin(x[2]);
        c = std::cos(x[2]);
        data.Ju_.col(0) << c, s, 0.; // partial_f/partial_u
        data.Ju_(2, 1) = 1.;

        data.xdot_.noalias() = data.Ju_ * u;
    }

    // Evaluate the vector field Jacobians
    void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                  ODEData &data) const override
    {
        // // Ju_ already computed
        // data.Jx_.setZero(); // partial_f/partial_x
        // double v = u[0];
        // double s, c;
        // s = std::sin(x[2]);
        // c = std::cos(x[2]);

        // data.Jx_.col(2) << -s * v, c * v, 0.;
    }
};

inline auto create_se2_problem(std::size_t nsteps, double timestep)
{
    auto space = SE2();
    const int nu = 2;
    const int ndx = space.ndx(); // ndx就相当于nv

    VectorXs x0(space.nx());
    // 给x0赋值
    {
        double theta = 0.15355;
        pinocchio::SINCOS(theta, &x0[2], &x0[3]);
        x0[0] = 0.7;
        x0[1] = -0.1;
    }
    const VectorXs x_target = space.neutral();

    MatrixXs w_x = MatrixXs::Zero(ndx, ndx);
    w_x.diagonal().array() = 0.01;
    MatrixXs w_term = w_x * 10;
    MatrixXs w_u = MatrixXs::Identity(nu, nu);
    w_u = w_u.transpose() * w_u;

    // Weighted sum of multiple cost components.
    auto rcost = aligator::CostStackTpl<double>(space, nu);
    auto rc1 = aligator::QuadraticStateCostTpl<double>(space, nu, x_target, w_x * timestep);
    auto rc2 = aligator::QuadraticControlCostTpl<double>(space, nu, w_u * timestep);
    rcost.addCost(rc1);
    rcost.addCost(rc2);

    auto ode = CarDynamics(); // xyz::polymorphic<CarDynamics>();
    // Explicit Euler integrator x_{k+1} = x_k + h * f(x_k, u_k).
    // auto discrete_dyn = aligator::dynamics::IntegratorSemiImplEulerTpl<double>(ode, timestep);
    auto discrete_dyn = aligator::dynamics::IntegratorEulerTpl<double>(ode, timestep);
    DynamicsFiniteDifference finite_diff_dyn(space, discrete_dyn, 1e-6);

    // 每一个StageModel都包含了一个cost和一个dynamics
    auto stage = aligator::context::StageModel(rcost, discrete_dyn);
    std::vector<xyz::polymorphic<aligator::context::StageModel>> stages(nsteps, stage);
    auto term_cost = aligator::QuadraticStateCostTpl<double>(space, nu, x_target, w_term);
    // 初始值，所有stage和终端cost
    return aligator::context::TrajOptProblem(x0, stages, term_cost);
}

int main()
{
    size_t nsteps = 40;
    double timestep = 0.05;

    aligator::context::TrajOptProblem problem = create_se2_problem(nsteps, timestep);
    // const double mu_init = 1e-2;
    // aligator::SolverProxDDPTpl<double> solver(1e-4, mu_init);
    // solver.verbose_ = proxsuite::nlp::VERYVERBOSE;
    // solver.sa_strategy_ = aligator::StepAcceptanceStrategy::FILTER;
    // solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    // solver.setNumThreads(4);
    // solver.rollout_type_ = aligator::RolloutType::LINEAR;
    // solver.setup(problem);
    // solver.run(problem);

    // fmt::print("{}\n", fmt::streamed(solver.results_));

    aligator::SolverFDDPTpl<double> fddp_solver(1e-4, aligator::VerboseLevel::VERBOSE);
    fddp_solver.setNumThreads(1);
    fddp_solver.setup(problem);
    fddp_solver.force_initial_condition_ = true;
    fddp_solver.run(problem);
    fmt::print("{}\n", fmt::streamed(fddp_solver.results_));

    return 0;
}
