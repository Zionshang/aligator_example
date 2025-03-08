#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <aligator/core/stage-model.hpp>
#include <aligator/modelling/dynamics/kinodynamics-fwd.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/multibody/centroidal-momentum-derivative.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/quad-residual-cost.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/centroidal/centroidal-friction-cone.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include <proxsuite-nlp/modelling/constraints/negative-orthant.hpp>
#include <proxsuite-nlp/modelling/constraints/equality-constraint.hpp>

#include "utils/project_path.hpp"
#include "utils/logger.hpp"

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

using pinocchio::Data;
using pinocchio::FrameIndex;
using pinocchio::Model;
using pinocchio::SE3;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using IntegratorEuler = aligator::dynamics::IntegratorEulerTpl<double>;
using KinodynamicsFwdDynamics = aligator::dynamics::KinodynamicsFwdDynamicsTpl<double>;
using StageModel = aligator::StageModelTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using CentroidalMomentumDerivativeResidual = aligator::CentroidalMomentumDerivativeResidualTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using FrameTranslationResidual = aligator::FrameTranslationResidualTpl<double>;
using CentroidalFrictionConeResidual = aligator::CentroidalFrictionConeResidualTpl<double>;
using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;
using SolverProxDDP = aligator::SolverProxDDPTpl<double>;

struct MPCSettings
{
    
};


// t_ss 为完整步长周期，ts 为当前步长
double ztraj(double swing_apex, double t_ss, double ts)
{
    return swing_apex * std::sin(ts / t_ss * M_PI);
}

double xtraj(double x_forward, double t_ss, double ts)
{
    return x_forward * ts / t_ss;
}

StageModel createStage(MultibodyPhaseSpace space, int nu, Vector3d gravity,
                       std::vector<bool> contact_states,
                       std::vector<FrameIndex> contact_ids,
                       int force_size,
                       VectorXd x0, MatrixXd w_x,
                       VectorXd u0, MatrixXd w_u,
                       MatrixXd w_cent_mom,
                       std::vector<Vector3d> cont_pos, MatrixXd w_trans, double dt,
                       double mu)
{
    const Model &model = space.getModel();
    CostStack cost(space, nu);
    CentroidalMomentumDerivativeResidual cent_mom_deriv(space.ndx(), model, gravity,
                                                        contact_states, contact_ids, force_size);

    cost.addCost(QuadraticStateCost(space, nu, x0, w_x));
    cost.addCost(QuadraticControlCost(space, u0, w_u));
    cost.addCost(QuadraticResidualCost(space, cent_mom_deriv, w_cent_mom));

    for (size_t i = 0; i < cont_pos.size(); i++)
    {
        if (!contact_states[i]) // 只考虑摆动腿轨迹跟踪
        {
            FrameTranslationResidual frame_res(space.ndx(), nu, model, cont_pos[i], contact_ids[i]);
            cost.addCost(QuadraticResidualCost(space, frame_res, w_trans));
        }
    }

    KinodynamicsFwdDynamics ode(space, model, gravity, contact_states, contact_ids, force_size);
    IntegratorEuler dyn_model(ode, dt);
    StageModel stage_model(cost, dyn_model);

    for (size_t i = 0; i < contact_states.size(); i++)
    {
        if (contact_states[i])
        {
            CentroidalFrictionConeResidual friction_residual(space.ndx(), nu, i, mu, 1e-5);
            stage_model.addConstraint(friction_residual, NegativeOrthant());
            FrameTranslationResidual frame_res(space.ndx(), nu, model, cont_pos[i], contact_ids[i]);
            stage_model.addConstraint(frame_res, EqualityConstraint());
        }
    }
    return stage_model;
}

int main(int argc, char const *argv[])
{
    std::string urdf_path = getProjectPath() + "/robot/galileo_mini/robot.urdf";
    Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    Data data(model);

    VectorXd q0(model.nq);
    q0 << 0, 0, 0.38, 0, 0, 0, 1,
        0, 0.72, -1.44,
        0, 0.72, -1.44,
        0, 0.72, -1.44,
        0, 0.72, -1.44;

    const int nq = model.nq;
    const int nv = model.nv;
    const int force_size = 3;
    const int nc = 4;                        // contact number
    const int nu = nv - 6 + nc * force_size; // input number

    MultibodyPhaseSpace space(model);
    const int ndx = space.ndx();

    VectorXd x0(nq + nv);
    x0 << q0, VectorXd::Zero(nv);
    VectorXd u0 = VectorXd::Zero(nu);
    Vector3d com0 = pinocchio::centerOfMass(model, data, x0.head(nq));

    double dt = 20e-3;             // Timestep
    Vector3d gravity(0, 0, -9.81); // Gravity
    double mu = 0.8;               // Friction coefficient
    double mass = pinocchio::computeTotalMass(model);
    Vector3d f_ref(0, 0, -mass * gravity[2] / 4.0); // Initial contact force

    const FrameIndex FL_id = model.getFrameId("FL_foot_link", pinocchio::BODY);
    const FrameIndex FR_id = model.getFrameId("FR_foot_link", pinocchio::BODY);
    const FrameIndex HL_id = model.getFrameId("HL_foot_link", pinocchio::BODY);
    const FrameIndex HR_id = model.getFrameId("HR_foot_link", pinocchio::BODY);

    std::vector<FrameIndex> feet_id = {FL_id, FR_id, HL_id, HR_id};
    pinocchio::forwardKinematics(model, data, q0);
    pinocchio::updateFramePlacements(model, data);
    SE3 FL_pose = data.oMf[FL_id];
    SE3 FR_pose = data.oMf[FR_id];
    SE3 HL_pose = data.oMf[HL_id];
    SE3 HR_pose = data.oMf[HR_id];

    VectorXd wx_diag = VectorXd::Ones(space.ndx()) * 1e-2;
    wx_diag.head(3).setZero();
    MatrixXd w_x = wx_diag.asDiagonal();              // 状态权重
    MatrixXd w_u = 1e-6 * MatrixXd::Identity(nu, nu); // 输入权重
    VectorXd w_trans_diag = VectorXd::Ones(3) * 100;  // 腿的平移权重
    MatrixXd w_trans = w_trans_diag.asDiagonal();
    VectorXd w_cent_mom_diag = VectorXd::Ones(6) * 1e-3;
    MatrixXd w_cent_mom = w_cent_mom_diag.asDiagonal();

    ////////////////////////// 添加生成接触状态与位姿 //////////////////////////////
    std::vector<std::vector<Vector3d>> contact_poses;
    std::vector<std::vector<bool>> contact_states;

    // 初始化当前各足平移
    std::vector<Vector3d> now_trans = {
        FL_pose.translation(), FR_pose.translation(),
        HL_pose.translation(), HR_pose.translation()};

    const int n_qs = 5;  // 离散时刻的全接触支持数量
    const int n_ds = 40; // 离散时刻的双足接触支持数量
    const int steps = 2; // 生成多少组步态

    double swing_apex = 0.05; // 抬腿高度
    double x_forward = 0.2;   // 前进距离

    // 最终生成 steps*(2*n_qs + 2*n_ds) 个离散时刻的接触状态与位姿
    for (int i = 0; i < steps; ++i)
    {
        // 第一阶段：全接触支持
        for (int j = 0; j < n_qs; ++j)
        {
            contact_states.push_back({true, true, true, true});
            contact_poses.push_back(now_trans); // 全接触支持时，各足平移不变
        }
        // 第二阶段：第一组双足接触（例如 swing 第 0 和 3 号足）
        for (int j = 0; j < n_ds; ++j)
        {
            contact_states.push_back({false, true, true, false});
            std::vector<Eigen::Vector3d> new_trans = now_trans;
            new_trans[0](0) = xtraj(x_forward, n_ds, j) + now_trans[0](0);
            new_trans[0](2) = ztraj(swing_apex, n_ds, j) + now_trans[0](2);
            new_trans[3](0) = xtraj(x_forward, n_ds, j) + now_trans[3](0);
            new_trans[3](2) = ztraj(swing_apex, n_ds, j) + now_trans[3](2);
            contact_poses.push_back(new_trans);
            if (j == n_ds - 1)
            {
                now_trans = new_trans;
            }
        }

        // 第三阶段：全接触支持
        for (int j = 0; j < n_qs; ++j)
        {
            contact_states.push_back({true, true, true, true});
            contact_poses.push_back(now_trans);
        }

        // 第四阶段：第二组双足接触（例如 swing 第 1 和 2 号足）
        for (int j = 0; j < n_ds; ++j)
        {
            contact_states.push_back({true, false, false, true});
            std::vector<Eigen::Vector3d> new_trans = now_trans;
            new_trans[1](0) = xtraj(x_forward, n_ds, j) + now_trans[1](0);
            new_trans[1](2) = ztraj(swing_apex, n_ds, j) + now_trans[1](2);
            new_trans[2](0) = xtraj(x_forward, n_ds, j) + now_trans[2](0);
            new_trans[2](2) = ztraj(swing_apex, n_ds, j) + now_trans[2](2);
            contact_poses.push_back(new_trans);
            if (j == n_ds - 1)
            {
                now_trans = new_trans;
            }
        }
    }

    int nsteps = contact_states.size(); // 离散时刻的数量

    CostStack term_cost(space, nu);
    term_cost.addCost(QuadraticStateCost(space, nu, x0, 10 * w_x)); // ? 为什么是x0
    std::vector<xyz::polymorphic<StageModel>> stages;
    for (size_t i = 0; i < nsteps; i++)
    {
        stages.push_back(createStage(space, nu, gravity, contact_states[i], feet_id, force_size,
                                     x0, w_x, u0, w_u, w_cent_mom, contact_poses[i], w_trans, dt, mu));
    }
    TrajOptProblem problem(x0, stages, term_cost);

    double TOL = 1e-5;
    double mu_init = 1e-8;
    size_t max_iters = 100;

    SolverProxDDP solver(TOL, mu_init, max_iters, proxsuite::nlp::VERBOSE);
    solver.rollout_type_ = aligator::RolloutType::LINEAR;
    solver.sa_strategy_ = aligator::StepAcceptanceStrategy::FILTER; // FILTER or LINESEARCH
    solver.force_initial_condition_ = true;
    solver.filter_.beta_ = 1e-5;
    solver.setNumThreads(4);
    solver.setup(problem);

    std::vector<VectorXd> xs_init(nsteps + 1, x0);
    VectorXd u_ref(nu);
    for (int i = 0; i < 4; ++i)
    {
        u_ref.segment(i * force_size, force_size) = f_ref;
    }
    u_ref.segment(4 * force_size, model.nv - 6).setZero();
    std::vector<VectorXd> us_init(nsteps, u_ref);

    solver.run(problem, xs_init, us_init);

    std::vector<VectorXd> xs = solver.results_.xs;

    saveVectorsToCsv("solo_kinodynamics_result.csv", xs);
    return 0;
}
