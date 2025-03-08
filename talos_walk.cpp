#include "aligator/core/traj-opt-problem.hpp"

#include <Eigen/Core>

#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/context.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/proximal.hpp>

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"
#include "aligator/modelling/multibody/frame-velocity.hpp"
#include "aligator/modelling/multibody/frame-placement.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/costs/quad-state-cost.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/core/enums.hpp"
#include <fmt/ostream.h>
namespace pin = pinocchio;

using aligator::context::TrajOptData;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Model = pin::ModelTpl<double, 0>;
using ProximalSettings = pin::ProximalSettingsTpl<double>;
using MultibodyConstraintFwdDynamics = aligator::dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
using ODEAbstract = aligator::dynamics::ODEAbstractTpl<double>;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using IntegratorSemiImplEuler = aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using StageModel = aligator::StageModelTpl<double>;
using FrameVelocityResidual = aligator::FrameVelocityResidualTpl<double>;
using FramePlacementResidual = aligator::FramePlacementResidualTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;

using aligator::SolverProxDDPTpl;

enum Support
{
    LEFT,
    RIGHT,
    DOUBLE
};

#define EXAMPLE_ROBOT_DATA_MODEL_DIR "/opt/openrobots/share/example-robot-data/robots"

void makeTalosReduced(Model &model_complete, Model &model,
                      Eigen::VectorXd &q0)
{
    const std::string talos_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf";
    const std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf";
    pin::urdf::buildModel(talos_path, pin::JointModelFreeFlyer(), model_complete);
    pin::srdf::loadReferenceConfigurations(model_complete, srdf_path, false);
    Eigen::VectorXd q0_complete = model_complete.referenceConfigurations["half_sitting"];
    std::vector<std::string> controlled_joints_names = {
        "root_joint",
        "leg_left_1_joint",
        "leg_left_2_joint",
        "leg_left_3_joint",
        "leg_left_4_joint",
        "leg_left_5_joint",
        "leg_left_6_joint",
        "leg_right_1_joint",
        "leg_right_2_joint",
        "leg_right_3_joint",
        "leg_right_4_joint",
        "leg_right_5_joint",
        "leg_right_6_joint",
        "torso_1_joint",
        "torso_2_joint",
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        "arm_left_4_joint",
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        "arm_right_4_joint",
    };
    // Check if listed joints belong to model
    for (std::vector<std::string>::const_iterator it =
             controlled_joints_names.begin();
         it != controlled_joints_names.end(); ++it)
    {
        const std::string &joint_name = *it;
        // std::cout << joint_name << std::endl;
        // std::cout << model_complete.getJointId(joint_name) << std::endl;
        if (not(model_complete.existJointName(joint_name)))
        {
            std::cout << "joint: " << joint_name << " does not belong to the model"
                      << std::endl;
        }
    }

    // making list of blocked joints
    std::vector<unsigned long> locked_joints_id;
    for (std::vector<std::string>::const_iterator it =
             model_complete.names.begin() + 1;
         it != model_complete.names.end(); ++it)
    {
        const std::string &joint_name = *it;
        if (std::find(controlled_joints_names.begin(),
                      controlled_joints_names.end(),
                      joint_name) == controlled_joints_names.end())
        {
            locked_joints_id.push_back(model_complete.getJointId(joint_name));
        }
    }
    model = pin::buildReducedModel(model_complete, locked_joints_id, q0_complete);
    q0 = model.referenceConfigurations["half_sitting"];
}

// 设置摆动腿轨迹
void foot_traj(Eigen::Vector3d &translation_init, const int &t_ss,
               const int &ts)
{
    double swing_apex = 0.05;
    translation_init[2] += swing_apex * sin(ts * M_PI / (double)t_ss);
}

// proximal_settings: 跟碰撞有关的设置
IntegratorSemiImplEuler
create_dynamics(MultibodyPhaseSpace &stage_space, Support &support,
                MatrixXd &actuation_matrix, ProximalSettings &proximal_settings,
                std::vector<pin::RigidConstraintModel> &constraint_models)
{
    // 存储着约束模型的vector，因为多点支撑有多个约束模型
    pinocchio::context::RigidConstraintModelVector cms;
    switch (support)
    {
    case LEFT:
        cms.push_back(constraint_models[0]);
        break;
    case RIGHT:
        cms.push_back(constraint_models[1]);
        break;
    case DOUBLE:
        cms.push_back(constraint_models[0]);
        cms.push_back(constraint_models[1]);
        break;
    }

    // Constraint multibody forward dynamics, using Pinocchio
    MultibodyConstraintFwdDynamics ode = MultibodyConstraintFwdDynamics(
        stage_space, actuation_matrix, cms, proximal_settings);
    IntegratorSemiImplEuler dyn_model = IntegratorSemiImplEuler(ode, 0.01);
    return dyn_model;
}

//  T_ss 是单腿支撑的步长，T_ds是双腿支撑的步长
TrajOptProblem defineLocomotionProblem(const std::size_t T_ss,
                                       const std::size_t T_ds)
{
    pin::Model rmodel;
    Eigen::VectorXd q0;
    {
        pin::Model rmodel_complete;
        makeTalosReduced(rmodel_complete, rmodel, q0);
    }
    pin::Data rdata = pin::Data(rmodel);

    // ? 为什么要先进行运动学更新？
    pin::forwardKinematics(rmodel, rdata, q0);
    pin::updateFramePlacements(rmodel, rdata);
    const int nq = rmodel.nq;
    const int nv = rmodel.nv;
    const int nu = nv - 6;

    Eigen::VectorXd x0(nq + nv);
    x0 << q0, Eigen::VectorXd::Zero(rmodel.nv);
    Eigen::VectorXd u0 = Eigen::VectorXd::Zero(nu);

    Eigen::MatrixXd w_x(nv * 2, nv * 2);
    // 状态权重
    w_x.setZero();
    // ? body的orientation是怎么表示的
    w_x.diagonal() << 0, 0, 0, 10000, 10000, 10000, // Base pos/ori
        10, 10, 10, 10, 10, 10,                     // Left leg
        10, 10, 10, 10, 10, 10,                     // Right leg
        1000, 1000,                                 // Torso
        1, 1, 1, 1,                                 // Left arm
        1, 1, 1, 1,                                 // Right arm
        100, 100, 100, 100, 100, 100,               // Base pos/ori vel
        10, 10, 10, 10, 1, 1,                       // Left leg vel
        10, 10, 10, 10, 1, 1,                       // Right leg vel
        1000, 1000,                                 // Torso vel
        10, 10, 10, 10,                             // Left arm vel
        10, 10, 10, 10;                             // Right arm vel

    // 输入权重
    Eigen::MatrixXd w_u(nu, nu);
    w_u.setIdentity();
    w_u *= 1e-3;

    // 摆动腿的操作空间权重
    Eigen::MatrixXd w_LFRF(6, 6);
    w_LFRF.setIdentity();
    w_LFRF *= 10000;

    // 将输入映射到状态
    Eigen::MatrixXd actuation_matrix(nv, nu);
    actuation_matrix.setZero();
    actuation_matrix.bottomRows(nu).setIdentity();

    ProximalSettings prox_settings = ProximalSettings(1e-9, 1e-10, 10);
    std::vector<std::string> foot_frame_name;
    foot_frame_name.push_back("left_sole_link");
    foot_frame_name.push_back("right_sole_link");
    std::vector<std::size_t> foot_frame_ids;
    foot_frame_ids.push_back(rmodel.getFrameId(foot_frame_name[0]));
    foot_frame_ids.push_back(rmodel.getFrameId(foot_frame_name[1]));
    std::vector<std::size_t> foot_joint_ids;
    foot_joint_ids.push_back(rmodel.frames[foot_frame_ids[0]].parentJoint);
    foot_joint_ids.push_back(rmodel.frames[foot_frame_ids[1]].parentJoint);

    // 存在约束的动力学模型，比如人形机器人的向前动力学，存在的约束是某个脚必须固定在地面上
    std::vector<pin::RigidConstraintModel> constraint_models;
    // 构建左脚接触地面的约束模型以及右脚接触地面的约束模型
    for (std::size_t i = 0; i < 2; i++)
    {
        pin::SE3 pl1 = rmodel.frames[foot_frame_ids[i]].placement;
        pin::SE3 pl2 = rdata.oMf[foot_frame_ids[i]];
        // ? 这两个pl的含义是什么？为什么约束模型要传两个joint_id进去？
        // 一个是连接脚的关节的id，一个是世界系原点的id
        // pl1 表示约束相对于连接脚的关节的位置，pl2表示约束相对于世界系原点的位置
        pin::RigidConstraintModel constraint_model = pin::RigidConstraintModel(
            pin::ContactType::CONTACT_6D, rmodel, foot_joint_ids[i], pl1, 0, pl2,
            pin::LOCAL_WORLD_ALIGNED);
        constraint_model.corrector.Kp << 0, 0, 100, 0, 0, 0;
        constraint_model.corrector.Kd << 50, 50, 50, 50, 50, 50;
        constraint_model.name = foot_frame_name[i];
        constraint_models.push_back(constraint_model);
    }

    std::vector<Support> double_phase;
    double_phase.assign(T_ds, Support::DOUBLE);
    std::vector<Support> right_phase;
    right_phase.assign(T_ss, Support::RIGHT);
    std::vector<Support> left_phase;
    left_phase.assign(T_ss, Support::LEFT);
    std::vector<Support> contact_phases;
    // 将double_phase, left_phase, double_phase, right_phase, double_phase依次插入到contact_phases中
    contact_phases.insert(contact_phases.end(), double_phase.begin(),
                          double_phase.end());
    contact_phases.insert(contact_phases.end(), left_phase.begin(),
                          left_phase.end());
    contact_phases.insert(contact_phases.end(), double_phase.begin(),
                          double_phase.end());
    contact_phases.insert(contact_phases.end(), right_phase.begin(),
                          right_phase.end());
    contact_phases.insert(contact_phases.end(), double_phase.begin(),
                          double_phase.end());

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    size_t ts = 0;
    for (std::vector<Support>::iterator phase = contact_phases.begin();
         phase < contact_phases.end(); phase++)
    {
        Support ph = *phase;
        ts += 1;
        // The tangent bundle of a multibody configuration group
        // Any point on the manifold is of the form x=(q,v),
        // where q is a configuration and v is a joint velocity vector.
        auto stage_space = MultibodyPhaseSpace(rmodel);

        auto rcost = CostStack(stage_space, nu);

        rcost.addCost("quad_state", QuadraticStateCost(stage_space, nu, x0, w_x));
        rcost.addCost("quad_control", QuadraticControlCost(stage_space, u0, w_u));
        pin::SE3 LF_placement = rdata.oMf[foot_frame_ids[0]];
        pin::SE3 RF_placement = rdata.oMf[foot_frame_ids[1]];
        // frame 的位置约束，相当于操作空间的约束
        std::shared_ptr<FramePlacementResidual> frame_fn_RF;
        std::shared_ptr<FramePlacementResidual> frame_fn_LF;
        switch (ph)
        {
        case LEFT:
            // RF_placement是期望位置
            foot_traj(RF_placement.translation(), T_ss, ts);
            frame_fn_RF = std::make_shared<FramePlacementResidual>(
                stage_space.ndx(), nu, rmodel, RF_placement, foot_frame_ids[1]);
            rcost.addCost("frame_fn_RF",
                          QuadraticResidualCost(stage_space, *frame_fn_RF, w_LFRF));
            break;
        case RIGHT:
            foot_traj(LF_placement.translation(), T_ss, ts);
            frame_fn_LF = std::make_shared<FramePlacementResidual>(
                stage_space.ndx(), nu, rmodel, LF_placement, foot_frame_ids[0]);
            rcost.addCost("frame_fn_LF",
                          QuadraticResidualCost(stage_space, *frame_fn_LF, w_LFRF));
            break;
        case DOUBLE:
            ts = 0;
            break;
        }
        stage_models.push_back(
            StageModel(rcost, create_dynamics(stage_space, ph, actuation_matrix,
                                              prox_settings, constraint_models)));
    }
    auto ter_space = MultibodyPhaseSpace(rmodel);
    auto term_cost = CostStack(ter_space, nu);
    term_cost.addCost("quad_state", QuadraticStateCost(ter_space, nu, x0, w_x));
    return TrajOptProblem(x0, stage_models, term_cost);
}

constexpr double TOL = 1e-4;
constexpr std::size_t max_iters = 100;
constexpr double mu_init = 1e-8;

int main(int, char **)
{

    std::size_t T_ss = 80;
    std::size_t T_ds = 20;

    std::size_t nsteps = T_ss * 2 + T_ds * 3;

    auto problem = defineLocomotionProblem(T_ss, T_ds);

    SolverProxDDPTpl<double> solver(TOL, mu_init, max_iters, proxsuite::nlp::VERBOSE);
    std::vector<VectorXd> xs_i, us_i;
    Eigen::VectorXd u0 = Eigen::VectorXd::Zero(22);
    xs_i.assign(nsteps + 1, problem.getInitState());
    us_i.assign(nsteps, u0);

    solver.rollout_type_ = aligator::RolloutType::LINEAR;
    solver.sa_strategy_ = aligator::StepAcceptanceStrategy::FILTER;
    solver.filter_.beta_ = 1e-5;
    solver.force_initial_condition_ = true;
    solver.reg_min = 1e-6;
    solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver.setNumThreads(4); // call before setup()
    solver.setup(problem);
    
    auto start = std::chrono::high_resolution_clock::now();
    solver.run(problem, xs_i, us_i);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    fmt::print("Execution time: {} milliseconds\n", duration.count());

    auto &res = solver.results_;
    fmt::print("Results: {}\n", res);
}
